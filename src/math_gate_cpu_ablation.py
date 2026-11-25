"""Small CPU gate ablations using verified, already computed text embeddings."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from .neural_policy import build_policy_text
from .train_fixed_math_policy import (audit_split, best_action, decision_metrics,
                                      digest, label_metrics, write_json)
from .utils import load_pickle, save_pickle

RECIPES = (
    {'name': 'small_unweighted', 'hidden': 16, 'dropout': .2, 'weight_decay': .01, 'balance': False},
    {'name': 'small_sqrt_weighted', 'hidden': 16, 'dropout': .2, 'weight_decay': .01, 'balance': True},
    {'name': 'linear_sqrt_weighted', 'hidden': 0, 'dropout': 0., 'weight_decay': .01, 'balance': True},
)
BUDGETS = (.05, .10, .20)
GRID = tuple(np.linspace(0, .2, 21)) + (.3, .5, 1.)


def read_cached_features(cache, rows, expected_fingerprint):
    manifest = json.loads((cache / 'manifest.json').read_text())
    fingerprint = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    if fingerprint != expected_fingerprint:
        raise ValueError('Feature manifest fingerprint mismatch')
    if manifest['ids'] != [r['id'] for r in rows]:
        raise ValueError('Feature row order or IDs changed')
    hashes = [hashlib.sha256(build_policy_text(r, max_text_chars=0, include_state=False).encode()).hexdigest()
              for r in rows]
    if hashes != manifest['text_sha256']:
        raise ValueError('Question/initial-answer text differs from the cached encoder input')
    if manifest['layers'] != [-2, -1] or manifest['pooling'] != 'mean_last':
        raise ValueError('Unexpected encoder layers or pooling')
    if digest(Path(__file__).with_name('neural_policy.py')) != manifest['neural_policy_sha256']:
        raise ValueError('Policy input/inference implementation changed since caching')
    completion = json.loads((cache / 'complete.json').read_text())
    if completion['fingerprint'] != fingerprint:
        raise ValueError('Feature extraction was not completed for this fingerprint')
    parts, next_start = [], 0
    for path in sorted(cache.glob('*.npy')):
        start, end = map(int, path.stem.split('_'))
        x = np.load(path, allow_pickle=False)
        if start != next_start or end <= start or x.shape != (end - start, 4096) or not np.isfinite(x).all():
            raise ValueError(f'Invalid/missing feature shard: {path}')
        parts.append(x)
        next_start = end
    if next_start != len(rows):
        raise ValueError('Incomplete feature row coverage')
    return np.concatenate(parts).astype('float32'), completion


def score_model(bundle, features):
    if list(bundle['action_labels']) not in (['direct', 'reflect_1'], ['reflect_1', 'reflect_2']):
        raise ValueError('Unexpected classifier label order')
    model = bundle['model'].cpu().eval()
    transformed = bundle['scaler'].transform(features).astype('float32')
    if not np.isfinite(transformed).all():
        raise ValueError('Nonfinite standardized features')
    with torch.inference_mode():
        return model(torch.from_numpy(transformed)).softmax(1).numpy()


def probability_thresholds(direct_prob, grid=GRID):
    """Convert label-free call quantiles to deployable thresholds, keeping ties intact."""
    p = np.asarray(direct_prob, dtype=np.float32)
    if p.ndim != 1 or not len(p) or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError('Invalid probabilities')
    values, counts = np.unique(p, return_counts=True)
    totals = counts.cumsum()
    thresholds = {0.}
    for budget in grid:
        allowed = np.flatnonzero(totals <= int(np.floor(len(p) * float(budget) + 1e-9)))
        if len(allowed):
            # nextafter in float32 matches numpy inference and the stored threshold exactly.
            thresholds.add(float(np.nextafter(values[allowed[-1]], np.float32(np.inf))))
    return sorted(thresholds)


def select_budget_point(points, budget, strong_cost):
    feasible = [p for p in points if p['metrics']['reflection_rate'] <= budget + 1e-12
                and p['metrics']['avg_token_cost'] < strong_cost]
    if not feasible:
        return None
    return max(feasible, key=lambda p: (p['metrics']['correct'], -p['metrics']['avg_token_cost'],
                                        -p['metrics']['reflection_rate'], -p['direct_threshold']))


def mild_weights(labels, enabled):
    counts = np.bincount(np.asarray(labels), minlength=2)
    if not counts.all():
        raise ValueError('Gate training requires both classes')
    return torch.tensor([1., min(5., float(np.sqrt(counts[0] / counts[1]))) if enabled else 1.])


def build_model(recipe):
    if recipe['hidden']:
        return torch.nn.Sequential(torch.nn.Linear(4096, recipe['hidden']), torch.nn.ReLU(),
                                   torch.nn.Dropout(recipe['dropout']), torch.nn.Linear(recipe['hidden'], 2))
    return torch.nn.Sequential(torch.nn.Linear(4096, 2))


def classification(model, x, vx, y, vy):
    model.eval()
    with torch.inference_mode():
        tr, va = model(x), model(vx)
        probabilities = va.softmax(1).numpy()
        score = probabilities[:, 1]
        result = {'train_loss': float(torch.nn.functional.cross_entropy(tr, y)),
                  'validation_loss': float(torch.nn.functional.cross_entropy(va, vy)),
                  'train': label_metrics(y.numpy(), tr.argmax(1).numpy()),
                  'validation': label_metrics(vy.numpy(), va.argmax(1).numpy()),
                  'validation_roc_auc': float(roc_auc_score(vy.numpy(), score)),
                  'validation_average_precision': float(average_precision_score(vy.numpy(), score))}
    return result, probabilities[:, 0]


def plot_history(history, destination, baselines):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    epochs = [h['epoch'] for h in history]
    for name in ('train_loss', 'validation_loss'):
        axes[0, 0].plot(epochs, [h['classification'][name] for h in history], label=name)
    axes[0, 1].plot(epochs, [h['classification']['validation_average_precision'] for h in history], label='AP')
    axes[0, 1].axhline(history[0]['positive_rate'], linestyle='--', label='Positive prevalence')
    for branch in ('original_round', 'fixed_r1'):
        for budget in BUDGETS:
            key = f'{branch}_{budget:.2f}'
            metrics = [h['selected'][key] for h in history]
            axes[1, 0].plot(epochs, [m['metrics']['accuracy'] if m else np.nan for m in metrics], label=key)
            axes[1, 1].plot(epochs, [m['metrics']['avg_token_cost'] if m else np.nan for m in metrics], label=key)
    for name in ('direct', 'independent'):
        axes[1, 0].axhline(baselines[name]['accuracy'], linestyle='--', label=name)
        axes[1, 1].axhline(baselines[name]['avg_token_cost'], linestyle='--', label=name)
    for ax, title in zip(axes.flat, ['Gate cross-entropy (unweighted reporting)', 'Validation positive-class ranking',
                                    'Validation final answer accuracy', 'Validation input + output tokens']):
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def run_recipe(name, rows, original_rounds, direct_provider, epochs, directory, args,
               baselines, bundle_provider, audit):
    directory.mkdir()
    history, best, best_keys = [], {}, {}
    branches = {'original_round': original_rounds, 'fixed_r1': np.full(len(rows), 'reflect_1')}
    for epoch in range(1, epochs + 1):
        cls, direct = direct_provider(epoch)
        thresholds = probability_thresholds(direct)
        entry = {'epoch': epoch, 'classification': cls, 'selected': {},
                 'positive_rate': sum(r['best_action'] != 'direct' for r in rows) / len(rows), 'curves': {}}
        for branch, round_actions in branches.items():
            points = []
            for threshold in thresholds:
                actions = np.where(direct >= threshold, 'direct', round_actions).tolist()
                metrics = decision_metrics(rows, actions, args.lambda_cost)
                points.append({'direct_threshold': threshold, 'metrics': metrics})
            entry['curves'][branch] = points
            for budget in BUDGETS:
                key = f'{branch}_{budget:.2f}'
                point = select_budget_point(points, budget, baselines['independent']['avg_token_cost'])
                entry['selected'][key] = point
                if point is None:
                    continue
                m = point['metrics']
                order = (m['correct'], -m['avg_token_cost'], -m['reflection_rate'])
                if key not in best_keys or order > best_keys[key]:
                    best_keys[key] = order
                    checkpoint = directory / f'gate_{key}.pkl'
                    bundle = dict(bundle_provider())
                    bundle.update(direct_threshold=point['direct_threshold'], best_epoch=epoch,
                                  fixed_split_audit=audit, device='cpu',
                                  selection_metric='validation_answer_accuracy_with_call_and_token_caps',
                                  deployment_branch=branch, validation_call_budget=budget,
                                  selection_scope='Exploratory internal validation, not independent testing')
                    save_pickle(bundle, checkpoint)
                    best[key] = {'recipe': name, 'branch': branch, 'budget': budget, 'epoch': epoch,
                                 'gate_model': str(checkpoint), **point,
                                 'accuracy_gap_to_strong_pp': (baselines['independent']['accuracy'] - m['accuracy']) * 100,
                                 'token_savings_vs_strong': 1 - m['avg_token_cost'] / baselines['independent']['avg_token_cost']}
        history.append(entry)
        write_json(directory / 'history.json', history)
        write_json(directory / 'selected_points.json', best)
        if cls:
            plot_history(history, directory / f'epoch_{epoch:02d}.png', baselines)
            plot_history(history, directory / 'training_curve.png', baselines)
        print(f'[ablation] {name} epoch={epoch}/{epochs} ' + ' | '.join(
            f"{k}: acc={v['metrics']['accuracy']:.4f} tokens={v['metrics']['avg_token_cost']:.1f}"
            for k, v in entry['selected'].items() if v is not None), flush=True)
    return list(best.values())


def verify_exports(points, features, rows, rounds, baselines, args):
    for p in points:
        b = load_pickle(p['gate_model'])
        direct = score_model(b, features)[:, 0]
        reflection = rounds if p['branch'] == 'original_round' else np.full(len(rows), 'reflect_1')
        actions = np.where(direct >= b['direct_threshold'], 'direct', reflection).tolist()
        observed = decision_metrics(rows, actions, args.lambda_cost)
        if observed != p['metrics']:
            raise ValueError(f"Exported checkpoint differs from reported decisions: {p['gate_model']}")
        if observed['reflection_rate'] > p['budget'] + 1e-12 or observed['avg_token_cost'] >= baselines['independent']['avg_token_cost']:
            raise ValueError('Exported policy violates validation budget')


def report(output, baselines, points, audit):
    lines = ['# CPU 门控对照：固定内部验证集', '',
             '**这是开发对照，不是独立测试结论。用户暂不设置精度容差，因此不宣布达标或选择论文主模型。**', '',
             f"训练 {audit['train']['observed']} 题，验证 {audit['validation']['observed']} 题；原失败题继续保留。",
             '仅复用冻结纯文本向量，无新模型编码、GPU 或生成调用。原 Round 固定；fixed_r1 是轮次消融，不改变生成轨迹。', '',
             '## 基线', '', '| 方法 | 准确率 | 总 token | 反思率 |', '| --- | ---: | ---: | ---: |']
    for name, m in baselines.items():
        lines.append(f"| {name} | {m['accuracy']:.2%} | {m['avg_token_cost']:.2f} | {m['reflection_rate']:.2%} |")
    lines += ['', '## 各配方与预算', '',
              '| Gate | 反思分支 | 预算 | epoch | 准确率 | 与强模型差(pp) | 输入 token | 输出 token | 总 token | 节省 token | 实际调用率 | 修复/引错 |',
              '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for p in points:
        m = p['metrics']
        lines.append(f"| {p['recipe']} | {p['branch']} | {p['budget']:.0%} | {p['epoch']} | {m['accuracy']:.2%} | "
                     f"{p['accuracy_gap_to_strong_pp']:.2f} | {m['avg_prompt_tokens']:.2f} | {m['avg_completion_tokens']:.2f} | "
                     f"{m['avg_token_cost']:.2f} | {p['token_savings_vs_strong']:.2%} | {m['reflection_rate']:.2%} | {m['fixes']}/{m['harms']} |")
    lines += ['', '## 解释边界', '',
              '- epoch 与阈值均在同一固定内部验证集选择，不能把这些数字作为显著性或非劣结论。',
              '- Gate 标签始终是完整动作集合的准确率优先标签；fixed_r1 分支没有另外训练 Gate。',
              '- 调用预算用不依赖标签的分数分位点生成逐题阈值，同分整组处理；验证预算不保证新分布部署调用率不超限。',
              '- 仅当验证平均 token 低于强模型独立回答且不超过调用预算时，才在可行点中按最终正确题数优先、成本次之选取；无可行点则不导出。',
              '- 不设精度容差，因此“预算可行”不等于“精度接近强模型”。没有默认部署模型，也没有覆盖原检查点。',
              '- 标准化只拟合训练题；没有人工状态特征、金答案或真实候选得分进入向量输入。',
              '- 缺失反思的正确训练题不补造结果；本次不报告全训练集系统准确率或错误引入率。',
              '- 原 Round 的训练条件分布偏差仍存在；本实验固定该组件以隔离 Gate 改动，并通过固定 R1 分支检验第二轮贡献。',
              '- token 包含已有执行路径的输入与输出；不等于实测延迟或跨模型统一 FLOPs/费用。',
              '- 所有导出头已用缓存逐题回放验证；各配方 history.json 和 epoch_XX.png 保留每轮结果及曲线。', '']
    (output / 'REPORT.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('outputs/fixed_math_policy_cpu_20260915_151132_seed42'))
    parser.add_argument('--root', type=Path, default=Path('outputs/math_full_training_v1'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lambda-cost', type=float, default=.0001)
    args = parser.parse_args()
    if args.threads < 1 or args.epochs < 1:
        parser.error('Threads and epochs must be positive')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    torch.set_num_threads(args.threads)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'run.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (args.output / 'protocol.json').exists():
            raise ValueError('Choose a new output directory; existing experiments are never overwritten')
        train, val, audit = audit_split(args.root, allow_partial=True)
        prior = json.loads((args.source / 'selected_validation.json').read_text())
        if prior['coverage'] != audit:
            raise ValueError('Data coverage or fixed split changed from the source experiment')
        protocol = {'recipes': RECIPES, 'epochs': args.epochs, 'seed': args.seed, 'learning_rate': .0003,
                    'batch_size': 128, 'positive_weight': 'min(5, sqrt(train_neg/train_pos))',
                    'gate_targets': 'best_action != direct; no labels changed between branches',
                    'call_budgets': BUDGETS, 'score_quantile_grid': GRID,
                    'cost_constraint': 'validation avg total tokens strictly less than strong independent baseline',
                    'accuracy_tolerance': None, 'main_model_selection': False,
                    'selection': 'per recipe/branch/budget: final correct count, cost, call rate, earliest epoch',
                    'fixed_split': audit, 'source': str(args.source), 'source_hashes': {
                        name: digest(args.source / name) for name in ('gate.pkl', 'round.pkl', 'selected_validation.json')},
                    'feature_fingerprint': prior['encoder']['fingerprint'],
                    'scope': 'Exploratory internal validation. No test evaluation or claims of noninferiority.',
                    'source_code_sha256': digest(__file__)}
        write_json(args.output / 'protocol.json', protocol)
        write_json(args.output / 'status.json', {'stage': 'checking_cache', 'test_evaluated': False})
        try:
            features, info = read_cached_features(Path(prior['encoder']['cache']), train + val, prior['encoder']['fingerprint'])
            print(f'[cache] verified and reused {len(features)} rows; encoder not loaded', flush=True)
            n = len(train)
            vf = features[n:]
            gate = load_pickle(args.source / 'gate.pkl')
            rounds = load_pickle(args.source / 'round.pkl')
            save_pickle(rounds, args.output / 'reference_round.pkl')
            round_actions = np.array(rounds['action_labels'])[score_model(rounds, vf).argmax(1)]
            original_direct = score_model(gate, vf)[:, 0]
            original_actions = np.where(original_direct >= gate['direct_threshold'], 'direct', round_actions).tolist()
            if original_actions != [r['action'] for r in prior['predictions']]:
                raise ValueError('Original checkpoint replay failed')
            baselines = {name: decision_metrics(val, [a] * len(val), args.lambda_cost) for name, a in
                         [('direct', 'direct'), ('fixed_reflect_1', 'reflect_1'), ('fixed_reflect_2', 'reflect_2'), ('independent', 'independent')]}
            baselines['oracle_accuracy_first'] = decision_metrics(val, [best_action(r) for r in val], args.lambda_cost)
            baselines['original_policy_unmodified'] = decision_metrics(val, original_actions, args.lambda_cost)
            write_json(args.output / 'baselines.json', baselines)
            points = run_recipe('original_gate_recalibrated', val, round_actions, lambda epoch: ({}, original_direct),
                                1, args.output / 'original_gate_recalibrated', args, baselines, lambda: gate, audit)
            # The same train-only scaler and feature input are used by all new gate recipes.
            scaler = StandardScaler().fit(features[:n])
            x = torch.from_numpy(scaler.transform(features[:n]).astype('float32'))
            vx = torch.from_numpy(scaler.transform(vf).astype('float32'))
            if not torch.isfinite(x).all() or not torch.isfinite(vx).all():
                raise ValueError('Nonfinite scaled feature')
            y = torch.tensor([r['best_action'] != 'direct' for r in train], dtype=torch.long)
            vy = torch.tensor([r['best_action'] != 'direct' for r in val], dtype=torch.long)
            for recipe in RECIPES:
                torch.manual_seed(args.seed)
                np.random.seed(args.seed)
                model = build_model(recipe).cpu()
                optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=recipe['weight_decay'])
                criterion = torch.nn.CrossEntropyLoss(weight=mild_weights(y.numpy(), recipe['balance']))
                write_json(args.output / 'status.json', {'stage': 'training_cpu', 'recipe': recipe['name'], 'test_evaluated': False})

                def step(epoch):
                    model.train()
                    for indices in torch.randperm(n).split(128):
                        optimizer.zero_grad()
                        criterion(model(x[indices]), y[indices]).backward()
                        optimizer.step()
                    return classification(model, x, vx, y, vy)

                def bundle():
                    return {'policy_type': 'torch_embedding_mlp', 'backend': 'hf_hidden_torch_mlp',
                            'model': model, 'scaler': scaler, 'encoder_info': info,
                            'encoder_name': gate['encoder_name'], 'layers': [-2, -1], 'pooling': 'mean_last',
                            'pure_text': True, 'max_text_chars': 0, 'max_length': gate['max_length'],
                            'encoder_batch_size': gate['encoder_batch_size'], 'device': 'cpu',
                            'action_labels': ['direct', 'reflect_1'], 'recipe': recipe,
                            'lambda_cost': args.lambda_cost, 'seed': args.seed}

                points += run_recipe(recipe['name'], val, round_actions, step, args.epochs,
                                     args.output / recipe['name'], args, baselines, bundle, audit)
            write_json(args.output / 'status.json', {'stage': 'replaying_exports', 'test_evaluated': False})
            verify_exports(points, vf, val, round_actions, baselines, args)
            write_json(args.output / 'summary.json', {'baselines': baselines, 'comparisons': points,
                       'coverage': audit, 'accuracy_tolerance': None, 'main_model_selected': False,
                       'all_checkpoint_replays_match': True, 'test_evaluated': False})
            report(args.output, baselines, points, audit)
            write_json(args.output / 'status.json', {'stage': 'completed', 'test_evaluated': False,
                                                    'all_checkpoint_replays_match': True})
            print(f'[ablation] completed: {args.output / "REPORT.md"}', flush=True)
        except Exception as exc:
            write_json(args.output / 'status.json', {'stage': 'failed', 'error': repr(exc)})
            raise


if __name__ == '__main__':
    main()
