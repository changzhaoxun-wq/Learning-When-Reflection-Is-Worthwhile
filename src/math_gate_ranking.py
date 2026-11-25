"""Controlled weighted-CE versus CE+pairwise-ranking gate experiment on CPU."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from .math_gate_cpu_ablation import (BUDGETS, GRID, RECIPES, build_model, classification,
                                      mild_weights, read_cached_features, report,
                                      run_recipe, score_model, verify_exports)
from .train_fixed_math_policy import audit_split, digest, write_json
from .utils import load_pickle, save_pickle


def pairwise_ranking_loss(logits, labels):
    """Softplus(s_negative - s_positive), using only pairs in the supplied split/batch."""
    if logits.ndim != 2 or logits.shape[1] != 2 or labels.shape != logits.shape[:1]:
        raise ValueError('Expected [n,2] logits and [n] binary labels')
    if not ((labels == 0) | (labels == 1)).all():
        raise ValueError('Labels must be binary')
    scores = logits[:, 1] - logits[:, 0]
    positive, negative = scores[labels == 1], scores[labels == 0]
    if not len(positive) or not len(negative):
        return logits.sum() * 0.
    return torch.nn.functional.softplus(negative.unsqueeze(0) - positive.unsqueeze(1)).mean()


def combined_loss(logits, labels, class_weights, ranking_coefficient):
    if ranking_coefficient < 0:
        raise ValueError('Ranking coefficient must be nonnegative')
    ce = torch.nn.functional.cross_entropy(logits, labels, weight=class_weights)
    rank = pairwise_ranking_loss(logits, labels)
    return ce + ranking_coefficient * rank, ce, rank


def weights_digest(model):
    h = hashlib.sha256()
    for name, parameter in model.state_dict().items():
        h.update(name.encode())
        h.update(parameter.cpu().numpy().tobytes())
    return h.hexdigest()


def compare_points(control, ranking):
    index = {(x['branch'], x['budget']): x for x in control}
    rows = []
    for r in ranking:
        key = (r['branch'], r['budget'])
        if key not in index:
            continue
        c = index[key]
        rows.append({'branch': r['branch'], 'budget': r['budget'],
                     'control': c, 'ranking': r,
                     'delta_correct': r['metrics']['correct'] - c['metrics']['correct'],
                     'delta_accuracy_pp': 100 * (r['metrics']['accuracy'] - c['metrics']['accuracy']),
                     'delta_avg_token_cost': r['metrics']['avg_token_cost'] - c['metrics']['avg_token_cost']})
    return rows


def write_comparison(path, pairs, reproduced):
    text = ['# Gate 排序损失对照', '',
            '两组使用相同冻结向量、固定划分、小 MLP、随机初始化、批次顺序及验证选择规则。仅改变训练损失：', '',
            '- Control：原加权交叉熵，负类权重 1、正类权重 5。',
            '- Ranking：相同交叉熵 + 0.5 × pairwise logistic ranking loss。',
            '- 分数 s = reflect logit - direct logit；排序项为 mean softplus(s_negative - s_positive)。',
            '- 正类仍为 best_action != direct。所有配对仅在训练批次内部，单类别批次排序项为零。', '',
            (f"原 seed 42 加权小 MLP 的六个预算检查点结果全部复现：{reproduced}。"
             if reproduced is not None else "本轮不是 seed 42，不与旧 seed 42 数值作精确复现检查。"), '',
            '| 分支 | 调用预算 | CE epoch | CE 准确率 | CE 总 token | 排序 epoch | 排序准确率 | 排序总 token | 多答对 |',
            '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for p in pairs:
        c, r = p['control'], p['ranking']
        text.append(f"| {p['branch']} | {p['budget']:.0%} | {c['epoch']} | {c['metrics']['accuracy']:.2%} | "
                    f"{c['metrics']['avg_token_cost']:.2f} | {r['epoch']} | {r['metrics']['accuracy']:.2%} | "
                    f"{r['metrics']['avg_token_cost']:.2f} | {p['delta_correct']:+d} |")
    text += ['', '## 边界', '',
             '每组/分支/预算各自在内部验证上选 epoch 和阈值，所以上表不是固定一个阈值的单步梯度消融。',
             '阈值分位网格、成本上限、调用预算与上一轮一致；不根据本轮结果追加排序权重搜索。',
             '完整 1492 题验证中的正确性与 token 用于开发评价，不参与梯度更新。5 个未完成生成异常仍保留。',
             '原 Round 全程不训练；固定 R1 分支没有另训 Gate。未读取验证错误清单、也未将其回流训练。',
             '不设精度容差、不自动选论文主模型；这些结果不是独立测试或显著性证据。',
             '分类 loss 曲线仍按未加权 CE 展示，history.json 另有 weighted_ce、ranking_loss 与 combined objective。',
             '训练批次排序项均值和全训练集全配对排序项不同，日志分别命名，不相互冒充。', '']
    path.write_text('\n'.join(text))


def plot_objectives(history, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for split in ('train', 'validation'):
        epochs = [h['epoch'] for h in history]
        axes[0].plot(epochs, [h['classification'][f'{split}_ranking_loss'] for h in history], label=split)
        axes[1].plot(epochs, [h['classification'][f'{split}_combined_objective'] for h in history], label=split)
    for ax, title in zip(axes, ['All-pair diagnostic ranking loss', 'Weighted CE + coefficient * all-pair loss']):
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.legend()
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', type=Path, default=Path('outputs/math_gate_cpu_ablation_20260916_seed42'))
    p.add_argument('--root', type=Path, default=Path('outputs/math_full_training_v1'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    if args.threads < 1:
        p.error('threads must be positive')
    if not 0 <= args.seed < 2**32:
        p.error('seed must be in [0, 2**32)')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    torch.set_num_threads(args.threads)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'run.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (args.output / 'protocol.json').exists():
            raise ValueError('Output already used; choose a new directory')
        old = json.loads((args.reference / 'summary.json').read_text())
        old_protocol = json.loads((args.reference / 'protocol.json').read_text())
        source = Path(old_protocol['source'])
        prior = json.loads((source / 'selected_validation.json').read_text())
        train, val, audit = audit_split(args.root, True)
        if audit != old['coverage'] or audit != prior['coverage']:
            raise ValueError('Data or fixed split has changed')
        recipe = dict(RECIPES[1])
        trials = [('weighted_ce_control', 0.), ('weighted_ce_pairwise', .5)]
        write_json(args.output / 'protocol.json', {
            'source': str(source), 'reference': str(args.reference), 'trials': trials,
            'ranking_score': 'logit_reflect - logit_direct',
            'ranking_loss': 'mean softplus(score_negative - score_positive) over all within-training-batch pairs',
            'recipe': recipe, 'epochs': 10, 'seed': args.seed, 'learning_rate': .0003, 'batch_size': 128,
            'class_weight': 'negative=1, positive=min(5,sqrt(train_negative/train_positive))',
            'sampling': 'Identical original-distribution shuffled minibatches; no resampling or validation pairs',
            'call_budgets': BUDGETS, 'score_quantile_grid': GRID,
            'cost_constraint': 'validation mean total tokens strictly below same-item strong independent baseline',
            'selection': 'per trial/branch/budget: final correct count, cost, call rate, earliest epoch',
            'accuracy_tolerance': None, 'main_model_selection': False, 'test_evaluated': False,
            'fixed_split': audit, 'feature_fingerprint': prior['encoder']['fingerprint'],
            'source_code_sha256': digest(__file__),
            'helper_sha256': digest(Path(__file__).with_name('math_gate_cpu_ablation.py')),
            'reference_summary_sha256': digest(args.reference / 'summary.json'),
            'reference_round_sha256': digest(args.reference / 'reference_round.pkl'),
            'scope': 'Exploratory internal validation, not confirmation'})
        write_json(args.output / 'status.json', {'stage': 'checking_cache', 'test_evaluated': False})
        try:
            features, info = read_cached_features(Path(prior['encoder']['cache']), train + val, prior['encoder']['fingerprint'])
            print(f'[ranking] reused {len(features)} cached vectors; encoder not loaded', flush=True)
            n = len(train)
            scaler = StandardScaler().fit(features[:n])
            x = torch.from_numpy(scaler.transform(features[:n]).astype('float32'))
            vx = torch.from_numpy(scaler.transform(features[n:]).astype('float32'))
            if not torch.isfinite(x).all() or not torch.isfinite(vx).all():
                raise ValueError('Nonfinite scaled feature')
            y = torch.tensor([r['best_action'] != 'direct' for r in train], dtype=torch.long)
            vy = torch.tensor([r['best_action'] != 'direct' for r in val], dtype=torch.long)
            weights = mild_weights(y.numpy(), True)
            round_bundle = load_pickle(args.reference / 'reference_round.pkl')
            save_pickle(round_bundle, args.output / 'reference_round.pkl')
            rounds = np.array(round_bundle['action_labels'])[score_model(round_bundle, features[n:]).argmax(1)]
            baselines = old['baselines']
            write_json(args.output / 'baselines.json', baselines)
            # The shared runner only consumes lambda_cost from args, never uses validation for gradients.
            args.lambda_cost = float(round_bundle.get('lambda_cost', .0001))
            points, initialization, batches, checks = [], {}, {}, {}
            for name, coefficient in trials:
                torch.manual_seed(args.seed)
                np.random.seed(args.seed)
                model = build_model(recipe).cpu()
                initialization[name] = weights_digest(model)
                optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.01)
                batch_hash = hashlib.sha256()
                write_json(args.output / 'status.json', {'stage': 'training_cpu', 'trial': name, 'test_evaluated': False})

                def step(epoch):
                    model.train()
                    zero_pair_batches, pairs, batch_rank_sum, steps = 0, 0, 0., 0
                    for indices in torch.randperm(n).split(128):
                        batch_hash.update(indices.numpy().tobytes())
                        labels = y[indices]
                        pos = int((labels == 1).sum())
                        pair_count = pos * (len(labels) - pos)
                        pairs += pair_count
                        zero_pair_batches += int(pair_count == 0)
                        optimizer.zero_grad()
                        loss, _, rank = combined_loss(model(x[indices]), labels, weights, coefficient)
                        loss.backward()
                        optimizer.step()
                        batch_rank_sum += float(rank.detach())
                        steps += 1
                    metrics, direct = classification(model, x, vx, y, vy)
                    with torch.inference_mode():
                        for split, f, labels in (('train', x, y), ('validation', vx, vy)):
                            total, ce, rank = combined_loss(model(f), labels, weights, coefficient)
                            metrics[f'{split}_weighted_ce'] = float(ce)
                            metrics[f'{split}_ranking_loss'] = float(rank)
                            metrics[f'{split}_combined_objective'] = float(total)
                    metrics.update(ranking_coefficient=coefficient, training_pairs=pairs,
                                   training_batches_without_pair=zero_pair_batches,
                                   training_batch_mean_ranking_loss=batch_rank_sum / steps)
                    return metrics, direct

                def bundle():
                    return {'policy_type': 'torch_embedding_mlp', 'backend': 'hf_hidden_torch_mlp',
                            'model': model, 'scaler': scaler, 'encoder_info': info,
                            'encoder_name': info['encoder_name'], 'layers': [-2, -1], 'pooling': 'mean_last',
                            'pure_text': True, 'max_text_chars': 0, 'max_length': info['max_length'],
                            'encoder_batch_size': 2, 'device': 'cpu', 'action_labels': ['direct', 'reflect_1'],
                            'recipe': recipe, 'ranking_coefficient': coefficient, 'lambda_cost': args.lambda_cost,
                            'seed': args.seed, 'loss': 'weighted_ce_plus_pairwise_logistic'}

                trial_points = run_recipe(name, val, rounds, step, 10, args.output / name,
                                          args, baselines, bundle, audit)
                points += trial_points
                batches[name] = batch_hash.hexdigest()
                history = json.loads((args.output / name / 'history.json').read_text())
                plot_objectives(history, args.output / name / 'objective_curve.png')
                if coefficient == 0 and args.seed == 42:
                    old_points = {(r['branch'], r['budget']): r for r in old['comparisons'] if r['recipe'] == 'small_sqrt_weighted'}
                    reproduced = len(trial_points) == len(old_points) and all(
                        r['metrics'] == old_points[(r['branch'], r['budget'])]['metrics']
                        and r['epoch'] == old_points[(r['branch'], r['budget'])]['epoch']
                        and r['direct_threshold'] == old_points[(r['branch'], r['budget'])]['direct_threshold']
                        for r in trial_points)
                    checks['control_reproduces_previous_six_points'] = reproduced
                    if not reproduced:
                        raise ValueError('Control failed to reproduce the source experiment; do not interpret loss ablation')
                elif coefficient == 0:
                    checks['control_reproduces_previous_six_points'] = None
            checks['identical_initialization'] = len(set(initialization.values())) == 1
            checks['identical_training_batch_order'] = len(set(batches.values())) == 1
            if not checks['identical_initialization'] or not checks['identical_training_batch_order']:
                raise ValueError('Initialization or minibatch-order mismatch')
            verify_exports(points, features[n:], val, rounds, baselines, args)
            checks['all_checkpoint_replays_match'] = True
            write_json(args.output / 'controlled_comparison_audit.json', {**checks, 'initialization_sha256': initialization,
                                                                       'batch_order_sha256': batches})
            comparison = compare_points([r for r in points if r['recipe'] == trials[0][0]],
                                        [r for r in points if r['recipe'] == trials[1][0]])
            write_json(args.output / 'summary.json', {'baselines': baselines, 'comparisons': points,
                       'coverage': audit, 'accuracy_tolerance': None, 'main_model_selected': False,
                       'test_evaluated': False, **checks, 'loss_comparison': comparison})
            report(args.output, baselines, points, audit)
            write_comparison(args.output / 'LOSS_COMPARISON.md', comparison, checks['control_reproduces_previous_six_points'])
            write_json(args.output / 'status.json', {'stage': 'completed', 'test_evaluated': False, **checks})
            print(f'[ranking] completed: {args.output / "LOSS_COMPARISON.md"}', flush=True)
        except Exception as exc:
            write_json(args.output / 'status.json', {'stage': 'failed', 'error': repr(exc)})
            raise


if __name__ == '__main__':
    main()
