"""Audit and summarize fixed-recipe Gate seeds; never select a winning seed."""
from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path

import numpy as np

from .train_fixed_math_policy import digest, write_json


SEEDS = (7, 21, 42)
TRIALS = ('weighted_ce_control', 'weighted_ce_pairwise')
BRANCHES = ('original_round', 'fixed_r1')
BUDGETS = (.05, .10, .20)
METRICS = ('accuracy', 'correct', 'avg_prompt_tokens', 'avg_completion_tokens',
           'avg_token_cost', 'reflection_rate', 'avg_reward', 'fixes', 'harms')
REQUIRED_CHECKS = ('identical_initialization', 'identical_training_batch_order',
                   'all_checkpoint_replays_match')


def mean_std(values):
    a = np.asarray(values, dtype=float)
    if len(a) < 2 or not np.isfinite(a).all():
        raise ValueError('At least two finite observations required')
    return {'mean': float(a.mean()), 'std': float(a.std(ddof=1))}


def validate_runs(runs):
    """Reject changed protocols, missing recipes, duplicate seeds, or unverified runs."""
    if sorted(r['protocol']['seed'] for r in runs) != list(SEEDS):
        raise ValueError('Exactly one completed run for each of seeds 7,21,42 is required')
    ref = runs[0]
    expected = set(itertools.product(TRIALS, BRANCHES, BUDGETS))
    for r in runs:
        p, s, status = r['protocol'], r['summary'], r['status']
        # Only seed and runner version may differ. An old completed seed 42 is reused as-is.
        ignored = {'seed', 'source_code_sha256'}
        if {k: v for k, v in p.items() if k not in ignored} != {
                k: v for k, v in ref['protocol'].items() if k not in ignored}:
            raise ValueError('Training protocol differs beyond seed/runner version')
        if (status.get('stage') != 'completed' or s.get('test_evaluated') is not False
                or p.get('test_evaluated') is not False or status.get('test_evaluated') is not False
                or s.get('main_model_selected') is not False or s.get('accuracy_tolerance') is not None):
            raise ValueError('Expected completed development-only run with no model promotion')
        if not all(s.get(k) is True and status.get(k) is True for k in REQUIRED_CHECKS):
            raise ValueError('Controlled comparison or replay audit failed')
        if p['seed'] == 42 and s.get('control_reproduces_previous_six_points') is not True:
            raise ValueError('Seed 42 must reproduce the previous control')
        if s['coverage'] != ref['summary']['coverage'] or s['baselines'] != ref['summary']['baselines']:
            raise ValueError('Data coverage or candidate baselines changed')
        keys = [(x['recipe'], x['branch'], x['budget']) for x in s['comparisons']]
        if len(keys) != len(expected) or set(keys) != expected:
            raise ValueError('Missing or duplicate recipe/branch/budget point')
        for x in s['comparisons']:
            m = x['metrics']
            if (m['n'] != s['baselines']['direct']['n'] or
                    m['reflection_rate'] > x['budget'] + 1e-12 or
                    m['avg_token_cost'] >= s['baselines']['independent']['avg_token_cost']):
                raise ValueError('Selected point violates shared sample or budget constraints')
    return True


def aggregate(runs):
    validate_runs(runs)
    runs = sorted(runs, key=lambda r: r['protocol']['seed'])
    index = [{(x['recipe'], x['branch'], x['budget']): x for x in r['summary']['comparisons']}
             for r in runs]
    groups, paired = [], []
    strong = runs[0]['summary']['baselines']['independent']
    for recipe, branch, budget in itertools.product(TRIALS, BRANCHES, BUDGETS):
        points = [idx[(recipe, branch, budget)] for idx in index]
        stats = {m: mean_std([p['metrics'][m] for p in points]) for m in METRICS}
        stats['accuracy_gap_to_strong_pp'] = mean_std([
            100 * (strong['accuracy'] - p['metrics']['accuracy']) for p in points])
        stats['token_savings_vs_strong'] = mean_std([
            1 - p['metrics']['avg_token_cost'] / strong['avg_token_cost'] for p in points])
        groups.append({'recipe': recipe, 'branch': branch, 'budget': budget, 'metrics': stats,
                       'seeds': [{'seed': seed, **point} for seed, point in zip(SEEDS, points)]})
    for branch, budget in itertools.product(BRANCHES, BUDGETS):
        deltas = [{m: idx[(TRIALS[1], branch, budget)]['metrics'][m]
                      - idx[(TRIALS[0], branch, budget)]['metrics'][m] for m in METRICS} for idx in index]
        paired.append({'branch': branch, 'budget': budget,
                       'ranking_minus_ce': {m: mean_std([d[m] for d in deltas]) for m in METRICS},
                       'per_seed': [{'seed': seed, **d} for seed, d in zip(SEEDS, deltas)]})
    return {'seeds': SEEDS, 'std_ddof': 1, 'scope': 'Gate training seed only; Round and LLM traces fixed',
            'test_evaluated': False, 'main_model_selected': False, 'accuracy_tolerance': None,
            'coverage': runs[0]['summary']['coverage'], 'baselines': runs[0]['summary']['baselines'],
            'groups': groups, 'paired_deltas': paired}


def write_report(path, s):
    lines = ['# Gate 三随机种子复核', '',
             '固定 seed 7、21、42，原 Round、缓存向量、候选轨迹、训练/验证题清单及全部超参数保持不变。',
             '以下误差为样本标准差（ddof=1），仅衡量 Gate 训练随机性，不是三次独立生成，也不是置信区间。',
             '每种配方、分支、预算按原规则在内部验证上选择 epoch/阈值；不选最优 seed，不设精度容差，不晋升主模型。', '',
             '## 固定基线', '', '| 方法 | 最终准确率 | 输入 token | 输出 token | 总 token |',
             '| --- | ---: | ---: | ---: | ---: |']
    for name in ('direct', 'fixed_reflect_1', 'fixed_reflect_2', 'independent', 'oracle_accuracy_first'):
        m = s['baselines'][name]
        lines.append(f"| {name} | {m['accuracy']:.2%} | {m['avg_prompt_tokens']:.2f} | "
                     f"{m['avg_completion_tokens']:.2f} | {m['avg_token_cost']:.2f} |")
    lines += ['', 'Oracle 不可部署，候选池不包含 independent。总 token 包含路径输入与输出；本地选择器计算开销未折算为 token。']
    for branch in BRANCHES:
        lines += ['', f'## {branch}', '',
                  '| 配方 | 预算 | 答案准确率 % | 总 token | 实际调用率 % | 修复/引错均值 | epoch (7,21,42) |',
                  '| --- | ---: | ---: | ---: | ---: | ---: | --- |']
        for g in s['groups']:
            if g['branch'] != branch:
                continue
            m = g['metrics']
            def fmt(key, scale=1):
                return f"{m[key]['mean'] * scale:.2f} +/- {m[key]['std'] * scale:.2f}"
            epochs = ','.join(str(p['epoch']) for p in g['seeds'])
            lines.append(f"| {g['recipe']} | {g['budget']:.0%} | {fmt('accuracy', 100)} | "
                         f"{fmt('avg_token_cost')} | {fmt('reflection_rate', 100)} | "
                         f"{m['fixes']['mean']:.2f}/{m['harms']['mean']:.2f} | {epochs} |")
    lines += ['', '## 同种子配对差值（排序减 CE）', '',
              '| 分支 | 预算 | 答对题数差 mean +/- std | 准确率差 pp | 总 token 差 | 各 seed 答对题数差 |',
              '| --- | ---: | ---: | ---: | ---: | --- |']
    for p in s['paired_deltas']:
        m = p['ranking_minus_ce']
        changes = ', '.join(f"{d['seed']}: {d['correct']:+d}" for d in p['per_seed'])
        lines.append(f"| {p['branch']} | {p['budget']:.0%} | {m['correct']['mean']:+.2f} +/- {m['correct']['std']:.2f} | "
                     f"{100*m['accuracy']['mean']:+.3f} +/- {100*m['accuracy']['std']:.3f} | "
                     f"{m['avg_token_cost']['mean']:+.2f} +/- {m['avg_token_cost']['std']:.2f} | {changes} |")
    lines += ['', '## 每种子结果', '',
              '| Seed | 配方 | 分支 | 预算 | epoch | 答对数 | 准确率 % | 输入 | 输出 | 总 token | 调用率 % | 修复/引错 |',
              '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for g in s['groups']:
        for p in g['seeds']:
            m = p['metrics']
            lines.append(f"| {p['seed']} | {g['recipe']} | {g['branch']} | {g['budget']:.0%} | {p['epoch']} | "
                         f"{m['correct']} | {m['accuracy']*100:.2f} | {m['avg_prompt_tokens']:.2f} | "
                         f"{m['avg_completion_tokens']:.2f} | {m['avg_token_cost']:.2f} | "
                         f"{m['reflection_rate']*100:.2f} | {m['fixes']}/{m['harms']} |")
    lines += ['', '## 证据边界', '',
              '- 数据为 5976 题训练、1492 题内部验证；原始 5 题异常仍未补齐，不能宣布全量干净测试完成。',
              '- 训练少数类仅 185 题。验证结果经历检查点与阈值选择，仍是探索性开发评价。',
              '- 同一验证题在三个 seed 中重复使用，不能视为 4476 个独立样本。未作独立测试显著性主张。',
              '- 验证预算不保证部署调用率；相同预算上限也不保证实际调用量相同。',
              '- 本次只改变 Gate 随机种子，不覆盖编码器、Round 或大模型生成的随机性。',
              '- 原 seed 42 文件只读复用，代码版本差异和哈希记录在 manifest.json，36 个导出模型均重新重放。', '']
    path.write_text('\n'.join(lines))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', nargs=3, type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--threads', type=int, default=4)
    args = p.parse_args()
    if args.threads < 1:
        p.error('threads must be positive')
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / 'status.json', {'stage': 'auditing', 'test_evaluated': False})
    try:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        import torch
        from .math_gate_cpu_ablation import read_cached_features, score_model, verify_exports
        from .math_gate_ranking import weights_digest
        from .train_fixed_math_policy import audit_split
        from .utils import load_pickle
        torch.set_num_threads(args.threads)
        runs = [{key: json.loads((d / f'{key}.json').read_text())
                 for key in ('protocol', 'summary', 'status')} for d in args.runs]
        summary = aggregate(runs)
        first = runs[0]['protocol']
        prior = json.loads((Path(first['source']) / 'selected_validation.json').read_text())
        root = Path(first['fixed_split']['train']['file']).parent
        train, val, audit = audit_split(root, True)
        if audit != summary['coverage']:
            raise ValueError('Current data differs from training snapshots')
        features, _ = read_cached_features(Path(prior['encoder']['cache']), train + val, first['feature_fingerprint'])
        vx = features[len(train):]
        artifacts = []
        for d, run in zip(args.runs, runs):
            protocol = run['protocol']
            reference_path = Path(protocol['reference']) / 'reference_round.pkl'
            if digest(reference_path) != protocol['reference_round_sha256']:
                raise ValueError('Source Round artifact hash changed')
            reference = load_pickle(reference_path)
            round_bundle = load_pickle(d / 'reference_round.pkl')
            # Torch pickle serialization changes byte hashes; compare actual parameters and preprocessing.
            if weights_digest(round_bundle['model']) != weights_digest(reference['model']):
                raise ValueError('Round weights changed')
            for key in ('action_labels', 'layers', 'pooling', 'pure_text', 'max_length', 'lambda_cost'):
                if round_bundle.get(key) != reference.get(key):
                    raise ValueError(f'Round metadata changed: {key}')
            if round_bundle['scaler'].get_params() != reference['scaler'].get_params():
                raise ValueError('Round scaler parameters changed')
            for key in ('mean_', 'scale_', 'var_', 'n_features_in_', 'n_samples_seen_'):
                if not np.array_equal(getattr(round_bundle['scaler'], key), getattr(reference['scaler'], key)):
                    raise ValueError(f'Round preprocessing changed: {key}')
            scores = score_model(round_bundle, vx)
            if not np.array_equal(scores, score_model(reference, vx)):
                raise ValueError('Round probability replay changed')
            rounds = np.array(round_bundle['action_labels'])[scores.argmax(1)]
            args.lambda_cost = float(round_bundle.get('lambda_cost', .0001))
            points = run['summary']['comparisons']
            verify_exports(points, vx, val, rounds, summary['baselines'], args)
            artifacts.append({'run': str(d), 'seed': protocol['seed'],
                              'training_runner_sha256': protocol['source_code_sha256'],
                              'protocol_sha256': digest(d / 'protocol.json'),
                              'summary_sha256': digest(d / 'summary.json'),
                              'round_copy_sha256': digest(d / 'reference_round.pkl'),
                              'round_semantically_identical_to_source': True,
                              'models': {x['gate_model']: digest(x['gate_model']) for x in points}})
            print(f"[multiseed] seed {protocol['seed']}: 12 checkpoints replayed exactly", flush=True)
        write_json(args.output / 'manifest.json', {'artifacts': artifacts, 'aggregator_sha256': digest(__file__),
                   'std_ddof': 1, 'required_seeds': SEEDS, 'all_36_checkpoint_replays_match': True,
                   'test_evaluated': False, 'new_llm_calls': 0, 'encoder_loaded': False})
        write_json(args.output / 'summary.json', summary)
        write_report(args.output / 'REPORT.md', summary)
        write_json(args.output / 'status.json', {'stage': 'completed', 'test_evaluated': False,
                   'all_36_checkpoint_replays_match': True})
        print(f'[multiseed] completed: {args.output / "REPORT.md"}', flush=True)
    except Exception as exc:
        write_json(args.output / 'status.json', {'stage': 'failed', 'error': repr(exc)})
        raise


if __name__ == '__main__':
    main()
