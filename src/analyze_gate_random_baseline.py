"""Frozen development-policy comparison to Direct and matched-call random gates."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from .analyze_decoupling_significance import _bootstrap, _mcnemar
from .math_gate_cpu_ablation import read_cached_features, score_model, verify_exports
from .summarize_math_gate_multiseed import aggregate, mean_std
from .train_fixed_math_policy import ACTIONS, audit_split, candidates, digest, write_json
from .utils import load_pickle


FIELDS = ('accuracy', 'avg_prompt_tokens', 'avg_completion_tokens', 'avg_token_cost',
          'reflection_rate', 'fixes', 'harms')
RANDOM_TYPES = ('uniform_gate', 'round_stratified_gate')
DRAW_SEED = 2026091601
BOOTSTRAP_SEED = 2026091602


def holm_adjust(p_values):
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1 or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('Expected finite probabilities')
    order = np.argsort(p)
    out = np.empty_like(p)
    out[order] = np.minimum(1., np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    return out.tolist()


def paired_interval(difference, iterations=10000, seed=BOOTSTRAP_SEED):
    x = np.asarray(difference, dtype=float)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError('Expected aligned finite question-level differences')
    result = _bootstrap(x, iterations, seed, chunk_size=250)
    # An uncentered bootstrap tail fraction is not a calibrated hypothesis-test p-value.
    return {k: result[k] for k in ('observed_difference', 'ci95_lower', 'ci95_upper')}


def random_inclusion(rounds, selected, mode):
    rounds, selected = np.asarray(rounds), np.asarray(selected)
    if rounds.ndim != 1 or rounds.shape != selected.shape or not len(rounds):
        raise ValueError('Round and selection arrays must align')
    if not np.isin(rounds, (1, 2)).all() or not np.isin(selected, (0, 1, 2)).all():
        raise ValueError('Invalid action indices')
    if np.any((selected != 0) & (selected != rounds)):
        raise ValueError('Selected reflection differs from fixed Round')
    if mode == 'uniform_gate':
        return np.full(len(rounds), np.count_nonzero(selected) / len(rounds))
    if mode != 'round_stratified_gate':
        raise ValueError('Unknown random gate type')
    probabilities = np.zeros(len(rounds))
    for action in (1, 2):
        pool = rounds == action
        if pool.any():
            probabilities[pool] = np.count_nonzero(selected == action) / pool.sum()
    return probabilities


def random_orders(ids, repetitions, rng):
    ids = np.asarray(ids, dtype=int)
    orders = np.empty((repetitions, len(ids)), dtype=np.int32)
    for i in range(repetitions):
        orders[i] = rng.permutation(ids)
    return orders


def sampled_metrics(base, reflected, selected, rounds, mode, orders):
    """Uniform subsets without replacement; no correctness/cost used in sampling."""
    probabilities = random_inclusion(rounds, selected, mode)
    expected = base + probabilities[:, None] * (reflected - base)
    indices = (orders['all'][:, :np.count_nonzero(selected)] if mode == 'uniform_gate' else
               np.concatenate([orders[a][:, :np.count_nonzero(selected == a)] for a in (1, 2)], axis=1))
    values = np.empty((len(indices), base.shape[1]))
    delta = reflected - base
    for j in range(base.shape[1]):
        values[:, j] = base[:, j].sum() + delta[indices, j].sum(axis=1)
    values[:, :5] /= len(base)
    return expected, values


def to_metrics(values):
    result = values.mean(axis=0)
    result[5:] *= len(values)
    return dict(zip(FIELDS, map(float, result)))


def make_tables(rows):
    n = len(rows)
    table = np.zeros((n, 3, len(FIELDS)))
    for i, row in enumerate(rows):
        cs = candidates(row)
        if set(cs) != set(ACTIONS):
            raise ValueError('Complete candidates required')
        for a, action in enumerate(ACTIONS):
            c = cs[action]
            table[i, a] = [c['correct'], c['prompt_tokens'], c['completion_tokens'], c['token_cost'],
                           a != 0, not cs['direct']['correct'] and c['correct'],
                           cs['direct']['correct'] and not c['correct']]
    return table


def write_report(path, groups, tests, baselines):
    lines = ['# 少量反思的质量提升与随机 Gate 对照', '',
             '**探索性内部验证报告：不改模型、阈值、训练数据或生成提示；不是独立测试显著性结论。**', '',
             '主比较为策略相对基础模型 Direct 的答案提升；固定反思提供质量/成本对照，强模型只作质量参照。',
             '仍不设置相对强模型的准确率容差，也不依据此报告选主模型。', '',
             '## 基线', '', '| 方法 | 答案准确率 | 输入 token | 输出 token | 总 token |',
             '| --- | ---: | ---: | ---: | ---: |']
    for name in ('direct', 'fixed_reflect_1', 'fixed_reflect_2', 'independent', 'oracle_accuracy_first'):
        m = baselines[name]
        lines.append(f"| {name} | {m['accuracy']:.2%} | {m['avg_prompt_tokens']:.2f} | "
                     f"{m['avg_completion_tokens']:.2f} | {m['avg_token_cost']:.2f} |")
    for branch in ('original_round', 'fixed_r1'):
        lines += ['', f'## {branch}', '',
                  '| 损失 | 预算上限 | 策略准确率 % (种子 SD) | 随机 Gate 期望 % | 同轮次数随机期望 % | 策略/随机/同轮次总 token | 实际调用率 |',
                  '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
        for g in groups:
            if g['branch'] != branch:
                continue
            m, u, t = g['learned'], g['random_expected']['uniform_gate'], g['random_expected']['round_stratified_gate']
            lines.append(f"| {g['recipe']} | {g['budget']:.0%} | {m['accuracy']['mean']*100:.2f} +/- {m['accuracy']['std']*100:.2f} | "
                         f"{u['accuracy']['mean']*100:.2f} | {t['accuracy']['mean']*100:.2f} | "
                         f"{m['avg_token_cost']['mean']:.2f}/{u['avg_token_cost']['mean']:.2f}/{t['avg_token_cost']['mean']:.2f} | "
                         f"{m['reflection_rate']['mean']:.2%} |")
    lines += ['', '## 按题配对区间', '',
              '先在同一题上平均三个冻结策略种子的差值，再按题有放回重采样 10,000 次。样本单位为 1492 题，不是 4476 条独立记录。',
              '随机参照使用解析期望，区间只衡量当前验证题的经验不确定性；固定当前调用概率，不重新搜索阈值或选择检查点。',
              '下表为准确率差，单位为百分点，括号中为 95% percentile bootstrap 区间。', '',
              '| 损失 | 分支 | 预算 | 策略 - Direct | 策略 - 随机 Gate | 策略 - 同轮次数随机 |',
              '| --- | --- | ---: | ---: | ---: | ---: |']
    for g in groups:
        def interval(name):
            d = g['paired_bootstrap'][name]['accuracy']
            return f"{100*d['observed_difference']:+.3f} [{100*d['ci95_lower']:+.3f}, {100*d['ci95_upper']:+.3f}]"
        lines.append(f"| {g['recipe']} | {g['branch']} | {g['budget']:.0%} | "
                     f"{interval('direct')} | {interval('uniform_gate')} | {interval('round_stratified_gate')} |")
    lines += ['', '## 每个冻结策略相对 Direct 的精确 McNemar', '',
              '双侧精确检验使用修复/引错这两个不一致对计数。不对三个种子的平均正确率作 McNemar，也不对随机期望这种非二值数值作 McNemar。',
              'Holm 校正的同一检验族包含 36 个策略点和 3 个固定参照，共 39 次比较；无法校正此前模型/阈值开发使用验证集造成的偏差。', '',
              '| 方法 | 修复/引错 | 原始 p | Holm p |', '| --- | ---: | ---: | ---: |']
    for t in tests:
        m = t['mcnemar']
        lines.append(f"| {t['name']} | {m['learned_correct_baseline_wrong']}/{m['learned_wrong_baseline_correct']} | "
                     f"{m['exact_two_sided_p']:.4g} | {t['holm_p']:.4g} |")
    lines += ['', '## 随机参照与成本口径', '',
              '- uniform_gate：从全部题均匀无放回选择 K 题，K 等于对应策略的实际调用数，然后执行原冻结 Round；fixed_r1 分支始终执行 R1。',
              '- round_stratified_gate：按原冻结 Round 预测的 R1/R2 分组，分别随机选同样 K1/K2 题；同时匹配总调用数和轮次数，不使用正确性或成本挑题。',
              '- 每个点各 10,000 次随机重复，不从中选择最优随机 seed。随机键跨方法复用以减少模拟噪声，各重复边际上保持合法均匀抽样。',
              '- 表中随机准确率/成本为解析期望；random_draws.npz 保存 Monte Carlo 实现，per_policy.json 保存模拟均值、SD、2.5%/97.5% 分位。随机分位带不是按题 bootstrap CI。',
              '- 相同调用数不等于相同 token 成本：被选问题长度不同，uniform_gate 的 R2 比例也可能不同。因此不声称本实验是严格等 token 对照。',
              '- 路径总成本包含初始回答及实际 R1/R2 的输入+输出；被拒绝调用不能退回只记 Direct。随机路径成本同样从保存轨迹读取。',
              '- 标准差报告 Gate 训练随机性；配对区间报告固定策略在当前题集的经验差值，两者不可替换。', '',
              '## 结论边界', '',
              '所有策略和阈值已在这批验证题上开发与选择，因此名义 p 值、Holm p 和 bootstrap 区间均只能用作探索性诊断，不能当作独立确认。',
              '没有训练新模型，没有新的生成调用，没有测试集调参。原始训练 2 题、验证 3 题失败仍未补齐。',
              'Oracle 不含强模型独立回答且不可部署。token 不代表跨模型等价 FLOPs 或完整延迟；本地选择器前向开销仍需实测。', '']
    path.write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('outputs/math_gate_ranking_multiseed_20260916'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('threads must be positive')
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / 'status.json', {'stage': 'auditing', 'test_evaluated': False})
    try:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        import torch
        torch.set_num_threads(args.threads)
        manifest = json.loads((args.source / 'manifest.json').read_text())
        source_status = json.loads((args.source / 'status.json').read_text())
        if source_status.get('stage') != 'completed' or manifest.get('all_36_checkpoint_replays_match') is not True:
            raise ValueError('Require completed multiseed replay audit')
        runs = []
        for a in manifest['artifacts']:
            path = Path(a['run'])
            for f in ('protocol', 'summary'):
                if digest(path / f'{f}.json') != a[f'{f}_sha256']:
                    raise ValueError(f'Changed {f} artifact')
            for path_string, checksum in a['models'].items():
                if digest(path_string) != checksum:
                    raise ValueError('Changed model checkpoint')
            if digest(path / 'reference_round.pkl') != a['round_copy_sha256']:
                raise ValueError('Changed Round')
            runs.append({key: json.loads((path / f'{key}.json').read_text())
                         for key in ('protocol', 'summary', 'status')})
        summary = aggregate(runs)
        p = runs[0]['protocol']
        train, val, audit = audit_split(Path(p['fixed_split']['train']['file']).parent, True)
        if audit != summary['coverage']:
            raise ValueError('Candidate data changed')
        write_json(args.output / 'protocol.json', {
            'source': str(args.source), 'source_manifest_sha256': digest(args.source / 'manifest.json'),
            'source_code_sha256': digest(__file__), 'coverage': audit, 'training_seeds': summary['seeds'],
            'random_gate_draws': 10000, 'random_gate_seed': DRAW_SEED,
            'bootstrap_iterations': 10000, 'bootstrap_seed': BOOTSTRAP_SEED,
            'bootstrap_unit': 'question, after averaging paired differences across frozen seeds',
            'random_types': RANDOM_TYPES, 'no_random_seed_selection': True, 'no_new_policy_selection': True,
            'matched_budget': 'actual number of reflection calls; stratified variant also matches R1/R2 counts',
            'bootstrap_random_reference': 'analytic expectation, conditional on fixed observed inclusion probabilities',
            'mcnemar_family': '36 policy-vs-direct tests plus fixed R1, fixed R2 and independent; Holm 39',
            'inference_scope': 'post-selection exploratory validation only; no confirmatory significance claim',
            'test_evaluated': False, 'new_llm_calls': 0, 'accuracy_tolerance': None})
        prior = json.loads((Path(p['source']) / 'selected_validation.json').read_text())
        features, _ = read_cached_features(Path(prior['encoder']['cache']), train + val, p['feature_fingerprint'])
        vf = features[len(train):]
        ref_path = Path(manifest['artifacts'][0]['run']) / 'reference_round.pkl'
        rb = load_pickle(ref_path)
        round_actions = np.array(rb['action_labels'])[score_model(rb, vf).argmax(1)]
        rounds = np.array([ACTIONS.index(a) for a in round_actions])
        args.lambda_cost = float(rb.get('lambda_cost', .0001))
        n = len(val)
        table = make_tables(val)
        base = table[:, 0]
        ids = np.arange(n)
        rng = np.random.default_rng(DRAW_SEED)
        orders = {'all': random_orders(ids, 10000, rng)}
        stratified_orders = {a: random_orders(ids[rounds == a], 10000, rng) for a in (1, 2)}
        groups, per_policy, tests, draws = {}, [], [], {}
        random_cache = {}
        with (args.output / 'decisions.jsonl').open('w') as stream:
            for run in sorted(runs, key=lambda r: r['protocol']['seed']):
                seed = run['protocol']['seed']
                points = run['summary']['comparisons']
                verify_exports(points, vf, val, round_actions, summary['baselines'], args)
                for point in points:
                    key = (point['recipe'], point['branch'], point['budget'])
                    name = f"seed{seed}_{point['recipe']}_{point['branch']}_{point['budget']:.2f}"
                    bundle = load_pickle(point['gate_model'])
                    direct_p = score_model(bundle, vf)[:, 0]
                    branch_rounds = rounds if point['branch'] == 'original_round' else np.ones(n, dtype=int)
                    selected = np.where(direct_p >= bundle['direct_threshold'], 0, branch_rounds)
                    learned = table[ids, selected]
                    expected, random_info = {}, {}
                    for mode in RANDOM_TYPES:
                        counts = tuple(int((selected == a).sum()) for a in (1, 2))
                        cache_key = (point['branch'], mode, sum(counts) if mode == 'uniform_gate' else counts)
                        if cache_key not in random_cache:
                            o = {**orders, **stratified_orders} if point['branch'] == 'original_round' else {
                                **orders, 1: orders['all'], 2: np.empty((10000, 0), dtype=np.int32)}
                            random_cache[cache_key] = sampled_metrics(
                                base, table[ids, branch_rounds], selected, branch_rounds, mode, o)
                        expectation, sampled = random_cache[cache_key]
                        expected[mode] = expectation
                        random_info[mode] = {'expected': to_metrics(expectation),
                            'simulation': {field: {**mean_std(sampled[:, j]),
                                'draw_q025': float(np.quantile(sampled[:, j], .025)),
                                'draw_q975': float(np.quantile(sampled[:, j], .975))} for j, field in enumerate(FIELDS)},
                            'selected_counts': {'reflect_1': counts[0], 'reflect_2': counts[1]},
                            'matching': 'total calls only' if mode == 'uniform_gate' else 'total calls and R1/R2 counts'}
                        draws[f'{name}_{mode}'] = sampled
                    groups.setdefault(key, []).append({'seed': seed, 'learned': learned, 'expected': expected})
                    tests.append({'name': name, 'mcnemar': _mcnemar(learned[:, 0], base[:, 0])})
                    per_policy.append({'name': name, 'seed': seed, **point, 'random': random_info})
                    for i, row in enumerate(val):
                        stream.write(json.dumps({'policy': name, 'id': row['id'], 'direct_correct': int(base[i, 0]),
                            'action': ACTIONS[selected[i]], 'final_correct': int(learned[i, 0]),
                            'prompt_tokens': int(learned[i, 1]), 'completion_tokens': int(learned[i, 2]),
                            'total_tokens': int(learned[i, 3]), 'direct_probability': float(direct_p[i]),
                            'direct_threshold': bundle['direct_threshold']}) + '\n')
                print(f'[random] seed {seed}: matched random baselines and 12 frozen decisions complete', flush=True)
        for name, correct in (
                ('fixed_reflect_1', table[:, 1, 0]), ('fixed_reflect_2', table[:, 2, 0]),
                ('independent', np.array([r['independent']['correct'] for r in val]))):
            tests.append({'name': name, 'mcnemar': _mcnemar(correct, base[:, 0])})
        for t, corrected in zip(tests, holm_adjust([t['mcnemar']['exact_two_sided_p'] for t in tests])):
            t['holm_p'] = corrected
        group_results = []
        for key, items in groups.items():
            learned = np.mean([x['learned'] for x in items], axis=0)
            references = {'direct': base, **{mode: np.mean([x['expected'][mode] for x in items], axis=0)
                                             for mode in RANDOM_TYPES}}
            intervals = {ref: {field: paired_interval(learned[:, j] - values[:, j])
                               for field, j in (('accuracy', 0), ('avg_token_cost', 3))}
                         for ref, values in references.items()}
            group_results.append({'recipe': key[0], 'branch': key[1], 'budget': key[2],
                'learned': {f: mean_std([to_metrics(x['learned'])[f] for x in items]) for f in FIELDS},
                'random_expected': {mode: {f: mean_std([to_metrics(x['expected'][mode])[f] for x in items])
                                          for f in FIELDS} for mode in RANDOM_TYPES},
                'paired_bootstrap': intervals})
            print(f'[bootstrap] {key}: paired question intervals complete', flush=True)
        np.savez_compressed(args.output / 'random_draws.npz', **draws)
        write_json(args.output / 'per_policy.json', per_policy)
        write_json(args.output / 'summary.json', {'groups': group_results, 'mcnemar_vs_direct': tests,
            'baselines': summary['baselines'], 'coverage': audit, 'draw_columns': FIELDS,
            'test_evaluated': False, 'inference_scope': 'exploratory post-selection',
            'all_36_checkpoint_replays_match': True})
        write_report(args.output / 'REPORT.md', group_results, tests, summary['baselines'])
        write_json(args.output / 'status.json', {'stage': 'completed', 'test_evaluated': False,
                   'all_36_checkpoint_replays_match': True})
        print(f'[random] completed: {args.output / "REPORT.md"}', flush=True)
    except Exception as exc:
        write_json(args.output / 'status.json', {'stage': 'failed', 'error': repr(exc)})
        raise


if __name__ == '__main__':
    main()
