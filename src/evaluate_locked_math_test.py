"""Evaluate all pre-locked policies on complete new test traces, without calibration."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from .analyze_decoupling_significance import _mcnemar
from .analyze_gate_random_baseline import (BOOTSTRAP_SEED, DRAW_SEED, FIELDS, RANDOM_TYPES,
                                          holm_adjust, make_tables, paired_interval,
                                          random_orders, sampled_metrics, to_metrics)
from .math_gate_cpu_ablation import score_model
from .summarize_math_gate_multiseed import mean_std
from .train_fixed_math_policy import ACTIONS, best_action, decision_metrics, digest, extract_features, write_json
from .utils import load_pickle, read_jsonl


def frozen_actions(bundle, features, round_actions, branch, threshold):
    if bundle['direct_threshold'] != threshold or bundle['action_labels'] != ['direct', 'reflect_1']:
        raise ValueError('Gate threshold or class order differs from the pre-test lock')
    if branch not in ('original_round', 'fixed_r1'):
        raise ValueError('Unknown frozen reflection branch')
    direct = score_model(bundle, features)[:, 0]
    reflection = round_actions if branch == 'original_round' else np.full(len(features), 'reflect_1')
    # Apply the locked per-question threshold, never test-side top-k or budget calibration.
    return np.where(direct >= threshold, 'direct', reflection), direct


def assert_complete_trace(root, lock, completion):
    if not completion.get('final_evaluation_ready') or completion.get('pending_ids') or completion.get('unresolved_call_logs'):
        raise ValueError('Unresolved test generation; final evaluation forbidden')
    if completion['policy_lock_sha256'] != digest(root / 'policy_lock.json'):
        raise ValueError('Policy lock changed after generation')
    if completion['protocol_sha256'] != digest(root / 'protocol.json'):
        raise ValueError('Generation protocol changed')
    if digest(root / 'policy_test.jsonl') != completion['test_trace_sha256']:
        raise ValueError('Test traces changed')
    if digest(root / 'test_items.jsonl') != lock['test_items_sha256']:
        raise ValueError('Test items changed')
    rows = read_jsonl(root / 'policy_test.jsonl')
    items = read_jsonl(root / 'test_items.jsonl')
    if len(rows) != lock['test_n'] or [r['id'] for r in rows] != [r['id'] for r in items]:
        raise ValueError('Missing, duplicated or misordered test rows')
    if any(r['policy_split'] != 'test' or r['candidate_coverage'] != 'complete' for r in rows):
        raise ValueError('Full test candidates required')
    return rows


def evaluate(root):
    root = Path(root)
    lock = json.loads((root / 'policy_lock.json').read_text())
    completion = json.loads((root / 'completion_audit.json').read_text())
    rows = assert_complete_trace(root, lock, completion)
    protocol = json.loads((root / 'protocol.json').read_text())
    for name, checksum in protocol['source_code_sha256'].items():
        if digest(Path(__file__).parent / name) != checksum:
            raise ValueError(f'Evaluation dependency changed after pre-test lock: {name}')
    if digest(lock['encoder_manifest']) != lock['encoder_manifest_sha256']:
        raise ValueError('Frozen encoder provenance changed')
    signature = json.loads(Path(lock['encoder_manifest']).read_text())
    for name, checksum in signature['encoder_files_sha256'].items():
        if digest(Path(signature['encoder_name']) / name) != checksum:
            raise ValueError('Encoder parameters/tokenizer changed')
    for point in lock['policies']:
        if digest(point['model']) != point['sha256']:
            raise ValueError('Frozen Gate changed')
    if digest(lock['round_model']) != lock['round_sha256']:
        raise ValueError('Frozen Round changed')
    output = root / 'evaluation'
    output.mkdir(exist_ok=True)
    if (output / 'summary.json').exists():
        existing = json.loads((output / 'summary.json').read_text())
        if existing['policy_lock_sha256'] != digest(root / 'policy_lock.json') or existing['test_trace_sha256'] != completion['test_trace_sha256']:
            raise ValueError('Existing evaluation belongs to different inputs')
        write_json(root / 'status.json', {'stage': 'completed', 'test_metrics_released': True,
                   'test_n': len(rows), 'summary': str(output / 'summary.json')})
        return
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import torch
    torch.set_num_threads(4)
    write_json(root / 'status.json', {'stage': 'encoding_test_cpu', 'test_n': len(rows),
                                    'test_metrics_released': False})
    features, info = extract_features(rows, lock['encoder']['encoder_name'], root / 'test_feature_cache',
                                      lock['encoder']['max_length'], 2)
    # Encoder clipping is distinct from LLM truncation. Record it; do not alter trained input length.
    write_json(output / 'encoder_audit.json', info)
    round_bundle = load_pickle(lock['round_model'])
    round_actions = np.array(round_bundle['action_labels'])[score_model(round_bundle, features).argmax(1)]
    round_ids = np.array([ACTIONS.index(a) for a in round_actions])
    n, ids = len(rows), np.arange(len(rows))
    table = make_tables(rows)
    base = table[:, 0]
    baselines = {name: decision_metrics(rows, [action] * n) for name, action in (
        ('direct', 'direct'), ('fixed_reflect_1', 'reflect_1'), ('fixed_reflect_2', 'reflect_2'), ('independent', 'independent'))}
    baselines['oracle_accuracy_first'] = decision_metrics(rows, [best_action(r) for r in rows])
    rng = np.random.default_rng(DRAW_SEED)
    orders = {'all': random_orders(ids, lock['random_repetitions'], rng)}
    stratified = {a: random_orders(ids[round_ids == a], lock['random_repetitions'], rng) for a in (1, 2)}
    groups, records, tests, random_cache, draws = {}, [], [], {}, {}
    write_json(root / 'status.json', {'stage': 'offline_test_evaluation', 'test_n': n, 'test_metrics_released': False})
    with (output / 'decisions.jsonl').open('w') as handle:
        for point in lock['policies']:
            bundle = load_pickle(point['model'])
            actions, probabilities = frozen_actions(bundle, features, round_actions, point['branch'], point['direct_threshold'])
            selected = np.array([ACTIONS.index(a) for a in actions])
            learned = table[ids, selected]
            metrics = decision_metrics(rows, actions.tolist())
            branch_round = round_ids if point['branch'] == 'original_round' else np.ones(n, dtype=int)
            expected, random_info = {}, {}
            for mode in RANDOM_TYPES:
                counts = tuple(int((selected == a).sum()) for a in (1, 2))
                cache_key = (point['branch'], mode, sum(counts) if mode == 'uniform_gate' else counts)
                if cache_key not in random_cache:
                    local_orders = {**orders, **stratified} if point['branch'] == 'original_round' else {
                        **orders, 1: orders['all'], 2: np.empty((lock['random_repetitions'], 0), dtype=np.int32)}
                    random_cache[cache_key] = sampled_metrics(base, table[ids, branch_round], selected, branch_round, mode, local_orders)
                expectation, sampled = random_cache[cache_key]
                expected[mode] = expectation
                random_info[mode] = {'expected': to_metrics(expectation),
                    'simulation': {field: {**mean_std(sampled[:, j]), 'draw_q025': float(np.quantile(sampled[:, j], .025)),
                                          'draw_q975': float(np.quantile(sampled[:, j], .975))} for j, field in enumerate(FIELDS)}}
                draws[f'{point["name"]}_{mode}'] = sampled
            key = (point['recipe'], point['branch'], point['validation_call_budget'])
            groups.setdefault(key, []).append({'seed': point['seed'], 'learned': learned, 'expected': expected})
            records.append({**point, 'metrics': metrics, 'random': random_info,
                            'test_rate_exceeds_validation_budget': metrics['reflection_rate'] > point['validation_call_budget'] + 1e-12})
            tests.append({'name': point['name'], 'mcnemar': _mcnemar(learned[:, 0], base[:, 0])})
            for i, row in enumerate(rows):
                handle.write(json.dumps({'policy': point['name'], 'id': row['id'], 'action': actions[i],
                    'direct_probability': float(probabilities[i]), 'direct_threshold': point['direct_threshold'],
                    'direct_correct': int(base[i, 0]), 'final_correct': int(learned[i, 0]),
                    'prompt_tokens': int(learned[i, 1]), 'completion_tokens': int(learned[i, 2]),
                    'total_tokens': int(learned[i, 3])}) + '\n')
            print(f'[test-eval] applied locked policy {point["name"]}', flush=True)
    independent = np.zeros_like(base)
    for i, row in enumerate(rows):
        c = row['independent']
        independent[i, :4] = [c['correct'], c['prompt_tokens'], c['completion_tokens'], c['token_cost']]
    for name, correctness in (('fixed_reflect_1', table[:, 1, 0]), ('fixed_reflect_2', table[:, 2, 0]), ('independent', independent[:, 0])):
        tests.append({'name': name, 'mcnemar': _mcnemar(correctness, base[:, 0])})
    for test, adjusted in zip(tests, holm_adjust([t['mcnemar']['exact_two_sided_p'] for t in tests])):
        test['holm_p'] = adjusted
    results = []
    for key, items in groups.items():
        if sorted(x['seed'] for x in items) != [7, 21, 42]:
            raise ValueError('All three frozen seeds must be reported')
        learned = np.mean([x['learned'] for x in items], axis=0)
        refs = {'direct': base, 'fixed_reflect_1': table[:, 1], 'independent': independent,
                **{m: np.mean([x['expected'][m] for x in items], axis=0) for m in RANDOM_TYPES}}
        results.append({'recipe': key[0], 'branch': key[1], 'validation_call_budget': key[2],
            'metrics': {f: mean_std([to_metrics(x['learned'])[f] for x in items]) for f in FIELDS},
            'random_expected': {m: {f: mean_std([to_metrics(x['expected'][m])[f] for x in items]) for f in FIELDS} for m in RANDOM_TYPES},
            'paired_bootstrap': {ref: {field: paired_interval(learned[:, j] - value[:, j], lock['bootstrap_iterations'], BOOTSTRAP_SEED)
                                      for field, j in (('accuracy', 0), ('avg_token_cost', 3))} for ref, value in refs.items()}})
    result = {'n': n, 'baselines': baselines, 'groups': results, 'policies': records, 'mcnemar_vs_direct': tests,
              'policy_lock_sha256': digest(root / 'policy_lock.json'), 'test_trace_sha256': completion['test_trace_sha256'],
              'encoder_info': info, 'no_test_calibration': True, 'random_draw_columns': FIELDS,
              'bootstrap_unit': 'question after averaging the three frozen seeds', 'main_model_selected': False,
              'historical_limitation': lock['historical_limitation'], 'training_coverage': lock['training_coverage']}
    lines = ['# GSM8K 全量测试：锁定策略评测', '',
             '所有模型/阈值在本轮测试生成前锁定；不按测试结果重新校准或挑选 seed。',
             lock['historical_limitation'], '',
             f'完整测试 {n} 题；生成未解决截断为零。策略编码器输入截断 {info["encoder_clipped_count"]} 题，未改变冻结的输入长度。',
             '训练/验证侧五道异常仍如实披露，未在测试生成过程中重训策略。', '',
             '## 基线', '', '| 方法 | 准确率 | 输入 token | 输出 token | 总 token | 修复/引错 |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    for name, m in baselines.items():
        lines.append(f"| {name} | {m['accuracy']:.2%} | {m['avg_prompt_tokens']:.2f} | {m['avg_completion_tokens']:.2f} | {m['avg_token_cost']:.2f} | {m['fixes']}/{m['harms']} |")
    lines += ['', '## 三种子策略', '', '| 配方 | 分支 | 验证预算 | 测试准确率 % (SD) | 测试调用率 | 总 token (SD) | 随机准确率 | 同轮次数随机准确率 |',
              '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for g in results:
        m, r = g['metrics'], g['random_expected']
        lines.append(f"| {g['recipe']} | {g['branch']} | {g['validation_call_budget']:.0%} | "
                     f"{100*m['accuracy']['mean']:.2f} +/- {100*m['accuracy']['std']:.2f} | {m['reflection_rate']['mean']:.2%} | "
                     f"{m['avg_token_cost']['mean']:.2f} +/- {m['avg_token_cost']['std']:.2f} | "
                     f"{r['uniform_gate']['accuracy']['mean']:.2%} | {r['round_stratified_gate']['accuracy']['mean']:.2%} |")
    lines += ['', '预算来自验证侧，不保证测试调用率。随机基线匹配每个测试策略的实际调用数，不是等 token 对照。',
              'summary.json 包含每个种子的完整指标、随机参照成本、按题配对 Bootstrap 区间和 39 次 McNemar 的 Holm 校正。',
              '3 个训练 seed 不等于 3 份独立测试题；历史测试反馈仍限制确认性解释。Oracle 不可部署，不包括独立强模型。',
              '成本包括每次调用及已记录重试的输入和输出。强模型独立回答不包含 Base 成本；策略本地前向耗时未折算为 token。', '']
    (output / 'REPORT.md').write_text('\n'.join(lines))
    np.savez_compressed(output / 'random_draws.npz', **draws)
    # Commit the summary last; a partial CPU evaluation is safely repeatable from cached vectors.
    write_json(output / 'summary.json', result)
    write_json(root / 'status.json', {'stage': 'completed', 'test_metrics_released': True, 'test_n': n,
               'summary': str(output / 'summary.json'), 'report': str(output / 'REPORT.md')})
    print(f'[test-eval] completed: {output / "REPORT.md"}', flush=True)
