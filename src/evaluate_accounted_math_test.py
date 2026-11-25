"""Evaluate locked policies with explicit zero-credit execution failures, not clean answers."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from .account_math_test_failures import verify_snapshot
from .analyze_decoupling_significance import _mcnemar
from .analyze_gate_random_baseline import (BOOTSTRAP_SEED, DRAW_SEED, FIELDS, RANDOM_TYPES,
    holm_adjust, make_tables, paired_interval, random_orders, sampled_metrics, to_metrics)
from .evaluate_locked_math_test import frozen_actions
from .math_gate_cpu_ablation import score_model
from .summarize_math_gate_multiseed import mean_std
from .train_fixed_math_policy import ACTIONS, best_action, decision_metrics, digest, extract_features, write_json
from .utils import load_pickle, read_jsonl


def validate_inputs(root):
    manifest = json.loads((root / 'manifest.json').read_text())
    verify_snapshot(manifest)
    source = Path(manifest['source'])
    audit = json.loads((root / 'accounting_audit.json').read_text())
    if audit['manifest_sha256'] != digest(root / 'manifest.json') or \
            audit['trace_sha256'] != digest(root / 'policy_test_accounted.jsonl') or \
            audit.get('all_action_outcomes_accounted') is not True:
        raise ValueError('Accounted trace/manifest changed or is incomplete')
    for name, checksum in audit['call_logs_sha256'].items():
        if digest(name) != checksum:
            raise ValueError(f'Paid call changed: {name}')
    lock = json.loads((source / 'policy_lock.json').read_text())
    protocol = json.loads((source / 'protocol.json').read_text())
    for name, checksum in protocol['source_code_sha256'].items():
        if digest(Path(__file__).parent / name) != checksum:
            raise ValueError(f'Frozen evaluation dependency changed: {name}')
    rows = read_jsonl(root / 'policy_test_accounted.jsonl')
    items = read_jsonl(source / 'test_items.jsonl')
    if len(rows) != lock['test_n'] or [r['id'] for r in rows] != [i['id'] for i in items]:
        raise ValueError('Missing/duplicate/misordered test rows')
    if digest(lock['encoder_manifest']) != lock['encoder_manifest_sha256']:
        raise ValueError('Encoder provenance changed')
    signature = json.loads(Path(lock['encoder_manifest']).read_text())
    for name, checksum in signature['encoder_files_sha256'].items():
        if digest(Path(signature['encoder_name']) / name) != checksum:
            raise ValueError('Encoder parameters/tokenizer changed')
    if digest(lock['round_model']) != lock['round_sha256']:
        raise ValueError('Frozen Round changed')
    if len(lock['policies']) != 36 or len({p['name'] for p in lock['policies']}) != 36:
        raise ValueError('Require all 36 locked policies')
    for point in lock['policies']:
        if digest(point['model']) != point['sha256']:
            raise ValueError('Frozen Gate changed')
    return source, lock, rows, audit


def chosen_candidate(row, action):
    return row['independent'] if action == 'independent' else next(
        c for c in row['candidate_results'] if c['action'] == action)


def failure_metrics(rows, actions):
    metrics = decision_metrics(rows, actions)
    failed_ids = [r['id'] for r, a in zip(rows, actions, strict=True)
                  if chosen_candidate(r, a)['execution_status'] != 'success']
    metrics.update({'execution_failure_count': len(failed_ids), 'execution_failure_ids': failed_ids,
                    'execution_failure_rate': len(failed_ids) / len(rows),
                    'semantic_wrong_count': len(rows) - metrics['correct'] - len(failed_ids),
                    'accuracy_if_selected_failures_all_correct': (metrics['correct'] + len(failed_ids)) / len(rows)})
    return metrics


def evaluate(root):
    root = Path(root)
    source, lock, rows, audit = validate_inputs(root)
    output = root / 'evaluation'
    output.mkdir(exist_ok=True)
    summary_path = output / 'summary.json'
    if summary_path.exists():
        result = json.loads(summary_path.read_text())
        if result['accounting_audit_sha256'] != digest(root / 'accounting_audit.json'):
            raise ValueError('Existing summary belongs to a different audit')
        print(f'[accounted-eval] already complete: {summary_path}', flush=True)
        return
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import torch
    torch.set_num_threads(4)
    write_json(root / 'status.json', {'stage': 'encoding_test_cpu', 'n': len(rows), 'test_metrics_released': False})
    features, info = extract_features(rows, lock['encoder']['encoder_name'], root / 'test_feature_cache',
                                     lock['encoder']['max_length'], 2)
    write_json(output / 'encoder_audit.json', info)
    round_bundle = load_pickle(lock['round_model'])
    round_actions = np.array(round_bundle['action_labels'])[score_model(round_bundle, features).argmax(1)]
    round_ids = np.array([ACTIONS.index(a) for a in round_actions])
    n, ids = len(rows), np.arange(len(rows))
    table = make_tables(rows)
    base = table[:, 0]
    baseline_actions = {name: [action] * n for name, action in (
        ('direct', 'direct'), ('fixed_reflect_1', 'reflect_1'), ('fixed_reflect_2', 'reflect_2'),
        ('independent', 'independent'))}
    baseline_actions['oracle_accuracy_first'] = [best_action(r) for r in rows]
    baselines = {name: failure_metrics(rows, actions) for name, actions in baseline_actions.items()}
    independent = np.zeros_like(base)
    for i, row in enumerate(rows):
        c = row['independent']
        independent[i, :4] = [c['correct'], c['prompt_tokens'], c['completion_tokens'], c['token_cost']]
    rng = np.random.default_rng(DRAW_SEED)
    orders = {'all': random_orders(ids, lock['random_repetitions'], rng)}
    strata = {a: random_orders(ids[round_ids == a], lock['random_repetitions'], rng) for a in (1, 2)}
    groups, records, tests, cache, draws = {}, [], [], {}, {}
    write_json(root / 'status.json', {'stage': 'offline_test_evaluation', 'n': n, 'test_metrics_released': False})
    with (output / 'decisions.jsonl').open('w') as handle:
        for point in lock['policies']:
            actions, probabilities = frozen_actions(load_pickle(point['model']), features, round_actions,
                                                     point['branch'], point['direct_threshold'])
            selected = np.array([ACTIONS.index(a) for a in actions])
            learned = table[ids, selected]
            metrics = failure_metrics(rows, actions.tolist())
            branch = round_ids if point['branch'] == 'original_round' else np.ones(n, dtype=int)
            expected, random_info = {}, {}
            for mode in RANDOM_TYPES:
                counts = tuple(int((selected == a).sum()) for a in (1, 2))
                key = (point['branch'], mode, sum(counts) if mode == 'uniform_gate' else counts)
                if key not in cache:
                    local = {**orders, **strata} if point['branch'] == 'original_round' else {
                        **orders, 1: orders['all'], 2: np.empty((lock['random_repetitions'], 0), dtype=np.int32)}
                    cache[key] = sampled_metrics(base, table[ids, branch], selected, branch, mode, local)
                expectation, sampled = cache[key]
                expected[mode] = expectation
                random_info[mode] = {'expected': to_metrics(expectation), 'simulation': {
                    field: {**mean_std(sampled[:, j]), 'draw_q025': float(np.quantile(sampled[:, j], .025)),
                            'draw_q975': float(np.quantile(sampled[:, j], .975))} for j, field in enumerate(FIELDS)}}
                draws[f'{point["name"]}_{mode}'] = sampled
            key = (point['recipe'], point['branch'], point['validation_call_budget'])
            groups.setdefault(key, []).append({'seed': point['seed'], 'learned': learned, 'expected': expected,
                                               'execution_failure_rate': metrics['execution_failure_rate']})
            records.append({**point, 'metrics': metrics, 'random': random_info,
                            'test_rate_exceeds_validation_budget': metrics['reflection_rate'] > point['validation_call_budget'] + 1e-12})
            tests.append({'name': point['name'], 'mcnemar': _mcnemar(learned[:, 0], base[:, 0])})
            for i, row in enumerate(rows):
                handle.write(json.dumps({'policy': point['name'], 'id': row['id'], 'action': actions[i],
                    'direct_probability': float(probabilities[i]), 'direct_threshold': point['direct_threshold'],
                    'direct_correct': int(base[i, 0]), 'final_correct': int(learned[i, 0]),
                    'execution_status': chosen_candidate(row, actions[i])['execution_status'],
                    'prompt_tokens': int(learned[i, 1]), 'completion_tokens': int(learned[i, 2]),
                    'total_tokens': int(learned[i, 3])}) + '\n')
            print(f'[accounted-eval] applied frozen policy {point["name"]}', flush=True)
    for name, values in (('fixed_reflect_1', table[:, 1]), ('fixed_reflect_2', table[:, 2]), ('independent', independent)):
        tests.append({'name': name, 'mcnemar': _mcnemar(values[:, 0], base[:, 0])})
    for test, adjusted in zip(tests, holm_adjust([t['mcnemar']['exact_two_sided_p'] for t in tests])):
        test['holm_p'] = adjusted
    results = []
    for key, members in groups.items():
        if sorted(x['seed'] for x in members) != [7, 21, 42]:
            raise ValueError('Missing frozen seeds')
        learned = np.mean([x['learned'] for x in members], axis=0)
        refs = {'direct': base, 'fixed_reflect_1': table[:, 1], 'independent': independent,
                **{m: np.mean([x['expected'][m] for x in members], axis=0) for m in RANDOM_TYPES}}
        results.append({'recipe': key[0], 'branch': key[1], 'validation_call_budget': key[2],
            'metrics': {f: mean_std([to_metrics(x['learned'])[f] for x in members]) for f in FIELDS},
            'execution_failure_rate': mean_std([x['execution_failure_rate'] for x in members]),
            'random_expected': {m: {f: mean_std([to_metrics(x['expected'][m])[f] for x in members]) for f in FIELDS}
                                for m in RANDOM_TYPES},
            'paired_bootstrap': {ref: {f: paired_interval(learned[:, j] - value[:, j], lock['bootstrap_iterations'], BOOTSTRAP_SEED)
                for f, j in (('accuracy', 0), ('avg_token_cost', 3))} for ref, value in refs.items()}})
    result = {'analysis_type': audit['analysis_type'], 'n': n, 'baselines': baselines,
              'groups': results, 'policies': records, 'mcnemar_vs_direct': tests,
              'unresolved_execution_failures': audit['unresolved_execution_failures'],
              'clean_generation_complete': audit['clean_generation_complete'],
              'accounting_audit_sha256': digest(root / 'accounting_audit.json'),
              'original_policy_lock_sha256': digest(source / 'policy_lock.json'),
              'no_test_calibration': True, 'main_model_selected': False,
              'bootstrap_unit': 'question after averaging the three frozen seeds',
              'historical_limitation': lock['historical_limitation'], 'training_coverage': lock['training_coverage'],
              'encoder_info': info, 'random_draw_columns': FIELDS}
    lines = ['# GSM8K locked-policy evaluation: execution failures included', '',
        '**Post-hoc failure-accounting addendum, NOT a clean no-truncation confirmation.**',
        f'All {n} test questions retained. Unresolved failed actions: {len(audit["unresolved_execution_failures"])}.',
        'Exhausted generation receives zero answer credit, with a null prediction and a separate execution-failure flag.',
        'This is not a claim that the unfinished reasoning is a semantically incorrect completed answer.',
        'All failed-attempt input/output tokens are charged. No fallback answer or extra retry is fabricated.',
        'The original clean-only experiment remains blocked and unchanged.', lock['historical_limitation'], '',
        '## Baselines', '',
        '| Method | Accuracy | Input tokens | Output tokens | Total tokens | Fix/Harm | Failed actions |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name, m in baselines.items():
        lines.append(f"| {name} | {m['accuracy']:.4f} | {m['avg_prompt_tokens']:.2f} | {m['avg_completion_tokens']:.2f} | "
                     f"{m['avg_token_cost']:.2f} | {m['fixes']}/{m['harms']} | {m['execution_failure_count']} |")
    lines += ['', '## All frozen three-seed groups', '',
        '| Recipe | Branch | Validation budget | Accuracy mean +/- SD | Tokens mean +/- SD | Call rate | Failure rate |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for g in results:
        m = g['metrics']
        lines.append(f"| {g['recipe']} | {g['branch']} | {g['validation_call_budget']:.0%} | "
            f"{m['accuracy']['mean']:.4f} +/- {m['accuracy']['std']:.4f} | "
            f"{m['avg_token_cost']['mean']:.2f} +/- {m['avg_token_cost']['std']:.2f} | "
            f"{m['reflection_rate']['mean']:.4f} | {g['execution_failure_rate']['mean']:.4f} |")
    lines += ['', 'No thresholds, weights, seeds or decoding settings were selected using these test results.',
        'Validation call budgets are not hard test-rate constraints. All 36 locked points are reported; no main model is selected here.',
        'summary.json includes matched-call random controls (not equal-token controls), paired bootstrap intervals and Holm-adjusted McNemar tests.',
        'Sensitivity upper values assume all selected execution failures could become correct; they are not observed accuracies or forecasts.',
        'Independent strong generation excludes Base cost. Policy cost includes Base and every executed reflection attempt.',
        'Local policy latency is not converted into tokens. Cross-model token counts are not equivalent FLOPs or money.',
        'The Oracle is a retrospective upper bound over observed Direct/R1/R2 paths, not a deployable method.', '']
    (output / 'REPORT.md').write_text('\n'.join(lines))
    np.savez_compressed(output / 'random_draws.npz', **draws)
    validate_inputs(root)
    write_json(summary_path, result)
    write_json(root / 'status.json', {'stage': 'completed_failure_inclusive', 'test_metrics_released': True,
        'n': n, 'clean_generation_complete': audit['clean_generation_complete'], 'report': str(output / 'REPORT.md')})
    print(f'[accounted-eval] complete: {output / "REPORT.md"}', flush=True)
