"""Decompose frozen validation decisions without tuning or new model generation."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import fcntl
import json
import os
from pathlib import Path

import numpy as np
import torch

from .math_gate_cpu_ablation import read_cached_features, score_model
from .train_fixed_math_policy import (ACTIONS, audit_split, best_action, candidates,
                                      decision_metrics, digest, write_json)
from .utils import load_pickle

FILTERS = ('gate_miss', 'round_miss', 'triggered_unrepairable', 'introduced_error',
           'correct_direct_redundant_reflection', 'nonbeneficial_second_round')


def classify_case(row, triggered, round_action, direct_probability=None, threshold=None):
    if round_action not in ACTIONS[1:]:
        raise ValueError('Round action must be reflect_1 or reflect_2')
    cs = candidates(row)
    if set(cs) != set(ACTIONS):
        raise ValueError(f"Complete observed candidates required: {row['id']}")
    d, r1, r2 = (bool(cs[a]['correct']) for a in ACTIONS)
    selected = round_action if triggered else 'direct'
    final = bool(cs[selected]['correct'])
    predicted_round_correct = bool(cs[round_action]['correct'])
    repairable = not d and (r1 or r2)
    flags = {
        'repairable': repairable,
        'fixed': not d and final,
        'gate_miss': repairable and not triggered,
        'round_miss': repairable and triggered and not final,
        'introduced_error': d and not final,
        'triggered_unrepairable': triggered and not d and not (r1 or r2),
        'correct_direct_redundant_reflection': triggered and d and final,
        'triggered_no_positive_accuracy_gain': triggered and not (not d and final),
        'gate_miss_round_would_succeed': repairable and not triggered and predicted_round_correct,
        'gate_miss_round_would_fail': repairable and not triggered and not predicted_round_correct,
        'harm_avoidable_by_round': d and not final and (r1 or r2),
        'harm_unavoidable_after_trigger': d and not final and not (r1 or r2),
    }
    second = selected == 'reflect_2'
    kind = ('benefit' if r2 and not r1 else 'harm' if r1 and not r2
            else 'both_correct' if r1 and r2 else 'both_wrong') if second else 'not_called'
    flags['nonbeneficial_second_round'] = second and kind != 'benefit'
    if flags['gate_miss']:
        category = 'gate_miss'
    elif flags['round_miss']:
        category = 'round_miss'
    elif flags['fixed']:
        category = 'fixed'
    elif flags['introduced_error']:
        category = 'introduced_error'
    elif not d:
        category = 'unrepairable_in_saved_candidates'
    else:
        category = 'correct_preserved'
    return {
        'id': row['id'], 'question': row['question'], 'gold_answer': row.get('gold_answer'),
        'initial_pred': cs['direct'].get('pred'), 'reflect_1_pred': cs['reflect_1'].get('pred'),
        'reflect_2_pred': cs['reflect_2'].get('pred'), 'selected_pred': cs[selected].get('pred'),
        'direct_correct': d, 'reflect_1_correct': r1, 'reflect_2_correct': r2,
        'triggered': bool(triggered), 'round_action': round_action, 'selected_action': selected,
        'final_correct': final, 'category': category, 'flags': flags,
        'direct_probability': None if direct_probability is None else float(direct_probability),
        'direct_threshold': threshold,
        'token_cost': cs[selected]['token_cost'], 'prompt_tokens': cs[selected]['prompt_tokens'],
        'completion_tokens': cs[selected]['completion_tokens'],
        'reflection_extra_tokens_over_direct': cs[selected]['token_cost'] - cs['direct']['token_cost'],
        'second_round_kind': kind,
        'second_round_extra_tokens_vs_saved_r1': cs['reflect_2']['token_cost'] - cs['reflect_1']['token_cost'] if second else 0,
    }


def summarize(rows, cases, lambda_cost=.0001):
    if not rows or len(rows) != len(cases) or [r['id'] for r in rows] != [r['id'] for r in cases]:
        raise ValueError('Decision rows must be aligned with source examples')
    flags = Counter({key: sum(bool(c['flags'][key]) for c in cases) for key in cases[0]['flags']})
    triggered = sum(c['triggered'] for c in cases)
    base_wrong = sum(not c['direct_correct'] for c in cases)
    selected = [c['selected_action'] for c in cases]
    metrics = decision_metrics(rows, selected, lambda_cost)
    oracle = decision_metrics(rows, [best_action(r) for r in rows], lambda_cost)
    gate_oracle_round = [best_action(r, ACTIONS[1:]) if c['triggered'] else 'direct' for r, c in zip(rows, cases)]
    round_oracle_gate = [c['round_action'] if not c['direct_correct'] and candidates(r)[c['round_action']]['correct']
                        else 'direct' for r, c in zip(rows, cases)]
    fixed_r1 = ['reflect_1' if c['triggered'] else 'direct' for c in cases]
    same_gate_r1 = decision_metrics(rows, fixed_r1, lambda_cost)
    if flags['repairable'] != flags['fixed'] + flags['gate_miss'] + flags['round_miss']:
        raise AssertionError('Repairable partition failed')
    gap = flags['gate_miss'] + flags['round_miss'] + flags['introduced_error']
    if oracle['correct'] - metrics['correct'] != gap:
        raise AssertionError('Oracle accuracy gap decomposition failed')
    if triggered != sum(flags[k] for k in ('fixed', 'round_miss', 'triggered_unrepairable',
                                          'correct_direct_redundant_reflection', 'introduced_error')):
        raise AssertionError('Triggered decision partition failed')
    if flags['gate_miss'] != flags['gate_miss_round_would_succeed'] + flags['gate_miss_round_would_fail']:
        raise AssertionError('Gate miss counterfactual partition failed')
    second = {}
    for kind in ('benefit', 'both_correct', 'both_wrong', 'harm'):
        group = [c for c in cases if c['second_round_kind'] == kind]
        second[kind] = {'n': len(group), 'extra_total_tokens_vs_saved_r1': sum(c['second_round_extra_tokens_vs_saved_r1'] for c in group)}
    extra_second_tokens = sum(c['second_round_extra_tokens_vs_saved_r1'] for c in cases)
    if not np.isclose(metrics['avg_token_cost'] - same_gate_r1['avg_token_cost'], extra_second_tokens / len(rows)):
        raise AssertionError('Second-round token decomposition failed')
    if metrics['correct'] - same_gate_r1['correct'] != second['benefit']['n'] - second['harm']['n']:
        raise AssertionError('Second-round correctness decomposition failed')
    return {
        'n': len(rows), 'base_wrong': base_wrong, 'base_correct': len(rows) - base_wrong,
        'repairable': flags['repairable'], 'unrepairable_in_saved_candidates': base_wrong - flags['repairable'],
        'triggered': triggered, 'counts': dict(flags), 'categories': dict(Counter(c['category'] for c in cases)),
        'metrics': metrics, 'oracle': oracle, 'oracle_gap_questions': gap,
        'gate_recall_on_repairable': (flags['repairable'] - flags['gate_miss']) / flags['repairable'] if flags['repairable'] else None,
        'repairable_fraction_among_triggered': (flags['repairable'] - flags['gate_miss']) / triggered if triggered else None,
        'counterfactuals_not_deployable': {
            'fixed_gate_oracle_round': decision_metrics(rows, gate_oracle_round, lambda_cost),
            'fixed_round_oracle_gate': decision_metrics(rows, round_oracle_gate, lambda_cost),
        },
        'same_gate_fixed_r1': same_gate_r1, 'second_round': second,
        'second_round_extra_total_tokens_vs_saved_r1': extra_second_tokens,
        'extra_reflection_tokens_over_direct_by_outcome': {
            key: sum(c['reflection_extra_tokens_over_direct'] for c in cases if c['flags'][key])
            for key in ('fixed', 'round_miss', 'triggered_unrepairable',
                        'correct_direct_redundant_reflection', 'introduced_error')},
        'identities_verified': True,
    }


def write_jsonl(path, rows):
    temporary = path.with_suffix('.jsonl.tmp')
    with temporary.open('w') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def make_report(path, results, audit):
    lines = ['# 固定策略的 Gate/Round 误差归因', '',
             '所有结果来自同一内部验证集，不重新训练、搜索阈值或生成答案。Oracle 仅用于事后诊断，不是可部署结果。',
             f"原定验证 {audit['validation']['expected']} 题，当前完整候选 {audit['validation']['observed']} 题。",
             '训练缺失 2 题、验证缺失 3 题仍保留于原生成审计，不属于下面可分类的完整候选。', '',
             '## 准确率差距分解', '',
             '| 策略 | 实际调用率 | 可修复 | 已修复 | Gate 漏检 | 已触发但 Round 选错 | 引错 | Oracle 差距题数 |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name, s in results.items():
        c = s['counts']
        lines.append(f"| {name} | {s['metrics']['reflection_rate']:.2%} | {s['repairable']} | {c['fixed']} | "
                     f"{c['gate_miss']} | {c['round_miss']} | {c['introduced_error']} | {s['oracle_gap_questions']} |")
    lines += ['', '对每个策略均核验两条恒等式：', '',
              '- 可修复题数 = 已修复 + Gate 漏检 + 已触发但 Round 选错。',
              '- 完整候选 Oracle 正确题数 - 策略正确题数 = Gate 漏检 + Round 误选 + 引入错误。', '',
              '这里 Round 误选专指 Direct 原本错误且存在正确反思的题；Direct 原本正确时另归入引错，并标注另一轮是否可避免，防止重复计数。', '',
              '## 调用没有带来正向准确率收益的情况', '',
              '| 策略 | 调用数 | 原本正确且保持正确 | 调用了但所有反思候选均错 | Round 误选 | 引错 |',
              '| --- | ---: | ---: | ---: | ---: | ---: |']
    for name, s in results.items():
        c = s['counts']
        lines.append(f"| {name} | {s['triggered']} | {c['correct_direct_redundant_reflection']} | "
                     f"{c['triggered_unrepairable']} | {c['round_miss']} | {c['introduced_error']} |")
    lines += ['', '这里“无正向准确率收益”是已保存候选的事后属性，不表示部署时已知这些题无需反思。', '',
              '## 第二轮相对同题保存的第一轮', '',
              '| 策略 | R2 调用 | R1错/R2对 | 两轮都对 | 两轮都错 | R1对/R2错 | 非正向第二轮额外总 token |',
              '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name, s in results.items():
        r = s['second_round']
        redundant_cost = sum(r[k]['extra_total_tokens_vs_saved_r1'] for k in ('both_correct', 'both_wrong', 'harm'))
        lines.append(f"| {name} | {sum(v['n'] for v in r.values())} | {r['benefit']['n']} | {r['both_correct']['n']} | "
                     f"{r['both_wrong']['n']} | {r['harm']['n']} | {redundant_cost} |")
    lines += ['', '第二轮成本按 R2 路径总 token 减同题 R1 路径总 token 计算，并与相同 Gate 的固定 R1 对照总差核对。', '',
              '## 文件说明', '',
              '- `summary.json` 包含所有计数、分母、成本分解和不可部署的局部 Oracle 诊断。',
              '- `examples.jsonl` 保存本次完整候选题的问题、标准答案及三种原始回答，供按 ID 查阅。',
              '- 每个策略目录的 `decisions.jsonl` 保存全部逐题归因、动作、概率、正确性和成本。',
              '- `gate_miss.jsonl`、`round_miss.jsonl`：分别为 Gate 漏检和真正的已触发轮次误选。',
              '- `triggered_unrepairable.jsonl`：触发后，两种已保存反思均未修复，不能归咎于 Round 选错。',
              '- `introduced_error.jsonl`：Direct 对、最终错；另标明改选另一轮是否能避免。',
              '- `correct_direct_redundant_reflection.jsonl`：原本正确、反思后仍正确的额外调用。',
              '- `nonbeneficial_second_round.jsonl`：与 R1 同正确性或更差的 R2 调用，区分两轮均对/均错/引错。',
              '- `decisions.csv` 是方便表格查看的同一逐题清单。各文件会有重叠标记，不能任意相加；总体统计用上述明确分区。', '',
              '## 解释限制', '',
              '局部 Oracle 只替换一个组件的决定，用金答案回溯，不是已经实现的可用模型，也不保证保留原调用预算。',
              'Gate 漏检题会同时标记当前 Round 在触发后是否能选对，避免把两组件同时修好的收益算给 Gate。',
              '逐题错误清单用于诊断，不回流训练集；现有内部验证已参与开发，后续结论仍需独立评估。',
              '“不可修复”仅指这两条已保存候选均错，不是断言模型永远不能修复。',
              '此分析没有设置精度容差、挑选新主模型或修改任何已有检查点。', '']
    path.write_text('\n'.join(lines))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ablation', type=Path, default=Path('outputs/math_gate_cpu_ablation_20260916_seed42'))
    p.add_argument('--root', type=Path, default=Path('outputs/math_full_training_v1'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--threads', type=int, default=4)
    args = p.parse_args()
    if args.threads < 1:
        p.error('threads must be positive')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    torch.set_num_threads(args.threads)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'run.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (args.output / 'manifest.json').exists():
            raise ValueError('Output already exists; use a new directory')
        experiment = json.loads((args.ablation / 'summary.json').read_text())
        protocol = json.loads((args.ablation / 'protocol.json').read_text())
        source = Path(protocol['source'])
        prior = json.loads((source / 'selected_validation.json').read_text())
        train, val, audit = audit_split(args.root, True)
        if audit != experiment['coverage'] or audit != prior['coverage']:
            raise ValueError('Data/split differs from the frozen experiments')
        x, info = read_cached_features(Path(prior['encoder']['cache']), train + val, prior['encoder']['fingerprint'])
        vx = x[len(train):]
        round_bundle = load_pickle(args.ablation / 'reference_round.pkl')
        if digest(args.ablation / 'reference_round.pkl') != digest(source / 'round.pkl'):
            # Pickling may serialize equivalent bundles differently; compare source scores too.
            source_round = load_pickle(source / 'round.pkl')
            if not np.array_equal(score_model(round_bundle, vx), score_model(source_round, vx)):
                raise ValueError('Round differs from source checkpoint')
        round_probs = score_model(round_bundle, vx)
        round_actions = np.array(round_bundle['action_labels'])[round_probs.argmax(1)]
        points = [point for point in experiment['comparisons'] if point['branch'] == 'original_round']
        points.append({'recipe': 'original_unmodified', 'budget': None, 'epoch': prior['best_epoch'],
                       'gate_model': str(source / 'gate.pkl'), 'direct_threshold': prior['direct_threshold'],
                       'metrics': prior['metrics']})
        write_json(args.output / 'manifest.json', {
            'scope': 'Read-only diagnostic on fixed internal validation; no tuning or generation',
            'coverage': audit, 'feature_fingerprint': info['fingerprint'],
            'ablation_summary_sha256': digest(args.ablation / 'summary.json'),
            'source_code_sha256': digest(__file__),
            'reference_round_sha256': digest(args.ablation / 'reference_round.pkl'),
            'gate_hashes': {point['gate_model']: digest(point['gate_model']) for point in points},
            'accuracy_tolerance': None, 'test_evaluated': False})
        write_json(args.output / 'status.json', {'stage': 'analyzing'})
        try:
            write_jsonl(args.output / 'examples.jsonl', ({
                'id': r['id'], 'question': r['question'], 'gold_answer': r.get('gold_answer'),
                'answers': {a: {k: c.get(k) for k in ('final_answer', 'pred', 'correct', 'prompt_tokens', 'completion_tokens', 'token_cost')}
                            for a, c in candidates(r).items()}}
                for r in val))
            results = {}
            for point in points:
                name = point['recipe'] + (f"__budget{round(point['budget'] * 100):02d}" if point['budget'] is not None else '')
                folder = args.output / name
                folder.mkdir()
                gate = load_pickle(point['gate_model'])
                direct = score_model(gate, vx)[:, 0]
                threshold = gate['direct_threshold']
                if threshold != point['direct_threshold']:
                    raise ValueError('Saved gate threshold differs from reported point')
                trigger = direct < threshold
                cases = [classify_case(r, bool(t), str(a), prob, threshold)
                         for r, t, a, prob in zip(val, trigger, round_actions, direct)]
                for case, probs in zip(cases, round_probs):
                    case['round_probabilities'] = dict(zip(round_bundle['action_labels'], map(float, probs)))
                s = summarize(val, cases, float(gate.get('lambda_cost', .0001)))
                if s['metrics'] != point['metrics']:
                    raise ValueError(f'Replay disagrees with reported outcomes: {name}')
                s['checkpoint'] = {k: point[k] for k in ('gate_model', 'recipe', 'epoch', 'budget')}
                results[name] = s
                write_json(folder / 'summary.json', s)
                write_jsonl(folder / 'decisions.jsonl', cases)
                for flag in FILTERS:
                    write_jsonl(folder / f'{flag}.jsonl', (c for c in cases if c['flags'][flag]))
                csv_keys = [k for k in cases[0] if k not in ('flags', 'round_probabilities')]
                with (folder / 'decisions.csv').open('w', newline='') as stream:
                    writer = csv.DictWriter(stream, fieldnames=csv_keys)
                    writer.writeheader()
                    writer.writerows({k: c[k] for k in csv_keys} for c in cases)
                c = s['counts']
                print(f"[errors] {name}: repairable={s['repairable']} fixed={c['fixed']} "
                      f"gate_miss={c['gate_miss']} round_miss={c['round_miss']} harm={c['introduced_error']}", flush=True)
            write_json(args.output / 'summary.json', {'coverage': audit, 'policies': results, 'test_evaluated': False})
            make_report(args.output / 'REPORT.md', results, audit)
            write_json(args.output / 'status.json', {'stage': 'completed', 'policies': len(results),
                                                    'all_decompositions_verified': True, 'test_evaluated': False})
        except Exception as exc:
            write_json(args.output / 'status.json', {'stage': 'failed', 'error': repr(exc)})
            raise


if __name__ == '__main__':
    main()
