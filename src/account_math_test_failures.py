"""Separate, post-hoc execution-failure accounting; never clear the clean-test audit."""
from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path

from . import math_reflector_pilot as pilot
from .math_full_test_trace import check_generation_protocol, validate_failure, validate_row
from .math_full_training_trace import CachedOnly, make_policy_row
from .math_trace_failures import exhausted_call, read_failures, wait_for_resources
from .prepare_qa_hard_development import locked_write
from .train_fixed_math_policy import digest, write_json
from .utils import read_jsonl

DEFAULT_SOURCE = pilot.PROJECT / 'outputs/math_full_test_v1'
DEFAULT_ROOT = pilot.PROJECT / 'outputs/math_full_test_failure_accounted_v1'
USAGE = ('prompt_tokens', 'completion_tokens', 'total_tokens')


def snapshot(source):
    files = [source / name for name in ('protocol.json', 'policy_lock.json', 'test_items.jsonl',
              'direct_test.jsonl', 'policy_test_partial.jsonl', 'completion_audit.json', 'status.json')]
    files += sorted((source / 'calls').glob('*/*/*.jsonl'))
    files += sorted((source / 'failures').glob('*/*.json'))
    return {str(p): digest(p) for p in files}


def verify_snapshot(manifest):
    if snapshot(Path(manifest['source'])) != manifest['source_files_sha256']:
        raise ValueError('Original traces/ledger changed; refusing to mix runs')
    for name, checksum in manifest['implementation_sha256'].items():
        if digest(Path(__file__).parent / name) != checksum:
            raise ValueError(f'Accounting implementation changed: {name}')


def accounted_candidate(action, calls, gold):
    failures = [c for c in calls if c.get('execution_status') == 'output_budget_exhausted']
    if not failures:
        return {**pilot.candidate(action, calls, gold), 'execution_status': 'success',
                'scored_as_semantic_answer': True}
    if len(failures) != 1 or calls[-1] is not failures[0]:
        raise ValueError('Only observed terminal execution failures are supported')
    p = sum(c['prompt_tokens'] for c in calls)
    c = sum(c['completion_tokens'] for c in calls)
    return {'action': action, 'final_answer': None, 'pred': None, 'correct': 0,
            'prompt_tokens': p, 'completion_tokens': c, 'total_tokens': p + c,
            'token_cost': p + c, 'reward': -.0001 * (p + c), 'calls': calls,
            'final_marker_present': False, 'execution_status': 'output_budget_exhausted',
            'scored_as_semantic_answer': False,
            'score_rule': 'execution_failure_zero_credit_not_semantic_incorrectness'}


def cached_outcome(root, tokenizer, key, action, prompt):
    failed = exhausted_call(root, 'qwen14', key, action, prompt, tokenizer)
    if failed:
        return {**failed, 'execution_status': 'output_budget_exhausted'}
    return pilot.complete_call(CachedOnly(), tokenizer, root, 'qwen14', key, action, prompt)


def reconstruct(item, base, source, root, tokenizer):
    key = item['id']
    calls = [base['initial_call']]
    results = [accounted_candidate('direct', calls, item['gold_answer'])]
    for round_id in (1, 2):
        if calls[-1].get('execution_status') == 'output_budget_exhausted':
            raise ValueError('Missing downstream reflection is not an observed failure')
        prompt = pilot.REFLECTION_TEMPLATE.format(question=item['question'],
                 current_answer=calls[-1]['answer_text'], round_id=round_id)
        action = f'reflect_{round_id}'
        calls = [*calls, cached_outcome(source, tokenizer, key, action, prompt)]
        results.append(accounted_candidate(action, calls, item['gold_answer']))
    prompt = pilot.MATH_DIRECT_PROMPT.format(question=item['question'])
    call_root = source if (source / 'calls/qwen14' / key / 'independent.jsonl').exists() else root
    independent = accounted_candidate('independent',
        [cached_outcome(call_root, tokenizer, key, 'independent', prompt)], item['gold_answer'])
    row = make_policy_row(base, results, independent)
    row['candidate_coverage'] = 'all_actions_accounted_including_execution_failures'
    row['label_evidence'] = 'posthoc_execution_failure_accounting_not_training_labels'
    row['execution_failures'] = [c['action'] for c in [*results, independent]
                               if c['execution_status'] != 'success']
    return row


def supplement(source, root, items, tokenizer, generate):
    missing = [i for i in items if not (source / 'calls/qwen14' / i['id'] / 'independent.jsonl').exists()]
    needed = []
    for item in missing:
        path = root / 'calls/qwen14' / item['id'] / 'independent.jsonl'
        prompt = pilot.MATH_DIRECT_PROMPT.format(question=item['question'])
        if path.exists():
            records = read_jsonl(path)
            if records and (records[-1]['metadata']['finish_reason'] == 'stop' or
                            exhausted_call(root, 'qwen14', item['id'], 'independent', prompt, tokenizer)):
                cached_outcome(root, tokenizer, item['id'], 'independent', prompt)
                continue
        needed.append(item)
    print(f'[failure-accounting] missing independent calls={len(missing)}, remaining={len(needed)}', flush=True)
    if needed and not generate:
        return False
    if needed:
        wait_for_resources(root, 'qwen14')
        with pilot.server(root, 'qwen14') as client:
            for item in needed:
                write_json(root / 'status.json', {'stage': 'supplement_independent', 'id': item['id'],
                           'test_metrics_released': False})
                prompt = pilot.MATH_DIRECT_PROMPT.format(question=item['question'])
                try:
                    pilot.complete_call(client, tokenizer, root, 'qwen14', item['id'], 'independent', prompt)
                except RuntimeError:
                    if not exhausted_call(root, 'qwen14', item['id'], 'independent', prompt, tokenizer):
                        raise
    return True


def export(root, source, items, base, tokenizer, manifest):
    rows = []
    for i, item in enumerate(items):
        rows.append(reconstruct(item, base[item['id']], source, root, tokenizer))
        if (i + 1) % 100 == 0:
            print(f'[failure-accounting] accounted {i + 1}/{len(items)}', flush=True)
    observed = {}
    for row in rows:
        for candidate in [*row['candidate_results'], row['independent']]:
            for call in candidate['calls']:
                path = call['call_log']
                if path in observed and observed[path] != call:
                    raise ValueError('Inconsistent shared call record')
                observed[path] = call
    paid = {str(p) for folder in (source, root) for p in (folder / 'calls').glob('*/*/*.jsonl')}
    if set(observed) != paid or len(observed) != 4 * len(items):
        raise ValueError('Missing/unaccounted/extra paid calls')
    for path, call in observed.items():
        if digest(path) != call['call_log_sha256']:
            raise ValueError('Call changed during reconstruction')
    verify_snapshot(manifest)
    trace = root / 'policy_test_accounted.jsonl'
    locked_write(trace, ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    failed = [{'id': r['id'], 'action': c['action'], 'call_log': c['calls'][-1]['call_log'],
               'path_total_tokens': c['token_cost']}
              for r in rows for c in [*r['candidate_results'], r['independent']]
              if c['execution_status'] != 'success']
    audit = {'n': len(rows), 'all_action_outcomes_accounted': True,
             'unresolved_execution_failures': failed, 'clean_generation_complete': not failed,
             'analysis_type': 'posthoc_failure_inclusive_not_clean_confirmation',
             'failure_rule': 'zero answer credit; null prediction; retain all actual usage',
             'manifest_sha256': digest(root / 'manifest.json'), 'trace_sha256': digest(trace),
             'logical_calls': len(observed), 'paid_attempts': sum(c['attempt_count'] for c in observed.values()),
             'generation_usage': {f: sum(c[f] for c in observed.values()) for f in USAGE},
             'call_logs_sha256': {p: c['call_log_sha256'] for p, c in observed.items()},
             'no_test_calibration': True, 'no_original_artifact_changes': True}
    pilot.locked_json(root / 'accounting_audit.json', audit)
    write_json(root / 'status.json', {'stage': 'accounted_ready_for_offline_evaluation',
               'n': len(rows), 'execution_failure_count': len(failed), 'test_metrics_released': False})
    print(f'[failure-accounting] exported {len(rows)} rows, {len(failed)} explicit failed actions', flush=True)


def run(source, root, generate=False, evaluate=False):
    if source == root or source in root.parents:
        raise ValueError('Use a separate sibling output directory')
    protocol = json.loads((source / 'protocol.json').read_text())
    check_generation_protocol(protocol['training_protocol'])
    for name, checksum in protocol['source_code_sha256'].items():
        if digest(pilot.PROJECT / 'src' / name) != checksum:
            raise ValueError(f'Frozen source changed: {name}')
    for role, spec in protocol['roles'].items():
        pilot.ROLES[role].update(spec)
    manifest = {'source': str(source), 'source_files_sha256': snapshot(source),
                'implementation_sha256': {n: digest(Path(__file__).parent / n) for n in
                    ('account_math_test_failures.py', 'evaluate_accounted_math_test.py')},
                'posthoc_amendment': True, 'protocol_unchanged': True,
                'only_new_generations': 'independent strong calls skipped by exhausted R2',
                'failure_score': 0, 'retain_failed_call_usage': True,
                'original_clean_evaluation_remains_blocked': True, 'no_test_calibration': True}
    pilot.locked_json(root / 'manifest.json', manifest)
    lock = json.loads((source / 'policy_lock.json').read_text())
    if digest(source / 'test_items.jsonl') != lock['test_items_sha256']:
        raise ValueError('Test items changed')
    items = read_jsonl(source / 'test_items.jsonl')
    bases = read_jsonl(source / 'direct_test.jsonl')
    if len(items) != lock['test_n'] or [r['id'] for r in bases] != [i['id'] for i in items]:
        raise ValueError('Require complete ordered base and test coverage')
    base = {r['id']: r for r in bases}
    completed = read_jsonl(source / 'policy_test_partial.jsonl')
    failures = read_failures(source, 'qwen14')
    completed_ids = [r['id'] for r in completed]
    if len(set(completed_ids)) != len(completed_ids) or set(completed_ids) & set(failures) or \
            set(completed_ids) | set(failures) != set(base):
        raise ValueError('Original completed and failed coverage differs')
    from transformers import AutoTokenizer
    tokenizers = {r: AutoTokenizer.from_pretrained(str(pilot.MODELS / pilot.ROLES[r]['model']),
                   local_files_only=True) for r in ('base', 'qwen14')}
    write_json(root / 'status.json', {'stage': 'auditing_original_artifacts', 'test_metrics_released': False})
    lookup = {i['id']: i for i in items}
    for row in bases:
        validate_row(source, row, lookup[row['id']], 'base', tokenizers)
    for i, row in enumerate(completed):
        validate_row(source, row, lookup[row['id']], 'qwen14', tokenizers, base[row['id']])
        if (i + 1) % 100 == 0:
            print(f'[failure-accounting] audited {i + 1}/{len(completed)} original complete rows', flush=True)
    for key, record in failures.items():
        validate_failure(source, 'qwen14', record, lookup[key], tokenizers, base)
    missing = [i['id'] for i in items if not (source / 'calls/qwen14' / i['id'] / 'independent.jsonl').exists()]
    if any(k not in failures or failures[k]['failed_call']['action'] != 'reflect_2' for k in missing):
        raise ValueError('New calls must be independent calls skipped after a documented R2 failure')
    if supplement(source, root, items, tokenizers['qwen14'], generate):
        export(root, source, items, base, tokenizers['qwen14'], manifest)
        if evaluate:
            from .evaluate_accounted_math_test import evaluate as evaluate_accounted
            evaluate_accounted(root)
    else:
        verify_snapshot(manifest)
        write_json(root / 'status.json', {'stage': 'audit_passed_awaiting_supplement',
                   'missing_independent_ids': missing, 'test_metrics_released': False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--generate', action='store_true')
    parser.add_argument('--evaluate', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.run.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            run(args.source.resolve(), root, args.generate, args.evaluate)
        except Exception as exc:
            write_json(root / 'last_error.json', {'type': type(exc).__name__, 'message': str(exc)})
            raise


if __name__ == '__main__':
    main()
