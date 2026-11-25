"""Generate complete frozen transfer paths, retaining terminal failure costs."""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
from pathlib import Path
import signal
import time

from . import math_reflector_pilot as pilot
from .account_math_test_failures import accounted_candidate
from .math_full_test_trace import check_generation_protocol
from .math_full_training_trace import CachedOnly, make_policy_row, role_placements
from .math_trace_failures import exhausted_call, wait_for_resources
from .prepare_qa_hard_development import file_hash, locked_write
from .train_fixed_math_policy import write_json
from .utils import read_jsonl

DEFAULT_ROOT = pilot.PROJECT / 'outputs/math_transfer_locked_v1'


def verify_inputs(root):
    manifest = json.loads((root / 'manifest.json').read_text())
    for name in ('policy_lock', 'generation_protocol'):
        if file_hash(root / f'{name}.json') != manifest[f'{name}_sha256']:
            raise ValueError(f'Frozen {name} changed')
    for path, sha in manifest['reference_files_sha256'].items():
        if file_hash(Path(path)) != sha:
            raise ValueError('Reference split changed')
    protocol = json.loads((root / 'generation_protocol.json').read_text())
    check_generation_protocol(protocol)
    lock = json.loads((root / 'policy_lock.json').read_text())
    if len(lock['policies']) != 36:
        raise ValueError('All 36 policies required')
    for point in lock['policies']:
        if file_hash(Path(point['model'])) != point['sha256']:
            raise ValueError('Gate changed')
    if file_hash(Path(lock['round_model'])) != lock['round_sha256']:
        raise ValueError('Round changed')
    items = {}
    for dataset, spec in manifest['datasets'].items():
        if file_hash(Path(spec['source'])) != spec['source_sha256']:
            raise ValueError('Source dataset changed')
        path = root / dataset / 'test_items.jsonl'
        if file_hash(path) != spec['items_sha256']:
            raise ValueError('Frozen items changed')
        items[dataset] = read_jsonl(path)
        if [r['id'] for r in items[dataset]] != spec['subsets']['full']:
            raise ValueError('Test order or denominator changed')
    return protocol, items


def resolve_call(root, role, item, action, prompt, tokenizer, get_client):
    diagnostic = exhausted_call(root, role, item['id'], action, prompt, tokenizer)
    if diagnostic:
        return {**diagnostic, 'execution_status': 'output_budget_exhausted'}
    path = root / 'calls' / role / item['id'] / f'{action}.jsonl'
    records = read_jsonl(path) if path.exists() else []
    # Replay completed calls without loading a model; incomplete paid retries are resumed.
    client = CachedOnly() if records and records[-1]['metadata']['finish_reason'] == 'stop' else get_client()
    try:
        return pilot.complete_call(client, tokenizer, root, role, item['id'], action, prompt)
    except RuntimeError:
        diagnostic = exhausted_call(root, role, item['id'], action, prompt, tokenizer)
        if diagnostic is None:
            raise  # Network, context, malformed output or unknown usage must stop, not become a wrong answer.
        return {**diagnostic, 'execution_status': 'output_budget_exhausted'}


def blocked_candidate(action, previous):
    if previous['execution_status'] == 'success':
        raise ValueError('Cannot block after success')
    return {**previous, 'action': action, 'execution_status': 'blocked_upstream_failure',
            'blocked_by': previous['action'], 'additional_calls_made': 0,
            'score_rule': 'upstream_failure_zero_system_credit_no_fictitious_downstream_call'}


def base_row(item, complete):
    call = complete('direct', pilot.MATH_DIRECT_PROMPT.format(question=item['question']))
    direct = accounted_candidate('direct', [call], item['gold_answer'])
    return {**item, 'policy_split': 'test', 'initial_call': call,
            'initial_answer': call.get('answer_text', ''), 'initial_pred': direct['pred'],
            'initial_execution_status': direct['execution_status'], 'candidate_results': [direct]}


def reflection_row(base, complete):
    results = list(base['candidate_results'])
    for round_id in (1, 2):
        action = f'reflect_{round_id}'
        previous = results[-1]
        if previous['execution_status'] != 'success':
            results.append(blocked_candidate(action, previous))
            continue
        prompt = pilot.REFLECTION_TEMPLATE.format(question=base['question'],
                     current_answer=previous['calls'][-1]['answer_text'], round_id=round_id)
        call = complete(action, prompt)
        results.append(accounted_candidate(action, [*previous['calls'], call], base['gold_answer']))
    # Independent solving does not depend on either reflection or initial completion.
    independent = accounted_candidate('independent', [complete('independent',
        pilot.MATH_DIRECT_PROMPT.format(question=base['question']))], base['gold_answer'])
    row = make_policy_row(base, results, independent)
    row.update(candidate_coverage='all_paths_accounted_including_execution_failures',
               label_evidence='test_only_not_training_labels',
               policy_input_available=base['initial_execution_status'] == 'success')
    return row


def row_path(root, dataset, role, key):
    return root / dataset / 'rows' / role / f'{key}.json'


def preflight(root, protocol, items, base_gpu, reflector_gpus, tokenizers):
    roles = role_placements(protocol, base_gpu, reflector_gpus)
    for role, spec in roles.items():
        pilot.ROLES[role].update(spec)
    source_files = ('run_locked_math_transfer.py', 'account_math_test_failures.py',
                    'math_full_training_trace.py', 'math_full_test_trace.py', 'math_trace_failures.py',
                    'train_fixed_math_policy.py', 'gsm8k_utils.py', 'utils.py',
                    'prepare_qa_hard_development.py')
    run = {'preparation_manifest_sha256': file_hash(root / 'manifest.json'), 'roles': roles,
           'implementation_sha256': {name: file_hash(pilot.PROJECT / 'src' / name) for name in source_files},
           'launcher_sha256': file_hash(pilot.PROJECT / 'scripts/run_locked_math_transfer.sh'),
           'generation_protocol_sha256': file_hash(root / 'generation_protocol.json'),
           'failure_accounting': 'terminal length gets zero system credit and null prediction; dependent paths blocked with upstream paid usage; independent still generated',
           'other_errors': 'stop and retain paid logs; no completed export until resolved',
           'initial_failure_deployment': 'no valid policy input; return failed Direct with its cost, do not invent an initial response',
           'gpu_note': 'Physical placement differs from GSM8K; same model/TP/decoding, not bitwise hardware equivalence',
           'evaluation': 'pending separate offline evaluation; no threshold fitting on targets'}
    pilot.locked_json(root / 'run_protocol.json', run)
    windows = {}
    for dataset, rows in items.items():
        windows[dataset] = {}
        for role, tok in tokenizers.items():
            windows[dataset][role] = max(pilot.check_window(len(tok.encode(pilot.wire_prompt(tok,
                pilot.MATH_DIRECT_PROMPT.format(question=r['question']), role), add_special_tokens=False))) for r in rows)
    pilot.locked_json(root / 'generation_preflight.json', {
        'direct_required_windows': windows, 'context': pilot.CONTEXT,
        'reflection_check': 'per call with actual prior visible answer; never silently crop',
        'run_protocol_sha256': file_hash(root / 'run_protocol.json')})
    print(f'[transfer] preflight passed: {windows}', flush=True)


def run_stage(root, role, items, tokenizer):
    count = sum(map(len, items.values()))
    started, completed = time.monotonic(), 0
    with contextlib.ExitStack() as stack:
        client = None

        def get_client():
            nonlocal client
            if client is None:
                wait_for_resources(root, role)
                client = stack.enter_context(pilot.server(root, role))
            return client

        for dataset, rows in items.items():
            for item in rows:
                path = row_path(root, dataset, role, item['id'])
                replay = path.exists()

                def complete(action, prompt):
                    def no_generation():
                        raise ValueError('Saved row references an incomplete call')
                    return resolve_call(root / dataset, role, item, action, prompt, tokenizer,
                                        no_generation if replay else get_client)

                write_json(root / 'status.json', {'stage': role, 'dataset': dataset, 'id': item['id'],
                           'completed': completed, 'total': count, 'phase': 'replay' if replay else 'generate',
                           'heartbeat_unix': time.time()})
                if role == 'base':
                    result = base_row(item, complete)
                else:
                    base = json.loads(row_path(root, dataset, 'base', item['id']).read_text())
                    result = reflection_row(base, complete)
                pilot.locked_json(path, result)
                completed += 1
                elapsed = time.monotonic() - started
                print(f'[transfer] {role} {completed}/{count} {dataset}/{item["id"]} '
                      f'{"replayed" if replay else "done"} elapsed={elapsed:.0f}s', flush=True)


def export(root, items):
    for dataset, entries in items.items():
        rows = [json.loads(row_path(root, dataset, 'qwen14', item['id']).read_text()) for item in entries]
        observed, failures = {}, []
        for item, row in zip(entries, rows):
            if any(row.get(k) != v for k, v in item.items()):
                raise ValueError('Export item mismatch')
            for c in [*row['candidate_results'], row['independent']]:
                if c['execution_status'] != 'success':
                    failures.append({'id': row['id'], 'action': c['action'], 'status': c['execution_status']})
                for call in c['calls']:
                    path = call['call_log']
                    if path in observed and observed[path] != call:
                        raise ValueError('Shared call mismatch')
                    if file_hash(Path(path)) != call['call_log_sha256']:
                        raise ValueError('Paid call changed')
                    observed[path] = call
        if set(observed) != {str(p) for p in (root / dataset / 'calls').glob('*/*/*.jsonl')}:
            raise ValueError('Unaccounted paid call')
        target = root / dataset / 'policy_test_accounted.jsonl'
        locked_write(target, ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
        pilot.locked_json(root / dataset / 'accounting_audit.json', {
            'n': len(rows), 'trace_sha256': file_hash(target), 'failed_or_blocked_paths': failures,
            'clean_generation_complete': not failures, 'all_paths_accounted': True,
            'logical_paid_calls': len(observed),
            'generation_usage': {f: sum(c[f] for c in observed.values()) for f in
                                 ('prompt_tokens', 'completion_tokens', 'total_tokens', 'attempt_count')},
            'call_logs_sha256': {p: c['call_log_sha256'] for p, c in observed.items()},
            'run_protocol_sha256': file_hash(root / 'run_protocol.json'),
            'no_policy_evaluation_yet': True})
    write_json(root / 'status.json', {'stage': 'generation_complete_pending_offline_evaluation',
               'counts': {name: len(rows) for name, rows in items.items()}, 'llm_calls_finished': True})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--base-gpu', default='0')
    parser.add_argument('--reflector-gpus', default='0,2')
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    def interrupt(signum, frame):
        raise KeyboardInterrupt(f'Received signal {signum}')
    signal.signal(signal.SIGTERM, interrupt)
    with (root / 'pipeline.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        protocol, items = verify_inputs(root)
        from transformers import AutoTokenizer
        tokenizers = {r: AutoTokenizer.from_pretrained(str(pilot.MODELS / pilot.ROLES[r]['model']),
                      local_files_only=True) for r in ('base', 'qwen14')}
        preflight(root, protocol, items, args.base_gpu, args.reflector_gpus, tokenizers)
        if args.preflight_only:
            return
        try:
            run_stage(root, 'base', items, tokenizers['base'])
            run_stage(root, 'qwen14', items, tokenizers['qwen14'])
            verify_inputs(root)
            export(root, items)
        except BaseException as exc:
            write_json(root / 'status.json', {'stage': 'stopped_resume_required',
                       'error_type': type(exc).__name__, 'error': str(exc), 'heartbeat_unix': time.time()})
            raise


if __name__ == '__main__':
    main()
