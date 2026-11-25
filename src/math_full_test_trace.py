"""Locked full GSM8K test generation, reusing the training generation protocol."""
from __future__ import annotations

import argparse
import contextlib
import datetime
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import time

from . import math_reflector_pilot as pilot
from .gsm8k_utils import normalize_example_item
from .math_full_training_trace import CachedOnly, append_row, check_call, make_policy_row, role_placements
from .math_trace_failures import (OutputCapFailure, blocked_record, exhausted_call, failure_record,
                                  handled_prefix, paid_logs, read_failures, wait_for_resources, write_failure)
from .prepare_math_reflector_pilot import DATA, question_key
from .prepare_qa_hard_development import file_hash, locked_write
from .utils import read_jsonl, save_json


PROJECT = pilot.PROJECT
DEFAULT_ROOT = PROJECT / 'outputs/math_full_test_v1'
TRAIN_ROOT = PROJECT / 'outputs/math_full_training_v1'
POLICY_ROOT = PROJECT / 'outputs/math_gate_ranking_multiseed_20260916'


def locked_copy(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if file_hash(source) != file_hash(target):
            raise ValueError(f'Locked artifact differs: {target}')
    else:
        temp = target.with_suffix(target.suffix + '.tmp')
        shutil.copyfile(source, temp)
        os.replace(temp, target)
    return file_hash(target)


def check_generation_protocol(protocol):
    actual = {'direct_prompt': pilot.MATH_DIRECT_PROMPT, 'reflection_template': pilot.REFLECTION_TEMPLATE,
              'context': pilot.CONTEXT, 'output_steps': list(pilot.OUTPUT_STEPS), 'safety': pilot.SAFETY,
              'temperature': 0., 'seed': 42, 'max_request_retries': 0,
              'server': {'port': pilot.PORT, 'gpu_memory_utilization': .46, 'max_num_seqs': 1,
                         'max_num_batched_tokens': 1024, 'enforce_eager': True}}
    for key, value in actual.items():
        if protocol[key] != value:
            raise ValueError(f'Train/test generation mismatch: {key}')
    for role in ('base', 'qwen14'):
        for key in ('model', 'tp', 'thinking'):
            if protocol['roles'][role][key] != pilot.ROLES[role][key]:
                raise ValueError(f'Role generation mismatch: {role}/{key}')
    for name, checksum in protocol['code_sha256'].items():
        if file_hash(PROJECT / 'src' / name) != checksum:
            raise ValueError(f'Generation/scoring code changed: {name}')
    for path, checksum in protocol['model_metadata_sha256'].items():
        if file_hash(Path(path)) != checksum:
            raise ValueError(f'Model/tokenizer metadata changed: {path}')
    for path, expected in protocol['model_shard_stats'].items():
        stat = Path(path).stat()
        if {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns} != expected:
            raise ValueError(f'Model shard changed: {path}')


def freeze(root, base_gpu='0', reflector_gpus='0,1'):
    original = json.loads((TRAIN_ROOT / 'protocol.json').read_text())
    check_generation_protocol(original)
    roles = role_placements(original, base_gpu, reflector_gpus)
    source_manifest = json.loads((POLICY_ROOT / 'manifest.json').read_text())
    source_summary = json.loads((POLICY_ROOT / 'summary.json').read_text())
    if json.loads((POLICY_ROOT / 'status.json').read_text()).get('stage') != 'completed':
        raise ValueError('Source policy audit is incomplete')
    if source_manifest.get('all_36_checkpoint_replays_match') is not True:
        raise ValueError('Require verified frozen policies')
    policies, round_sources = [], []
    for artifact in source_manifest['artifacts']:
        run = Path(artifact['run'])
        for name in ('protocol', 'summary'):
            if file_hash(run / f'{name}.json') != artifact[f'{name}_sha256']:
                raise ValueError('Source training artifacts changed')
        if file_hash(run / 'reference_round.pkl') != artifact['round_copy_sha256']:
            raise ValueError('Source Round changed')
        if not artifact.get('round_semantically_identical_to_source'):
            raise ValueError('Round identity was not audited')
        round_sources.append(run / 'reference_round.pkl')
        for point in json.loads((run / 'summary.json').read_text())['comparisons']:
            source = Path(point['gate_model'])
            if file_hash(source) != artifact['models'][str(source)]:
                raise ValueError('Source Gate changed')
            name = f"seed{artifact['seed']}_{point['recipe']}_{point['branch']}_{point['budget']:.2f}"
            dest = root / 'locked_models' / f'{name}.pkl'
            policies.append({'name': name, 'seed': artifact['seed'], 'recipe': point['recipe'],
                             'branch': point['branch'], 'validation_call_budget': point['budget'],
                             'epoch': point['epoch'], 'direct_threshold': point['direct_threshold'],
                             'model': str(dest), 'sha256': locked_copy(source, dest)})
    if len(policies) != 36 or len({p['name'] for p in policies}) != 36:
        raise ValueError('Expected all 36 existing checkpoints, no test-side selection')
    round_path = root / 'locked_models' / 'round.pkl'
    round_hash = locked_copy(round_sources[0], round_path)
    for split in ('train', 'validation'):
        if file_hash(Path(source_summary['coverage'][split]['file'])) != source_summary['coverage'][split]['sha256']:
            raise ValueError('Training/validation data changed after policy fitting')
    train_manifest = json.loads((TRAIN_ROOT / 'manifest.json').read_text())
    if file_hash(DATA / 'test.jsonl') != train_manifest['input_sha256']['test.jsonl']:
        raise ValueError('Official test source changed')
    items = [{**normalize_example_item(r, i, 'test'), 'policy_split': 'test'}
             for i, r in enumerate(read_jsonl(DATA / 'test.jsonl'))]
    train_items = read_jsonl(TRAIN_ROOT / 'train_items.jsonl') + read_jsonl(TRAIN_ROOT / 'validation_items.jsonl')
    ids = [r['id'] for r in items]
    keys = [question_key(r['question']) for r in items]
    if len(items) != 1319 or len(set(ids)) != 1319 or len(set(keys)) != 1319:
        raise ValueError('Expected unique full 1319 GSM8K test problems')
    if set(ids) & {r['id'] for r in train_items} or set(keys) & {question_key(r['question']) for r in train_items}:
        raise ValueError('Test/training overlap')
    locked_write(root / 'test_items.jsonl', ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in items))
    # The timestamp remains unchanged on resume, and is written before any generation begins.
    lock_path = root / 'policy_lock.json'
    old_lock = json.loads(lock_path.read_text()) if lock_path.exists() else {}
    prior_path = PROJECT / 'outputs/fixed_math_policy_cpu_20260915_151132_seed42/selected_validation.json'
    encoder = json.loads(prior_path.read_text())['encoder']
    feature_manifest = Path(encoder['cache']) / 'manifest.json'
    pilot.locked_json(lock_path, {
        'locked_at_utc': old_lock.get('locked_at_utc', datetime.datetime.now(datetime.timezone.utc).isoformat()),
        'policies': policies, 'round_model': str(round_path), 'round_sha256': round_hash,
        'source_policy_manifest_sha256': file_hash(POLICY_ROOT / 'manifest.json'),
        'source_policy_summary_sha256': file_hash(POLICY_ROOT / 'summary.json'),
        'training_coverage': source_summary['coverage'], 'encoder': encoder,
        'encoder_manifest': str(feature_manifest), 'encoder_manifest_sha256': file_hash(feature_manifest),
        'test_items_sha256': file_hash(root / 'test_items.jsonl'), 'test_n': 1319,
        'main_model_selected': False, 'accuracy_tolerance': None, 'no_test_threshold_search': True,
        'primary_reference': 'base Direct', 'secondary_references': ['fixed R1/R2', 'matched-call random gates', 'independent Qwen3-14B'],
        'test_call_budget': 'apply frozen validation threshold; report actual test rate without retuning',
        'random_repetitions': 10000, 'bootstrap_iterations': 10000,
        'historical_limitation': 'Official test was observed during older project development; not a pristine confirmation set',
        'training_limit': '2 train and 3 validation generation failures unresolved; fitted checkpoints remain frozen'})
    dependencies = ('math_full_test_trace.py', 'evaluate_locked_math_test.py', 'math_full_training_trace.py',
                    'math_trace_failures.py', 'train_fixed_math_policy.py', 'math_gate_cpu_ablation.py',
                    'analyze_gate_random_baseline.py', 'analyze_decoupling_significance.py',
                    'summarize_math_gate_multiseed.py', 'neural_policy.py', 'transformer_policy.py',
                    'gsm8k_utils.py', 'utils.py', 'prepare_qa_hard_development.py', 'prepare_math_reflector_pilot.py')
    protocol = {'name': 'locked_gsm8k_full_test_v1', 'training_protocol': original,
                'training_protocol_sha256': file_hash(TRAIN_ROOT / 'protocol.json'), 'roles': roles,
                'test_items_sha256': file_hash(root / 'test_items.jsonl'), 'policy_lock_sha256': file_hash(lock_path),
                'source_code_sha256': {name: file_hash(PROJECT / 'src' / name) for name in dependencies},
                'coverage': 'all test questions: Direct, R1, R2, independent Qwen3-14B',
                'failure_policy': 'quarantine exhausted calls; keep all usage; no final evaluation with missing rows',
                'generation_role': 'test', 'no_training': True}
    pilot.locked_json(root / 'protocol.json', protocol)
    for role, spec in roles.items():
        pilot.ROLES[role].update(spec)
    return items, protocol


def validate_row(root, row, item, role, tokenizers, base=None):
    if any(row.get(k) != v for k, v in item.items()):
        raise ValueError('Test item changed')
    first = row['initial_call']
    check_call(first, 'base', item['id'], 'direct', pilot.MATH_DIRECT_PROMPT.format(question=item['question']), tokenizers['base'])
    if Path(first['call_log']).parents[3] != root:
        raise ValueError('Test must not reuse other-generation calls')
    if row['initial_answer'] != first['answer_text'] or row['initial_pred'] != pilot.extract_final_number(first['answer_text']):
        raise ValueError('Initial answer mismatch')
    expected = [pilot.candidate('direct', [first], item['gold_answer'])]
    if role == 'base':
        if row['candidate_results'] != expected:
            raise ValueError('Direct candidate mismatch')
        return
    if base is None or base['initial_call'] != first or len(row['candidate_results']) != 3:
        raise ValueError('Complete matched candidates required')
    calls = [first]
    for round_id in (1, 2):
        call = row['candidate_results'][round_id]['calls'][-1]
        prompt = pilot.REFLECTION_TEMPLATE.format(question=item['question'], current_answer=calls[-1]['answer_text'], round_id=round_id)
        check_call(call, 'qwen14', item['id'], f'reflect_{round_id}', prompt, tokenizers['qwen14'])
        if Path(call['call_log']).parents[3] != root:
            raise ValueError('Foreign test reflection call')
        calls = [*calls, call]
        expected.append(pilot.candidate(f'reflect_{round_id}', calls, item['gold_answer']))
    call = row['independent']['calls'][0]
    check_call(call, 'qwen14', item['id'], 'independent', pilot.MATH_DIRECT_PROMPT.format(question=item['question']), tokenizers['qwen14'])
    if Path(call['call_log']).parents[3] != root:
        raise ValueError('Foreign independent call')
    independent = pilot.candidate('independent', [call], item['gold_answer'])
    if row != make_policy_row(base, expected, independent):
        raise ValueError('Test scoring or cumulative usage mismatch')


def validate_failure(root, role, record, item, tokenizers, base):
    if record['reason'] == 'base_output_unresolved':
        if role != 'qwen14' or item['id'] in base or record != blocked_record(root, item):
            raise ValueError('Bad blocked test failure')
        return
    action = record['failed_call']['action']
    if action not in (('direct',) if role == 'base' else ('reflect_1', 'reflect_2', 'independent')):
        raise ValueError('Unexpected failed test action')
    prompt = pilot.MATH_DIRECT_PROMPT.format(question=item['question'])
    if action.startswith('reflect_'):
        current = base[item['id']]['initial_answer']
        if action == 'reflect_2':
            first_prompt = pilot.REFLECTION_TEMPLATE.format(question=item['question'], current_answer=current, round_id=1)
            current = pilot.complete_call(CachedOnly(), tokenizers[role], root, role, item['id'], 'reflect_1', first_prompt)['answer_text']
        prompt = pilot.REFLECTION_TEMPLATE.format(question=item['question'], current_answer=current, round_id=int(action[-1]))
    diagnostic = exhausted_call(root, role, item['id'], action, prompt, tokenizers[role])
    if diagnostic is None or record != failure_record(root, role, item, diagnostic):
        raise ValueError('Failure ledger mismatch')


def build_test_row(item, role, complete, base=None):
    if role == 'base':
        first = complete(item, 'direct', pilot.MATH_DIRECT_PROMPT.format(question=item['question']))
        direct = pilot.candidate('direct', [first], item['gold_answer'])
        return {**item, 'initial_call': first, 'initial_answer': first['answer_text'],
                'initial_pred': direct['pred'], 'candidate_results': [direct]}
    # Always generate both reflections, including initially correct test answers.
    calls, results = [base['initial_call']], base['candidate_results'][:]
    for round_id in (1, 2):
        prompt = pilot.REFLECTION_TEMPLATE.format(question=item['question'], current_answer=calls[-1]['answer_text'], round_id=round_id)
        calls = [*calls, complete(item, f'reflect_{round_id}', prompt)]
        results.append(pilot.candidate(f'reflect_{round_id}', calls, item['gold_answer']))
    independent = complete(item, 'independent', pilot.MATH_DIRECT_PROMPT.format(question=item['question']))
    return make_policy_row(base, results, pilot.candidate('independent', [independent], item['gold_answer']))


def run_stage(root, role, items, tokenizers, base=None):
    base = base or {}
    path = root / ('direct_test.jsonl' if role == 'base' else 'policy_test_partial.jsonl')
    rows = read_jsonl(path) if path.exists() else []
    failures = read_failures(root, role)
    prefix = handled_prefix(items, rows, failures)
    lookup = {r['id']: r for r in items}
    for row in rows:
        validate_row(root, row, lookup[row['id']], role, tokenizers, base.get(row['id']))
    for key, record in failures.items():
        validate_failure(root, role, record, lookup[key], tokenizers, base)
    started = time.monotonic()
    print(f'[math-test] {role} verified={len(rows)} processed={prefix}/{len(items)} failures={len(failures)}', flush=True)
    with contextlib.ExitStack() as stack:
        client = None

        def complete(item, action, prompt):
            nonlocal client
            diagnostic = exhausted_call(root, role, item['id'], action, prompt, tokenizers[role])
            if diagnostic:
                raise OutputCapFailure(diagnostic)
            call_path = root / 'calls' / role / item['id'] / f'{action}.jsonl'
            cached = read_jsonl(call_path) if call_path.exists() else []
            if cached and cached[-1]['metadata']['finish_reason'] == 'stop':
                return pilot.complete_call(CachedOnly(), tokenizers[role], root, role, item['id'], action, prompt)
            if client is None:
                wait_for_resources(root, role)
                client = stack.enter_context(pilot.server(root, role))
            try:
                return pilot.complete_call(client, tokenizers[role], root, role, item['id'], action, prompt)
            except RuntimeError:
                diagnostic = exhausted_call(root, role, item['id'], action, prompt, tokenizers[role])
                if diagnostic:
                    raise OutputCapFailure(diagnostic)
                raise

        for count, item in enumerate(items[prefix:], prefix + 1):
            save_json({'stage': role, 'phase': 'generation', 'processed': count - 1, 'total': len(items),
                       'id': item['id'], 'test_metrics_released': False}, root / 'status.json')
            if role == 'qwen14' and item['id'] not in base:
                record = blocked_record(root, item)
                failures[item['id']] = record
                write_failure(root, role, record)
            else:
                try:
                    row = build_test_row(item, role, complete, base.get(item['id']))
                except OutputCapFailure as exc:
                    record = failure_record(root, role, item, exc.diagnostic)
                    failures[item['id']] = record
                    write_failure(root, role, record)
                    print(f'[math-test] pending {item["id"]}: {exc}', flush=True)
                else:
                    append_row(path, row)
                    rows.append(row)
            elapsed = time.monotonic() - started
            eta = elapsed / (count - prefix) * (len(items) - count)
            save_json({'stage': role, 'processed': count, 'completed': len(rows), 'total': len(items),
                       'pending': len(failures), 'elapsed_seconds': elapsed, 'stage_eta_seconds': eta,
                       'test_metrics_released': False}, root / 'status.json')
            print(f'[math-test] {role} {count}/{len(items)} complete={len(rows)} pending={len(failures)} elapsed={elapsed:.0f}s stage_eta={eta:.0f}s', flush=True)
    return rows


def finalize(root, items, tokenizers, base_rows, rows):
    base = {r['id']: r for r in base_rows}
    pending = set()
    for role, stage_rows in (('base', base_rows), ('qwen14', rows)):
        failures = read_failures(root, role)
        if handled_prefix(items, stage_rows, failures) != len(items):
            raise ValueError('Incomplete stage prefix')
        pending.update(failures)
        for item in items:
            if item['id'] in failures:
                validate_failure(root, role, failures[item['id']], item, tokenizers, base)
    lookup = {r['id']: r for r in items}
    for row in rows:
        validate_row(root, row, lookup[row['id']], 'qwen14', tokenizers, base[row['id']])
    logs = [entry for role in ('base', 'qwen14') for item in items for entry in paid_logs(root, role, item['id'])]
    all_paths = {str(p) for p in (root / 'calls').rglob('*.jsonl')}
    if all_paths != {c['path'] for c in logs}:
        raise ValueError('Unaccounted generation logs')
    unresolved = [x['path'] for x in logs if x['final_finish_reason'] != 'stop']
    if unresolved and not pending:
        raise ValueError('Unreported unfinished calls')
    if not pending and (len(rows) != len(items) or len(logs) != 4 * len(items)):
        raise ValueError('Expected four logical calls per test question')
    report = {'expected': len(items), 'completed': len(rows), 'pending_ids': sorted(pending),
              'unresolved_call_logs': unresolved, 'final_evaluation_ready': not pending,
              'logical_calls': len(logs), 'paid_attempts': sum(x['attempt_count'] for x in logs),
              'generation_usage': {f: sum(x[f] for x in logs) for f in ('prompt_tokens', 'completion_tokens', 'total_tokens')},
              'protocol_sha256': file_hash(root / 'protocol.json'), 'policy_lock_sha256': file_hash(root / 'policy_lock.json')}
    if not pending:
        locked_copy(root / 'policy_test_partial.jsonl', root / 'policy_test.jsonl')
        report['test_trace_sha256'] = file_hash(root / 'policy_test.jsonl')
    save_json(report, root / 'completion_audit.json')
    save_json({'stage': 'awaiting_failure_resolution' if pending else 'generation_complete',
               **report, 'test_metrics_released': False}, root / 'status.json')
    return not pending


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--base-gpu', default='0')
    parser.add_argument('--reflector-gpus', default='0,1')
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    def interrupt(signum, frame):
        raise KeyboardInterrupt(f'Received signal {signum}')
    signal.signal(signal.SIGTERM, interrupt)
    with (root / 'pipeline.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            items, protocol = freeze(root, args.base_gpu, args.reflector_gpus)
            from transformers import AutoTokenizer
            tokenizers = {role: AutoTokenizer.from_pretrained(str(pilot.MODELS / spec['model']),
                          local_files_only=True, trust_remote_code=False) for role, spec in protocol['roles'].items()}
            windows = {role: max(pilot.check_window(len(tok.encode(pilot.wire_prompt(tok,
                        pilot.MATH_DIRECT_PROMPT.format(question=i['question']), role), add_special_tokens=False)))
                        for i in items) for role, tok in tokenizers.items()}
            save_json({'n': len(items), 'required_direct_windows': windows, 'context': pilot.CONTEXT,
                       'reflection_window_check': 'per call with real prior response', 'no_generation': True}, root / 'preflight.json')
            print(f'[math-test] locked 36 policies, {len(items)} test items; windows={windows}', flush=True)
            if args.preflight_only:
                return
            base = run_stage(root, 'base', items, tokenizers)
            rows = run_stage(root, 'qwen14', items, tokenizers, {r['id']: r for r in base})
            if finalize(root, items, tokenizers, base, rows):
                from .evaluate_locked_math_test import evaluate
                evaluate(root)
        except BaseException as exc:
            save_json({'stage': 'failed_or_interrupted', 'error': repr(exc), 'resume_supported': True,
                       'partial_results_not_final': True}, root / 'status.json')
            raise


if __name__ == '__main__':
    main()
