"""Offline frozen-policy transfer evaluation; no generative model requests."""
import argparse
from collections import defaultdict
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from . import math_reflector_pilot as pilot
from . import run_locked_math_transfer as runner
from .analyze_decoupling_significance import _mcnemar
from .analyze_gate_random_baseline import holm_adjust
from .evaluate_accounted_math_test import failure_metrics, chosen_candidate
from .evaluate_locked_math_test import frozen_actions
from .math_gate_cpu_ablation import score_model
from .math_test_gpu_features import pool_hidden
from .neural_policy import build_policy_text, _ensure_torchvision_nms_compat
from .prepare_math_reflector_pilot import question_key
from .prepare_qa_hard_development import file_hash
from .resume_math_transfer_single_gpu import preserved_files
from .summarize_math_gate_multiseed import mean_std
from .train_fixed_math_policy import best_action, write_json
from .utils import read_jsonl, load_pickle


def cluster_interval(differences, questions, iterations=10000, seed=42):
    groups = defaultdict(list)
    for i, q in enumerate(questions):
        groups[question_key(q)].append(i)
    sizes = np.array([len(g) for g in groups.values()])
    totals = np.array([np.asarray(differences)[g].sum() for g in groups.values()])
    rng = np.random.default_rng(seed)
    samples = []
    for start in range(0, iterations, 100):
        indices = rng.integers(0, len(sizes), (min(100, iterations-start), len(sizes)))
        samples.extend(totals[indices].sum(1) / sizes[indices].sum(1))
    return {'difference': float(np.mean(differences)), 'ci95': np.quantile(samples, [.025, .975]).tolist(),
            'clusters': len(sizes), 'iterations': iterations, 'seed': seed,
            'unit': 'normalized-question cluster; paired; row-weighted estimand'}


def validate(root):
    _, items = runner.verify_inputs(root)
    lock = json.loads((root / 'policy_lock.json').read_text())
    amendment = json.loads((root / 'single_gpu_amendment.json').read_text())
    provenance = json.loads((root / 'single_gpu_export_provenance.json').read_text())
    if provenance['amendment_sha256'] != file_hash(root / 'single_gpu_amendment.json'):
        raise ValueError('Runtime amendment changed')
    if preserved_files(root) != amendment['preserved_files_sha256']:
        raise ValueError('Original Base/protocol changed')
    if file_hash(Path(lock['encoder_manifest'])) != lock['encoder_manifest_sha256']:
        raise ValueError('Encoder manifest changed')
    encoder = json.loads(Path(lock['encoder_manifest']).read_text())
    for name, sha in encoder['encoder_files_sha256'].items():
        if file_hash(Path(encoder['encoder_name']) / name) != sha:
            raise ValueError('Encoder weights/tokenizer changed')
    from transformers import AutoTokenizer
    tokenizers = {r: AutoTokenizer.from_pretrained(str(pilot.MODELS / pilot.ROLES[r]['model']),
                  local_files_only=True) for r in ('base', 'qwen14')}
    rows_by_dataset, source_hashes = {}, {}
    def forbidden():
        raise ValueError('Missing cached response; offline evaluation cannot generate')
    for name, entries in items.items():
        audit_path = root / name / 'accounting_audit.json'
        if file_hash(audit_path) != provenance['accounting_audits_sha256'][name]:
            raise ValueError('Accounting audit changed')
        audit = json.loads(audit_path.read_text())
        path = root / name / 'policy_test_accounted.jsonl'
        if file_hash(path) != audit['trace_sha256'] or not audit['all_paths_accounted']:
            raise ValueError('Incomplete/changed trace')
        rows = read_jsonl(path)
        if [r['id'] for r in rows] != [r['id'] for r in entries]:
            raise ValueError('Trace coverage/order differs')
        for p, sha in audit['call_logs_sha256'].items():
            if file_hash(Path(p)) != sha:
                raise ValueError('Paid call changed')
        for item, row in zip(entries, rows):
            def complete(role):
                return lambda action, prompt: runner.resolve_call(root / name, role, item, action,
                                     prompt, tokenizers[role], forbidden)
            reconstructed = runner.reflection_row(runner.base_row(item, complete('base')), complete('qwen14'))
            if reconstructed != row:
                raise ValueError(f'Replayed scoring/path mismatch: {item["id"]}')
        source_hashes[str(path)] = file_hash(path)
        source_hashes[str(audit_path)] = file_hash(audit_path)
        rows_by_dataset[name] = rows
        print(f'[transfer-eval] replay audit {name}: {len(rows)} rows', flush=True)
    return lock, encoder, rows_by_dataset, source_hashes


def encode(rows, encoder, lock, output, gpu):
    from transformers import AutoModel, AutoTokenizer
    from torch.nn.attention import sdpa_kernel, SDPBackend
    import transformers
    texts = [build_policy_text(r, max_text_chars=0, include_state=False) for r in rows]
    signature = {'ids': [r['id'] for r in rows],
        'text_sha256': [hashlib.sha256(t.encode()).hexdigest() for t in texts],
        'encoder_files_sha256': encoder['encoder_files_sha256'], 'layers': [-2, -1],
        'pooling': 'mean_last', 'dtype': 'float32', 'physical_gpu': gpu, 'batch_size': 2,
        'max_length': lock['encoder']['max_length'], 'sdpa_backend': 'math', 'tf32': False,
        'torch': torch.__version__, 'transformers': transformers.__version__, 'code_sha256': file_hash(Path(__file__))}
    cache = output / 'features'
    cache.mkdir(parents=True, exist_ok=True)
    pilot.locked_json(cache / 'manifest.json', signature)
    tok = AutoTokenizer.from_pretrained(encoder['encoder_name'], local_files_only=True)
    tok.padding_side = 'right'
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    lengths = [len(tok(t, truncation=False)['input_ids']) for t in texts]
    pilot.locked_json(cache / 'length_audit.json', {'lengths': lengths, 'max_length': signature['max_length'],
        'clipped_ids': [r['id'] for r, n in zip(rows, lengths) if n > signature['max_length']]})
    parts, model = [], None
    for start in range(0, len(rows), 32):
        end = min(start + 32, len(rows))
        path = cache / f'{start:06d}_{end:06d}.npy'
        if path.exists():
            values = np.load(path, allow_pickle=False)
        else:
            if model is None:
                if not torch.cuda.is_available():
                    raise RuntimeError('CUDA unavailable; no silent device substitution')
                _ensure_torchvision_nms_compat(torch)
                model = AutoModel.from_pretrained(encoder['encoder_name'], local_files_only=True,
                    dtype=torch.float32, attn_implementation='sdpa').to('cuda:0').eval().requires_grad_(False)
            batches = []
            for b in range(start, end, 2):
                batch = tok(texts[b:min(b+2, end)], padding=True, truncation=True,
                            max_length=signature['max_length'], return_tensors='pt').to('cuda:0')
                with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
                    hs = model(**batch, output_hidden_states=True, use_cache=False).hidden_states
                    batches.append(pool_hidden(hs, batch['attention_mask']).cpu().numpy())
                del hs, batch
            values = np.concatenate(batches).astype('float32')
            with path.with_suffix('.tmp').open('wb') as f:
                np.save(f, values, allow_pickle=False)
            path.with_suffix('.tmp').replace(path)
        if values.shape != (end-start, 4096) or not np.isfinite(values).all():
            raise ValueError('Invalid feature shard')
        parts.append(values)
        print(f'[transfer-encode] {output.name} {end}/{len(rows)}', flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    pilot.locked_json(cache / 'complete.json', {'shards_sha256': {str(p): file_hash(p) for p in cache.glob('*.npy')},
                      'gpu_name': torch.cuda.get_device_name(0), 'encoder_clipped_count': sum(n > signature['max_length'] for n in lengths),
                      'note': 'Same FP32 GPU representation as recent GSM8K evaluation; not a proof of CPU bitwise equality'})
    return np.concatenate(parts)


def evaluate_subset(rows, indices, decisions, lock, unknown):
    selected = [rows[i] for i in indices]
    ids = {r['id'] for r in selected}
    unknown_here = [u for u in unknown if u['id'] in ids]
    actions = {name: [a] * len(selected) for name, a in (
        ('direct', 'direct'), ('fixed_reflect_1', 'reflect_1'), ('fixed_reflect_2', 'reflect_2'), ('independent', 'independent'))}
    actions['oracle_accuracy_first'] = [best_action(r) for r in selected]
    baselines = {name: failure_metrics(selected, a) for name, a in actions.items()}
    for name, metric in baselines.items():
        missing = [u for u in unknown_here if u['action'] == 'independent' and name == 'independent']
        metric['token_cost_complete'] = not missing
        metric['unknown_cost_attempts'] = missing
        metric['token_interpretation'] = 'known_cost_lower_bound' if missing else 'all_recorded_path_attempts'
        metric['reward_interpretation'] = 'upper_bound_due_to_missing_cost' if missing else 'observed_reward'
    records, groups, tests = [], defaultdict(list), []
    base_correct = np.array([chosen_candidate(r, 'direct')['correct'] for r in selected])
    unique_questions = len({question_key(r['question']) for r in selected}) == len(selected)
    for point in lock['policies']:
        a = [decisions[point['name']]['actions'][i] for i in indices]
        metric = failure_metrics(selected, a)
        records.append({**point, 'metrics': metric})
        values = np.array([[chosen_candidate(r, action)[field] for field in ('correct', 'token_cost')]
                           for r, action in zip(selected, a)])
        key = (point['recipe'], point['branch'], point['validation_call_budget'])
        groups[key].append((point['seed'], metric, values))
        if unique_questions:
            tests.append({'name': point['name'], 'mcnemar': _mcnemar(values[:, 0], base_correct)})
    for name in ('fixed_reflect_1', 'fixed_reflect_2', 'independent'):
        if unique_questions:
            values = [chosen_candidate(r, a)['correct'] for r, a in zip(selected, actions[name])]
            tests.append({'name': name, 'mcnemar': _mcnemar(values, base_correct)})
    if tests:
        for t, p in zip(tests, holm_adjust([t['mcnemar']['exact_two_sided_p'] for t in tests])):
            t['holm_p_within_subset_39_tests'] = p
    grouped = []
    for key, members in groups.items():
        if sorted(x[0] for x in members) != [7, 21, 42]:
            raise ValueError('Missing seeds')
        values = np.mean([m[2] for m in members], axis=0)
        comparisons = {}
        for name in ('direct', 'fixed_reflect_1', 'independent'):
            reference = np.array([[chosen_candidate(r, a)[f] for f in ('correct', 'token_cost')]
                                  for r, a in zip(selected, actions[name])])
            comparisons[name] = {field: cluster_interval(values[:, j]-reference[:, j],
                [r['question'] for r in selected]) for j, field in enumerate(('accuracy', 'avg_token_cost'))}
            if name == 'independent' and unknown_here:
                comparisons[name]['cost_caveat'] = 'Difference uses lower-bound independent cost, not complete-cost difference'
        grouped.append({'recipe': key[0], 'branch': key[1], 'validation_call_budget': key[2],
            'metrics': {f: mean_std([m[1][f] for m in members]) for f in
                ('accuracy', 'avg_prompt_tokens', 'avg_completion_tokens', 'avg_token_cost', 'avg_reward',
                 'reflection_rate', 'fixes', 'harms', 'execution_failure_rate')}, 'paired_bootstrap': comparisons})
    return {'n': len(selected), 'ids': [r['id'] for r in selected], 'baselines': baselines,
            'policies': records, 'groups': grouped, 'mcnemar_vs_direct': tests,
            'mcnemar_note': '39-test Holm family per subset; no cross-dataset/subset family correction' if unique_questions else
                           'Omitted on duplicate-question full set; use cluster bootstrap and unique subset',
            'unknown_cost_attempts': unknown_here}


def run(root, gpu):
    output = root / 'evaluation'
    output.mkdir(exist_ok=True)
    if (output / 'summary.json').exists():
        raise ValueError('Results already committed; do not overwrite')
    lock, encoder, datasets, hashes = validate(root)
    preparation = json.loads((root / 'manifest.json').read_text())
    warning = json.loads((root / 'cost_completeness_warning.json').read_text())
    protocol = {'source_sha256': hashes, 'policy_lock_sha256': file_hash(root / 'policy_lock.json'),
        'runtime_amendment_sha256': file_hash(root / 'single_gpu_amendment.json'),
        'cost_warning_sha256': file_hash(root / 'cost_completeness_warning.json'),
        'code_sha256': {name: file_hash(Path(__file__).with_name(name)) for name in
            ('evaluate_math_transfer.py', 'evaluate_locked_math_test.py', 'evaluate_accounted_math_test.py',
             'math_gate_cpu_ablation.py', 'math_test_gpu_features.py', 'neural_policy.py',
             'analyze_decoupling_significance.py', 'analyze_gate_random_baseline.py', 'summarize_math_gate_multiseed.py')},
        'physical_gpu': gpu, 'no_llm_calls': True, 'no_training_or_threshold_search': True,
        'main_model_selected': False, 'bootstrap_iterations': 10000, 'bootstrap_seed': 42,
        'random_matched_call_controls': 'not included in this transfer evaluation',
        'reporting': 'all 36 policies; all predefined subsets; exploratory, not pristine confirmation'}
    pilot.locked_json(output / 'protocol.json', protocol)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    results = {}
    for name, rows in datasets.items():
        folder = output / name
        folder.mkdir(exist_ok=True)
        features = encode(rows, encoder, lock, folder, gpu)
        round_bundle = load_pickle(lock['round_model'])
        round_actions = np.array(round_bundle['action_labels'])[score_model(round_bundle, features).argmax(1)]
        decisions = {}
        for point in lock['policies']:
            a, p = frozen_actions(load_pickle(point['model']), features, round_actions, point['branch'], point['direct_threshold'])
            invalid = np.array([not r['policy_input_available'] for r in rows])
            a[invalid] = 'direct'
            decisions[point['name']] = {'actions': a.tolist(), 'direct_probability': [None if invalid[i] else float(v) for i, v in enumerate(p)]}
        pilot.locked_json(folder / 'decisions.json', {'ids': [r['id'] for r in rows], 'policies': decisions})
        positions = {r['id']: i for i, r in enumerate(rows)}
        results[name] = {}
        for subset, ids in preparation['datasets'][name]['subsets'].items():
            result = evaluate_subset(rows, [positions[i] for i in ids], decisions, lock, warning['unknown_attempts'])
            pilot.locked_json(folder / f'{subset}.json', result)
            results[name][subset] = result
            print(f'[transfer-eval] {name}/{subset} n={len(ids)} all 36 policies done', flush=True)
    for path, sha in hashes.items():
        if file_hash(Path(path)) != sha:
            raise ValueError('Input changed during evaluation')
    lines = ['# Locked Mathematical Transfer Results', '',
        'Offline application of all 36 frozen policies; no threshold tuning or new LLM calls.',
        'All three predefined subsets are saved in per-dataset JSON. Main tables below use full datasets.',
        'Historical target feedback and TP=1 versus prior GSM8K TP=2 limit confirmation claims.',
        'GSM1K independent tokens are a LOWER BOUND: one pre-crash response usage was lost; reward is an upper bound.',
        'Local encoder/head latency is not included in generation-token counts.',
        'Bootstrap intervals are paired question-cluster intervals after averaging seeds; not simultaneous intervals.',
        'Random matched-call controls and end-to-end latency are not evaluated here.', '']
    for name, subsets in results.items():
        full = subsets['full']
        lines += [f'## {name} (n={full["n"]})', '',
            '| Baseline | Accuracy | Input | Output | Total known tokens | Fix/Harm | Failed/blocked |',
            '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
        for method, m in full['baselines'].items():
            lines.append(f"| {method} | {m['accuracy']:.4f} | {m['avg_prompt_tokens']:.2f} | {m['avg_completion_tokens']:.2f} | {m['avg_token_cost']:.2f} | {m['fixes']}/{m['harms']} | {m['execution_failure_count']} |")
        lines += ['', '| Recipe | Branch | Validation budget | Accuracy mean +/- SD | Tokens mean +/- SD | Actual call rate |',
                  '| --- | --- | ---: | ---: | ---: | ---: |']
        for g in full['groups']:
            m = g['metrics']
            lines.append(f"| {g['recipe']} | {g['branch']} | {g['validation_call_budget']:.0%} | {m['accuracy']['mean']:.4f} +/- {m['accuracy']['std']:.4f} | {m['avg_token_cost']['mean']:.2f} +/- {m['avg_token_cost']['std']:.2f} | {m['reflection_rate']['mean']:.4f} |")
        lines += ['']
    from .prepare_qa_hard_development import locked_write
    locked_write(output / 'REPORT.md', '\n'.join(lines))
    pilot.locked_json(output / 'summary.json', {'protocol_sha256': file_hash(output / 'protocol.json'),
        'datasets': results, 'cost_warning': warning, 'no_test_calibration': True, 'main_model_selected': False})
    write_json(output / 'status.json', {'stage': 'completed', 'datasets': list(results), 'report': str(output / 'REPORT.md')})
    print(f'[transfer-eval] COMPLETE {output / "REPORT.md"}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=runner.DEFAULT_ROOT)
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    root = args.root.resolve()
    with (root / 'pipeline.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(root, args.gpu)


if __name__ == '__main__':
    main()
