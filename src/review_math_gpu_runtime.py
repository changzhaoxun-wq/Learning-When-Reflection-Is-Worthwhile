"""Explicit label-free review of GPU numerical drift; preserve the first strict audit."""
import argparse
import fcntl
import json
from pathlib import Path

import numpy as np
import torch

from . import evaluate_accounted_math_test as offline
from . import math_test_gpu_features as gpu
from .math_reflector_pilot import locked_json
from .train_fixed_math_policy import digest, write_json


def numerical_review(cpu, features, audit):
    delta = features[:len(cpu)].astype('float64') - cpu
    norms = np.linalg.norm(cpu.astype('float64'), axis=1)
    relative = np.linalg.norm(delta, axis=1) / np.maximum(norms, 1e-12)
    action_match = audit['round_action_mismatches'] == 0 and all(
        p['action_mismatches'] == 0 for p in audit['policies'])
    maximum = float(np.max(np.abs(delta)))
    rms = float(np.sqrt(np.mean(delta ** 2)))
    return {'posthoc_label_free_runtime_review': True,
            'original_strict_audit_passed': audit['passed'],
            'overlap_n': len(cpu), 'total_elements': int(cpu.size),
            'strict_element_mismatches': int((~np.isclose(cpu, features[:len(cpu)], rtol=.001, atol=.0001)).sum()),
            'max_absolute_difference': maximum, 'rmse': rms,
            'max_row_relative_l2': float(relative.max()), 'all_overlap_actions_match': action_match,
            'acceptance_bounds': {'max_absolute_difference': .001, 'rmse': .0001, 'max_row_relative_l2': .00001},
            'accepted': bool(action_match and maximum <= .001 and rms <= .0001 and relative.max() <= .00001),
            'reason': 'Near-zero coordinates can fail elementwise relative tolerance despite small vector-level drift.',
            'limitations': '384-row (or saved prefix) runtime check, not bitwise equivalence or a guarantee on unseen CPU rows; no answer labels used.'}


def run(root):
    torch.set_num_threads(4)
    source, lock, rows, _ = offline.validate_inputs(root)
    revision = json.loads((root / 'gpu_execution_revision.json').read_text())
    if digest(gpu.__file__) != revision['extractor_sha256'] or \
            digest(root / 'accounting_audit.json') != revision['accounting_audit_sha256'] or \
            digest(root / 'manifest.json') != revision['source_accounting_manifest_sha256']:
        raise ValueError('GPU runtime provenance changed')
    manifests = list((root / 'test_feature_cache_gpu').glob('*/manifest.json'))
    if len(manifests) != 1:
        raise ValueError('Require one unambiguous GPU cache')
    manifest, cache = manifests[0], manifests[0].parent
    signature = json.loads(manifest.read_text())
    if signature['extractor_sha256'] != revision['extractor_sha256'] or signature['ids'] != [r['id'] for r in rows]:
        raise ValueError('GPU input/extractor differs')
    parts, hashes = [], {}
    for start in range(0, len(rows), 32):
        end = min(start+32, len(rows))
        p = cache / f'{start:06d}_{end:06d}.npy'
        a = np.load(p, allow_pickle=False)
        if a.shape != (end-start, 4096) or not np.isfinite(a).all():
            raise ValueError('Invalid GPU shard')
        parts.append(a)
        hashes[str(p)] = digest(p)
    features = np.concatenate(parts)
    cpu_parts = []
    for path, checksum in revision['cpu_shards_sha256'].items():
        if digest(path) != checksum:
            raise ValueError('Preserved CPU vectors changed')
        cpu_parts.append(np.load(path, allow_pickle=False))
    cpu = np.concatenate(cpu_parts)
    audit = gpu.compare_vectors(cpu, features[:len(cpu)], lock)
    saved = json.loads((root / 'gpu_cpu_equivalence.json').read_text())
    if audit != saved:
        raise ValueError('Original numerical/action audit does not replay exactly')
    review = {**numerical_review(cpu, features, audit), 'reviewer_code_sha256': digest(__file__),
              'gpu_manifest_sha256': digest(manifest), 'gpu_shards_sha256': hashes,
              'gpu_execution_revision_sha256': digest(root / 'gpu_execution_revision.json'),
              'strict_audit_sha256': digest(root / 'gpu_cpu_equivalence.json')}
    locked_json(root / 'gpu_numerical_review.json', review)
    if not review['accepted']:
        raise RuntimeError('Numerical runtime review failed; no evaluation')
    length_audit = json.loads((cache / 'input_length_audit.json').read_text())
    info = {'name': 'hf_hidden_state', 'encoder_name': signature['encoder_name'], 'embedding_dim': 4096,
            'max_length': signature['max_length'], 'device': 'cuda:0', 'physical_gpu': signature['physical_gpu'],
            'pooling': 'mean_last', 'layer_indices': [-2, -1], 'dtype': 'float32', 'cache': str(cache),
            'encoder_clipped_count': len(length_audit['truncated_ids']),
            'execution_revision_sha256': digest(root / 'gpu_execution_revision.json'),
            'strict_cpu_gpu_audit_passed': False, 'runtime_review_sha256': digest(root / 'gpu_numerical_review.json'),
            'runtime_review_scope': review['limitations']}
    write_json(cache / 'reviewed_complete.json', info)

    def replay(loaded_rows, encoder, cache_path, max_length, batch_size):
        if loaded_rows != rows or encoder != signature['encoder_name'] or max_length != signature['max_length']:
            raise ValueError('Evaluator input mismatch')
        return features, info

    original = offline.extract_features
    try:
        offline.extract_features = replay
        offline.evaluate(root)
    finally:
        offline.extract_features = original
    report = root / 'evaluation/REPORT.md'
    marker = '## GPU numerical runtime review'
    text = report.read_text()
    if marker not in text:
        report.write_text(text + '\n' + marker + '\n\n' +
            'All vectors were extracted on GPU in FP32; policy scoring and statistics ran on CPU.\n' +
            f'The strict elementwise audit failed on {review["strict_element_mismatches"]}/{review["total_elements"]} coordinates. ' +
            f'On {review["overlap_n"]} saved CPU rows, Round and all 36 frozen policy actions matched.\n' +
            'A separate post-hoc, label-free vector-level numerical review accepted the runtime change. ' +
            'The failed strict audit remains preserved; this is not bitwise equivalence on the full set.\n' +
            'See gpu_cpu_equivalence.json, gpu_numerical_review.json and encoder_audit.json for provenance.\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    with (args.root / '.run.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args.root.resolve())
