"""Audited GPU-only vector extraction; the frozen policy/statistics remain on CPU."""
from __future__ import annotations

import argparse
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from . import evaluate_accounted_math_test as offline
from . import math_reflector_pilot as pilot
from .neural_policy import build_policy_text, _ensure_torchvision_nms_compat
from .train_fixed_math_policy import digest, write_json
from .utils import load_pickle


def pool_hidden(hidden_states, mask):
    last = mask.sum(1) - 1
    pooled = []
    for layer in (-2, -1):
        hidden = hidden_states[layer].float()
        mean = (hidden * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
        pooled.extend([mean, hidden[torch.arange(len(last), device=mask.device), last]])
    return torch.cat(pooled, 1)


def compare_vectors(cpu, gpu, lock):
    if cpu.shape != gpu.shape or not len(cpu) or not np.isfinite(cpu).all() or not np.isfinite(gpu).all():
        raise ValueError('Invalid matched CPU/GPU features')
    difference = gpu.astype('float64') - cpu.astype('float64')
    result = {'n': len(cpu), 'max_absolute_difference': float(np.max(np.abs(difference))),
              'rmse': float(np.sqrt(np.mean(difference ** 2))), 'rtol': .001, 'atol': .0001,
              'features_close': bool(np.allclose(cpu, gpu, rtol=.001, atol=.0001)), 'policies': []}
    bundle = load_pickle(lock['round_model'])
    labels = np.array(bundle['action_labels'])
    cpu_round = labels[offline.score_model(bundle, cpu).argmax(1)]
    gpu_round = labels[offline.score_model(bundle, gpu).argmax(1)]
    result['round_action_mismatches'] = int(np.count_nonzero(cpu_round != gpu_round))
    for point in lock['policies']:
        bundle = load_pickle(point['model'])
        left, lp = offline.frozen_actions(bundle, cpu, cpu_round, point['branch'], point['direct_threshold'])
        right, rp = offline.frozen_actions(bundle, gpu, gpu_round, point['branch'], point['direct_threshold'])
        result['policies'].append({'name': point['name'], 'action_mismatches': int(np.count_nonzero(left != right)),
                                   'max_direct_probability_difference': float(np.max(np.abs(lp-rp)))})
    result['passed'] = result['features_close'] and result['round_action_mismatches'] == 0 and all(
        p['action_mismatches'] == 0 for p in result['policies'])
    result['scope'] = 'Saved CPU prefix only; not a proof of bitwise equality on uncomputed rows'
    return result


def encode_gpu(rows, lock, root, revision):
    if not torch.cuda.is_available():
        raise RuntimeError('GPU requested but CUDA unavailable; no silent CPU fallback')
    _ensure_torchvision_nms_compat(torch)
    from transformers import AutoTokenizer, AutoModel
    import transformers
    from torch.nn.attention import sdpa_kernel, SDPBackend

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    texts = [build_policy_text(r, max_text_chars=0, include_state=False) for r in rows]
    old = json.loads(Path(revision['cpu_manifest']).read_text())
    text_hashes = [hashlib.sha256(t.encode()).hexdigest() for t in texts]
    if old['ids'] != [r['id'] for r in rows] or old['text_sha256'] != text_hashes:
        raise ValueError('Original CPU cache input differs from test rows')
    if (old['layers'], old['pooling'], old['dtype'], old['device'], old['max_length']) != (
            [-2, -1], 'mean_last', 'float32', 'cpu', lock['encoder']['max_length']):
        raise ValueError('Unexpected frozen CPU representation')
    for name, checksum in old['encoder_files_sha256'].items():
        if digest(Path(old['encoder_name']) / name) != checksum:
            raise ValueError('Test cache encoder changed')
    signature = {**old, 'version': 2, 'device': 'cuda:0', 'physical_gpu': revision['physical_gpu'],
                 'gpu_name': torch.cuda.get_device_name(0), 'extractor_sha256': digest(__file__),
                 'torch': torch.__version__, 'transformers': transformers.__version__,
                 'attention_backend': 'sdpa_math', 'tf32': False, 'batch_size': 2,
                 'source_cpu_manifest_sha256': digest(revision['cpu_manifest'])}
    fingerprint = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    cache = root / 'test_feature_cache_gpu' / fingerprint[:20]
    cache.mkdir(parents=True, exist_ok=True)
    pilot.locked_json(cache / 'manifest.json', signature)
    tokenizer = AutoTokenizer.from_pretrained(old['encoder_name'], local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    lengths = [len(tokenizer(t, truncation=False)['input_ids']) for t in texts]
    write_json(cache / 'input_length_audit.json', {'max_length': old['max_length'], 'lengths': lengths,
        'truncated_ids': [r['id'] for r, n in zip(rows, lengths) if n > old['max_length']]})
    model, parts, start_time = None, [], time.monotonic()
    for start in range(0, len(rows), 32):
        end = min(start + 32, len(rows))
        path = cache / f'{start:06d}_{end:06d}.npy'
        if path.exists():
            values = np.load(path, allow_pickle=False)
        else:
            if model is None:
                model = AutoModel.from_pretrained(old['encoder_name'], local_files_only=True,
                    dtype=torch.float32, attn_implementation='sdpa').to('cuda:0').eval().requires_grad_(False)
            batches = []
            for b in range(start, end, 2):
                batch = tokenizer(texts[b:min(b+2, end)], padding=True, truncation=True,
                                  max_length=old['max_length'], return_tensors='pt').to('cuda:0')
                with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
                    hidden = model(**batch, output_hidden_states=True, use_cache=False).hidden_states
                    batches.append(pool_hidden(hidden, batch['attention_mask']).cpu().numpy())
                del hidden, batch
            values = np.concatenate(batches).astype('float32')
            if values.shape != (end-start, 4096) or not np.isfinite(values).all():
                raise ValueError('Invalid GPU feature shard')
            temporary = path.with_suffix('.tmp')
            with temporary.open('wb') as handle:
                np.save(handle, values, allow_pickle=False)
            temporary.replace(path)
        if values.shape != (end-start, 4096) or not np.isfinite(values).all():
            raise ValueError('Invalid cached GPU vectors')
        parts.append(values)
        write_json(root / 'status.json', {'stage': 'encoding_test_gpu', 'physical_gpu': revision['physical_gpu'],
                   'completed': end, 'n': len(rows), 'test_metrics_released': False})
        print(f'[encode gpu] {end}/{len(rows)} elapsed={time.monotonic()-start_time:.0f}s', flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    features = np.concatenate(parts)
    cpu_parts = []
    end = 0
    for path, checksum in revision['cpu_shards_sha256'].items():
        if digest(path) != checksum:
            raise ValueError('Preserved CPU vectors changed')
        start, stop = map(int, Path(path).stem.split('_'))
        values = np.load(path, allow_pickle=False)
        if start != end or values.shape != (stop-start, 4096):
            raise ValueError('CPU shard prefix is not contiguous')
        cpu_parts.append(values)
        end = stop
    comparison = compare_vectors(np.concatenate(cpu_parts), features[:end], lock)
    write_json(root / 'gpu_cpu_equivalence.json', comparison)
    if not comparison['passed']:
        raise RuntimeError('CPU/GPU numerical or action audit failed; no policy evaluation released')
    info = {'name': 'hf_hidden_state', 'encoder_name': old['encoder_name'], 'embedding_dim': 4096,
            'max_length': old['max_length'], 'device': 'cuda:0', 'physical_gpu': revision['physical_gpu'],
            'pooling': 'mean_last', 'layer_indices': [-2, -1], 'dtype': 'float32', 'cache': str(cache),
            'fingerprint': fingerprint, 'encoder_clipped_count': sum(n > old['max_length'] for n in lengths),
            'execution_revision_sha256': digest(root / 'gpu_execution_revision.json'),
            'cpu_gpu_audit_sha256': digest(root / 'gpu_cpu_equivalence.json')}
    write_json(cache / 'complete.json', info)
    return features, info


def run(root, gpu):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    source, lock, rows, audit = offline.validate_inputs(root)
    if (root / 'evaluation/summary.json').exists():
        raise ValueError('Evaluation already committed; refusing to replace released results')
    cpu_manifests = list((root / 'test_feature_cache').glob('*/manifest.json'))
    if len(cpu_manifests) != 1:
        raise ValueError('Require the single existing CPU test cache for comparison')
    cpu_manifest = cpu_manifests[0]
    shards = sorted(cpu_manifest.parent.glob('*.npy'))
    if not shards:
        raise ValueError('Require saved CPU vectors before switching runtime device')
    revision = {'change': 'GPU FP32 extraction then unchanged CPU policy/statistical evaluation',
                'physical_gpu': str(gpu),
                'prior_cpu_pid': json.loads((root / 'launch.json').read_text())['pid'],
                'source_accounting_manifest_sha256': digest(root / 'manifest.json'),
                'accounting_audit_sha256': digest(root / 'accounting_audit.json'),
                'extractor_sha256': digest(__file__), 'cpu_manifest': str(cpu_manifest),
                'cpu_manifest_sha256': digest(cpu_manifest),
                'cpu_shards_sha256': {str(p): digest(p) for p in shards},
                'no_regeneration': True, 'no_retraining': True, 'no_threshold_change': True,
                'evaluator_dependency_injection': 'only extract_features; return audited GPU vectors'}
    pilot.locked_json(root / 'gpu_execution_revision.json', revision)
    write_json(root / 'status.json', {'stage': 'encoding_test_gpu', 'physical_gpu': str(gpu),
                                    'n': len(rows), 'test_metrics_released': False})
    features, info = encode_gpu(rows, lock, root, revision)

    def replay(loaded_rows, encoder, cache, max_length, batch_size):
        if loaded_rows != rows or encoder != lock['encoder']['encoder_name'] or max_length != lock['encoder']['max_length']:
            raise ValueError('Evaluator requested different inputs')
        return features, info

    # The original locked evaluator is unmodified; only its extraction provider changes.
    original = offline.extract_features
    try:
        offline.extract_features = replay
        offline.evaluate(root)
    finally:
        offline.extract_features = original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=pilot.PROJECT / 'outputs/math_full_test_failure_accounted_v1')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()
    root = args.root.resolve()
    with (root / '.run.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            run(root, args.gpu)
        except Exception as exc:
            write_json(root / 'last_gpu_error.json', {'type': type(exc).__name__, 'message': str(exc)})
            raise


if __name__ == '__main__':
    main()
