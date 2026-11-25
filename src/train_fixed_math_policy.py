"""CPU-only frozen-vector Gate/Round training on an existing problem split."""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler

from .neural_policy import build_policy_text, _ensure_torchvision_nms_compat
from .transformer_policy import TorchPolicyMLP
from .utils import read_jsonl, save_pickle

ACTIONS = ('direct', 'reflect_1', 'reflect_2')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def candidates(row):
    result = {c['action']: c for c in row['candidate_results']}
    if len(result) != len(row['candidate_results']):
        raise ValueError(f"Duplicate candidate: {row['id']}")
    for c in result.values():
        if c['correct'] not in (0, 1) or c['token_cost'] != c['prompt_tokens'] + c['completion_tokens']:
            raise ValueError(f"Invalid correctness/token accounting: {row['id']}")
    return result


def best_action(row, actions=ACTIONS):
    cs = candidates(row)
    # Missing counterfactuals are never synthesized, scored as wrong, or given zero cost.
    if not set(actions).issubset(cs):
        raise ValueError(f"Missing candidates: {row['id']}")
    return max(actions, key=lambda a: (cs[a]['correct'], -cs[a]['token_cost']))


def audit_split(root, allow_partial=False):
    manifest = json.loads((root / 'manifest.json').read_text())
    output, audit = {}, {'manifest_sha256': digest(root / 'manifest.json')}
    for split in ('train', 'validation'):
        source = root / f'policy_{split}_partial.jsonl'
        rows = read_jsonl(str(source))
        items = read_jsonl(str(root / f'{split}_items.jsonl'))
        expected = {r['id']: r for r in items}
        if set(expected) != set(manifest[f'{split}_ids']) or len(expected) != len(items):
            raise ValueError(f'{split} items do not match the locked manifest')
        observed = set()
        for row in rows:
            ident = row['id']
            if ident in observed or ident not in expected:
                raise ValueError(f'Duplicate/foreign ID: {ident}')
            observed.add(ident)
            if row['question'] != expected[ident]['question'] or row['policy_split'] != split:
                raise ValueError(f'Question/split changed: {ident}')
            cs = candidates(row)
            if set(cs) == set(ACTIONS):
                label = best_action(row)
            elif (split == 'train' and set(cs) == {'direct'} and cs['direct']['correct'] == 1
                  and row.get('candidate_coverage') == 'direct_only_known_correct'):
                label = 'direct'
            else:
                raise ValueError(f'Unsupported candidate coverage: {ident}')
            if label != row['best_action']:
                raise ValueError(f'Incorrect accuracy-first label: {ident}')
        missing = sorted(set(expected) - observed)
        if missing and not allow_partial:
            raise ValueError('Unresolved examples remain; explicitly pass --allow-partial')
        output[split] = rows
        audit[split] = {'file': str(source), 'sha256': digest(source), 'expected': len(expected),
                        'observed': len(rows), 'missing_ids': missing,
                        'labels': dict(Counter(r['best_action'] for r in rows))}
    if set(manifest['train_ids']) & set(manifest['validation_ids']):
        raise ValueError('Train/validation ID overlap')
    normalize = lambda q: ' '.join(q.lower().split())
    if {normalize(r['question']) for r in output['train']} & {normalize(r['question']) for r in output['validation']}:
        raise ValueError('Train/validation question overlap')
    audit['provisional'] = any(audit[s]['missing_ids'] for s in output)
    return output['train'], output['validation'], audit


def extract_features(rows, encoder, cache, max_length, batch_size):
    """Content-addressed shards allow CPU extraction to resume without re-encoding."""
    _ensure_torchvision_nms_compat(torch)
    from transformers import AutoModel, AutoTokenizer
    import transformers

    cache.mkdir(parents=True, exist_ok=True)
    texts = [build_policy_text(r, max_text_chars=0, include_state=False) for r in rows]
    encoder = Path(encoder).resolve()
    model_files = sorted(p for p in encoder.iterdir() if p.is_file() and
                         (p.suffix in ('.json', '.safetensors', '.bin', '.txt', '.model', '.py')))
    if not any(p.suffix in ('.safetensors', '.bin') for p in model_files):
        raise ValueError('No local encoder weights found')
    signature = {'version': 1, 'encoder_name': str(encoder),
                 'encoder_files_sha256': {p.name: digest(p) for p in model_files},
                 'layers': [-2, -1], 'pooling': 'mean_last', 'dtype': 'float32',
                 'device': 'cpu', 'max_length': max_length, 'max_text_chars': 0,
                 'torch': torch.__version__, 'transformers': transformers.__version__,
                 'neural_policy_sha256': digest(Path(__file__).with_name('neural_policy.py')),
                 'extractor_sha256': digest(__file__),
                 'ids': [r['id'] for r in rows],
                 'text_sha256': [hashlib.sha256(t.encode()).hexdigest() for t in texts]}
    fingerprint = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    cache = cache / fingerprint[:20]
    cache.mkdir(exist_ok=True)
    lock = (cache / 'cache.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = cache / 'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text()) != signature:
        raise ValueError('Cache fingerprint mismatch')
    write_json(manifest, signature)
    tokenizer = AutoTokenizer.from_pretrained(str(encoder), local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    lengths = [len(tokenizer(t, truncation=False)['input_ids']) for t in texts]
    write_json(cache / 'input_length_audit.json', {
        'max_length': max_length, 'maximum_observed': max(lengths), 'lengths': lengths,
        'truncated_ids': [r['id'] for r, n in zip(rows, lengths) if n > max_length],
        'note': 'Encoder input clipping only; no LLM generation is performed.'})
    print(f'[cache] {cache} rows={len(rows)} encoder_clipped={sum(n > max_length for n in lengths)}', flush=True)
    model, parts, started, new_count = None, [], time.monotonic(), 0
    shard_size = 32
    for start in range(0, len(texts), shard_size):
        end = min(start + shard_size, len(texts))
        shard = cache / f'{start:06d}_{end:06d}.npy'
        if shard.exists():
            values = np.load(shard, allow_pickle=False)
        else:
            if model is None:
                model = AutoModel.from_pretrained(str(encoder), local_files_only=True,
                                                 dtype=torch.float32, attn_implementation='sdpa')
                model.to('cpu').eval().requires_grad_(False)
            encoded = []
            for b in range(start, end, batch_size):
                batch = tokenizer(texts[b:min(b + batch_size, end)], padding=True,
                                  truncation=True, max_length=max_length, return_tensors='pt')
                with torch.inference_mode():
                    hs = model(**batch, output_hidden_states=True, use_cache=False).hidden_states
                    mask = batch['attention_mask']
                    last = mask.sum(1) - 1
                    pooled = []
                    for layer in (-2, -1):
                        hidden = hs[layer].float()
                        mean = (hidden * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
                        pooled.extend([mean, hidden[torch.arange(len(last)), last]])
                    encoded.append(torch.cat(pooled, 1).numpy())
            values = np.concatenate(encoded).astype('float32')
            if not np.isfinite(values).all():
                raise ValueError(f'Nonfinite encoder features in rows {start}:{end}')
            temporary = shard.with_suffix('.tmp')
            with temporary.open('wb') as stream:
                np.save(stream, values, allow_pickle=False)
            temporary.replace(shard)
            new_count += end - start
        if values.shape != (end - start, 4096) or not np.isfinite(values).all():
            raise ValueError(f'Invalid cache shard: {shard}')
        parts.append(values)
        elapsed = time.monotonic() - started
        eta = elapsed / new_count * (len(texts) - end) if new_count else 0
        print(f'[encode cpu] {end}/{len(texts)} elapsed={elapsed:.0f}s eta={eta:.0f}s', flush=True)
    info = {'name': 'hf_hidden_state', 'encoder_name': str(encoder), 'embedding_dim': 4096,
            'max_length': max_length, 'device': 'cpu', 'pooling': 'mean_last',
            'layer_indices': [-2, -1], 'cache': str(cache), 'fingerprint': fingerprint,
            'encoder_clipped_count': sum(n > max_length for n in lengths)}
    write_json(cache / 'complete.json', info)
    lock.close()
    return np.concatenate(parts), info


def decision_metrics(rows, actions, lambda_cost=0.0001):
    chosen = []
    for row, action in zip(rows, actions, strict=True):
        cs = candidates(row)
        if action == 'independent':
            chosen.append(row['independent'])
        elif action in cs:
            chosen.append(cs[action])
        else:
            raise ValueError(f"Cannot evaluate unobserved action {action}: {row['id']}")
    correct = np.array([c['correct'] for c in chosen])
    base = np.array([candidates(r)['direct']['correct'] for r in rows])
    prompt = np.array([c['prompt_tokens'] for c in chosen])
    completion = np.array([c['completion_tokens'] for c in chosen])
    total = np.array([c['token_cost'] for c in chosen])
    if not np.array_equal(prompt + completion, total):
        raise ValueError('Inconsistent total token count')
    fix, harm = int(((base == 0) & (correct == 1)).sum()), int(((base == 1) & (correct == 0)).sum())
    return {'n': len(rows), 'correct': int(correct.sum()), 'accuracy': float(correct.mean()),
            'avg_prompt_tokens': float(prompt.mean()), 'avg_completion_tokens': float(completion.mean()),
            'avg_token_cost': float(total.mean()), 'avg_reward': float((correct - lambda_cost * total).mean()),
            'reflection_rate': sum(a in ACTIONS[1:] for a in actions) / len(rows),
            'actions': dict(Counter(actions)), 'fixes': fix, 'harms': harm, 'net_fix': fix - harm,
            'fix_rate': fix / int((base == 0).sum()) if (base == 0).any() else None,
            'harm_rate': harm / int((base == 1).sum()) if (base == 1).any() else None}


def select_threshold(rows, direct_prob, round_actions, lambda_cost):
    # Fixed grid, actual predicted Round outcomes. No oracle reflection proxy.
    choices = []
    for threshold in [*np.linspace(0, 1, 21), 1.000001]:
        actions = ['direct' if p >= threshold else a for p, a in zip(direct_prob, round_actions)]
        metrics = decision_metrics(rows, actions, lambda_cost)
        choices.append((float(threshold), metrics, actions))
    return max(choices, key=lambda x: (x[1]['correct'], -x[1]['avg_token_cost'], -x[0]))


def label_metrics(labels, predictions):
    p, r, f, _ = precision_recall_fscore_support(labels, predictions, average='binary', zero_division=0)
    return {'accuracy': float(accuracy_score(labels, predictions)), 'positive_precision': float(p),
            'positive_recall': float(r), 'positive_f1': float(f)}


def plot_history(history, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    epochs = [h['epoch'] for h in history]
    for head in ('gate', 'round'):
        for split in ('train', 'validation'):
            axes[0, 0].plot(epochs, [h[head][f'{split}_loss'] for h in history], label=f'{head} {split}')
    for key, label in [('system', 'calibrated Gate/Round'), ('uncalibrated', 'threshold 0.5')]:
        axes[0, 1].plot(epochs, [h[key]['accuracy'] for h in history], label=label)
        axes[1, 0].plot(epochs, [h[key]['avg_token_cost'] for h in history], label=label)
    for key in ('direct', 'fixed_reflect_1', 'independent'):
        baseline = history[0]['baselines'][key]
        axes[0, 1].axhline(baseline['accuracy'], linestyle='--', label=key)
        axes[1, 0].axhline(baseline['avg_token_cost'], linestyle='--', label=key)
    for key in ('fixes', 'harms'):
        axes[1, 1].plot(epochs, [h['system'][key] for h in history], label=key)
    for ax, title in zip(axes.flat, ['Head cross-entropy', 'Validation final answer accuracy',
                                    'Validation input + output tokens', 'Validation fixes and harms']):
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def train_heads(train, val, features, info, audit, args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    n = len(train)
    round_indices = np.array([i for i, r in enumerate(train) if set(candidates(r)) == set(ACTIONS)])
    sets = {'gate': (np.arange(n), np.array([r['best_action'] != 'direct' for r in train], dtype=np.int64),
                     np.array([r['best_action'] != 'direct' for r in val], dtype=np.int64)),
            'round': (round_indices, np.array([best_action(train[i], ACTIONS[1:]) == 'reflect_2' for i in round_indices], dtype=np.int64),
                      np.array([best_action(r, ACTIONS[1:]) == 'reflect_2' for r in val], dtype=np.int64))}
    heads = {}
    for name, (indices, labels, val_labels) in sets.items():
        scaler = StandardScaler().fit(features[indices])
        model = TorchPolicyMLP(4096, (64, 16), 2).cpu()
        counts = np.bincount(labels, minlength=2)
        if not counts.all():
            raise ValueError(f'{name} has fewer than two observed classes')
        weight = torch.tensor(len(labels) / (2 * counts), dtype=torch.float32) if name == 'round' else None
        heads[name] = {'model': model, 'scaler': scaler,
                       'x': torch.from_numpy(scaler.transform(features[indices]).astype('float32')),
                       'v': torch.from_numpy(scaler.transform(features[n:]).astype('float32')),
                       'y': torch.from_numpy(labels), 'vy': torch.from_numpy(val_labels),
                       'optimizer': torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.001),
                       'criterion': torch.nn.CrossEntropyLoss(weight=weight)}
        print(f'[train] {name} n={len(labels)} labels={counts.tolist()} balance={name == "round"}', flush=True)
    baselines = {name: decision_metrics(val, [a] * len(val), args.lambda_cost) for name, a in
                 [('direct', 'direct'), ('fixed_reflect_1', 'reflect_1'), ('fixed_reflect_2', 'reflect_2'), ('independent', 'independent')]}
    baselines['oracle_accuracy_first'] = decision_metrics(val, [best_action(r) for r in val], args.lambda_cost)
    write_json(args.output / 'baselines.json', baselines)
    history, best_key = [], None
    for epoch in range(1, args.epochs + 1):
        entry = {'epoch': epoch, 'baselines': baselines}
        probabilities = {}
        for name, head in heads.items():
            model = head['model']
            model.train()
            for batch in torch.randperm(len(head['y'])).split(128):
                head['optimizer'].zero_grad()
                loss = head['criterion'](model(head['x'][batch]), head['y'][batch])
                loss.backward()
                head['optimizer'].step()
            model.eval()
            with torch.inference_mode():
                tr, va = model(head['x']), model(head['v'])
                entry[name] = {'train_loss': float(torch.nn.functional.cross_entropy(tr, head['y'])),
                               'validation_loss': float(torch.nn.functional.cross_entropy(va, head['vy'])),
                               'train': label_metrics(head['y'].numpy(), tr.argmax(1).numpy()),
                               'validation': label_metrics(head['vy'].numpy(), va.argmax(1).numpy())}
                probabilities[name] = va.softmax(1).numpy()
        rounds = [ACTIONS[1 + int(i)] for i in probabilities['round'].argmax(1)]
        threshold, system, actions = select_threshold(val, probabilities['gate'][:, 0], rounds, args.lambda_cost)
        entry['system'], entry['direct_threshold'] = system, threshold
        entry['uncalibrated'] = decision_metrics(val, ['direct' if p >= .5 else a for p, a in
                                                       zip(probabilities['gate'][:, 0], rounds)], args.lambda_cost)
        entry['gate_calibrated'] = label_metrics(sets['gate'][2], np.array([a != 'direct' for a in actions]))
        entry['train_system_metrics'] = None
        entry['train_system_note'] = 'Unobserved reflection outcomes on correct Direct training rows; no full train system score.'
        history.append(entry)
        write_json(args.output / 'history.json', history)
        plot_history(history, args.output / f'epoch_{epoch:02d}.png')
        plot_history(history, args.output / 'training_curve.png')
        key = (system['correct'], -system['avg_token_cost'])
        if best_key is None or key > best_key:
            best_key = key
            for name, head in heads.items():
                bundle = {'policy_type': 'torch_embedding_mlp', 'backend': 'hf_hidden_torch_mlp',
                          'model': head['model'], 'scaler': head['scaler'], 'encoder_info': info,
                          'encoder_name': info['encoder_name'], 'layers': [-2, -1], 'pooling': 'mean_last',
                          'pure_text': True, 'max_text_chars': 0, 'max_length': args.max_length,
                          'encoder_batch_size': args.encoder_batch_size, 'device': 'cpu',
                          'action_labels': ['direct', 'reflect_1'] if name == 'gate' else ['reflect_1', 'reflect_2'],
                          'hidden_sizes': [64, 16], 'best_epoch': epoch, 'fixed_split_audit': audit,
                          'selection_metric': 'joint_validation_answer_accuracy_then_minimum_tokens',
                          'lambda_cost': args.lambda_cost}
                if name == 'gate':
                    bundle['direct_threshold'] = threshold
                save_pickle(bundle, str(args.output / f'{name}.pkl'))
            write_json(args.output / 'selected_validation.json', {
                'best_epoch': epoch, 'direct_threshold': threshold, 'metrics': system,
                'baselines': baselines, 'coverage': audit, 'encoder': info,
                'expected_validation_accuracy_bounds': [system['correct'] / audit['validation']['expected'],
                    (system['correct'] + len(audit['validation']['missing_ids'])) / audit['validation']['expected']],
                'scope': 'Model selection on complete-case internal validation; NOT independent test evidence.',
                'predictions': [{'id': r['id'], 'action': a} for r, a in zip(val, actions)]})
        print(f"[epoch {epoch}/{args.epochs}] answer_acc={system['accuracy']:.4f} tokens={system['avg_token_cost']:.1f} "
              f"reflect={system['reflection_rate']:.4f} fixes/harms={system['fixes']}/{system['harms']} "
              f"direct_threshold={threshold:.4f}", flush=True)
    write_json(args.output / 'status.json', {'stage': 'completed', 'epochs': args.epochs,
                                           'provisional': audit['provisional'], 'test_evaluated': False})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('outputs/math_full_training_v1'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cache', type=Path, default=Path('outputs/math_fixed_features'))
    p.add_argument('--encoder', default='Qwen/Qwen3-0.6B')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--max-length', type=int, default=2048)
    p.add_argument('--encoder-batch-size', type=int, default=2)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--lambda-cost', type=float, default=.0001)
    p.add_argument('--allow-partial', action='store_true')
    args = p.parse_args()
    if min(args.threads, args.max_length, args.encoder_batch_size, args.epochs) < 1:
        p.error('Thread counts, lengths, batch sizes and epochs must be positive')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    torch.set_num_threads(args.threads)
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / 'run.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (args.output / 'history.json').exists():
        raise ValueError('Training history already exists; choose a new output directory (features remain reusable).')
    train, val, audit = audit_split(args.root, args.allow_partial)
    write_json(args.output / 'split_audit.json', audit)
    write_json(args.output / 'training_protocol.json', {
        'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'input': 'Question + initial answer only', 'encoder': 'fully frozen, last two hidden layers, mean_last',
        'gate_balance': False, 'round_balance': 'inverse class frequency on observed train candidates only',
        'threshold_grid': [*np.linspace(0, 1, 21), 1.000001], 'round_decision': 'argmax before reflection',
        'selection': 'paired same-epoch heads, actual validation final correct count then lowest tokens; earliest tie',
        'limitations': ['Incomplete candidates retained in coverage audit', 'Internal validation used for selection, not confirmation',
                        'Missing train counterfactuals preclude full train end-to-end metrics',
                        'Encoder and heads incur CPU latency not represented by LLM token metrics'],
        'source_sha256': digest(__file__)})
    write_json(args.output / 'status.json', {'stage': 'encoding_cpu', 'provisional': audit['provisional']})
    try:
        features, info = extract_features(train + val, args.encoder, args.cache, args.max_length, args.encoder_batch_size)
        write_json(args.output / 'status.json', {'stage': 'training_cpu', 'provisional': audit['provisional']})
        train_heads(train, val, features, info, audit, args)
    except Exception as exc:
        write_json(args.output / 'status.json', {'stage': 'failed', 'error': repr(exc)})
        raise


if __name__ == '__main__':
    main()
