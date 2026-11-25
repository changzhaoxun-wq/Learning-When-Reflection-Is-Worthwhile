"""Training-only Gate hyperparameter development; fixed labels, Round and threshold."""
import argparse
import copy
import fcntl
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from .math_gate_cpu_ablation import build_model, classification, read_cached_features, score_model
from .math_reflector_pilot import PROJECT, locked_json
from .train_fixed_math_policy import audit_split, digest, decision_metrics, write_json
from .utils import load_pickle, save_pickle

BASE = {'lr': .0003, 'weight_decay': .01, 'dropout': .2, 'positive_weight': 5., 'batch_size': 128}
TRIALS = [('baseline', BASE), *[(name, {**BASE, key: value}) for name, key, value in (
    ('lr_1e4', 'lr', .0001), ('lr_5e5', 'lr', .00005),
    ('wd_003', 'weight_decay', .03), ('dropout_035', 'dropout', .35),
    ('weight_2', 'positive_weight', 2.), ('unweighted', 'positive_weight', 1.),
    ('batch_64', 'batch_size', 64))]]


def selection_key(metrics):
    return (metrics['correct'], -metrics['avg_token_cost'], -metrics['reflection_rate'])


def plot(history, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    epochs = [r['epoch'] for r in history]
    for name in ('train_loss', 'validation_loss'):
        axes[0].plot(epochs, [r['classification'][name] for r in history], label=name)
    axes[0].set_title('Unweighted cross-entropy')
    axes[0].legend()
    axes[1].plot(epochs, [r['metrics']['accuracy'] for r in history])
    axes[1].set_title('Validation final answer accuracy')
    axes[2].plot(epochs, [r['metrics']['avg_token_cost'] for r in history])
    axes[2].set_title('Validation input + output tokens')
    for ax in axes:
        ax.set_xlabel('Epoch')
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def train_trial(name, config, seed, directory, train, val, features, rounds, threshold, encoder):
    directory.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    n = len(train)
    scaler = StandardScaler().fit(features[:n])
    x = torch.from_numpy(scaler.transform(features[:n]).astype('float32'))
    vx = torch.from_numpy(scaler.transform(features[n:]).astype('float32'))
    y = torch.tensor([r['best_action'] != 'direct' for r in train], dtype=torch.long)
    vy = torch.tensor([r['best_action'] != 'direct' for r in val], dtype=torch.long)
    model = build_model({'hidden': 16, 'dropout': config['dropout']}).cpu()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    weights = torch.tensor([1., config['positive_weight']])
    history, best, state = [], None, None
    for epoch in range(1, 11):
        model.train()
        for indices in torch.randperm(n).split(config['batch_size']):
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(x[indices]), y[indices], weight=weights)
            loss.backward()
            optimizer.step()
        cls, direct = classification(model, x, vx, y, vy)
        actions = np.where(direct >= threshold, 'direct', rounds).tolist()
        metrics = decision_metrics(val, actions)
        entry = {'epoch': epoch, 'classification': cls, 'metrics': metrics}
        if best is None or selection_key(metrics) > selection_key(best['metrics']):
            best = entry
            state = copy.deepcopy(model.state_dict())
        history.append(entry)
        write_json(directory / 'history.json', history)
        plot(history, directory / f'epoch_{epoch:02d}.png')
        print(f'[gate-tune] {name} seed={seed} epoch={epoch} '
              f"acc={metrics['accuracy']:.4f} tokens={metrics['avg_token_cost']:.1f} calls={metrics['reflection_rate']:.3f}", flush=True)
    model.load_state_dict(state)
    bundle = {'model': model.eval(), 'scaler': scaler, 'action_labels': ['direct', 'reflect_1'],
              'direct_threshold': threshold, 'best_epoch': best['epoch'], 'encoder': encoder,
              'device': 'cpu', 'deployment_branch': 'original_round',
              'selection_scope': 'existing internal validation; not independent confirmation',
              'hyperparameters': config, 'seed': seed}
    path = directory / 'gate.pkl'
    save_pickle(bundle, path)
    direct = score_model(load_pickle(path), features[n:])[:, 0]
    replay = decision_metrics(val, np.where(direct >= threshold, 'direct', rounds).tolist())
    if replay != best['metrics']:
        raise ValueError('Saved checkpoint metrics mismatch')
    result = {'name': name, 'config': config, 'seed': seed, 'epoch': best['epoch'],
              'metrics': best['metrics'], 'model': str(path), 'sha256': digest(path),
              'curve': str(directory / 'epoch_10.png')}
    write_json(directory / 'result.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=PROJECT / 'outputs/math_gate_hparam_v1')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    with (output / 'run.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output / 'protocol.json').exists():
            raise ValueError('Output already used; use a fresh output directory')
        lock_path = PROJECT / 'outputs/math_full_test_v1/policy_lock.json'
        lock = json.loads(lock_path.read_text())
        point = next(p for p in lock['policies'] if p['seed'] == 42 and p['recipe'] == 'weighted_ce_control'
                     and p['branch'] == 'original_round' and p['validation_call_budget'] == .10)
        train, val, audit = audit_split(PROJECT / 'outputs/math_full_training_v1', True)
        for split in ('train', 'validation'):
            if audit[split]['sha256'] != lock['training_coverage'][split]['sha256']:
                raise ValueError('Fixed data split changed')
        if digest(point['model']) != point['sha256'] or digest(lock['round_model']) != lock['round_sha256']:
            raise ValueError('Frozen reference model changed')
        features, encoder = read_cached_features(Path(lock['encoder']['cache']), train + val,
                                                 lock['encoder']['fingerprint'])
        round_bundle = load_pickle(lock['round_model'])
        rounds = np.array(round_bundle['action_labels'])[score_model(round_bundle, features[len(train):]).argmax(1)]
        threshold = point['direct_threshold']
        frozen = score_model(load_pickle(point['model']), features[len(train):])[:, 0]
        reference = decision_metrics(val, np.where(frozen >= threshold, 'direct', rounds).tolist())
        protocol = {'trials': TRIALS, 'epochs': 10, 'stage1_seed': 42, 'confirmation_seeds': [7, 21, 42],
            'fixed_threshold': threshold, 'threshold_source_point': point, 'threshold_search': False,
            'fixed_round': lock['round_model'], 'round_sha256': lock['round_sha256'],
            'fixed_split': audit, 'encoder': encoder, 'architecture': '4096 -> 16 ReLU -> Dropout -> 2',
            'loss': 'same cross-entropy; class weight is the only loss hyperparameter varied',
            'selection': 'validation correct count, then lower tokens, then lower call rate, earliest epoch',
            'stage2': 'baseline and top two nonbaseline seed42 recipes; repeat seeds7/21; reuse seed42',
            'test_access': False, 'no_label_or_network_width_change': True, 'no_round_training': True,
            'development_only': True, 'source_code_sha256': digest(__file__),
            'feature_shards_sha256': {str(p): digest(p) for p in Path(encoder['cache']).glob('*.npy')},
            'scope': 'Repeated validation development after prior test feedback; not new independent confirmation'}
        locked_json(output / 'protocol.json', protocol)
        write_json(output / 'baselines.json', {'frozen_gate_round': reference,
            'direct': decision_metrics(val, ['direct'] * len(val)),
            'fixed_reflect_1': decision_metrics(val, ['reflect_1'] * len(val)),
            'fixed_reflect_2': decision_metrics(val, ['reflect_2'] * len(val))})
        screening = []
        for name, config in TRIALS:
            screening.append(train_trial(name, config, 42, output / name / 'seed42', train, val,
                                         features, rounds, threshold, encoder))
        selected = sorted(screening[1:], key=lambda r: selection_key(r['metrics']), reverse=True)[:2]
        write_json(output / 'stage1.json', {'all_trials': screening, 'selected_for_multiseed': selected})
        groups = []
        for item in [screening[0], *selected]:
            members = [item]
            for seed in (7, 21):
                members.append(train_trial(item['name'], item['config'], seed,
                    output / item['name'] / f'seed{seed}', train, val, features, rounds, threshold, encoder))
            metrics = {f: {'mean': float(np.mean([r['metrics'][f] for r in members])),
                           'std': float(np.std([r['metrics'][f] for r in members], ddof=1))}
                       for f in ('accuracy', 'avg_token_cost', 'reflection_rate', 'fixes', 'harms')}
            groups.append({'name': item['name'], 'metrics': metrics, 'members': members})
        write_json(output / 'summary.json', {'groups': groups, 'screening': screening,
                   'frozen_validation_reference': reference, 'test_evaluated': False,
                   'round_unchanged': True, 'threshold_unchanged': True})
        lines = ['# Gate Training Parameter Development', '',
                 'Internal validation only; no test evaluation or threshold search. Round and labels unchanged.',
                 'Stage 1 screens eight single-factor recipes at seed42; stage2 reuses seed42 and adds seeds7/21.',
                 'All epoch curves and unweighted CE diagnostics are saved. Best epochs selected by answer quality then cost.',
                 'These are validation-selected development results, not independent multi-seed confirmation.', '',
                 '| Recipe | Accuracy mean +/- SD | Tokens mean +/- SD | Call rate |',
                 '| --- | ---: | ---: | ---: |']
        for g in groups:
            m = g['metrics']
            lines.append(f"| {g['name']} | {m['accuracy']['mean']:.4f} +/- {m['accuracy']['std']:.4f} | {m['avg_token_cost']['mean']:.2f} +/- {m['avg_token_cost']['std']:.2f} | {m['reflection_rate']['mean']:.4f} |")
        from .prepare_qa_hard_development import locked_write
        locked_write(output / 'REPORT.md', '\n'.join(lines) + '\n')
        print(f'[gate-tune] COMPLETE {output / "REPORT.md"}', flush=True)


if __name__ == '__main__':
    main()
