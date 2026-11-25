import argparse
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.neural_policy import build_policy_text
from src.train_fixed_math_policy import (ACTIONS, audit_split, best_action, candidates, extract_features,
                                         decision_metrics, select_threshold, train_heads)
from src.utils import load_pickle


def row(ident, scores=(0, 1, 1), split='train'):
    cs = [{'action': a, 'correct': c, 'prompt_tokens': 10 * (i + 1),
           'completion_tokens': 10 * (i + 1), 'token_cost': 20 * (i + 1)}
          for i, (a, c) in enumerate(zip(ACTIONS, scores))]
    result = {'id': ident, 'question': f'Question {ident}?', 'initial_answer': 'Final answer: 1',
              'policy_split': split, 'candidate_results': cs, 'candidate_coverage': 'complete',
              'independent': {'action': 'independent', 'correct': 1, 'prompt_tokens': 15,
                              'completion_tokens': 15, 'token_cost': 30}}
    result['best_action'] = best_action(result)
    return result


class FixedPolicyTests(unittest.TestCase):
    def test_accuracy_first_before_cost(self):
        r = row('a', (0, 0, 1))
        r['candidate_results'][2].update(prompt_tokens=100000, token_cost=100030)
        self.assertEqual(best_action(r), 'reflect_2')

    def test_cheapest_on_tie(self):
        self.assertEqual(best_action(row('a', (1, 1, 1))), 'direct')
        self.assertEqual(best_action(row('a', (0, 0, 0)), ACTIONS[1:]), 'reflect_1')

    def test_missing_candidate_never_zero_filled(self):
        r = row('a', (1, 1, 1))
        r['candidate_results'] = r['candidate_results'][:1]
        with self.assertRaisesRegex(ValueError, 'unobserved'):
            decision_metrics([r], ['reflect_1'])
        with self.assertRaisesRegex(ValueError, 'Missing'):
            best_action(r)

    def test_joint_uses_predicted_round_not_oracle(self):
        r = row('a', (0, 1, 0))
        threshold, metrics, actions = select_threshold([r], np.array([.8]), ['reflect_2'], .0001)
        self.assertEqual(metrics['correct'], 0)
        self.assertEqual(actions, ['direct'])

    def test_fix_harm_and_full_cost(self):
        rows = [row('a', (0, 1, 1)), row('b', (1, 0, 1))]
        m = decision_metrics(rows, ['reflect_1', 'reflect_1'])
        self.assertEqual((m['fixes'], m['harms']), (1, 1))
        self.assertEqual(m['avg_token_cost'], 40)
        self.assertEqual(m['avg_prompt_tokens'] + m['avg_completion_tokens'], 40)

    def test_bad_cost_rejected(self):
        r = row('a')
        r['candidate_results'][0]['token_cost'] = 1
        with self.assertRaises(ValueError):
            candidates(r)

    def test_input_no_gold_labels_or_features(self):
        r = row('a')
        before = build_policy_text(r, max_text_chars=0, include_state=False)
        r.update(gold_answer='SECRET', best_action='reflect_2', difficulty='hard',
                 state_features={'verifier_score': 0.99})
        r['candidate_results'] = []
        self.assertEqual(build_policy_text(r, max_text_chars=0, include_state=False), before)
        self.assertNotIn('SECRET', before)

    def fixture(self, root):
        train, val = [row('a'), row('b', (0, 0, 1))], [row('v', split='validation')]
        manifest = {'train_ids': ['a', 'b', 'pending'], 'validation_ids': ['v']}
        (root / 'manifest.json').write_text(json.dumps(manifest))
        for split, rows in [('train', train), ('validation', val)]:
            (root / f'policy_{split}_partial.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
            items = rows + ([row('pending')] if split == 'train' else [])
            (root / f'{split}_items.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in items))

    def test_fixed_manifest_preserves_missing(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.fixture(root)
            tr, va, audit = audit_split(root, True)
            self.assertEqual([r['id'] for r in tr], ['a', 'b'])
            self.assertEqual([r['id'] for r in va], ['v'])
            self.assertEqual(audit['train']['missing_ids'], ['pending'])
            with self.assertRaisesRegex(ValueError, 'Unresolved'):
                audit_split(root)

    def test_wrong_split_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.fixture(root)
            r = row('v', split='train')
            (root / 'policy_validation_partial.jsonl').write_text(json.dumps(r) + '\n')
            with self.assertRaisesRegex(ValueError, 'split changed'):
                audit_split(root, True)

    def test_feature_cache_resume_and_masked_pooling(self):
        class Tokenizer:
            pad_token, eos_token = 'pad', 'eos'

            def __call__(self, texts, **kw):
                if isinstance(texts, str):
                    return {'input_ids': list(range(len(texts.split())))}
                lengths = [min(len(t.split()), kw['max_length']) for t in texts]
                ids = torch.zeros((len(texts), max(lengths)), dtype=torch.long)
                for i, n in enumerate(lengths):
                    ids[i, :n] = torch.arange(1, n + 1)
                return {'input_ids': ids, 'attention_mask': (ids > 0).long()}

        class Encoder(torch.nn.Module):
            def forward(self, input_ids, attention_mask, **kw):
                hidden = input_ids.float().unsqueeze(-1).expand(-1, -1, 1024)
                return SimpleNamespace(hidden_states=(hidden, hidden * 2))

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            model_dir = root / 'model'
            model_dir.mkdir()
            (model_dir / 'model.safetensors').write_bytes(b'fake unit-test weights')
            rows = [row('a'), row('b')]
            rows[0].update(question='one two three', initial_answer='four five six')
            rows[1].update(question='one', initial_answer='two')
            with patch('transformers.AutoTokenizer.from_pretrained', return_value=Tokenizer()), \
                    patch('transformers.AutoModel.from_pretrained', return_value=Encoder()) as loader:
                first, info = extract_features(rows, model_dir, root / 'cache', 16, 2)
                second, _ = extract_features(rows, model_dir, root / 'cache', 16, 2)
                self.assertEqual(loader.call_count, 1)
                np.testing.assert_array_equal(first, second)
                text = build_policy_text(rows[1], max_text_chars=0, include_state=False)
                n = len(text.split())
                np.testing.assert_array_equal(first[1, :1024], np.full(1024, (n + 1) / 2))
                np.testing.assert_array_equal(first[1, 1024:2048], np.full(1024, n))
                np.testing.assert_array_equal(first[1, 2048:3072], np.full(1024, n + 1))
                np.testing.assert_array_equal(first[1, 3072:], np.full(1024, 2 * n))
                np.save(next(Path(info['cache']).glob('*.npy')), np.zeros((1, 2)))
                with self.assertRaisesRegex(ValueError, 'Invalid cache shard'):
                    extract_features(rows, model_dir, root / 'cache', 16, 2)

    def test_cpu_training_scalers_export_and_curves(self):
        torch.set_num_threads(1)
        tr = [row(str(i), scores) for i, scores in enumerate([(1, 1, 1), (0, 1, 1), (0, 0, 1), (0, 0, 0)])]
        tr[0]['candidate_results'] = tr[0]['candidate_results'][:1]
        va = [row('v0', (0, 1, 0), 'validation'), row('v1', (1, 1, 1), 'validation')]
        f = np.random.default_rng(4).normal(size=(6, 4096)).astype('float32')
        f[4:] += 20  # Validation shift must not enter either fitted scaler.
        audit = {'provisional': True, 'validation': {'expected': 3, 'missing_ids': ['v2']}}
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            args = argparse.Namespace(seed=42, epochs=2, lambda_cost=.0001, output=out,
                                      max_length=1024, encoder_batch_size=2)
            train_heads(tr, va, f, {'encoder_name': '/unused'}, audit, args)
            gate, round_head = load_pickle(out / 'gate.pkl'), load_pickle(out / 'round.pkl')
            np.testing.assert_allclose(gate['scaler'].mean_, f[:4].mean(0), atol=1e-6)
            np.testing.assert_allclose(round_head['scaler'].mean_, f[1:4].mean(0), atol=1e-6)
            self.assertEqual(gate['best_epoch'], round_head['best_epoch'])
            self.assertEqual(next(gate['model'].parameters()).device.type, 'cpu')
            histories = json.loads((out / 'history.json').read_text())
            self.assertIsNone(histories[0]['train_system_metrics'])
            with torch.inference_mode():
                gp = gate['model'](torch.from_numpy(gate['scaler'].transform(f[4:]).astype('float32'))).softmax(1).numpy()[:, 0]
                rp = round_head['model'](torch.from_numpy(round_head['scaler'].transform(f[4:]).astype('float32'))).argmax(1).numpy()
            actions = ['direct' if p >= gate['direct_threshold'] else ACTIONS[1 + int(r)] for p, r in zip(gp, rp)]
            selected = json.loads((out / 'selected_validation.json').read_text())
            self.assertEqual(decision_metrics(va, actions), selected['metrics'])
            self.assertTrue((out / 'epoch_02.png').exists())
            self.assertTrue((out / 'training_curve.png').exists())


if __name__ == '__main__':
    unittest.main()
