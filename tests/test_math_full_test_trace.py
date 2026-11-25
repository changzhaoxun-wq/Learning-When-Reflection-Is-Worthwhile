import contextlib
import json
import itertools
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from src import math_full_test_trace as full
from src.evaluate_locked_math_test import assert_complete_trace, frozen_actions
from src import evaluate_locked_math_test as offline


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return messages[0]['content']

    def encode(self, text, **kwargs):
        return list(range(len(text.split())))


class FullTestTests(unittest.TestCase):
    def setUp(self):
        self.items = [{'id': key, 'question': key, 'gold_answer': '1', 'policy_split': 'test'}
                      for key in ('correct', 'wrong')]
        self.tokenizers = {r: Tokenizer() for r in ('base', 'qwen14')}

    def fake_server(self, generated, length=False):
        @contextlib.contextmanager
        def server(root, role):
            def generate(prompt, **kwargs):
                generated.append((role, prompt, kwargs))
                answer = '0' if role == 'base' and 'wrong' in prompt else '1'
                p, c = len(prompt.split()), 5
                return SimpleNamespace(text=f'Final answer: {answer}', prompt_tokens=p, completion_tokens=c,
                    total_tokens=p+c, metadata={'finish_reason': 'length' if length else 'stop',
                        'raw_usage': {'prompt_tokens': p, 'completion_tokens': c, 'total_tokens': p+c},
                        'request': {'temperature': 0., 'seed': 42, 'effective_max_tokens': kwargs['max_tokens']}})
            yield SimpleNamespace(generate=generate)
        return server

    def test_complete_test_coverage_resume_usage_and_tampering(self):
        calls = []
        with tempfile.TemporaryDirectory() as folder, patch.object(full.pilot, 'server', self.fake_server(calls)), patch.object(full, 'wait_for_resources'):
            root = Path(folder)
            (root / 'protocol.json').write_text('{}')
            (root / 'policy_lock.json').write_text('{}')
            base = full.run_stage(root, 'base', self.items, self.tokenizers)
            lookup = {r['id']: r for r in base}
            rows = full.run_stage(root, 'qwen14', self.items, self.tokenizers, lookup)
            self.assertEqual(len(calls), 8)
            self.assertTrue(all(r['candidate_coverage'] == 'complete' and 'independent' in r for r in rows))
            self.assertTrue(full.finalize(root, self.items, self.tokenizers, base, rows))
            audit = json.loads((root / 'completion_audit.json').read_text())
            self.assertEqual(audit['logical_calls'], 8)
            self.assertEqual(audit['generation_usage']['total_tokens'], sum(
                r['candidate_results'][2]['token_cost'] + r['independent']['token_cost'] for r in rows))
            # R2 sees only R1's visible answer, and independent sees only the direct prompt.
            self.assertIn('Current answer:\nFinal answer: 1', calls[-2][1])
            self.assertNotIn('Current answer:', calls[-1][1])
            self.assertEqual(full.run_stage(root, 'base', self.items, self.tokenizers), base)
            self.assertEqual(full.run_stage(root, 'qwen14', self.items, self.tokenizers, lookup), rows)
            self.assertEqual(len(calls), 8)
            rows[0]['candidate_results'][1]['token_cost'] += 1
            with self.assertRaises(ValueError):
                full.validate_row(root, rows[0], self.items[0], 'qwen14', self.tokenizers, lookup['correct'])

    def test_exhausted_length_is_pending_never_scored_or_exported(self):
        calls, items = [], self.items[:1]
        with tempfile.TemporaryDirectory() as folder, patch.object(full.pilot, 'server', self.fake_server(calls, True)), patch.object(full, 'wait_for_resources'):
            root = Path(folder)
            (root / 'protocol.json').write_text('{}')
            (root / 'policy_lock.json').write_text('{}')
            base = full.run_stage(root, 'base', items, self.tokenizers)
            rows = full.run_stage(root, 'qwen14', items, self.tokenizers, {})
            self.assertEqual(base, [])
            self.assertEqual(rows, [])
            self.assertEqual([c[2]['max_tokens'] for c in calls], [4096, 8192, 16384])
            self.assertFalse(full.finalize(root, items, self.tokenizers, base, rows))
            audit = json.loads((root / 'completion_audit.json').read_text())
            self.assertEqual(audit['paid_attempts'], 3)
            self.assertEqual(audit['pending_ids'], ['correct'])
            self.assertFalse((root / 'policy_test.jsonl').exists())
            full.run_stage(root, 'base', items, self.tokenizers)
            self.assertEqual(len(calls), 3)
            with self.assertRaises(ValueError):
                assert_complete_trace(root, {}, audit)

    def test_paid_interrupted_row_calls_are_reused(self):
        calls = []
        with tempfile.TemporaryDirectory() as folder, patch.object(full.pilot, 'server', self.fake_server(calls)), patch.object(full, 'wait_for_resources'):
            root = Path(folder)
            full.run_stage(root, 'base', self.items[:1], self.tokenizers)
            # Simulate a crash after durable call logging but before committing the row.
            (root / 'direct_test.jsonl').unlink()
            with patch.object(full.pilot, 'server', side_effect=AssertionError('Must use saved call')):
                rows = full.run_stage(root, 'base', self.items[:1], self.tokenizers)
            self.assertEqual(len(rows), 1)
            self.assertEqual(len(calls), 1)

    def test_locked_copy_refuses_different_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, target = root / 'source', root / 'target'
            source.write_bytes(b'original')
            first = full.locked_copy(source, target)
            self.assertEqual(first, full.locked_copy(source, target))
            source.write_bytes(b'changed')
            with self.assertRaises(ValueError):
                full.locked_copy(source, target)
            self.assertEqual(target.read_bytes(), b'original')

    def test_threshold_is_applied_without_budget_enforcement(self):
        bundle = {'direct_threshold': .8, 'action_labels': ['direct', 'reflect_1']}
        features = np.zeros((3, 2))
        scores = np.array([[.1, .9], [.2, .8], [.3, .7]])
        with patch('src.evaluate_locked_math_test.score_model', return_value=scores):
            actions, _ = frozen_actions(bundle, features, np.array(['reflect_2']*3), 'original_round', .8)
            self.assertEqual(actions.tolist(), ['reflect_2']*3)
            fixed, _ = frozen_actions(bundle, features, np.array(['reflect_2']*3), 'fixed_r1', .8)
            self.assertEqual(fixed.tolist(), ['reflect_1']*3)
            with self.assertRaises(ValueError):
                frozen_actions(bundle, features, actions, 'fixed_r1', .9)

    def test_threshold_tie_returns_direct(self):
        bundle = {'direct_threshold': .8, 'action_labels': ['direct', 'reflect_1']}
        with patch('src.evaluate_locked_math_test.score_model', return_value=np.array([[.8, .2]])):
            actions, _ = frozen_actions(bundle, np.zeros((1, 2)), np.array(['reflect_2']), 'original_round', .8)
            self.assertEqual(actions.tolist(), ['direct'])

    def test_offline_evaluation_reports_all_frozen_points_without_retuning(self):
        rows = []
        for i, item in enumerate(self.items):
            cs = [{'action': a, 'correct': int(i == 0 or j > 0), 'prompt_tokens': 10*(j+1),
                   'completion_tokens': 5*(j+1), 'token_cost': 15*(j+1)}
                  for j, a in enumerate(('direct', 'reflect_1', 'reflect_2'))]
            rows.append({**item, 'candidate_coverage': 'complete', 'candidate_results': cs,
                         'independent': {'correct': 1, 'prompt_tokens': 12, 'completion_tokens': 8, 'token_cost': 20}})
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            model = root / 'gate.pkl'
            model.write_bytes(b'trusted fake gate')
            round_path = root / 'round.pkl'
            round_path.write_bytes(b'trusted fake round')
            encoder_manifest = root / 'encoder_manifest.json'
            encoder_manifest.write_text(json.dumps({'encoder_files_sha256': {}, 'encoder_name': str(root)}))
            policies = [{'name': f'{seed}_{recipe}_{branch}_{budget}', 'seed': seed, 'recipe': recipe,
                         'branch': branch, 'validation_call_budget': budget, 'model': str(model),
                         'sha256': offline.digest(model), 'epoch': 1, 'direct_threshold': .5}
                        for seed, recipe, branch, budget in itertools.product((7, 21, 42), ('ce', 'ranking'),
                                ('original_round', 'fixed_r1'), (.05, .1, .2))]
            lock = {'policies': policies, 'round_model': str(round_path), 'round_sha256': offline.digest(round_path),
                    'encoder_manifest': str(encoder_manifest), 'encoder_manifest_sha256': offline.digest(encoder_manifest),
                    'encoder': {'encoder_name': str(root), 'max_length': 2048}, 'random_repetitions': 20,
                    'bootstrap_iterations': 20, 'historical_limitation': 'test fixture', 'training_coverage': {}}
            (root / 'policy_lock.json').write_text(json.dumps(lock))
            (root / 'protocol.json').write_text(json.dumps({'source_code_sha256': {}}))
            (root / 'completion_audit.json').write_text(json.dumps({'test_trace_sha256': 'frozen-trace'}))
            gate = {'action_labels': ['direct', 'reflect_1'], 'direct_threshold': .5}
            round_bundle = {'action_labels': ['reflect_1', 'reflect_2']}
            def load(path):
                return round_bundle if str(path) == str(round_path) else gate
            with patch.object(offline, 'assert_complete_trace', return_value=rows), \
                    patch.object(offline, 'extract_features', return_value=(np.zeros((2, 2)), {'encoder_clipped_count': 0})) as encode, \
                    patch.object(offline, 'load_pickle', side_effect=load), \
                    patch.object(offline, 'score_model', return_value=np.array([[.9, .1], [.1, .9]])):
                offline.evaluate(root)
                offline.evaluate(root)
                encode.assert_called_once()
            result = json.loads((root / 'evaluation/summary.json').read_text())
            self.assertEqual(len(result['policies']), 36)
            self.assertEqual(len(result['groups']), 12)
            self.assertEqual(len(result['mcnemar_vs_direct']), 39)
            self.assertTrue(result['no_test_calibration'])
            self.assertTrue(all(p['metrics']['reflection_rate'] == .5 for p in result['policies']))
            self.assertTrue(all(p['test_rate_exceeds_validation_budget'] for p in result['policies']))
            self.assertEqual(json.loads((root / 'status.json').read_text())['stage'], 'completed')


if __name__ == '__main__':
    unittest.main()
