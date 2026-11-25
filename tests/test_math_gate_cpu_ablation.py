import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from src.math_gate_cpu_ablation import (RECIPES, build_model, mild_weights,
                                        probability_thresholds, read_cached_features,
                                        score_model, select_budget_point)
from src.neural_policy import build_policy_text
from src.train_fixed_math_policy import digest
from sklearn.preprocessing import StandardScaler


class GateAblationTests(unittest.TestCase):
    def test_ties_keep_budget_without_breaking_by_label(self):
        p = np.array([.1, .1, .2, .3], dtype='float32')
        ts = probability_thresholds(p, [.25])
        self.assertEqual(ts, [0.])
        ts = probability_thresholds(p, [.5])
        self.assertEqual(int((p < ts[-1]).sum()), 2)

    def test_endpoints_match_float32_inference(self):
        p = np.array([0., .99999994, 1.], dtype='float32')
        ts = probability_thresholds(p, [0, 1])
        self.assertEqual(int((p < ts[0]).sum()), 0)
        self.assertEqual(int((p < ts[-1]).sum()), 3)

    def test_invalid_probability_rejected(self):
        for p in ([np.nan], [-1.], [2.]):
            with self.assertRaises(ValueError):
                probability_thresholds(p)

    def test_accuracy_priority_only_within_budget(self):
        def point(correct, cost, rate):
            return {'direct_threshold': .5, 'metrics': {'correct': correct, 'avg_token_cost': cost,
                                                        'reflection_rate': rate}}
        points = [point(99, 500, 1.), point(97, 700, .1), point(96, 450, .1), point(96, 400, .05)]
        chosen = select_budget_point(points, .1, 641)
        self.assertEqual(chosen['metrics']['avg_token_cost'], 400)
        self.assertIsNone(select_budget_point(points, .01, 641))

    def test_mild_class_weights_are_bounded(self):
        y = np.array([0] * 100 + [1])
        np.testing.assert_array_equal(mild_weights(y, True).numpy(), [1., 5.])
        np.testing.assert_array_equal(mild_weights(y, False).numpy(), [1., 1.])
        with self.assertRaises(ValueError):
            mild_weights(np.zeros(5, dtype=int), True)

    def test_smaller_model_and_cpu_inference(self):
        x = np.random.default_rng(0).normal(size=(4, 4096)).astype('float32')
        for recipe, size in zip(RECIPES, [65586, 65586, 8194]):
            model = build_model(recipe)
            self.assertEqual(sum(p.numel() for p in model.parameters()), size)
            b = {'model': model, 'scaler': StandardScaler().fit(x[:2]), 'action_labels': ['direct', 'reflect_1']}
            p = score_model(b, x[2:])
            self.assertEqual(p.shape, (2, 2))
            np.testing.assert_allclose(p.sum(1), 1., atol=1e-6)
            self.assertFalse(model.training)
            self.assertEqual(next(model.parameters()).device.type, 'cpu')

    def test_cache_input_order_and_text_are_verified(self):
        rows = [{'id': 'a', 'question': 'one?', 'initial_answer': '1'},
                {'id': 'b', 'question': 'two?', 'initial_answer': '2'}]
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)
            manifest = {'ids': ['a', 'b'], 'text_sha256': [hashlib.sha256(build_policy_text(r, 0, False).encode()).hexdigest() for r in rows],
                        'layers': [-2, -1], 'pooling': 'mean_last',
                        'neural_policy_sha256': digest(Path('src/neural_policy.py'))}
            fingerprint = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
            (path / 'manifest.json').write_text(json.dumps(manifest))
            (path / 'complete.json').write_text(json.dumps({'fingerprint': fingerprint}))
            np.save(path / '000000_000002.npy', np.zeros((2, 4096), dtype='float32'))
            x, _ = read_cached_features(path, rows, fingerprint)
            self.assertEqual(x.shape, (2, 4096))
            with self.assertRaisesRegex(ValueError, 'order'):
                read_cached_features(path, list(reversed(rows)), fingerprint)
            rows[0]['initial_answer'] = 'Changed'
            with self.assertRaisesRegex(ValueError, 'text differs'):
                read_cached_features(path, rows, fingerprint)


if __name__ == '__main__':
    unittest.main()
