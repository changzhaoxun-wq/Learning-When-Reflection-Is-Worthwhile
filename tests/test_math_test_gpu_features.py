import unittest
from unittest.mock import patch

import numpy as np
import torch

from src import math_test_gpu_features as gpu
from src.review_math_gpu_runtime import numerical_review


class GPUFeatureTests(unittest.TestCase):
    def test_pooling_uses_last_two_layers_and_ignores_padding(self):
        mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
        a = torch.tensor([[[1., 2.], [3., 4.], [999., 999.]], [[5., 6.], [7., 8.], [9., 10.]]])
        result = gpu.pool_hidden([a*0, a, a*2], mask)
        expected = torch.tensor([[2, 3, 3, 4, 4, 6, 6, 8], [7, 8, 9, 10, 14, 16, 18, 20.]])
        torch.testing.assert_close(result, expected)

    def test_comparison_checks_actions_as_well_as_small_numeric_error(self):
        lock = {'round_model': 'round', 'policies': [{'name': 'gate', 'model': 'gate',
                'branch': 'fixed_r1', 'direct_threshold': .5}]}
        bundle = {'action_labels': ['reflect_1', 'reflect_2']}
        cpu = np.ones((2, 4), dtype='float32')
        changed = cpu + .00001
        actions = (np.array(['direct', 'reflect_1']), np.array([.8, .2]))
        with patch.object(gpu, 'load_pickle', return_value=bundle), \
                patch.object(gpu.offline, 'score_model', return_value=np.array([[.8, .2], [.8, .2]])), \
                patch.object(gpu.offline, 'frozen_actions', return_value=actions):
            result = gpu.compare_vectors(cpu, changed, lock)
            self.assertTrue(result['passed'])
            self.assertEqual(result['round_action_mismatches'], 0)
            with patch.object(gpu.offline, 'frozen_actions', side_effect=[actions,
                    (np.array(['reflect_1', 'reflect_1']), np.array([.4999, .2]))]):
                result = gpu.compare_vectors(cpu, changed, lock)
                self.assertFalse(result['passed'])
                self.assertEqual(result['policies'][0]['action_mismatches'], 1)
            self.assertFalse(gpu.compare_vectors(cpu, cpu*2, lock)['passed'])

    def test_invalid_vector_audits_fail_closed(self):
        with self.assertRaises(ValueError):
            gpu.compare_vectors(np.ones((2, 3)), np.ones((1, 3)), {})
        with self.assertRaises(ValueError):
            gpu.compare_vectors(np.ones((2, 3)), np.full((2, 3), np.nan), {})

    def test_label_free_review_keeps_failed_strict_audit_visible(self):
        cpu = np.array([[10., -.002]], dtype='float32')
        other = cpu.copy()
        other[0, 1] += .00005
        audit = {'passed': False, 'round_action_mismatches': 0, 'policies': [{'action_mismatches': 0}]}
        result = numerical_review(cpu, other, audit)
        self.assertTrue(result['accepted'])
        self.assertFalse(result['original_strict_audit_passed'])
        self.assertFalse(numerical_review(cpu, cpu+1, audit)['accepted'])
        audit['round_action_mismatches'] = 1
        self.assertFalse(numerical_review(cpu, other, audit)['accepted'])


if __name__ == '__main__':
    unittest.main()
