import unittest
from unittest.mock import patch
import numpy as np

from src.evaluate_math_transfer import cluster_interval, evaluate_subset
from src.run_locked_math_transfer import base_row, reflection_row


class EvaluationTest(unittest.TestCase):
    def test_cluster_interval_duplicate_rows(self):
        result = cluster_interval(np.array([1., 1., -1.]), ['SAME', 'same', 'other'], iterations=1000)
        self.assertEqual(result['clusters'], 2)
        self.assertAlmostEqual(result['difference'], 1/3)
        self.assertEqual(result['ci95'], [-1., 1.])

    def test_zero_difference(self):
        self.assertEqual(cluster_interval(np.zeros(3), ['a', 'b', 'c'], 100)['ci95'], [0., 0.])

    def test_missing_cost_annotation_and_duplicate_test_omission(self):
        def call(*args):
            return {'answer_text': 'Final answer: 2', 'prompt_tokens': 2, 'completion_tokens': 3, 'total_tokens': 5}
        rows = [reflection_row(base_row({'id': str(i), 'question': 'Same', 'gold_answer': '2'}, call), call) for i in range(2)]
        points = [{'name': str(seed), 'seed': seed, 'recipe': 'test', 'branch': 'fixed_r1', 'validation_call_budget': .1} for seed in (7, 21, 42)]
        decisions = {p['name']: {'actions': ['direct', 'reflect_1']} for p in points}
        with patch('src.evaluate_math_transfer.cluster_interval', return_value={'ci95': [0, 0]}):
            result = evaluate_subset(rows, [0, 1], decisions, {'policies': points}, [{'id': '0', 'action': 'independent'}])
            self.assertFalse(result['baselines']['independent']['token_cost_complete'])
            self.assertTrue(result['baselines']['direct']['token_cost_complete'])
            self.assertEqual(result['mcnemar_vs_direct'], [])
            subset = evaluate_subset(rows, [1], decisions, {'policies': points}, [{'id': '0', 'action': 'independent'}])
            self.assertTrue(subset['baselines']['independent']['token_cost_complete'])
            self.assertEqual(len(subset['mcnemar_vs_direct']), 6)


if __name__ == '__main__':
    unittest.main()
