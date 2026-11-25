import itertools
import unittest

import numpy as np

from src.analyze_gate_random_baseline import (holm_adjust, paired_interval, random_inclusion,
                                             random_orders, sampled_metrics, to_metrics)
from src.analyze_decoupling_significance import _mcnemar


class RandomBaselineTests(unittest.TestCase):
    def setUp(self):
        self.rounds = np.array([1, 1, 2, 2])
        self.selected = np.array([1, 0, 2, 0])
        self.base = np.array([[1, 10, 20, 30, 0, 0, 0], [0, 20, 30, 50, 0, 0, 0],
                              [1, 5, 10, 15, 0, 0, 0], [0, 7, 8, 15, 0, 0, 0]], dtype=float)
        self.ref = np.array([[0, 20, 40, 60, 1, 0, 1], [1, 40, 50, 90, 1, 1, 0],
                             [1, 20, 25, 45, 1, 0, 0], [1, 30, 35, 65, 1, 1, 0]], dtype=float)
        self.orders = {'all': np.array(list(itertools.permutations(range(4))))}

    def test_exact_uniform_expectation_matches_all_permutations(self):
        expected, draws = sampled_metrics(self.base, self.ref, self.selected, self.rounds,
                                          'uniform_gate', self.orders)
        np.testing.assert_allclose(draws.mean(0), list(to_metrics(expected).values()))
        np.testing.assert_allclose(draws[:, 4], .5)
        np.testing.assert_allclose(draws[:, 1] + draws[:, 2], draws[:, 3])

    def test_exact_stratified_counts_and_expectation(self):
        first, second = zip(*itertools.product([(0, 1), (1, 0)], [(2, 3), (3, 2)]))
        orders = {1: np.array(first), 2: np.array(second)}
        expected, draws = sampled_metrics(self.base, self.ref, self.selected, self.rounds,
                                          'round_stratified_gate', orders)
        np.testing.assert_allclose(draws.mean(0), list(to_metrics(expected).values()))
        np.testing.assert_allclose(draws[:, 4], .5)

    def test_different_stratum_counts(self):
        selection = np.array([1, 1, 0, 0])
        np.testing.assert_array_equal(random_inclusion(self.rounds, selection, 'round_stratified_gate'), [1, 1, 0, 0])
        np.testing.assert_array_equal(random_inclusion(self.rounds, selection, 'uniform_gate'), [.5] * 4)

    def test_zero_and_full_calls(self):
        for selected, expected in ((np.zeros(4, dtype=int), self.base), (self.rounds, self.ref)):
            rows, draws = sampled_metrics(self.base, self.ref, selected, self.rounds, 'uniform_gate', self.orders)
            np.testing.assert_array_equal(rows, expected)
            np.testing.assert_allclose(draws, np.tile(list(to_metrics(expected).values()), (24, 1)))

    def test_fixed_r1_empty_second_stratum(self):
        rounds = np.ones(4, dtype=int)
        selected = np.array([1, 0, 1, 0])
        orders = {**self.orders, 1: self.orders['all'], 2: np.empty((24, 0), dtype=int)}
        a = sampled_metrics(self.base, self.ref, selected, rounds, 'uniform_gate', orders)
        b = sampled_metrics(self.base, self.ref, selected, rounds, 'round_stratified_gate', orders)
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x, y)

    def test_invalid_action_and_mismatched_round_rejected(self):
        for selection in (np.array([2, 0, 2, 0]), np.array([3, 0, 0, 0])):
            with self.assertRaises(ValueError):
                random_inclusion(self.rounds, selection, 'uniform_gate')
        with self.assertRaises(ValueError):
            random_inclusion(self.rounds, self.selected, 'unknown')

    def test_random_orders_reproducible_without_replacement(self):
        a = random_orders([1, 3, 5], 20, np.random.default_rng(4))
        b = random_orders([1, 3, 5], 20, np.random.default_rng(4))
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(np.sort(a, axis=1), np.tile([1, 3, 5], (20, 1)))

    def test_holm_preserves_order_and_monotonic_adjustment(self):
        np.testing.assert_allclose(holm_adjust([.04, .01, .03]), [.06, .03, .06])
        np.testing.assert_allclose(holm_adjust([.8, 1]), [1, 1])
        for values in ([-1], [float('nan')]):
            with self.assertRaises(ValueError):
                holm_adjust(values)

    def test_bootstrap_zero_and_constant_difference(self):
        for value in (0., .25):
            result = paired_interval(np.full(10, value), iterations=50, seed=3)
            self.assertEqual(set(result), {'observed_difference', 'ci95_lower', 'ci95_upper'})
            self.assertTrue(all(v == value for v in result.values()))

    def test_paired_mcnemar_uses_discordant_pairs(self):
        result = _mcnemar(np.ones(8), np.zeros(8))
        self.assertEqual(result['exact_two_sided_p'], .0078125)
        self.assertEqual(result['learned_correct_baseline_wrong'], 8)
        self.assertEqual(_mcnemar(np.ones(8), np.ones(8))['exact_two_sided_p'], 1.)


if __name__ == '__main__':
    unittest.main()
