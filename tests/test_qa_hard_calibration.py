import unittest

import numpy as np

from src.calibrate_qa_hard_development import cached_scores, route, scan, summarize


class HardCalibrationTests(unittest.TestCase):
    def test_source_rejection_can_choose_other_candidate(self):
        selected, invoked = route([0.8], [[0.2, 0.9, 0.8]], [0.5, 0.9, 0.5])
        self.assertEqual(selected.tolist(), [2])
        self.assertEqual(invoked.tolist(), [True])

    def test_invocation_without_adoption_still_charges_both(self):
        selected, invoked = route([0.8], [[0.2, 0.3, 0.4]], [0.5, 2, 2])
        usage = np.array([[[6, 4, 10], [12, 8, 20], [14, 11, 25]]])
        data = (np.ones((1, 3), dtype=int), np.ones((1, 3)), np.ones((1, 3)),
                np.ones((1, 3)), usage, np.array([[20, 15, 35]]))
        row = summarize("test", selected, invoked, data)
        self.assertEqual(row["avg_total_tokens"], 35)
        self.assertEqual(row["adoption_rate"], 0)
        self.assertEqual(row["invocation_rate"], 1)

    def test_probability_errors_rejected(self):
        for gate, scores in [([float("nan")], [[0, 0, 0]]),
                             ([0.2], [[0.4, 0.5, 0.7]]),
                             ([1.1], [[-0.1, 0.5, 0.7]])]:
            with self.assertRaises(ValueError):
                route(gate, scores, [0.5, 0, 0])

    def test_no_positive_fix_falls_back_to_direct(self):
        n = 20
        gate = np.linspace(0.05, 0.95, n)
        scores = np.column_stack([1 - gate, np.full(n, 0.8), np.full(n, 0.7)])
        correct = np.ones((n, 3), dtype=int)
        result = scan(gate, scores, correct, correct, np.ones(n) * 10, np.ones(n) * 30)
        self.assertTrue(result["fallback_to_direct"])
        self.assertEqual(result["invocation_count"], 0)

    def test_positive_fix_respects_integer_budget(self):
        n = 20
        gate = np.linspace(0.05, 0.95, n)
        scores = np.column_stack([1 - gate, np.full(n, 0.99), np.full(n, 0.1)])
        correct = np.ones((n, 3), dtype=int)
        correct[-1, 0] = 0
        result = scan(gate, scores, correct, correct.astype(float), np.ones(n) * 10, np.ones(n) * 30)
        self.assertFalse(result["fallback_to_direct"])
        self.assertEqual(result["net_fixes"], 1)
        self.assertLessEqual(result["invocation_count"], 2)
        self.assertEqual(result["invocation_count"], 1)

    def test_probability_id_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            cached_scores([{"id": "a"}], [{"id": "b"}])
        with self.assertRaises(ValueError):
            cached_scores([{"id": "a"}], [{"id": "a"}, {"id": "a"}])

    def test_cost_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            scan([0.5], [[0.5, 0.7, 0.6]], [[0, 1, 0]], [[0, 1, 0]], [10], [5])


if __name__ == "__main__":
    unittest.main()
