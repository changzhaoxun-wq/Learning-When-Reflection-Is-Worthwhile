import unittest

import numpy as np

from src.selective_advantage_policy import _gate_threshold, build_targets


def _candidate(action, correct, utility, tokens):
    return {
        "action": action,
        "correct": correct,
        "qa_utility_score": utility,
        "token_cost": tokens,
    }


class SelectiveAdvantagePolicyTests(unittest.TestCase):
    def test_selective_targets_mask_missing_counterfactuals(self):
        rows = [
            {
                "id": "correct-direct-only",
                "candidate_results": [_candidate("direct", 1, 1.0, 100)],
            },
            {
                "id": "repairable-error",
                "candidate_results": [
                    _candidate("direct", 0, 0.0, 100),
                    _candidate("reflect_1", 1, 1.0, 200),
                    _candidate("reflect_2", 1, 1.0, 300),
                ],
            },
        ]
        targets = build_targets(rows, lambda_cost=0.001)
        self.assertEqual(targets["complete"].tolist(), [False, True])
        self.assertEqual(targets["direct_error"].tolist(), [0.0, 1.0])
        self.assertEqual(targets["gate_target"].tolist(), [0, 1])
        self.assertTrue(np.isclose(targets["advantage"][1], 0.9))
        self.assertTrue(np.isclose(targets["round_delta"][1], -0.1))

    def test_gate_threshold_uses_supplied_oof_values(self):
        values = np.asarray([0.05, 0.10, 0.80, 0.90])
        threshold, metrics = _gate_threshold(
            values, np.asarray([0, 0, 1, 1]), beta=1.5
        )
        self.assertEqual((values >= threshold).tolist(), [False, False, True, True])
        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 1.0)


if __name__ == "__main__":
    unittest.main()
