import unittest

import numpy as np

from src.accuracy_first_qa_policy import (
    build_acceptance_text,
    build_accuracy_targets,
    select_accuracy_threshold,
    select_joint_accuracy_thresholds,
)


def _candidate(action, correct, answer):
    return {
        "action": action,
        "correct": correct,
        "final_answer": f"Final answer: {answer}",
        "pred": answer,
    }


def _row(row_id, direct_correct, reflect_correct=None):
    candidates = [_candidate("direct", direct_correct, "direct span")]
    if reflect_correct is not None:
        candidates.append(
            _candidate("reflect_1", reflect_correct, "reflected span")
        )
    return {
        "id": row_id,
        "task_type": "qa",
        "question": (
            "Context:\nAlpha supports Beta.\n\nQuestion:\n"
            "What does Alpha support?"
        ),
        "initial_answer": "Final answer: direct span",
        "gold_answer": "secret gold",
        "candidate_results": candidates,
    }


class AccuracyFirstQAPolicyTests(unittest.TestCase):
    def test_targets_only_mark_successful_error_repair(self):
        rows = [
            _row("harm", 1, 0),
            _row("fix", 0, 1),
            _row("both_wrong", 0, 0),
            _row("direct_only", 1, None),
        ]
        targets = build_accuracy_targets(rows)
        self.assertEqual(targets["target"].tolist(), [0, 1, 0, 0])
        self.assertEqual(
            targets["complete"].tolist(), [True, True, True, False]
        )

    def test_threshold_maximizes_selected_answer_accuracy(self):
        probability = np.asarray([0.95, 0.80, 0.10], dtype="float32")
        direct_correct = np.asarray([0, 1, 1], dtype="int64")
        reflect_correct = np.asarray([1, 0, 1], dtype="int64")
        mask = np.ones(3, dtype=bool)
        threshold, metrics = select_accuracy_threshold(
            probability,
            direct_correct,
            reflect_correct,
            mask,
            max_reflection_rate=1.0,
        )
        self.assertGreater(threshold, 0.80)
        self.assertLessEqual(threshold, 0.95 + 1e-6)
        self.assertEqual(metrics["fixes"], 1)
        self.assertEqual(metrics["harms"], 0)
        self.assertEqual(metrics["accuracy"], 1.0)

    def test_acceptance_threshold_can_reject_gate_harm(self):
        probability = np.asarray([0.90, 0.20], dtype="float32")
        direct_correct = np.asarray([0, 1], dtype="int64")
        reflect_correct = np.asarray([1, 0], dtype="int64")
        mask = np.ones(2, dtype=bool)
        active = np.asarray([True, True])
        threshold, metrics = select_accuracy_threshold(
            probability,
            direct_correct,
            reflect_correct,
            mask,
            max_reflection_rate=1.0,
            active=active,
        )
        self.assertGreater(threshold, 0.20)
        self.assertEqual(metrics["net_fixes"], 1)
        self.assertEqual(metrics["accuracy"], 1.0)

    def test_joint_thresholds_can_use_acceptance_to_remove_gate_harm(self):
        gate_probability = np.asarray(
            [0.90, 0.85, 0.10], dtype="float32"
        )
        acceptance_probability = np.asarray(
            [0.95, 0.05, 0.20], dtype="float32"
        )
        direct_correct = np.asarray([0, 1, 1], dtype="int64")
        reflect_correct = np.asarray([1, 0, 1], dtype="int64")
        mask = np.ones(3, dtype=bool)
        gate_threshold, acceptance_threshold, metrics = (
            select_joint_accuracy_thresholds(
                gate_probability,
                acceptance_probability,
                direct_correct,
                reflect_correct,
                mask,
                max_reflection_rate=1.0,
            )
        )
        self.assertLessEqual(gate_threshold, 0.90 + 1e-6)
        self.assertGreater(acceptance_threshold, 0.05)
        self.assertEqual(metrics["fixes"], 1)
        self.assertEqual(metrics["harms"], 0)
        self.assertEqual(metrics["accuracy"], 1.0)


    def test_acceptance_text_does_not_expose_gold_or_reward(self):
        row = _row("row", 0, 1)
        row["candidate_results"][1]["reward"] = 1.0
        text = build_acceptance_text(row)
        self.assertIn("Direct response:", text)
        self.assertIn("Reflected response:", text)
        self.assertIn("What does Alpha support?", text)
        self.assertNotIn("secret gold", text)
        self.assertNotIn("reward", text.lower())


if __name__ == "__main__":
    unittest.main()

