import unittest

import numpy as np

from src.qa_dual_candidate_policy import (
    align_dual_rows,
    build_candidate_text,
    build_dual_targets,
    build_usage_arrays,
    dual_selection_metrics,
    dual_usage,
    select_dual_thresholds,
)
from src.evaluate_qa_dual_candidate_offline import (
    _diagnostics_from_saved_probabilities,
)


def _candidate(action, correct, answer, prompt, completion):
    return {
        "action": action,
        "correct": correct,
        "answer_f1": float(correct),
        "enhanced_answer_f1": float(correct),
        "final_answer": f"Final answer: {answer}",
        "pred": answer,
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "token_cost": prompt + completion,
        "reflected_rounds": int(action != "direct"),
    }


def _row(row_id, direct_correct, reflect_correct=None, version="old"):
    candidates = [_candidate("direct", direct_correct, "initial", 10, 5)]
    if reflect_correct is not None:
        if version == "old":
            candidates.append(
                _candidate("reflect_1", reflect_correct, "old answer", 30, 10)
            )
        else:
            candidates.append(
                _candidate("reflect_1", reflect_correct, "new answer", 35, 12)
            )
    return {
        "id": row_id,
        "task_type": "qa",
        "question": (
            "Context:\nAlpha supports Beta.\n\nQuestion:\n"
            "What does Alpha support?"
        ),
        "initial_answer": "Final answer: initial",
        "gold_answer": "secret gold",
        "candidate_results": candidates,
    }


class QADualCandidatePolicyTests(unittest.TestCase):
    def test_alignment_and_union_repair_target(self):
        old = [
            _row("fix-old", 0, 1, "old"),
            _row("fix-new", 0, 0, "old"),
            _row("direct-only", 1, None, "old"),
        ]
        new = [
            _row("fix-old", 0, 0, "new"),
            _row("fix-new", 0, 1, "new"),
            _row("direct-only", 1, None, "new"),
        ]
        pairs = align_dual_rows(old, new)
        targets = build_dual_targets(pairs)
        self.assertEqual(targets["gate_target"].tolist(), [1, 1, 0])
        self.assertEqual(targets["direct_error_target"].tolist(), [1, 1, 0])
        self.assertEqual(targets["complete"].tolist(), [True, True, False])

    def test_candidate_text_excludes_gold_and_reward(self):
        old = _row("row", 0, 1, "old")
        new = _row("row", 0, 1, "new")
        new["candidate_results"][1]["reward"] = 1.0
        pair = align_dual_rows([old], [new], require_complete=True)[0]
        text = build_candidate_text(pair, 2)
        self.assertIn("Candidate source: v7_reflect_1", text)
        self.assertIn("What does Alpha support?", text)
        self.assertNotIn("secret gold", text)
        self.assertNotIn("reward", text.lower())

    def test_dual_usage_counts_base_once_and_both_reflections(self):
        pair = align_dual_rows(
            [_row("row", 0, 1, "old")],
            [_row("row", 0, 1, "new")],
            require_complete=True,
        )[0]
        self.assertEqual(dual_usage(pair), (55.0, 17.0, 72.0))

    def test_thresholds_choose_fix_and_reject_harm(self):
        old = [
            _row("fix", 0, 1, "old"),
            _row("harm", 1, 0, "old"),
            _row("easy", 1, 1, "old"),
        ]
        new = [
            _row("fix", 0, 0, "new"),
            _row("harm", 1, 0, "new"),
            _row("easy", 1, 1, "new"),
        ]
        pairs = align_dual_rows(old, new, require_complete=True)
        targets = build_dual_targets(pairs)
        usage = build_usage_arrays(pairs)
        gate = np.asarray([0.95, 0.85, 0.05], dtype="float32")
        candidate = np.asarray(
            [
                [0.10, 0.95, 0.20],
                [0.70, 0.75, 0.72],
                [0.90, 0.80, 0.85],
            ],
            dtype="float32",
        )
        gate_threshold, margin_threshold, metrics = select_dual_thresholds(
            gate,
            candidate,
            targets,
            usage,
            max_invocation_rate=1.0,
            grid_points=21,
        )
        self.assertGreater(gate_threshold, 0.85)
        self.assertLessEqual(gate_threshold, 0.95 + 1e-6)
        self.assertGreater(margin_threshold, 0.05)
        self.assertEqual(metrics["fixes"], 1)
        self.assertEqual(metrics["harms"], 0)
        self.assertEqual(metrics["accuracy"], 1.0)

    def test_invocation_cost_is_charged_when_candidate_is_rejected(self):
        pairs = align_dual_rows(
            [_row("row", 1, 0, "old")],
            [_row("row", 1, 0, "new")],
            require_complete=True,
        )
        targets = build_dual_targets(pairs)
        usage = build_usage_arrays(pairs)
        metrics = dual_selection_metrics(
            np.asarray([0.9], dtype="float32"),
            np.asarray([[0.9, 0.1, 0.2]], dtype="float32"),
            targets,
            usage,
            gate_threshold=0.5,
            margin_threshold=0.0,
        )
        self.assertEqual(metrics["candidate_adoption_rate"], 0.0)
        self.assertEqual(metrics["gate_invocation_rate"], 1.0)
        self.assertEqual(metrics["avg_token_cost"], 72.0)

    def test_saved_probabilities_reapply_new_thresholds(self):
        old = [_row("row", 0, 1, "old")]
        new = [_row("row", 0, 0, "new")]
        decisions = [
            {
                "id": "row",
                "gate_probability": 0.9,
                "candidate_probabilities": {
                    "direct": 0.1,
                    "v5_reflect_1": 0.8,
                    "v7_reflect_1": 0.2,
                },
            }
        ]
        bundle = {
            "gate_threshold": 0.5,
            "margin_threshold": 0.3,
            "selection_score_mode": "correctness_margin",
        }
        _, diagnostics = _diagnostics_from_saved_probabilities(
            bundle, old, new, decisions
        )
        self.assertEqual(diagnostics["invoked"].tolist(), [True])
        self.assertEqual(diagnostics["adopted"].tolist(), [True])
        self.assertEqual(diagnostics["selected_source"].tolist(), [1])


if __name__ == "__main__":
    unittest.main()
