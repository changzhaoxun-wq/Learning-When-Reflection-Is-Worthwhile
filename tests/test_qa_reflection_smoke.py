import unittest

from src.analyze_qa_reflection_prompt import compare
from src.prepare_qa_reflection_smoke import prepare_smoke_rows


def _candidate(action, correct, pred, decision=None, token_cost=10):
    calls = [{"round_id": 0, "output": "Final answer: direct"}]
    if action != "direct":
        calls.append(
            {
                "round_id": 1 if action == "reflect_1" else 2,
                "output": (
                    "Evidence chain: compact evidence\n"
                    f"Decision: {decision}\n"
                    f"Final answer: {pred}"
                ),
            }
        )
    return {
        "action": action,
        "pred": pred,
        "correct": correct,
        "enhanced_correct": correct,
        "answer_f1": float(correct),
        "enhanced_answer_f1": float(correct),
        "token_cost": token_cost,
        "metadata": {"calls": calls},
    }


def _complete_row(row_id, direct_correct, r1_correct, r2_correct, version="old"):
    direct_pred = "direct"
    return {
        "id": row_id,
        "reflection_prompt_version": version,
        "candidate_results": [
            _candidate("direct", direct_correct, direct_pred),
            _candidate(
                "reflect_1",
                r1_correct,
                direct_pred if r1_correct == direct_correct else "r1",
                "KEEP" if r1_correct == direct_correct else "REVISE",
                20,
            ),
            _candidate(
                "reflect_2",
                r2_correct,
                direct_pred if r2_correct == direct_correct else "r2",
                "KEEP" if r2_correct == direct_correct else "REVISE",
                30,
            ),
        ],
    }


class QAReflectionSmokeTests(unittest.TestCase):
    def test_balanced_sample_freezes_only_direct_candidate(self):
        rows = [
            _complete_row("c1", 1, 1, 1),
            _complete_row("c2", 1, 1, 1),
            _complete_row("w1", 0, 1, 0),
            _complete_row("w2", 0, 0, 1),
            {
                "id": "direct-only",
                "candidate_results": [_candidate("direct", 1, "direct")],
            },
        ]
        sample, summary = prepare_smoke_rows(rows, per_group=1, seed=42)
        self.assertEqual(len(sample), 2)
        self.assertEqual(summary["complete_source_rows"], 4)
        self.assertEqual(
            {row["reflection_smoke"]["stratum"] for row in sample},
            {"direct_correct", "direct_incorrect"},
        )
        self.assertTrue(
            all(
                [candidate["action"] for candidate in row["candidate_results"]]
                == ["direct"]
                for row in sample
            )
        )

    def test_comparison_reports_lower_harm_and_more_repairs(self):
        old = [
            _complete_row("c1", 1, 0, 0, "old"),
            _complete_row("c2", 1, 1, 0, "old"),
            _complete_row("w1", 0, 1, 0, "old"),
            _complete_row("w2", 0, 0, 0, "old"),
        ]
        new = [
            _complete_row("c1", 1, 1, 1, "new"),
            _complete_row("c2", 1, 1, 0, "new"),
            _complete_row("w1", 0, 1, 1, "new"),
            _complete_row("w2", 0, 0, 0, "new"),
        ]
        report = compare(old, new)
        self.assertTrue(report["direct_frozen"])
        self.assertEqual(report["old"]["actions"]["reflect_1"]["harmed_n"], 1)
        self.assertEqual(report["new"]["actions"]["reflect_1"]["harmed_n"], 0)
        self.assertEqual(report["old"]["candidate_ceiling"]["accuracy"], 0.75)
        self.assertEqual(report["new"]["candidate_ceiling"]["accuracy"], 0.75)
        self.assertEqual(report["new"]["actions"]["reflect_2"]["repaired_n"], 1)


if __name__ == "__main__":
    unittest.main()
