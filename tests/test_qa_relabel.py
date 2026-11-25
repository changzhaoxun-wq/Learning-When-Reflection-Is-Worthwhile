import unittest

from src.answer_extractor import (
    normalize_qa_equivalence_answer,
    qa_equivalence_answers_equal,
)
from src.prepare_hierarchical_policy_dataset import build as build_hierarchical
from src.relabel_qa_policy_dataset import (
    _context_aliases,
    relabel_rows,
    suspect_gold_reasons,
)
from src.reward import compute_match_metrics
from src.evaluate_policy_offline import _candidate_reward, _candidate_utility
from src.transformer_policy import _reward_for_action


class QAEquivalenceTests(unittest.TestCase):
    def test_safe_numeric_unit_and_initialism_equivalence(self):
        self.assertEqual(normalize_qa_equivalence_answer("six weeks"), "6 week")
        self.assertTrue(qa_equivalence_answers_equal("16 years old", "16-year-old"))
        self.assertTrue(qa_equivalence_answers_equal("U.S. 60", "US 60"))
        self.assertTrue(qa_equivalence_answers_equal("2", "twice"))

    def test_standard_and_enhanced_metrics_are_both_preserved(self):
        metrics = compute_match_metrics(
            "16 years old",
            "16-year-old",
            task_type="qa",
        )
        self.assertLess(metrics["answer_f1"], 1.0)
        self.assertEqual(metrics["enhanced_exact_match"], 1.0)
        self.assertEqual(metrics["enhanced_answer_f1"], 1.0)


class QAOfflineRelabelTests(unittest.TestCase):
    def _row(self):
        return {
            "id": "hotpotqa-train-x",
            "task_type": "qa",
            "source": "bridge",
            "gold_answer": "Barack Hussein Obama II",
            "gold_aliases": [],
            "candidate_results": [
                {"action": "direct", "pred": "Barack Obama", "token_cost": 100},
                {"action": "reflect_1", "pred": "John Smith", "token_cost": 300},
                {"action": "reflect_2", "pred": "Jane Smith", "token_cost": 500},
            ],
        }

    def test_context_aliases_only_accept_parenthetical_disambiguation(self):
        item = {
            "context": {
                "title": [
                    "List of United States Senators from Nevada",
                    "Jonathan Stark (tennis)",
                ],
                "sentences": [[], []],
            }
        }
        self.assertEqual(_context_aliases(item, "Nevada"), [])
        self.assertEqual(
            _context_aliases(item, "Jonathan Stark"),
            ["Jonathan Stark (tennis)"],
        )

    def test_context_alias_changes_utility_without_overwriting_standard_metric(self):
        clean, quarantine, summary = relabel_rows(
            [self._row()],
            {"hotpotqa-train-x": ["Barack Obama"]},
            qa_f1_threshold=0.8,
            lambda_cost=0.0001,
        )
        self.assertFalse(quarantine)
        self.assertEqual(summary["label_distribution"], {"direct": 1})
        direct = clean[0]["candidate_results"][0]
        self.assertEqual(direct["correct"], 0)
        self.assertEqual(direct["alias_exact_match"], 1.0)
        self.assertEqual(direct["enhanced_answer_f1"], 1.0)
        self.assertAlmostEqual(direct["reward"], 0.99)
        self.assertEqual(clean[0]["derived_gold_aliases"], ["Barack Obama"])

    def test_high_confidence_long_comparison_gold_is_quarantined(self):
        row = self._row()
        row["source"] = "comparison"
        row["gold_answer"] = "one two three four five six seven eight nine"
        self.assertIn("comparison_gold_exceeds_8_words", suspect_gold_reasons(row))
        clean, quarantine, _ = relabel_rows(
            [row], {}, qa_f1_threshold=0.8, lambda_cost=0.0001
        )
        self.assertFalse(clean)
        self.assertEqual(len(quarantine), 1)

    def test_offline_evaluation_recomputes_reward_from_continuous_utility(self):
        candidate = {
            "correct": 0,
            "qa_utility_score": 0.8,
            "token_cost": 100,
            "reward": 0.79,
        }
        self.assertEqual(_candidate_utility(candidate), 0.8)
        self.assertAlmostEqual(_candidate_reward(candidate, 0.0002), 0.78)

    def test_transformer_training_metrics_use_stored_continuous_reward(self):
        row = self._row()
        row["candidate_results"][0].update(correct=0, reward=0.73)
        self.assertEqual(_reward_for_action(row, "direct", 0.0001), 0.73)

    def test_hierarchical_builder_can_use_stored_continuous_reward(self):
        row = self._row()
        for candidate, reward in zip(row["candidate_results"], (0.1, 0.6, 0.4)):
            candidate["correct"] = 0
            candidate["reward"] = reward
        gate, rounds = build_hierarchical(
            [row], objective="stored_reward", lambda_cost=0.0001
        )
        self.assertEqual(gate[0]["best_action"], "reflect_1")
        self.assertEqual(rounds[0]["best_action"], "reflect_1")


if __name__ == "__main__":
    unittest.main()
