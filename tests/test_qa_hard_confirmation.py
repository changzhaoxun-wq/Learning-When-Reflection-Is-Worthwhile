import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.qa_hard_confirmation import confirm, confirmation_config, paired_intervals, validate_split


class HardConfirmationTests(unittest.TestCase):
    def test_config_only_changes_dataset_and_role(self):
        cfg = {"data": {"data_path": "dev", "eval_data_path": "dev", "train_limit": None},
               "experiment_protocol": {"train_data": "dev", "eval_data": "dev", "prompt_sha": "same"},
               "base_llm": {"max_tokens": 1024, "enable_thinking": False},
               "qa_reflection_profile": {"round_1": "fixed"}}
        original = copy.deepcopy(cfg)
        new = confirmation_config(cfg, Path("confirmation.jsonl"))
        self.assertEqual(cfg, original)
        self.assertEqual(new["base_llm"], cfg["base_llm"])
        self.assertEqual(new["qa_reflection_profile"], cfg["qa_reflection_profile"])
        self.assertIsNone(new["data"]["train_limit"])
        self.assertEqual(new["experiment_protocol"]["prompt_sha"], "same")
        self.assertEqual(new["data"]["data_path"], "confirmation.jsonl")

    def test_split_requires_reserved_order_without_overlap(self):
        manifest = {"development_ids": ["d"], "confirmation_ids": ["a", "b"]}
        self.assertEqual(validate_split(manifest, [{"id": "a"}, {"id": "b"}]), ["a", "b"])
        for items in ([{"id": "b"}, {"id": "a"}], [{"id": "a"}], [{"id": "a"}, {"id": "a"}]):
            with self.assertRaises(ValueError):
                validate_split(manifest, items)
        with self.assertRaises(ValueError):
            validate_split({"development_ids": ["a"], "confirmation_ids": ["a"]}, [{"id": "a"}])

    def test_paired_intervals_reproducible_and_no_uncentered_pvalue(self):
        first = paired_intervals({"constant": np.ones(4), "mixed": np.array([-1, 0, 1, 1])})
        self.assertEqual(first, paired_intervals({"constant": np.ones(4), "mixed": np.array([-1, 0, 1, 1])}))
        self.assertEqual(first["constant"], {"observed_difference": 1., "ci95_lower": 1., "ci95_upper": 1.})
        with self.assertRaises(ValueError):
            paired_intervals({"bad": []})

    def test_confirmation_fixed_rule_costs_cache_and_no_search(self):
        pairs = [{"id": "a"}, {"id": "b"}]
        decisions = [{"id": key, "gate_probability": .8,
                      "candidate_probabilities": {"direct": .2, "v5_reflect_1": .9, "v7_reflect_1": .8}}
                     for key in ("a", "b")]
        correct = np.array([[0, 1, 1], [1, 0, 1]])
        usage = np.array([[[6, 4, 10], [12, 8, 20], [14, 11, 25]]] * 2)
        data = (correct, correct.astype(float), correct.astype(float), correct.astype(float),
                usage, np.array([[20, 15, 35]] * 2))
        policy = {"reference_policy": "unused.pkl", "gate_threshold": .5,
                  "v5_margin_threshold": 2., "v7_margin_threshold": 2.}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("v5.jsonl", "v7.jsonl", "policy_lock.json", "manifest.json"):
                (root / name).write_text("{}\n")
            (root / "generation_complete.json").write_text(json.dumps({"raw_rows": 2, "aligned_rows": 2}))
            with patch("src.qa_hard_confirmation.evaluate", return_value=([], [], [], {}, decisions)) as inference, \
                 patch("src.qa_hard_confirmation.outcomes", return_value=data), \
                 patch("src.qa_hard_confirmation._oracle_source", return_value=2), \
                 patch("src.calibrate_qa_hard_development.scan", side_effect=AssertionError("No confirmation search")), \
                 patch("builtins.print"):
                confirm(root, policy, [], [], pairs)
                report = json.loads((root / "confirmation_report.json").read_text())
                selected = report["methods"][3]
                self.assertEqual(selected["avg_total_tokens"], 35)
                self.assertEqual(selected["adoption_rate"], 0)
                self.assertEqual(selected["invocation_rate"], 1)
                self.assertFalse(report["threshold_search"])
                digest = (root / "confirmation_report.json").read_bytes()
                confirm(root, policy, [], [], pairs)
                self.assertEqual(inference.call_count, 1)
                self.assertEqual(digest, (root / "confirmation_report.json").read_bytes())
                (root / "reference_decisions.jsonl").write_text("{}\n")
                with self.assertRaisesRegex(ValueError, "provenance mismatch"):
                    confirm(root, policy, [], [], pairs)


if __name__ == "__main__":
    unittest.main()
