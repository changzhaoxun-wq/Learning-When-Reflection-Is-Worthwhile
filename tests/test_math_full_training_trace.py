import contextlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src import math_full_training_trace as full


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return messages[0]["content"]

    def encode(self, text, **kwargs):
        return list(range(len(text.split())))


class FullMathTests(unittest.TestCase):
    def test_split_is_outcome_independent_and_keeps_pilot_in_train(self):
        train = [{"id": str(i), "question": f"problem {i}", "correct": i % 2} for i in range(20)]
        first = full.split_items(train, [], {"0", "1"}, validation_size=4)
        changed = [{**r, "correct": 1 - r["correct"]} for r in reversed(train)]
        second = full.split_items(changed, [], {"0", "1"}, validation_size=4)
        a = {r["id"]: r["policy_split"] for r in first}
        self.assertEqual(a, {r["id"]: r["policy_split"] for r in second})
        self.assertEqual(sum(v == "validation" for v in a.values()), 4)
        self.assertEqual(a["0"], "train")
        self.assertEqual(a["1"], "train")

    def test_overlap_and_duplicate_fail_closed(self):
        train = [{"id": str(i), "question": f"problem {i}"} for i in range(5)]
        with self.assertRaises(ValueError):
            full.split_items(train, [{"id": "other", "question": "  PROBLEM 1 "}], set(), validation_size=1)
        with self.assertRaises(ValueError):
            full.split_items(train + [train[0]], [], set(), validation_size=1)
        with self.assertRaises(ValueError):
            full.split_items(train, [], {"missing"}, validation_size=1)

    def test_sparse_correct_training_requires_complete_validation(self):
        direct = {"action": "direct", "correct": 1, "token_cost": 20}
        base = {"policy_split": "train", "question": "q", "initial_answer": "a"}
        row = full.make_policy_row(base, [direct])
        self.assertEqual(row["best_action"], "direct")
        self.assertEqual(row["missing_actions"], ["reflect_1", "reflect_2"])
        self.assertEqual(set(row["state_features"]), {"question", "initial_answer"})
        with self.assertRaises(ValueError):
            full.make_policy_row({**base, "policy_split": "validation"}, [direct])
        with self.assertRaises(ValueError):
            full.make_policy_row(base, [{**direct, "correct": 0}])

    def test_accuracy_first_not_reward_oracle(self):
        base = {"policy_split": "train", "question": "q", "initial_answer": "a"}
        candidates = [{"action": "direct", "correct": 0, "token_cost": 1, "reward": -.0001},
                      {"action": "reflect_1", "correct": 1, "token_cost": 20000, "reward": -1},
                      {"action": "reflect_2", "correct": 1, "token_cost": 30000, "reward": -2}]
        self.assertEqual(full.make_policy_row(base, candidates)["best_action"], "reflect_1")

    def test_fake_end_to_end_resume_and_tamper_detection(self):
        items = [{"id": key, "question": key, "gold_answer": "1", "policy_split": split}
                 for key, split in (("correct", "train"), ("wrong", "train"), ("heldout", "validation"))]
        tokenizers = {r: Tokenizer() for r in ("base", "qwen14")}
        generated = []

        @contextlib.contextmanager
        def fake_server(root, role):
            def generate(prompt, **kwargs):
                generated.append((role, prompt))
                answer = "0" if role == "base" and "wrong" in prompt else "1"
                p, c = len(prompt.split()), 5
                return SimpleNamespace(text=f"Final answer: {answer}", prompt_tokens=p, completion_tokens=c,
                                       total_tokens=p + c, metadata={"finish_reason": "stop", "raw_usage": {
                                           "prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c}})
            yield SimpleNamespace(generate=generate)

        with tempfile.TemporaryDirectory() as folder, patch.object(full.pilot, "server", fake_server), patch.object(full, "wait_for_resources"):
            root = Path(folder)
            (root / "protocol.json").write_text("{}")
            base_rows = full.run_stage(root, "base", items, tokenizers, {})
            base = {r["id"]: r for r in base_rows}
            rows = full.run_stage(root, "qwen14", items, tokenizers, {}, base)
            self.assertEqual(len(generated), 8)  # 3 Base + 2 train reflections + 3 validation calls.
            self.assertEqual(rows[0]["candidate_coverage"], "direct_only_known_correct")
            self.assertEqual(rows[1]["best_action"], "reflect_1")
            self.assertEqual(rows[2]["missing_actions"], [])
            self.assertIn("independent", rows[2])
            full.finalize(root, items, tokenizers, base, rows)
            report = json.loads((root / "completion_audit.json").read_text())
            self.assertEqual(report["unique_logical_calls"], 8)
            self.assertEqual(report["reused_generation_tokens"], 0)
            self.assertEqual(full.run_stage(root, "qwen14", items, tokenizers, {}, base), rows)
            self.assertEqual(len(generated), 8)
            rows[1]["candidate_results"][1]["pred"] = "999"
            with self.assertRaises(ValueError):
                full.validate_row(rows[1], items[1], "qwen14", tokenizers, base["wrong"])


if __name__ == "__main__":
    unittest.main()
