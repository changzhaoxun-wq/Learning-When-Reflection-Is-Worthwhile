import copy
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from src.build_frozen_base_reflector_trace import build_trace
from src.llm_client import LLMResponse
from src.prepare_qa_hard_development import question_key, select_splits, locked_write
from src.qa_reflection_profile import profile_hash, text_hash, validate_profile
from src.reflector import run_reflection
from src.strategies import execute_strategy


def make_profile():
    profile = {
        "version": "test_v5", "round_1": "Independent:\n{question}",
        "round_2": "Audit:\n{question}",
        "format_repair": "Extract:\n{question}\nDraft:\n{draft_answer}",
        "completion_suffix": "Answer: ", "prepend_decision": False,
    }
    profile["reflection_sha256"] = text_hash(profile["round_1"] + "\n" + profile["round_2"])
    profile["format_repair_sha256"] = text_hash(profile["format_repair"])
    return validate_profile(profile)


class Client:
    def __init__(self, responses):
        self.responses, self.calls = responses, []

    def generate(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        text, finish = self.responses.pop(0)
        return LLMResponse(text, 5, 3, 8, {"finish_reason": finish})


class HardDevelopmentTests(unittest.TestCase):
    def sample_rows(self):
        return [{"id": f"hotpotqa-train-{i}", "question": f"Question {i}?",
                 "gold_answer": "yes", "metadata": {"level": "hard", "type": source}}
                for i, source in enumerate(["bridge"] * 30 + ["comparison"] * 10)]

    def test_disjoint_stratified_reproducible_and_outcome_independent(self):
        data = self.sample_rows()
        dev, confirm, _ = select_splits(data, {"0"}, {question_key(data[1])}, 10, 10, 42)
        self.assertEqual(Counter(r["metadata"]["type"] for r in dev), {"bridge": 8, "comparison": 2})
        self.assertEqual(Counter(r["metadata"]["type"] for r in confirm), {"bridge": 8, "comparison": 2})
        self.assertFalse({r["id"] for r in dev} & {r["id"] for r in confirm})
        self.assertNotIn(data[0], dev + confirm)
        self.assertNotIn(data[1], dev + confirm)
        altered = copy.deepcopy(data)
        for row in altered:
            row["gold_answer"] = "changed"
            row["correct"] = 0
        again = select_splits(list(reversed(altered)), {"0"}, {question_key(data[1])}, 10, 10, 42)
        self.assertEqual([r["id"] for r in dev], [r["id"] for r in again[0]])

    def test_reject_nontraining_ids_and_insufficient_pool(self):
        data = self.sample_rows()
        data[0]["id"] = "hotpotqa-validation-0"
        with self.assertRaisesRegex(ValueError, "official HotpotQA train"):
            select_splits(data, set(), set(), 10, 10, 42)
        with self.assertRaisesRegex(ValueError, "Not enough"):
            select_splits(self.sample_rows()[:10], set(), set(), 10, 10, 42)

    def test_locked_artifact_rejects_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            locked_write(path, "first")
            locked_write(path, "first")
            with self.assertRaisesRegex(ValueError, "Locked artifact differs"):
                locked_write(path, "second")
            self.assertEqual(path.read_text(), "first")

    def test_profile_format_repair_and_total_accounting(self):
        profile = make_profile()
        client = Client([("Long draft without marker", "stop"), ("Paris", "stop")])
        result = execute_strategy("reflect_1", client, "Where?", "Final answer: London", 10,
                                  initial_prompt_tokens=7, initial_completion_tokens=3,
                                  task_type="qa", qa_profile=profile)
        self.assertEqual(client.calls[0][0], "Independent:\nWhere?")
        self.assertNotIn("London", client.calls[0][0])
        self.assertEqual(client.calls[0][1]["completion_prompt_suffix"], "Answer: ")
        self.assertEqual(client.calls[1][0], "Extract:\nWhere?\nDraft:\nLong draft without marker")
        self.assertEqual((result.prompt_tokens, result.completion_tokens, result.token_cost), (17, 9, 26))
        self.assertEqual(result.pred, "Paris")

    def test_profile_truncation_recovery_keeps_failed_attempt_cost(self):
        client = Client([("partial", "length"), ("Paris", "stop")])
        response = run_reflection(client, "Where?", "Final answer: London", 1,
                                  task_type="qa", qa_profile=make_profile())
        self.assertEqual(response.total_tokens, 16)
        self.assertEqual(response.metadata["finish_reason"], "stop")
        self.assertTrue(response.metadata["qa_truncation_recovery"]["applied"])
        self.assertEqual(response.text, "Final answer: Paris")

    def test_profile_validation_and_frozen_trace_resume(self):
        profile = make_profile()
        row = {"id": "hotpotqa-train-a", "question": "Where?", "task_type": "qa",
               "gold_answer": "London", "initial_answer": "Final answer: London",
               "initial_call": {"prompt": "Where?", "output": "Final answer: London",
                                "prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
               "candidate_results": [{"action": "direct", "final_answer": "Final answer: London",
                                      "prompt_tokens": 7, "completion_tokens": 3, "token_cost": 10}]}
        with tempfile.TemporaryDirectory() as tmp:
            canonical, output = Path(tmp) / "canonical.jsonl", Path(tmp) / "trace.jsonl"
            canonical.write_text(json.dumps(row) + "\n")
            config = {"llm": {"mode": "fake"}, "qa_reflection_profile": profile}
            kwargs = dict(canonical_path=canonical, target_path=None, output_path=output, config=config,
                          dry_run=False, resume=True, limit=None, incorrect_only=False,
                          reflection_actions=("reflect_1",))
            with patch("src.build_frozen_base_reflector_trace.build_llm_client", return_value=Client([("Final answer: Paris", "stop")])):
                build_trace(**kwargs)
            result = json.loads(output.read_text())
            self.assertEqual(result["qa_profile_sha256"], profile_hash(profile))
            self.assertEqual(result["initial_call"], row["initial_call"])
            self.assertEqual(result["reflection_prompt_version"], profile["version"])
            config["qa_reflection_profile"] = {**profile, "completion_suffix": "Changed: "}
            with self.assertRaisesRegex(ValueError, "mix QA profiles"):
                build_trace(**kwargs)
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            validate_profile({**profile, "round_1": "Changed {question}"})


if __name__ == "__main__":
    unittest.main()
