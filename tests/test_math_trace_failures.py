import contextlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src import math_full_training_trace as full
from src import math_trace_failures as failures
from src.prepare_qa_hard_development import file_hash
from tests.test_math_full_training_trace import Tokenizer


class FailureQueueTests(unittest.TestCase):
    def run_case(self, fail_role, fail_action):
        items = [{"id": key, "question": key, "gold_answer": "1", "policy_split": split}
                 for key, split in (("first", "train"), ("bad", "validation"), ("last", "train"))]
        tokenizers = {r: Tokenizer() for r in ("base", "qwen14")}
        generated = []

        @contextlib.contextmanager
        def server(root, role):
            def generate(prompt, **kwargs):
                action = "reflect_2" if "Reflection round: 2" in prompt else (
                    "reflect_1" if "Reflection round: 1" in prompt else ("direct" if role == "base" else "independent"))
                length = role == fail_role and action == fail_action and "bad" in prompt
                p = len(prompt.split())
                c = kwargs["max_tokens"] if length else 5
                generated.append((role, action, p + c))
                return SimpleNamespace(text="repeated output" if length else "Final answer: 1", prompt_tokens=p,
                                       completion_tokens=c, total_tokens=p + c, metadata={
                                           "finish_reason": "length" if length else "stop",
                                           "request": {"temperature": 0., "seed": 42, "effective_max_tokens": kwargs["max_tokens"]},
                                           "raw_usage": {"prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c}})
            yield SimpleNamespace(generate=generate)

        with tempfile.TemporaryDirectory() as folder, patch.object(full.pilot, "server", server), patch.object(full, "wait_for_resources"):
            root = Path(folder)
            (root / "protocol.json").write_text("{}")
            base_rows = full.run_stage(root, "base", items, tokenizers, {})
            base = {r["id"]: r for r in base_rows}
            rows = full.run_stage(root, "qwen14", items, tokenizers, {}, base)
            self.assertEqual([r["id"] for r in rows], ["first", "last"])
            before = len(generated)
            self.assertEqual(full.run_stage(root, "base", items, tokenizers, {}), base_rows)
            self.assertEqual(full.run_stage(root, "qwen14", items, tokenizers, {}, base), rows)
            self.assertEqual(len(generated), before)
            full.finalize(root, items, tokenizers, base, rows)
            report = json.loads((root / "coverage_audit.json").read_text())
            status = json.loads((root / "status.json").read_text())
            self.assertEqual(status["stage"], "awaiting_failure_resolution")
            self.assertFalse(status["final_evaluation_ready"])
            self.assertEqual(report["pending_ids"], ["bad"])
            self.assertEqual(report["unresolved_truncation"], 1)
            self.assertEqual(report["unresolved_length_attempts"], 3)
            self.assertEqual(report["new_generation_tokens"], sum(g[2] for g in generated))
            self.assertEqual(report["splits"]["validation"]["expected_n"], 1)
            self.assertEqual(report["splits"]["validation"]["pending_n"], 1)
            self.assertFalse((root / "policy_validation.jsonl").exists())
            self.assertFalse((root / "completion_audit.json").exists())
            entry = failures.read_failures(root, fail_role)["bad"]
            self.assertNotIn("best_action", entry)
            self.assertNotIn("correct", entry)
            target = Path(entry["failed_call"]["call_log"])
            records = [json.loads(x) for x in target.read_text().splitlines()]
            records[0]["metadata"]["request"]["seed"] = 999
            target.write_text("".join(json.dumps(r) + "\n" for r in records))
            with self.assertRaises(ValueError):
                full.run_stage(root, fail_role, items, tokenizers, {}, base if fail_role == "qwen14" else None)

    def test_base_exhaustion_preserves_denominator_and_blocks_downstream(self):
        self.run_case("base", "direct")

    def test_second_reflection_exhaustion_keeps_prior_round_cost(self):
        self.run_case("qwen14", "reflect_2")

    def test_independent_exhaustion_keeps_both_reflection_costs(self):
        self.run_case("qwen14", "independent")

    def test_service_error_is_not_quarantined(self):
        @contextlib.contextmanager
        def broken_server(root, role):
            raise RuntimeError("Engine initialization failed")
            yield

        with tempfile.TemporaryDirectory() as folder, patch.object(full.pilot, "server", broken_server), patch.object(full, "wait_for_resources"):
            root = Path(folder)
            item = {"id": "a", "question": "q", "gold_answer": "1", "policy_split": "train"}
            with self.assertRaisesRegex(RuntimeError, "Engine initialization"):
                full.run_stage(root, "base", [item], {"base": Tokenizer()}, {})
            self.assertEqual(failures.read_failures(root, "base"), {})

    def test_cached_cap_from_old_run_is_quarantined_without_loading_model(self):
        tokenizer = Tokenizer()
        item = {"id": "cached", "question": "q", "gold_answer": "1", "policy_split": "train"}
        prompt = full.pilot.MATH_DIRECT_PROMPT.format(question="q")
        p = len(prompt.split())

        def generate(text, **kwargs):
            c = kwargs["max_tokens"]
            return SimpleNamespace(text="loop", prompt_tokens=p, completion_tokens=c, total_tokens=p + c,
                                   metadata={"finish_reason": "length", "request": {
                                       "temperature": 0., "seed": 42, "effective_max_tokens": c},
                                       "raw_usage": {"prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c}})

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with self.assertRaises(RuntimeError):
                full.pilot.complete_call(SimpleNamespace(generate=generate), tokenizer, root, "base", "cached", "direct", prompt)
            path = root / "calls/base/cached/direct.jsonl"
            sha = file_hash(path)
            with patch.object(full.pilot, "server") as server, patch.object(full, "wait_for_resources") as wait:
                rows = full.run_stage(root, "base", [item], {"base": tokenizer}, {})
                self.assertEqual(rows, [])
                server.assert_not_called()
                wait.assert_not_called()
            self.assertEqual(file_hash(path), sha)
            self.assertEqual(failures.read_failures(root, "base")["cached"]["failed_call"]["attempt_count"], 3)

    def test_duplicate_or_missing_progress_rejected(self):
        items = [{"id": str(i)} for i in range(4)]
        self.assertEqual(failures.handled_prefix(items, [items[0], items[2]], {"1": {}}), 3)
        for rows, ledger in (([items[0]], {"0": {}}), ([items[2]], {}), ([items[0], items[0]], {})):
            with self.assertRaises(ValueError):
                failures.handled_prefix(items, rows, ledger)

    def test_gpu_wait_does_not_kill_other_processes(self):
        outputs = [SimpleNamespace(stdout="0, 49140, 1000\n"), SimpleNamespace(stdout="0, 49140, 48000\n")]
        with tempfile.TemporaryDirectory() as folder, patch.dict(full.pilot.ROLES["base"], {"gpus": "0"}), \
                patch.object(failures.subprocess, "run", side_effect=outputs) as run, patch.object(failures.time, "sleep") as sleep:
            failures.wait_for_resources(Path(folder), "base")
            self.assertEqual(run.call_count, 2)
            sleep.assert_called_once_with(60)
            self.assertTrue(all(c.args[0][0] == "nvidia-smi" for c in run.call_args_list))

    def test_protocol_migration_rejects_prompt_changes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            archive = root / "infrastructure_revisions/exhausted_output_quarantine_v1"
            archive.mkdir(parents=True)
            (archive / "math_full_training_trace.py").write_text("old source")
            old = {"pipeline_code_sha256": {"math_full_training_trace.py": file_hash(archive / "math_full_training_trace.py")},
                   "roles": {"base": {"gpus": "1"}}, "direct_prompt": "original", "hardware_change": "old",
                   "output_steps": [4096, 8192, 16384]}
            (root / "protocol.json").write_text(json.dumps(old))
            (archive / "protocol.json").write_text(json.dumps(old))
            (archive / "before.json").write_text(json.dumps({"preserved_artifacts_sha256": {}}))
            new = {**old, "pipeline_code_sha256": {"math_full_training_trace.py": "new", "math_trace_failures.py": "helper"},
                   "roles": {"base": {"gpus": "0"}}, "hardware_change": "placement", "exception_policy": "quarantine",
                   "resource_policy": "wait", "output_steps": (4096, 8192, 16384)}
            with self.assertRaises(ValueError):
                full.lock_protocol(root, new)
            with self.assertRaisesRegex(ValueError, "unapproved"):
                full.lock_protocol(root, {**new, "direct_prompt": "tampered"}, True)
            full.lock_protocol(root, new, True)
            self.assertEqual(json.loads((root / "protocol.json").read_text()), json.loads(json.dumps(new)))
            self.assertTrue((archive / "audit.json").exists())
            full.lock_protocol(root, new)


if __name__ == "__main__":
    unittest.main()
