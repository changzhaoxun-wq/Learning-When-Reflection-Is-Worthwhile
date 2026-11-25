import json
import tempfile
import unittest
from pathlib import Path

from src import math_full_training_trace as full
from src.prepare_qa_hard_development import file_hash


class PlacementTests(unittest.TestCase):
    def test_placement_persists_without_mutating_model_specs(self):
        previous = {"roles": {"base": {"gpus": "0", "tp": 1, "model": "base"},
                              "qwen14": {"gpus": "1,2", "tp": 2, "model": "reflector"}}}
        roles = full.role_placements(previous, "0", "1,3")
        self.assertEqual(roles["qwen14"], {"gpus": "1,3", "tp": 2, "model": "reflector"})
        self.assertEqual(previous["roles"]["qwen14"]["gpus"], "1,2")
        self.assertEqual(full.role_placements(previous, "0", saved={"roles": roles}), roles)
        for gpu in ("1", "1,1", "1,01", "1,2,3", "", "a,3"):
            with self.assertRaises(ValueError):
                full.role_placements(previous, "0", gpu)

    def test_migration_rejects_tp_prompt_and_budget_changes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            archive = root / "infrastructure_revisions/reflector_gpu_placement_v1"
            archive.mkdir(parents=True)
            (archive / "math_full_training_trace.py").write_text("old source")
            old = {"pipeline_code_sha256": {"math_full_training_trace.py": file_hash(archive / "math_full_training_trace.py")},
                   "roles": {"qwen14": {"gpus": "1,2", "tp": 2}}, "direct_prompt": "original", "output_steps": [4096, 8192, 16384]}
            for path in (root / "protocol.json", archive / "protocol.json"):
                path.write_text(json.dumps(old))
            (archive / "before.json").write_text(json.dumps({"preserved_artifacts_sha256": {}}))
            new = {**old, "pipeline_code_sha256": {"math_full_training_trace.py": "new"},
                   "roles": {"qwen14": {"gpus": "1,3", "tp": 2}}}
            with self.assertRaises(ValueError):
                full.lock_protocol(root, new)
            for change in ({"direct_prompt": "changed"}, {"output_steps": [1024]},
                           {"roles": {"qwen14": {"gpus": "1,3", "tp": 1}}}):
                with self.assertRaisesRegex(ValueError, "unapproved"):
                    full.lock_protocol(root, {**new, **change}, accept_placement_revision=True)
            full.lock_protocol(root, new, accept_placement_revision=True)
            full.lock_protocol(root, new)
            self.assertEqual(json.loads((root / "protocol.json").read_text()), new)


if __name__ == "__main__":
    unittest.main()
