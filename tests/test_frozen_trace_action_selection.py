import json
import tempfile
import unittest
from pathlib import Path

from src.build_frozen_base_reflector_trace import build_trace


class FrozenTraceActionSelectionTests(unittest.TestCase):
    def test_reflect_one_only_preserves_selected_counterfactual_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            canonical = root / "canonical.jsonl"
            output = root / "output.jsonl"
            rows = []
            for index, selected in enumerate((True, False)):
                answer = str(index + 1)
                initial = f"Final answer: {answer}"
                candidates = [
                    {
                        "action": "direct",
                        "final_answer": initial,
                        "pred": answer,
                        "correct": 1,
                        "token_cost": 8,
                        "prompt_tokens": 5,
                        "completion_tokens": 3,
                        "reflected_rounds": 0,
                    }
                ]
                if selected:
                    candidates.append({"action": "reflect_1"})
                rows.append(
                    {
                        "id": f"row-{index}",
                        "question": f"What is {answer}?",
                        "gold_answer": answer,
                        "task_type": "math",
                        "initial_answer": initial,
                        "initial_call": {
                            "prompt": f"Question: What is {answer}?",
                            "output": initial,
                            "prompt_tokens": 5,
                            "completion_tokens": 3,
                            "total_tokens": 8,
                        },
                        "candidate_results": candidates,
                    }
                )
            canonical.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )

            counts = build_trace(
                canonical,
                None,
                output,
                {"lambda_cost": 0.0001, "llm": {"mode": "fake"}},
                False,
                False,
                None,
                False,
                max_workers=2,
                reflection_actions=("reflect_1",),
                canonical_reflections_only=True,
            )

            generated = [
                json.loads(line) for line in output.read_text().splitlines()
            ]
            self.assertEqual(
                [len(row["candidate_results"]) for row in generated],
                [2, 1],
            )
            self.assertEqual(
                generated[0]["action_space"], ["direct", "reflect_1"]
            )
            self.assertEqual(generated[1]["best_action"], "direct")
            self.assertEqual(counts["reflect_1_regenerated"], 1)
            self.assertEqual(counts["reflect_1_skipped_not_selected"], 1)


if __name__ == "__main__":
    unittest.main()
