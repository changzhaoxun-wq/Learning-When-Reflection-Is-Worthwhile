import json
import tempfile
import unittest
from pathlib import Path

from src.build_frozen_base_reflector_trace import build_trace


class FrozenTraceParallelTests(unittest.TestCase):
    def test_parallel_generation_preserves_input_order_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            canonical = root / "canonical.jsonl"
            output = root / "output.jsonl"
            rows = []
            for index in range(8):
                answer = str(index + 1)
                initial = f"Final answer: {answer}"
                rows.append({
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
                    "candidate_results": [{
                        "action": "direct",
                        "final_answer": initial,
                        "pred": answer,
                        "correct": 1,
                        "token_cost": 8,
                        "prompt_tokens": 5,
                        "completion_tokens": 3,
                        "reflected_rounds": 0,
                    }],
                })
            canonical.write_text("".join(json.dumps(row) + "\n" for row in rows))
            config = {"lambda_cost": 0.0001, "llm": {"mode": "fake"}}
            counts = build_trace(
                canonical, None, output, config, False, False, None, False,
                max_workers=4,
            )
            generated = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual([row["id"] for row in generated], [row["id"] for row in rows])
            self.assertTrue(all(len(row["candidate_results"]) == 3 for row in generated))
            self.assertEqual(counts["rows_written"], 8)
            resumed = build_trace(
                canonical, None, output, config, False, True, None, False,
                max_workers=4,
            )
            self.assertEqual(resumed["rows_skipped"], 8)
            self.assertEqual(len(output.read_text().splitlines()), 8)


if __name__ == "__main__":
    unittest.main()
