import json
import tempfile
import unittest
from pathlib import Path

from src.prepare_qa_counterfactual_training import merge_augmented, prepare_sample
from src.utils import read_jsonl, write_jsonl


def _row(index, correct, complete):
    direct = {
        "action": "direct",
        "final_answer": f"Final answer: {index}",
        "pred": str(index),
        "correct": correct,
        "prompt_tokens": 10,
        "completion_tokens": 2,
        "token_cost": 12,
    }
    candidates = [direct]
    if complete:
        candidates.extend([
            {"action": "reflect_1", "correct": 1, "token_cost": 20},
            {"action": "reflect_2", "correct": 1, "token_cost": 30},
        ])
    return {
        "id": f"row-{index}",
        "question": f"Question {index}",
        "initial_answer": direct["final_answer"],
        "initial_pred": str(index),
        "gold_answer": str(index),
        "task_type": "qa",
        "initial_call": {
            "prompt": f"Prompt {index}",
            "output": direct["final_answer"],
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "total_tokens": 12,
        },
        "candidate_results": candidates,
    }


class PrepareQACounterfactualTrainingTests(unittest.TestCase):
    def test_sample_and_merge_preserve_source_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.jsonl"
            sample = root / "sample.jsonl"
            completed = root / "completed.jsonl"
            output = root / "output.jsonl"
            rows = [_row(0, 0, True), *[_row(i, 1, False) for i in range(1, 6)]]
            write_jsonl(rows, source)
            manifest = prepare_sample(source, sample, root / "sample.json", 2, 42)
            self.assertEqual(manifest["existing_complete_rows"], 1)
            selected = read_jsonl(sample)
            self.assertEqual(len(selected), 2)
            completed_rows = []
            for row in selected:
                updated = dict(row)
                updated["candidate_results"] = [
                    *row["candidate_results"],
                    {"action": "reflect_1", "correct": 1, "token_cost": 20},
                    {"action": "reflect_2", "correct": 1, "token_cost": 30},
                ]
                completed_rows.append(updated)
            write_jsonl(completed_rows, completed)
            merged = merge_augmented(source, completed, output, root / "merge.json")
            self.assertEqual(merged["replaced_direct_correct_rows"], 2)
            output_rows = read_jsonl(output)
            self.assertEqual([row["id"] for row in output_rows], [row["id"] for row in rows])
            self.assertEqual(sum(len(row["candidate_results"]) == 3 for row in output_rows), 3)


if __name__ == "__main__":
    unittest.main()
