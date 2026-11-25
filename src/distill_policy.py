from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List

from .train_policy import featurize_row
from .utils import load_pickle, project_root, read_jsonl, write_jsonl


def distill(dataset_path: str, teacher_path: str, output_path: str) -> int:
    rows = read_jsonl(dataset_path)
    bundle = load_pickle(teacher_path)
    model = bundle["model"]
    feature_names = bundle.get("feature_names")
    features = [featurize_row(row, feature_names=feature_names) for row in rows]
    predictions = [str(value) for value in model.predict(features)]
    output: List[Dict[str, Any]] = []
    for row, action in zip(rows, predictions):
        item = dict(row)
        if action == "verify_reflect":
            action = "reflect_1"
        item["teacher_action"] = action
        item["best_action"] = action
        output.append(item)
    write_jsonl(output, output_path)
    return len(output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create policy data labeled by a trained RF teacher.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--teacher_model", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    count = distill(args.dataset, args.teacher_model, args.output)
    print(f"Wrote {count} RF-distillation examples to {args.output}")


if __name__ == "__main__":
    main()
