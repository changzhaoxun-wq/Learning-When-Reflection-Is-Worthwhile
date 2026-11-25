from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, List


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    output: List[Dict[str, Any]] = []
    for path in sorted(root.glob("lambda_*/results_seed*.json")):
        lambda_tag = path.parent.name.removeprefix("lambda_")
        rows = json.loads(path.read_text())
        by_method = {str(row["method"]): row for row in rows}
        for method, row in by_method.items():
            item = next((x for x in output if x["lambda_tag"] == lambda_tag and x["method"] == method), None)
            if item is None:
                item = {"lambda_tag": lambda_tag, "method": method, "values": {}}
                output.append(item)
            for key in ["accuracy", "avg_prompt_tokens", "avg_completion_tokens", "avg_token_cost", "reflection_rate", "avg_reward"]:
                item["values"].setdefault(key, []).append(float(row[key]))

    summary = []
    for item in sorted(output, key=lambda x: (x["lambda_tag"], x["method"])):
        row = {"lambda_tag": item["lambda_tag"], "method": item["method"], "n_seeds": len(next(iter(item["values"].values())))}
        for key, values in item["values"].items():
            row[f"mean_{key}"] = mean(values)
            row[f"std_{key}"] = stdev(values) if len(values) > 1 else 0.0
        summary.append(row)

    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(summary, indent=2))
    print("lambda_tag | method          | accuracy | avg_token_cost | reflection_rate | avg_reward")
    print("-----------|-----------------|----------|----------------|-----------------|----------")
    for row in summary:
        print(f"{row['lambda_tag']:<10} | {row['method']:<15} | {row['mean_accuracy']:.4f} | {row['mean_avg_token_cost']:.1f} | {row['mean_reflection_rate']:.4f} | {row['mean_avg_reward']:.4f}")
    print(f"Wrote lambda summary to {target}")


if __name__ == "__main__":
    main()
