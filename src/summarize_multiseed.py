from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, List


METRICS = ["accuracy", "avg_prompt_tokens", "avg_completion_tokens", "avg_token_cost", "reflection_rate", "avg_reward"]


def summarize(input_dir: str, output: str) -> Dict[str, Any]:
    root = Path(input_dir)
    paths = sorted({
        *root.glob("results_seed*.json"),
        *root.glob("eval_results_seed*.json"),
    })
    if not paths:
        raise FileNotFoundError(
            f"No results_seed*.json or eval_results_seed*.json found in {input_dir}"
        )
    by_method: Dict[str, List[Dict[str, Any]]] = {}
    seeds: List[int] = []
    for path in paths:
        seed_text = path.stem.removeprefix("eval_results_seed").removeprefix("results_seed")
        seed = int(seed_text)
        seeds.append(seed)
        rows = json.loads(path.read_text())
        for row in rows:
            by_method.setdefault(str(row["method"]), []).append(row)

    summary: List[Dict[str, Any]] = []
    for method, rows in sorted(by_method.items()):
        item: Dict[str, Any] = {"method": method, "n_seeds": len(rows)}
        for metric in METRICS:
            values = [float(row[metric]) for row in rows]
            item[f"mean_{metric}"] = mean(values)
            item[f"std_{metric}"] = stdev(values) if len(values) > 1 else 0.0
            item[f"values_{metric}"] = values
        summary.append(item)

    result = {"seeds": sorted(seeds), "runs": [str(path) for path in paths], "summary": summary}
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2))

    print("method                 | accuracy       | avg_token_cost | reflection_rate | avg_reward")
    print("-----------------------|----------------|----------------|-----------------|----------")
    for row in summary:
        print(
            f"{row['method']:<22} | "
            f"{row['mean_accuracy']:.4f} +/- {row['std_accuracy']:.4f} | "
            f"{row['mean_avg_token_cost']:.1f} +/- {row['std_avg_token_cost']:.1f} | "
            f"{row['mean_reflection_rate']:.4f} +/- {row['std_reflection_rate']:.4f} | "
            f"{row['mean_avg_reward']:.4f} +/- {row['std_avg_reward']:.4f}"
        )
    print(f"Wrote multi-seed summary to {output}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate repeated hierarchical policy evaluations.")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    summarize(args.input_dir, args.output)


if __name__ == "__main__":
    main()
