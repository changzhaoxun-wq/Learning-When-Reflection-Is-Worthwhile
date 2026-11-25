"""Build final tables and paired bootstrap comparisons from saved traces.

This report is fully offline: it reuses candidate traces and policy bundles,
so it does not call any LLM endpoint.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

from src.evaluate_policy_offline import _candidate_usage, _policy_row
from src.reward import compute_reward
from src.train_policy import predict_action
from src.utils import load_pickle, read_jsonl


METHODS = ["direct", "fixed_reflect_1", "fixed_reflect_2", "learned_policy", "oracle"]
FIXED_ACTIONS = {
    "direct": "direct",
    "fixed_reflect_1": "reflect_1",
    "fixed_reflect_2": "reflect_2",
}


def _oracle_action(candidates: Dict[str, Dict[str, Any]], lambda_cost: float) -> str:
    return max(
        [candidates[action] for action in ("direct", "reflect_1", "reflect_2")],
        key=lambda item: (
            compute_reward(
                int(float(item.get("correct", 0))),
                int(round(_candidate_usage(item)[2])),
                lambda_cost,
            ),
            -_candidate_usage(item)[2],
        ),
    )["action"]


def _records_for_trace(
    rows: Sequence[Dict[str, Any]],
    gate: Any,
    round_policy: Any,
    lambda_cost: float,
) -> Dict[str, List[Dict[str, float]]]:
    records: Dict[str, List[Dict[str, float]]] = {method: [] for method in METHODS}
    for trace in rows:
        candidates = {str(item.get("action")): item for item in trace.get("candidate_results", [])}
        gate_action = predict_action(gate, _policy_row(trace))
        if gate_action == "direct":
            learned_action = "direct"
        else:
            learned_action = predict_action(round_policy, _policy_row(trace))
            if learned_action not in {"reflect_1", "reflect_2"}:
                learned_action = "reflect_1"

        actions = dict(FIXED_ACTIONS)
        actions["learned_policy"] = learned_action
        actions["oracle"] = _oracle_action(candidates, lambda_cost)
        for method, action in actions.items():
            candidate = candidates[action]
            prompt_tokens, completion_tokens, total_tokens = _candidate_usage(candidate)
            correct = float(candidate.get("correct", 0))
            records[method].append(
                {
                    "correct": correct,
                    "prompt_tokens": float(prompt_tokens),
                    "completion_tokens": float(completion_tokens),
                    "token_cost": float(total_tokens),
                    "reflection": float(action != "direct"),
                    "reward": float(compute_reward(int(correct), int(round(total_tokens)), lambda_cost)),
                }
            )
    return records


def _aggregate(records: Sequence[Dict[str, float]]) -> Dict[str, float]:
    if not records:
        raise ValueError("Cannot aggregate an empty trace")
    keys = ["correct", "prompt_tokens", "completion_tokens", "token_cost", "reflection", "reward"]
    return {
        "n": len(records),
        "accuracy": float(np.mean([row["correct"] for row in records])),
        "avg_prompt_tokens": float(np.mean([row["prompt_tokens"] for row in records])),
        "avg_completion_tokens": float(np.mean([row["completion_tokens"] for row in records])),
        "avg_token_cost": float(np.mean([row["token_cost"] for row in records])),
        "reflection_rate": float(np.mean([row["reflection"] for row in records])),
        "avg_reward": float(np.mean([row["reward"] for row in records])),
    }


def _bootstrap_diff(
    learned: Sequence[Dict[str, float]],
    baseline: Sequence[Dict[str, float]],
    metric: str,
    iterations: int,
    seed: int,
) -> Dict[str, Any]:
    if len(learned) != len(baseline):
        raise ValueError("Paired comparisons require equal-length traces")
    differences = np.asarray([a[metric] - b[metric] for a, b in zip(learned, baseline)], dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(differences), size=(iterations, len(differences)))
    boot = differences[samples].mean(axis=1)
    observed = float(differences.mean())
    lower, upper = np.quantile(boot, [0.025, 0.975])
    return {
        "metric": metric,
        "observed_difference": observed,
        "ci95_lower": float(lower),
        "ci95_upper": float(upper),
        "learned_better_fraction": float(np.mean(differences > 0)),
        "ties_fraction": float(np.mean(differences == 0)),
        "baseline_better_fraction": float(np.mean(differences < 0)),
        "n": len(differences),
    }


def _fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def build_report(
    dataset_traces: Dict[str, str],
    policy_root: str,
    seeds: Sequence[int],
    output_dir: str,
    lambda_cost: float,
    device: str,
    bootstrap_iterations: int,
    bootstrap_seed: int,
) -> Dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    all_summaries: List[Dict[str, Any]] = []
    all_comparisons: List[Dict[str, Any]] = []

    loaded_traces = {name: read_jsonl(path) for name, path in dataset_traces.items()}
    for seed in seeds:
        gate_path = str(Path(policy_root) / f"gate_calibrated_seed{seed}.pkl")
        round_path = str(Path(policy_root) / f"round_seed{seed}.pkl")
        gate = load_pickle(gate_path)
        round_policy = load_pickle(round_path)
        for dataset, rows in loaded_traces.items():
            records = _records_for_trace(rows, gate, round_policy, lambda_cost)
            for method in METHODS:
                item = _aggregate(records[method])
                item.update({"dataset": dataset, "seed": seed, "method": method})
                all_summaries.append(item)

            for baseline in ("direct", "fixed_reflect_1", "fixed_reflect_2", "oracle"):
                for metric in ("correct", "token_cost", "reward"):
                    item = _bootstrap_diff(
                        records["learned_policy"],
                        records[baseline],
                        metric,
                        bootstrap_iterations,
                        bootstrap_seed + seed,
                    )
                    item.update({"dataset": dataset, "seed": seed, "baseline": baseline})
                    all_comparisons.append(item)

    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for row in all_summaries:
        grouped.setdefault((row["dataset"], row["method"]), []).append(row)
    summary_rows: List[Dict[str, Any]] = []
    for (dataset, method), rows in sorted(grouped.items()):
        item = {"dataset": dataset, "method": method, "n_seeds": len(rows)}
        for metric in ("accuracy", "avg_prompt_tokens", "avg_completion_tokens", "avg_token_cost", "reflection_rate", "avg_reward"):
            values = [float(row[metric]) for row in rows]
            item[f"mean_{metric}"] = float(np.mean(values))
            item[f"std_{metric}"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        summary_rows.append(item)

    report = {
        "lambda_cost": lambda_cost,
        "policy_root": policy_root,
        "seeds": list(seeds),
        "datasets": dataset_traces,
        "bootstrap_iterations": bootstrap_iterations,
        "summary": summary_rows,
        "paired_bootstrap": all_comparisons,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))

    lines = [
        "# Adaptive Reflection Final Report",
        "",
        f"Cost coefficient: `{lambda_cost}`; seeds: `{', '.join(map(str, seeds))}`.",
        "All results reuse saved candidate traces; no LLM calls are made.",
        "",
        "## Main Results",
        "",
        "| Dataset | Method | Accuracy | Token Cost | Reflection Rate | Reward |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['dataset']} | {row['method']} | "
            f"{_fmt(row['mean_accuracy'])} +/- {_fmt(row['std_accuracy'])} | "
            f"{row['mean_avg_token_cost']:.1f} +/- {row['std_avg_token_cost']:.1f} | "
            f"{_fmt(row['mean_reflection_rate'])} | "
            f"{_fmt(row['mean_avg_reward'])} +/- {_fmt(row['std_avg_reward'])} |"
        )
    lines += [
        "",
        "## Paired Bootstrap: Learned Policy Minus Baseline",
        "",
        "The interval is the 95% paired bootstrap confidence interval over examples.",
        "",
        "| Dataset | Baseline | Metric | Difference | 95% CI | Learned Better |",
        "|---|---|---|---:|---:|---:|",
    ]
    # Report seed-averaged comparisons for readability; raw seed values remain in report.json.
    comparison_groups: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for row in all_comparisons:
        comparison_groups.setdefault((row["dataset"], row["baseline"], row["metric"]), []).append(row)
    for key, rows in sorted(comparison_groups.items()):
        dataset, baseline, metric = key
        diffs = [row["observed_difference"] for row in rows]
        lowers = [row["ci95_lower"] for row in rows]
        uppers = [row["ci95_upper"] for row in rows]
        better = [
            row["learned_better_fraction"]
            if metric != "token_cost"
            else row["baseline_better_fraction"]
            for row in rows
        ]
        lines.append(
            f"| {dataset} | {baseline} | {metric} | {np.mean(diffs):.4f} | "
            f"[{np.mean(lowers):.4f}, {np.mean(uppers):.4f}] | {np.mean(better):.3f} |"
        )
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote JSON report to {output / 'report.json'}")
    print(f"Wrote Markdown report to {output / 'report.md'}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build final offline results and paired bootstrap report.")
    parser.add_argument("--policy_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--lambda_cost", type=float, default=0.00005)
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 21, 42])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bootstrap_iterations", type=int, default=10000)
    parser.add_argument("--bootstrap_seed", type=int, default=20260723)
    parser.add_argument("--dataset", action="append", nargs=2, metavar=("NAME", "TRACE"), required=True)
    args = parser.parse_args()
    build_report(
        dataset_traces=dict(args.dataset),
        policy_root=args.policy_root,
        seeds=args.seeds,
        output_dir=args.output_dir,
        lambda_cost=args.lambda_cost,
        device=args.device,
        bootstrap_iterations=args.bootstrap_iterations,
        bootstrap_seed=args.bootstrap_seed,
    )


if __name__ == "__main__":
    main()
