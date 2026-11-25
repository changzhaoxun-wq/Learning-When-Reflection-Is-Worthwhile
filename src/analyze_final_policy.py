"""Audit the frozen policy: token accounting, McNemar, and hard-split errors.

All inputs are saved traces and frozen policy bundles. This module performs
no LLM calls and treats the verifier as a deploy-time decision cost only for
the learned policy, because fixed baselines do not need a verifier call.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence

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


def oracle_action(candidates: Dict[str, Dict[str, Any]], lambda_cost: float) -> str:
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


def learned_action(trace: Dict[str, Any], gate: Any, round_policy: Any) -> str:
    action = predict_action(gate, _policy_row(trace))
    if action == "direct":
        return "direct"
    action = predict_action(round_policy, _policy_row(trace))
    return action if action in {"reflect_1", "reflect_2"} else "reflect_1"


def per_item_records(
    rows: Sequence[Dict[str, Any]],
    gate: Any,
    round_policy: Any,
    lambda_cost: float,
    policy_uses_verifier: bool,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for trace in rows:
        candidates = {str(item.get("action")): item for item in trace.get("candidate_results", [])}
        actions = dict(FIXED_ACTIONS)
        actions["learned_policy"] = learned_action(trace, gate, round_policy)
        actions["oracle"] = oracle_action(candidates, lambda_cost)
        state = trace.get("state_features", {})
        verifier_cost = float(
            state.get("verifier_token_cost", trace.get("verifier_token_cost", 0.0)) or 0.0
        )
        item: Dict[str, Any] = {
            "id": trace.get("id"),
            "difficulty": trace.get("difficulty", state.get("difficulty", "unknown")),
            "verifier_token_cost": verifier_cost,
            "actions": actions,
            "methods": {},
        }
        for method, action in actions.items():
            candidate = candidates[action]
            prompt_tokens, completion_tokens, total_tokens = _candidate_usage(candidate)
            correct = int(float(candidate.get("correct", 0)))
            # The local gate/round MLP has no LLM token usage. The learned
            # policy's deploy-time total adds the verifier call used by state.
            decision_cost = verifier_cost if method == "learned_policy" and policy_uses_verifier else 0.0
            item["methods"][method] = {
                "action": action,
                "correct": correct,
                "prompt_tokens": float(prompt_tokens),
                "completion_tokens": float(completion_tokens),
                "candidate_token_cost": float(total_tokens),
                "decision_token_cost": decision_cost,
                "total_system_token_cost": float(total_tokens) + decision_cost,
                "reflection": int(action != "direct"),
            }
        records.append(item)
    return records


def aggregate(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    output = []
    for method in METHODS:
        rows = [item["methods"][method] for item in records]
        output.append(
            {
                "method": method,
                "n": len(rows),
                "accuracy": sum(row["correct"] for row in rows) / len(rows),
                "avg_prompt_tokens": sum(row["prompt_tokens"] for row in rows) / len(rows),
                "avg_completion_tokens": sum(row["completion_tokens"] for row in rows) / len(rows),
                "avg_candidate_token_cost": sum(row["candidate_token_cost"] for row in rows) / len(rows),
                "avg_decision_token_cost": sum(row["decision_token_cost"] for row in rows) / len(rows),
                "avg_total_system_token_cost": sum(row["total_system_token_cost"] for row in rows) / len(rows),
                "reflection_rate": sum(row["reflection"] for row in rows) / len(rows),
            }
        )
    return output


def mcnemar(learned: Sequence[Dict[str, Any]], baseline: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    learned_correct_baseline_wrong = sum(
        int(a["correct"] == 1 and b["correct"] == 0) for a, b in zip(learned, baseline)
    )
    learned_wrong_baseline_correct = sum(
        int(a["correct"] == 0 and b["correct"] == 1) for a, b in zip(learned, baseline)
    )
    discordant = learned_correct_baseline_wrong + learned_wrong_baseline_correct
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(discordant, i) for i in range(min(learned_correct_baseline_wrong, learned_wrong_baseline_correct) + 1))
        p_value = min(1.0, 2.0 * tail / (2.0 ** discordant))
    return {
        "learned_correct_baseline_wrong": learned_correct_baseline_wrong,
        "learned_wrong_baseline_correct": learned_wrong_baseline_correct,
        "discordant_pairs": discordant,
        "exact_two_sided_p": p_value,
        "accuracy_difference": sum(a["correct"] - b["correct"] for a, b in zip(learned, baseline)) / len(learned),
    }


def repair_analysis(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    output: Dict[str, Any] = {"n": len(records), "by_action": {}}
    for action, label in (("reflect_1", "reflect_1"), ("reflect_2", "reflect_2")):
        direct_wrong = sum(item["methods"]["direct"]["correct"] == 0 for item in records)
        direct_correct = sum(item["methods"]["direct"]["correct"] == 1 for item in records)
        repaired = sum(
            item["methods"]["direct"]["correct"] == 0
            and item["methods"][label]["correct"] == 1
            for item in records
        )
        introduced = sum(
            item["methods"]["direct"]["correct"] == 1
            and item["methods"][label]["correct"] == 0
            for item in records
        )
        output["by_action"][action] = {
            "direct_wrong": direct_wrong,
            "direct_correct": direct_correct,
            "successful_repairs": repaired,
            "repair_rate_given_direct_wrong": repaired / direct_wrong if direct_wrong else 0.0,
            "introduced_errors": introduced,
            "error_introduction_rate_given_direct_correct": introduced / direct_correct if direct_correct else 0.0,
            "net_accuracy_difference": sum(
                item["methods"][label]["correct"] - item["methods"]["direct"]["correct"]
                for item in records
            ) / len(records),
        }

    learned_reflections = [
        item
        for item in records
        if item["methods"]["learned_policy"]["action"] != "direct"
    ]
    direct_wrong = sum(item["methods"]["direct"]["correct"] == 0 for item in records)
    direct_correct = sum(item["methods"]["direct"]["correct"] == 1 for item in records)
    repaired = sum(
        item["methods"]["direct"]["correct"] == 0
        and item["methods"]["learned_policy"]["correct"] == 1
        and item["methods"]["learned_policy"]["action"] != "direct"
        for item in records
    )
    introduced = sum(
        item["methods"]["direct"]["correct"] == 1
        and item["methods"]["learned_policy"]["correct"] == 0
        and item["methods"]["learned_policy"]["action"] != "direct"
        for item in records
    )
    output["learned_policy"] = {
        "reflection_calls": len(learned_reflections),
        "successful_repairs": repaired,
        "repair_rate_given_direct_wrong": repaired / direct_wrong if direct_wrong else 0.0,
        "introduced_errors": introduced,
        "error_introduction_rate_given_direct_correct": introduced / direct_correct if direct_correct else 0.0,
    }
    return output


def load_policy(path: Path, device: str) -> Any:
    bundle = load_pickle(path)
    bundle["device"] = device
    return bundle


def analyze(
    dataset_traces: Dict[str, str],
    policy_root: str,
    seeds: Sequence[int],
    output_dir: str,
    lambda_cost: float,
    device: str,
    policy_uses_verifier: bool,
) -> Dict[str, Any]:
    # Override the training device so the audit can run on a chosen GPU or CPU.
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    all_summary: Dict[str, Any] = {}
    all_mcnemar: Dict[str, Any] = {}
    all_hard: Dict[str, Any] = {}

    for dataset, trace_path in dataset_traces.items():
        rows = read_jsonl(trace_path)
        dataset_summary = []
        dataset_mcnemar = []
        hard_records_by_seed: List[List[Dict[str, Any]]] = []
        for seed in seeds:
            gate = load_policy(Path(policy_root) / f"gate_calibrated_seed{seed}.pkl", device)
            round_policy = load_policy(Path(policy_root) / f"round_seed{seed}.pkl", device)
            records = per_item_records(rows, gate, round_policy, lambda_cost, policy_uses_verifier)
            summary = aggregate(records)
            for row in summary:
                row["seed"] = seed
            dataset_summary.extend(summary)
            for baseline in ("direct", "fixed_reflect_1", "fixed_reflect_2", "oracle"):
                result = mcnemar(
                    [item["methods"]["learned_policy"] for item in records],
                    [item["methods"][baseline] for item in records],
                )
                result.update({"seed": seed, "baseline": baseline})
                dataset_mcnemar.append(result)
            if dataset == "gsm8k":
                hard_records_by_seed.append([item for item in records if item["difficulty"] == "hard"])

        all_summary[dataset] = dataset_summary
        all_mcnemar[dataset] = dataset_mcnemar
        if dataset == "gsm8k":
            all_hard[dataset] = [
                {"seed": seed, **repair_analysis(records)}
                for seed, records in zip(seeds, hard_records_by_seed)
            ]

    report = {
        "lambda_cost": lambda_cost,
        "policy_root": policy_root,
        "seeds": list(seeds),
        "datasets": dataset_traces,
        "token_accounting": {
            "candidate_token_cost": "prompt_tokens + completion_tokens for the selected candidate",
            "decision_token_cost": "verifier_token_cost for learned_policy only when --policy_uses_verifier is set; current pure_text policy uses zero LLM decision tokens",
            "total_system_token_cost": "candidate_token_cost + decision_token_cost",
            "policy_uses_verifier": policy_uses_verifier,
            "fixed_baselines": "do not add verifier cost because they do not require a verifier call",
            "oracle": "analysis-only upper bound; no deploy-time decision cost",
        },
        "summary": all_summary,
        "mcnemar": all_mcnemar,
        "hard_split": all_hard,
    }
    (output / "final_analysis.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))

    lines = [
        "# Final Policy Audit",
        "",
        "## Token Accounting",
        "",
        "`candidate_token_cost = prompt_tokens + completion_tokens`.",
        "For learned_policy, `total_system_token_cost` additionally includes the LLM verifier call.",
        "The gate and round heads are local models and contribute zero LLM tokens.",
        "",
        "| Dataset | Method | Candidate Cost | Decision Cost | Total System Cost | Accuracy |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for dataset, rows in all_summary.items():
        for method in METHODS:
            values = [row for row in rows if row["method"] == method]
            if not values:
                continue
            lines.append(
                f"| {dataset} | {method} | "
                f"{sum(row['avg_candidate_token_cost'] for row in values)/len(values):.1f} | "
                f"{sum(row['avg_decision_token_cost'] for row in values)/len(values):.1f} | "
                f"{sum(row['avg_total_system_token_cost'] for row in values)/len(values):.1f} | "
                f"{sum(row['accuracy'] for row in values)/len(values):.4f} |"
            )
    lines += [
        "",
        "## McNemar Exact Test",
        "",
        "`p` is the exact two-sided McNemar p-value over paired examples.",
        "",
        "| Dataset | Baseline | Accuracy Difference | Learned Correct / Baseline Wrong | Learned Wrong / Baseline Correct | p-value |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for dataset, rows in all_mcnemar.items():
        for baseline in ("direct", "fixed_reflect_1", "fixed_reflect_2", "oracle"):
            values = [row for row in rows if row["baseline"] == baseline]
            if not values:
                continue
            lines.append(
                f"| {dataset} | {baseline} | "
                f"{sum(row['accuracy_difference'] for row in values)/len(values):.4f} | "
                f"{sum(row['learned_correct_baseline_wrong'] for row in values)/len(values):.1f} | "
                f"{sum(row['learned_wrong_baseline_correct'] for row in values)/len(values):.1f} | "
                f"{sum(row['exact_two_sided_p'] for row in values)/len(values):.6f} |"
            )
    lines += [
        "",
        "## GSM8K Hard Split Repair Analysis",
        "",
        "Repair rate is conditioned on direct being wrong. Error introduction rate is conditioned on direct being correct.",
        "",
        "| Seed | Action | Successful Repairs | Repair Rate | Introduced Errors | Error Introduction Rate |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for seed_row in all_hard.get("gsm8k", []):
        seed = seed_row["seed"]
        for action in ("reflect_1", "reflect_2", "learned_policy"):
            row = seed_row["by_action"][action] if action in seed_row["by_action"] else seed_row[action]
            lines.append(
                f"| {seed} | {action} | {row['successful_repairs']} | "
                f"{row['repair_rate_given_direct_wrong']:.4f} | {row['introduced_errors']} | "
                f"{row['error_introduction_rate_given_direct_correct']:.4f} |"
            )
    (output / "final_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {output / 'final_analysis.json'}")
    print(f"Wrote {output / 'final_analysis.md'}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit frozen policy token costs and paired accuracy errors.")
    parser.add_argument("--policy_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--lambda_cost", type=float, default=0.0)
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 21, 42])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--policy_uses_verifier", action="store_true", help="Count verifier tokens for learned_policy; default false for pure-text policy")
    parser.add_argument("--dataset", action="append", nargs=2, metavar=("NAME", "TRACE"), required=True)
    args = parser.parse_args()
    analyze(
        dataset_traces=dict(args.dataset),
        policy_root=args.policy_root,
        seeds=args.seeds,
        output_dir=args.output_dir,
        lambda_cost=args.lambda_cost,
        device=args.device,
        policy_uses_verifier=args.policy_uses_verifier,
    )


if __name__ == "__main__":
    main()
