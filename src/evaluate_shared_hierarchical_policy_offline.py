from __future__ import annotations

import argparse
from collections import defaultdict
from typing import Any, Dict, List

from src.evaluate_policy_offline import _add, _candidate_usage, _policy_row, _rows_from_stats
from src.reward import compute_reward
from src.shared_hierarchical_policy import SharedHierarchicalMLP, predict  # noqa: F401
from src.utils import load_pickle, print_table, read_jsonl, save_json


def evaluate(rows: List[Dict[str, Any]], policy_path: str, lambda_cost: float) -> List[Dict[str, Any]]:
    policy = load_pickle(policy_path)
    methods = ["direct", "fixed_reflect_1", "fixed_reflect_2", "learned_policy", "oracle"]
    stats: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for trace in rows:
        candidates = {str(item.get("action")): item for item in trace.get("candidate_results", [])}
        action = predict(policy, _policy_row(trace))
        oracle_action = max(
            [candidates[name] for name in ("direct", "reflect_1", "reflect_2")],
            key=lambda item: (
                compute_reward(int(float(item.get("correct", 0))), int(round(_candidate_usage(item)[2])), lambda_cost),
                -_candidate_usage(item)[2],
            ),
        )["action"]
        selected = {
            "direct": candidates["direct"],
            "fixed_reflect_1": candidates["reflect_1"],
            "fixed_reflect_2": candidates["reflect_2"],
            "learned_policy": candidates[action],
            "oracle": candidates[oracle_action],
        }
        for method, candidate in selected.items():
            _add(stats[method], candidate, lambda_cost)
    return _rows_from_stats({"overall": stats}, None, methods)


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline evaluation for a shared hierarchical policy.")
    parser.add_argument("--trace", required=True)
    parser.add_argument("--policy_model", required=True)
    parser.add_argument("--lambda_cost", type=float, default=0.0001)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    results = evaluate(read_jsonl(args.trace), args.policy_model, args.lambda_cost)
    save_json(results, args.output)
    print(print_table(results, ["method", "n", "accuracy", "avg_prompt_tokens", "avg_completion_tokens", "avg_token_cost", "reflection_rate", "avg_reward"]))
    print("Read saved traces; no LLM calls were made.")
    print(f"Wrote shared-policy results to {args.output}")


if __name__ == "__main__":
    main()
