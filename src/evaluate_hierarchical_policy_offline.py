from __future__ import annotations

import argparse
from collections import defaultdict
from typing import Any, Dict, List

from src.evaluate_policy_offline import _add, _candidate_reward, _candidate_usage, _policy_row, _rows_from_stats
from src.train_policy import predict_action
from src.utils import load_pickle, print_table, read_jsonl, save_json


def _predict_many(bundle: Dict[str, Any], rows: List[Dict[str, Any]]) -> List[str]:
    if bundle.get("policy_type") == "torch_embedding_mlp":
        from src.transformer_policy import predict_transformer_actions

        return predict_transformer_actions(bundle, rows)
    return [predict_action(bundle, row) for row in rows]


def evaluate(rows: List[Dict[str, Any]], gate_model: str, round_model: str, lambda_cost: float) -> List[Dict[str, Any]]:
    gate = load_pickle(gate_model)
    round_policy = load_pickle(round_model)
    methods = ["direct", "fixed_reflect_1", "fixed_reflect_2", "learned_policy", "oracle"]
    stats: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    policy_rows = [_policy_row(trace) for trace in rows]
    gate_actions = _predict_many(gate, policy_rows)
    reflect_indices = [index for index, action in enumerate(gate_actions) if action != "direct"]
    round_actions = _predict_many(
        round_policy,
        [policy_rows[index] for index in reflect_indices],
    )
    round_by_index = dict(zip(reflect_indices, round_actions))

    for index, trace in enumerate(rows):
        candidates = {str(x.get("action")): x for x in trace.get("candidate_results", [])}
        if gate_actions[index] == "direct":
            learned_action = "direct"
        else:
            learned_action = round_by_index.get(index, "reflect_1")
            if learned_action not in {"reflect_1", "reflect_2"}:
                learned_action = "reflect_1"

        oracle_action = max(
            [candidates[action] for action in ("direct", "reflect_1", "reflect_2")],
            key=lambda item: (
                _candidate_reward(item, lambda_cost),
                -_candidate_usage(item)[2],
            ),
        )["action"]
        selected = {
            "direct": candidates["direct"],
            "fixed_reflect_1": candidates["reflect_1"],
            "fixed_reflect_2": candidates["reflect_2"],
            "learned_policy": candidates[learned_action],
            "oracle": candidates[oracle_action],
        }
        for method, candidate in selected.items():
            _add(stats[method], candidate, lambda_cost)
    return _rows_from_stats({"overall": stats}, None, methods)

def main() -> None:
    parser = argparse.ArgumentParser(description="Offline evaluation for a hierarchical Transformer policy.")
    parser.add_argument("--trace", required=True)
    parser.add_argument("--gate_model", required=True)
    parser.add_argument("--round_model", required=True)
    parser.add_argument("--lambda_cost", type=float, default=0.0001)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    results = evaluate(read_jsonl(args.trace), args.gate_model, args.round_model, args.lambda_cost)
    save_json(results, args.output)
    print(print_table(results, [
        "method", "n", "accuracy", "avg_exact_match", "avg_answer_f1",
        "avg_prompt_tokens", "avg_completion_tokens", "avg_token_cost",
        "reflection_rate", "avg_reward",
    ]))
    print(f"Read saved traces; no LLM calls were made.")
    print(f"Wrote hierarchical results to {args.output}")


if __name__ == "__main__":
    main()
