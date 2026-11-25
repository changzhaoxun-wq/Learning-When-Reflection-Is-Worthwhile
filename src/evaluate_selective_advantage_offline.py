from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from typing import Any, Dict, List

import numpy as np

from .evaluate_policy_offline import (
    _add,
    _candidate_reward,
    _candidate_usage,
    _policy_row,
    _rows_from_stats,
)
from .selective_advantage_policy import predict_actions
from .utils import load_pickle, print_table, read_jsonl, save_json, write_jsonl


METHODS = ["direct", "fixed_reflect_1", "fixed_reflect_2", "learned_policy", "oracle"]


def evaluate(
    traces: List[Dict[str, Any]],
    model_path: str,
    lambda_cost: float,
    feature_cache: str | None,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
    bundle = load_pickle(model_path)
    if bundle.get("policy_type") != "selective_advantage_mlp":
        raise ValueError(f"Expected selective_advantage_mlp, got {bundle.get('policy_type')}")
    policy_rows = [_policy_row(trace) for trace in traces]
    actions, diagnostics = predict_actions(bundle, policy_rows, cache_path=feature_cache)

    overall: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    by_difficulty: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(float))
    )
    by_source: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(float))
    )
    decisions = []
    action_counts: Counter[str] = Counter()
    oracle_counts: Counter[str] = Counter()
    fixes = harms = unchanged_correct = unchanged_wrong = 0

    for index, (trace, action) in enumerate(zip(traces, actions)):
        candidates = {
            str(item.get("action")): item
            for item in trace.get("candidate_results", [])
        }
        missing = [candidate for candidate in ("direct", "reflect_1", "reflect_2") if candidate not in candidates]
        if missing:
            raise ValueError(f"Trace {trace.get('id')} misses actions {missing}")
        if action not in candidates:
            action = "direct"
        oracle_action = max(
            candidates.values(),
            key=lambda item: (_candidate_reward(item, lambda_cost), -_candidate_usage(item)[2]),
        )["action"]
        selected = {
            "direct": candidates["direct"],
            "fixed_reflect_1": candidates["reflect_1"],
            "fixed_reflect_2": candidates["reflect_2"],
            "learned_policy": candidates[action],
            "oracle": candidates[str(oracle_action)],
        }
        difficulty = str(trace.get("difficulty") or "unknown")
        source = str(trace.get("source") or trace.get("dataset") or "unknown")
        for method, candidate in selected.items():
            _add(overall[method], candidate, lambda_cost)
            _add(by_difficulty[difficulty][method], candidate, lambda_cost)
            _add(by_source[source][method], candidate, lambda_cost)

        direct_correct = float(candidates["direct"].get("correct", 0)) >= 0.5
        selected_correct = float(candidates[action].get("correct", 0)) >= 0.5
        if not direct_correct and selected_correct:
            fixes += 1
        elif direct_correct and not selected_correct:
            harms += 1
        elif direct_correct:
            unchanged_correct += 1
        else:
            unchanged_wrong += 1
        action_counts[action] += 1
        oracle_counts[str(oracle_action)] += 1
        decisions.append({
            "id": trace.get("id"),
            "action": action,
            "oracle_action": oracle_action,
            "direct_correct": int(direct_correct),
            "selected_correct": int(selected_correct),
            "gate_score": float(diagnostics["gate_score"][index]),
            "direct_error_probability": float(diagnostics["direct_error_probability"][index]),
            "predicted_advantage": float(diagnostics["predicted_advantage"][index]),
            "predicted_round_delta": float(diagnostics["predicted_round_delta"][index]),
        })

    n = len(traces)
    analysis = {
        "n": n,
        "policy_type": bundle["policy_type"],
        "training_rows": bundle.get("n_rows"),
        "training_complete_counterfactual": bundle.get("n_complete_counterfactual"),
        "training_direct_only": bundle.get("n_direct_only"),
        "training_positive_advantage": bundle.get("n_positive_advantage"),
        "training_assumption": bundle.get("training_assumption"),
        "gate_threshold": bundle.get("gate_threshold"),
        "round_threshold": bundle.get("round_threshold"),
        "oof_gate_metrics": bundle.get("oof_gate_metrics"),
        "oof_round_weighted_accuracy": bundle.get("oof_round_weighted_accuracy"),
        "action_counts": dict(action_counts),
        "oracle_action_counts": dict(oracle_counts),
        "reflection_rate": (action_counts["reflect_1"] + action_counts["reflect_2"]) / n,
        "direct_to_reflect_fixes": fixes,
        "direct_to_reflect_harms": harms,
        "net_accuracy_changes": fixes - harms,
        "unchanged_correct": unchanged_correct,
        "unchanged_wrong": unchanged_wrong,
        "mean_gate_score": float(np.mean(diagnostics["gate_score"])),
        "mean_predicted_advantage": float(np.mean(diagnostics["predicted_advantage"])),
        "test_labels_used_for_action_selection": False,
    }
    return (
        _rows_from_stats({"overall": overall}, None, METHODS),
        _rows_from_stats(by_difficulty, "difficulty", METHODS),
        _rows_from_stats(by_source, "source", METHODS),
        analysis,
        decisions,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="One-shot frozen-trace evaluation of selective advantage policy.")
    parser.add_argument("--trace", required=True)
    parser.add_argument("--policy_model", required=True)
    parser.add_argument("--lambda_cost", type=float, default=0.0001)
    parser.add_argument("--feature_cache")
    parser.add_argument("--output", required=True)
    parser.add_argument("--difficulty_output", required=True)
    parser.add_argument("--source_output", required=True)
    parser.add_argument("--analysis_output", required=True)
    parser.add_argument("--decisions_output", required=True)
    args = parser.parse_args()
    results = evaluate(
        read_jsonl(args.trace), args.policy_model, args.lambda_cost, args.feature_cache
    )
    overall, difficulty, source, analysis, decisions = results
    save_json(overall, args.output)
    save_json(difficulty, args.difficulty_output)
    save_json(source, args.source_output)
    save_json(analysis, args.analysis_output)
    write_jsonl(decisions, args.decisions_output)
    print(print_table(overall, [
        "method", "n", "accuracy", "avg_exact_match", "avg_answer_f1",
        "avg_prompt_tokens", "avg_completion_tokens", "avg_token_cost",
        "reflection_rate", "avg_reward",
    ]))
    print(
        f"fixes={analysis['direct_to_reflect_fixes']} "
        f"harms={analysis['direct_to_reflect_harms']} "
        f"net={analysis['net_accuracy_changes']} actions={analysis['action_counts']}"
    )
    print(f"Read {analysis['n']} frozen traces; no LLM calls were made.")


if __name__ == "__main__":
    main()
