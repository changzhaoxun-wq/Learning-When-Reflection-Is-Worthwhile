from __future__ import annotations

import argparse
import copy
from collections import Counter, defaultdict
from typing import Any, Dict, List

from .accuracy_first_qa_policy import predict_accuracy_first
from .evaluate_policy_offline import _add, _rows_from_stats
from .utils import (
    load_pickle,
    print_table,
    read_jsonl,
    save_json,
    write_jsonl,
)


METHODS = [
    "direct",
    "fixed_reflect_1",
    "learned_gate",
    "learned_gate_acceptance",
    "binary_accuracy_oracle",
]


def _correct(candidate: Dict[str, Any]) -> bool:
    return float(candidate.get("correct", 0)) >= 0.5


def _rejected_reflection_candidate(
    direct: Dict[str, Any], reflected: Dict[str, Any]
) -> Dict[str, Any]:
    """Keep the direct answer but charge the reflection call that was made."""

    result = copy.deepcopy(direct)
    for key in (
        "prompt_tokens",
        "completion_tokens",
        "token_cost",
        "metadata",
    ):
        if reflected.get(key) is not None:
            result[key] = copy.deepcopy(reflected[key])
    result["reflected_rounds"] = max(
        1, int(reflected.get("reflected_rounds", 1))
    )
    return result


def _transition_counts(
    traces: List[Dict[str, Any]],
    reflected: List[bool],
) -> Dict[str, int]:
    counts = {
        "fixes": 0,
        "harms": 0,
        "unchanged_correct": 0,
        "unchanged_wrong": 0,
    }
    for trace, use_reflection in zip(traces, reflected):
        candidates = {
            str(item.get("action")): item
            for item in trace.get("candidate_results", [])
        }
        direct_correct = _correct(candidates["direct"])
        selected_correct = _correct(
            candidates["reflect_1"] if use_reflection else candidates["direct"]
        )
        if not direct_correct and selected_correct:
            counts["fixes"] += 1
        elif direct_correct and not selected_correct:
            counts["harms"] += 1
        elif direct_correct:
            counts["unchanged_correct"] += 1
        else:
            counts["unchanged_wrong"] += 1
    counts["net_fixes"] = counts["fixes"] - counts["harms"]
    return counts


def evaluate(
    traces: List[Dict[str, Any]],
    policy_model: str,
    gate_cache: str | None,
    acceptance_cache: str | None,
    lambda_cost: float,
) -> tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    Dict[str, Any],
    List[Dict[str, Any]],
]:
    bundle = load_pickle(policy_model)
    diagnostics = predict_accuracy_first(
        bundle,
        traces,
        gate_cache=gate_cache,
        acceptance_cache=acceptance_cache,
    )
    gate_reflect = diagnostics["gate_reflect"].tolist()
    joint_gate_reflect = diagnostics["joint_gate_reflect"].tolist()
    accepted_reflect = diagnostics["accepted_reflect"].tolist()

    overall: Dict[str, Dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    by_difficulty: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(float))
    )
    by_source: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(float))
    )
    decisions = []
    gate_actions: Counter[str] = Counter()
    joint_gate_actions: Counter[str] = Counter()
    accepted_actions: Counter[str] = Counter()
    oracle_actions: Counter[str] = Counter()

    for index, trace in enumerate(traces):
        candidates = {
            str(item.get("action")): item
            for item in trace.get("candidate_results", [])
        }
        missing = [
            action
            for action in ("direct", "reflect_1")
            if action not in candidates
        ]
        if missing:
            raise ValueError(
                f"Trace {trace.get('id')} is missing actions {missing}"
            )
        direct = candidates["direct"]
        reflected = candidates["reflect_1"]
        gate_action = "reflect_1" if gate_reflect[index] else "direct"
        accepted_action = (
            "reflect_1" if accepted_reflect[index] else "direct"
        )
        joint_gate_action = (
            "reflect_1" if joint_gate_reflect[index] else "direct"
        )
        oracle_action = (
            "reflect_1"
            if (not _correct(direct) and _correct(reflected))
            else "direct"
        )
        accepted_candidate = candidates[accepted_action]
        if joint_gate_reflect[index] and not accepted_reflect[index]:
            accepted_candidate = _rejected_reflection_candidate(
                direct, reflected
            )
        selected = {
            "direct": direct,
            "fixed_reflect_1": reflected,
            "learned_gate": candidates[gate_action],
            "learned_gate_acceptance": accepted_candidate,
            "binary_accuracy_oracle": candidates[oracle_action],
        }
        difficulty = str(trace.get("difficulty") or "unknown")
        source = str(trace.get("source") or trace.get("dataset") or "unknown")
        for method, candidate in selected.items():
            _add(overall[method], candidate, lambda_cost)
            _add(by_difficulty[difficulty][method], candidate, lambda_cost)
            _add(by_source[source][method], candidate, lambda_cost)

        gate_actions[gate_action] += 1
        joint_gate_actions[joint_gate_action] += 1
        accepted_actions[accepted_action] += 1
        oracle_actions[oracle_action] += 1
        decisions.append(
            {
                "id": trace.get("id"),
                "gate_action": gate_action,
                "joint_gate_action": joint_gate_action,
                "accepted_action": accepted_action,
                "binary_accuracy_oracle_action": oracle_action,
                "gate_probability": float(
                    diagnostics["gate_probability"][index]
                ),
                "acceptance_probability": float(
                    diagnostics["acceptance_probability"][index]
                ),
                "direct_correct": int(_correct(direct)),
                "reflect_1_correct": int(_correct(reflected)),
                "test_label_used_for_action_selection": False,
            }
        )

    analysis = {
        "n": len(traces),
        "policy_type": bundle.get("policy_type"),
        "max_length": bundle.get("max_length"),
        "training_rows": bundle.get("n_rows"),
        "training_complete": bundle.get("n_complete"),
        "training_positive": bundle.get("n_positive"),
        "training_objective": bundle.get("training_objective"),
        "gate_threshold": bundle.get("gate_threshold"),
        "joint_gate_threshold": bundle.get("joint_gate_threshold"),
        "acceptance_threshold": bundle.get("acceptance_threshold"),
        "gate_oof_metrics": bundle.get("gate_oof_metrics"),
        "combined_oof_metrics": bundle.get("combined_oof_metrics"),
        "binary_accuracy_oracle_oof": bundle.get("binary_accuracy_oracle"),
        "gate_action_counts": dict(gate_actions),
        "joint_gate_action_counts": dict(joint_gate_actions),
        "accepted_action_counts": dict(accepted_actions),
        "oracle_action_counts": dict(oracle_actions),
        "gate_transitions": _transition_counts(traces, gate_reflect),
        "accepted_transitions": _transition_counts(
            traces, accepted_reflect
        ),
        "test_labels_used_for_action_selection": False,
        "llm_calls": 0,
    }
    return (
        _rows_from_stats({"overall": overall}, None, METHODS),
        _rows_from_stats(by_difficulty, "difficulty", METHODS),
        _rows_from_stats(by_source, "source", METHODS),
        analysis,
        decisions,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the accuracy-first QA gate and reflection acceptance "
            "ensemble from saved traces."
        )
    )
    parser.add_argument("--trace", required=True)
    parser.add_argument("--policy_model", required=True)
    parser.add_argument("--gate_cache")
    parser.add_argument("--acceptance_cache")
    parser.add_argument("--lambda_cost", type=float, default=0.0001)
    parser.add_argument("--output", required=True)
    parser.add_argument("--difficulty_output", required=True)
    parser.add_argument("--source_output", required=True)
    parser.add_argument("--analysis_output", required=True)
    parser.add_argument("--decisions_output", required=True)
    args = parser.parse_args()

    results = evaluate(
        read_jsonl(args.trace),
        args.policy_model,
        args.gate_cache,
        args.acceptance_cache,
        args.lambda_cost,
    )
    overall, difficulty, source, analysis, decisions = results
    save_json(overall, args.output)
    save_json(difficulty, args.difficulty_output)
    save_json(source, args.source_output)
    save_json(analysis, args.analysis_output)
    write_jsonl(decisions, args.decisions_output)
    print(
        print_table(
            overall,
            [
                "method",
                "n",
                "accuracy",
                "avg_answer_f1",
                "avg_prompt_tokens",
                "avg_completion_tokens",
                "avg_token_cost",
                "reflection_rate",
                "avg_reward",
            ],
        )
    )
    print(
        f"gate={analysis['gate_transitions']} "
        f"gate_acceptance={analysis['accepted_transitions']}"
    )
    print(
        f"Read {analysis['n']} saved traces; no LLM calls were made."
    )


if __name__ == "__main__":
    main()

