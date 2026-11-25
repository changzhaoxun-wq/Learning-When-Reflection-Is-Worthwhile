from __future__ import annotations

import argparse
from collections import defaultdict
from typing import Any, Dict, Iterable, List

from .reward import compute_utility_reward
from .train_policy import predict_action
from .utils import load_pickle, print_table, project_root, read_jsonl, save_json

# Import the class before unpickling actor policies so pickle can resolve it.
from .actor_policy import ActorMLP  # noqa: F401
from .transformer_policy import TorchPolicyMLP  # noqa: F401


FIXED_ACTIONS = {
    "direct": "direct",
    "fixed_reflect_1": "reflect_1",
    "fixed_reflect_2": "reflect_2",
}
DEFAULT_ACTIONS = ["direct", "reflect_1", "reflect_2"]


def _parse_actions(value: str) -> List[str]:
    actions = [item.strip() for item in value.split(",") if item.strip()]
    if "direct" not in actions or len(actions) < 2:
        raise ValueError("Action space must include direct and at least one reflection action")
    invalid = [action for action in actions if action not in DEFAULT_ACTIONS]
    if invalid:
        raise ValueError(f"Unknown actions: {invalid}")
    return actions


def _method_for_action(action: str) -> str:
    return "direct" if action == "direct" else f"fixed_{action}"


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _candidate_usage(candidate: Dict[str, Any]) -> tuple[float, float, float]:
    prompt = candidate.get("prompt_tokens")
    completion = candidate.get("completion_tokens")
    metadata = candidate.get("metadata") or {}
    initial = metadata.get("initial_usage") or {}
    rounds = metadata.get("reflection_round_usage") or []
    if prompt is None:
        prompt = _float(initial.get("prompt_tokens"))
        prompt += sum(_float(item.get("prompt_tokens")) for item in rounds)
    if completion is None:
        completion = _float(initial.get("completion_tokens"))
        completion += sum(_float(item.get("completion_tokens")) for item in rounds)
    total = _float(candidate.get("token_cost"), _float(prompt) + _float(completion))
    return _float(prompt), _float(completion), total


def _policy_row(trace: Dict[str, Any]) -> Dict[str, Any]:
    # Deliberately exclude gold_answer, candidate_results, correct, reward and
    # oracle_action: these fields are only for offline scoring.
    return {
        "question": trace.get("question", ""),
        "initial_answer": trace.get("initial_answer", ""),
        "initial_pred": trace.get("initial_pred"),
        "state_features": trace.get("state_features", {}),
        "task_type": trace.get("task_type", "math"),
        "difficulty": trace.get("difficulty"),
        "difficulty_score": trace.get("difficulty_score", 0.0),
    }


def _candidate_utility(candidate: Dict[str, Any]) -> float:
    if candidate.get("qa_utility_score") is not None:
        return _float(candidate.get("qa_utility_score"))
    if candidate.get("enhanced_answer_f1") is not None:
        return _float(candidate.get("enhanced_answer_f1"))
    return _float(candidate.get("correct"))


def _candidate_reward(candidate: Dict[str, Any], lambda_cost: float) -> float:
    return compute_utility_reward(
        _candidate_utility(candidate),
        int(round(_candidate_usage(candidate)[2])),
        lambda_cost,
    )


def _add(stats: Dict[str, float], candidate: Dict[str, Any], lambda_cost: float) -> None:
    prompt, completion, total = _candidate_usage(candidate)
    correct = _float(candidate.get("correct"))
    utility = _candidate_utility(candidate)
    # Recompute from the stored utility so lambda sweeps remain valid.
    reward = compute_utility_reward(utility, int(round(total)), lambda_cost)
    stats["n"] += 1
    stats["correct"] += correct
    stats["prompt_tokens"] += prompt
    stats["completion_tokens"] += completion
    stats["token_cost"] += total
    stats["reflection"] += float(_float(candidate.get("reflected_rounds")) > 0)
    stats["reward"] += reward
    stats["utility"] += utility
    stats["exact_match"] += _float(candidate.get("exact_match"))
    stats["answer_f1"] += _float(candidate.get("answer_f1"))


def _rows_from_stats(stats: Dict[str, Dict[str, float]], include_group: str | None, methods: List[str]) -> List[Dict[str, Any]]:
    output = []
    for group in sorted(stats):
        for method in methods:
            values = stats[group].get(method)
            if not values or not values["n"]:
                continue
            n = values["n"]
            row: Dict[str, Any] = {
                "method": method,
                "n": int(n),
                "accuracy": values["correct"] / n,
                "avg_prompt_tokens": values["prompt_tokens"] / n,
                "avg_completion_tokens": values["completion_tokens"] / n,
                "avg_token_cost": values["token_cost"] / n,
                "reflection_rate": values["reflection"] / n,
                "avg_reward": values["reward"] / n,
                "avg_policy_utility": values["utility"] / n,
                "avg_exact_match": values["exact_match"] / n,
                "avg_answer_f1": values["answer_f1"] / n,
            }
            if include_group:
                row[include_group] = group
            output.append(row)
    return output


def evaluate_offline(trace_rows: List[Dict[str, Any]], policy_model: str, lambda_cost: float, actions: List[str] | None = None) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    bundle = load_pickle(policy_model)
    actions = actions or list(bundle.get("action_labels") or DEFAULT_ACTIONS)
    methods = [_method_for_action(action) for action in actions] + ["learned_policy", "oracle"]
    overall: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    by_difficulty: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    by_source: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))

    for trace in trace_rows:
        candidates = {
            str(candidate.get("action")): candidate
            for candidate in (trace.get("candidate_results") or [])
        }
        if not candidates:
            raise ValueError(f"Trace has no candidate_results: {trace.get('id')}")
        missing = [action for action in actions if action not in candidates]
        if missing:
            raise ValueError(f"Trace {trace.get('id')} is missing actions: {missing}")

        learned_action = predict_action(bundle, _policy_row(trace))
        if learned_action not in actions:
            learned_action = "direct"
        # Recompute the oracle for the requested lambda_cost.
        oracle_action = max(
            [candidates[action] for action in actions],
            key=lambda item: (
                _candidate_reward(item, lambda_cost),
                -_candidate_usage(item)[2],
            ),
        ).get("action")
        selected = {_method_for_action(action): candidates[action] for action in actions}
        selected["learned_policy"] = candidates.get(learned_action, candidates["direct"])
        selected["oracle"] = candidates[str(oracle_action)]
        difficulty = str(trace.get("difficulty") or "unknown")
        source = str(trace.get("source") or trace.get("dataset") or "unknown")
        for method, candidate in selected.items():
            _add(overall[method], candidate, lambda_cost)
            _add(by_difficulty[difficulty][method], candidate, lambda_cost)
            _add(by_source[source][method], candidate, lambda_cost)

    return (
        _rows_from_stats({"overall": overall}, None, methods),
        _rows_from_stats(by_difficulty, "difficulty", methods),
        _rows_from_stats(by_source, "source", methods),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate policies from saved LLM traces without new LLM calls.")
    parser.add_argument("--trace", required=True)
    parser.add_argument("--policy_model", required=True)
    parser.add_argument("--lambda_cost", type=float, default=0.0001)
    parser.add_argument("--actions", default=",".join(DEFAULT_ACTIONS), help="Comma-separated evaluation actions")
    parser.add_argument("--output", default=str(project_root() / "outputs" / "offline_eval_results.json"))
    parser.add_argument("--difficulty_output", default=str(project_root() / "outputs" / "offline_eval_results_by_difficulty.json"))
    parser.add_argument("--source_output", default=str(project_root() / "outputs" / "offline_eval_results_by_source.json"))
    args = parser.parse_args()

    traces = read_jsonl(args.trace)
    rows, difficulty_rows, source_rows = evaluate_offline(
        traces,
        policy_model=args.policy_model,
        lambda_cost=args.lambda_cost,
        actions=_parse_actions(args.actions),
    )
    save_json(rows, args.output)
    save_json(difficulty_rows, args.difficulty_output)
    save_json(source_rows, args.source_output)
    columns = [
        "method", "n", "accuracy", "avg_prompt_tokens",
        "avg_completion_tokens", "avg_token_cost",
        "reflection_rate", "avg_reward",
    ]
    print(print_table(rows, columns))
    print(f"Read {len(traces)} saved traces; no LLM calls were made.")
    print(f"Wrote offline results to {args.output}")


if __name__ == "__main__":
    main()
