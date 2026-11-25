from __future__ import annotations

import argparse
from collections import Counter
from typing import Any, Dict, List

from .reward import compute_reward
from .utils import project_root, read_jsonl, write_jsonl


DEFAULT_ACTIONS = ["direct", "reflect_1", "reflect_2"]


def _parse_actions(value: str) -> List[str]:
    actions = [item.strip() for item in value.split(",") if item.strip()]
    if "direct" not in actions or len(actions) < 2:
        raise ValueError("Action space must include direct and at least one reflection action")
    invalid = [action for action in actions if action not in DEFAULT_ACTIONS]
    if invalid:
        raise ValueError(f"Unknown actions: {invalid}")
    if len(set(actions)) != len(actions):
        raise ValueError(f"Duplicate actions are not allowed: {actions}")
    return actions


def convert_rows(
    rows: List[Dict[str, Any]],
    lambda_cost: float,
    actions: List[str],
    objective: str = "accuracy",
) -> List[Dict[str, Any]]:
    """Re-label saved candidate results without making new model calls.

    Accuracy objective first prefers a correct candidate, then prefers the
    lower-cost candidate among candidates with the same correctness. Reward
    remains recomputed and stored for later cost analysis.
    """
    if objective not in {"reward", "accuracy"}:
        raise ValueError("objective must be reward or accuracy")

    output = []
    for row in rows:
        candidates = {str(item.get("action")): dict(item) for item in row.get("candidate_results", [])}
        missing = [action for action in actions if action not in candidates]
        if missing:
            raise ValueError(f"Row {row.get('id')} is missing candidates: {missing}")

        selected = []
        for action in actions:
            item = candidates[action]
            correct = int(float(item.get("correct", 0)))
            token_cost = int(float(item.get("token_cost", 0)))
            item["reward"] = compute_reward(correct, token_cost, lambda_cost)
            selected.append(item)

        if objective == "accuracy":
            best = max(
                selected,
                key=lambda item: (
                    int(float(item.get("correct", 0))),
                    -int(float(item.get("token_cost", 0))),
                ),
            )
        else:
            best = max(
                selected,
                key=lambda item: (
                    float(item["reward"]),
                    -int(float(item.get("token_cost", 0))),
                ),
            )

        new_row = dict(row)
        new_row["candidate_results"] = selected
        new_row["best_action"] = best["action"]
        new_row["action_space"] = list(actions)
        new_row["policy_objective"] = objective
        new_row["binary_policy"] = len(actions) == 2
        output.append(new_row)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-label a saved multi-action policy dataset without new LLM calls."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--lambda_cost", type=float, default=0.0001)
    parser.add_argument("--actions", default=",".join(DEFAULT_ACTIONS))
    parser.add_argument("--objective", choices=["reward", "accuracy"], default="accuracy")
    args = parser.parse_args()

    actions = _parse_actions(args.actions)
    rows = read_jsonl(args.input)
    converted = convert_rows(
        rows,
        lambda_cost=args.lambda_cost,
        actions=actions,
        objective=args.objective,
    )
    output = project_root() / args.output if not str(args.output).startswith("/") else args.output
    write_jsonl(converted, output)
    counts = Counter(row["best_action"] for row in converted)
    mode = "binary" if len(actions) == 2 else "multi-action"
    print(f"Wrote {len(converted)} {mode} policy examples to {output}")
    print(f"Actions: {actions}")
    print(f"Objective: {args.objective}")
    print(f"Action distribution: {dict(counts)}")


if __name__ == "__main__":
    main()
