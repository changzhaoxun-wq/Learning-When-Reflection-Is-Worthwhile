from __future__ import annotations

import argparse
from collections import Counter
from typing import Any, Dict, List

from .reward import compute_reward
from .utils import project_root, read_jsonl, write_jsonl

DEFAULT_ACTIONS = ["direct", "reflect_1"]


def _parse_actions(value: str) -> List[str]:
    actions = [item.strip() for item in value.split(",") if item.strip()]
    if actions != DEFAULT_ACTIONS:
        raise ValueError(f"Binary mode requires exactly: {DEFAULT_ACTIONS}; got {actions}")
    return actions


def convert_rows(
    rows: List[Dict[str, Any]],
    lambda_cost: float,
    actions: List[str],
    objective: str = "reward",
) -> List[Dict[str, Any]]:
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
                key=lambda item: (int(float(item.get("correct", 0))), -int(item.get("token_cost", 0))),
            )
        else:
            best = max(selected, key=lambda item: (float(item["reward"]), -int(item.get("token_cost", 0))))
        new_row = dict(row)
        new_row["candidate_results"] = selected
        new_row["best_action"] = best["action"]
        new_row["action_space"] = actions
        new_row["binary_policy"] = True
        output.append(new_row)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Reuse a multi-action policy dataset as a direct/reflect_1 binary dataset.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--lambda_cost", type=float, default=0.0001)
    parser.add_argument("--actions", default="direct,reflect_1")
    parser.add_argument("--objective", choices=["reward", "accuracy"], default="reward")
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    actions = _parse_actions(args.actions)
    converted = convert_rows(
        rows,
        lambda_cost=args.lambda_cost,
        actions=actions,
        objective=args.objective,
    )
    output = project_root() / args.output if not str(args.output).startswith("/") else args.output
    write_jsonl(converted, output)
    counts = Counter(row["best_action"] for row in converted)
    print(f"Wrote {len(converted)} binary policy examples to {output}")
    print(f"Objective: {args.objective}")
    print(f"Action distribution: {dict(counts)}")


if __name__ == "__main__":
    main()
