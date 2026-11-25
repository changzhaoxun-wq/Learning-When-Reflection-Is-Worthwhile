from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from src.reward import compute_reward
from src.utils import read_jsonl, write_jsonl


def _pick(items: List[Dict[str, Any]], objective: str, lambda_cost: float) -> Dict[str, Any]:
    def key(item: Dict[str, Any]) -> tuple[float, float]:
        correct = int(float(item.get("correct", 0)))
        cost = float(item.get("token_cost", 0))
        if objective == "accuracy":
            return float(correct), -cost
        if objective == "stored_reward":
            return float(item.get("reward", -1e9)), -cost
        return compute_reward(correct, int(cost), lambda_cost), -cost

    return max(items, key=key)


def build(rows: List[Dict[str, Any]], objective: str, lambda_cost: float) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    gate_rows: List[Dict[str, Any]] = []
    round_rows: List[Dict[str, Any]] = []
    for row in rows:
        candidates = {str(x.get("action")): dict(x) for x in row.get("candidate_results", [])}
        if "direct" not in candidates:
            raise ValueError(f"Row {row.get('id')} is missing the direct candidate")
        direct = candidates["direct"]
        missing_reflections = [
            action for action in ("reflect_1", "reflect_2")
            if action not in candidates
        ]
        if missing_reflections:
            if int(float(direct.get("correct", 0))) != 1:
                raise ValueError(
                    f"Row {row.get('id')} has an incorrect direct answer but is missing "
                    f"reflection candidates: {missing_reflections}"
                )
            gate_row = dict(row)
            gate_row["candidate_results"] = [dict(direct)]
            gate_row["action_space"] = ["direct", "reflect_1"]
            gate_row["best_action"] = "direct"
            gate_row["policy_stage"] = "reflection_gate"
            gate_row["hierarchical_policy"] = True
            gate_row["selective_training_row"] = True
            gate_rows.append(gate_row)
            continue

        reflect_best = _pick([candidates["reflect_1"], candidates["reflect_2"]], objective, lambda_cost)

        gate_reflect = dict(reflect_best)
        gate_reflect["action"] = "reflect_1"
        gate_reflect["source_action"] = reflect_best["action"]
        gate_candidates = [dict(direct), gate_reflect]
        gate_best = _pick(gate_candidates, objective, lambda_cost)
        gate_row = dict(row)
        gate_row["candidate_results"] = gate_candidates
        gate_row["action_space"] = ["direct", "reflect_1"]
        gate_row["best_action"] = gate_best["action"]
        gate_row["policy_stage"] = "reflection_gate"
        gate_row["hierarchical_policy"] = True
        gate_rows.append(gate_row)

        round_candidates = [dict(candidates["reflect_1"]), dict(candidates["reflect_2"])]
        round_best = _pick(round_candidates, objective, lambda_cost)
        round_row = dict(row)
        round_row["candidate_results"] = round_candidates
        round_row["action_space"] = ["reflect_1", "reflect_2"]
        round_row["best_action"] = round_best["action"]
        round_row["policy_stage"] = "reflection_round"
        round_row["hierarchical_policy"] = True
        round_rows.append(round_row)
    return gate_rows, round_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build hierarchical policy datasets from saved three-action traces.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--objective", choices=["accuracy", "reward", "stored_reward"], default="accuracy")
    parser.add_argument("--lambda_cost", type=float, default=0.0001)
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    gate_rows, round_rows = build(rows, args.objective, args.lambda_cost)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    gate_path = output_dir / "gate.jsonl"
    round_path = output_dir / "round.jsonl"
    write_jsonl(gate_rows, gate_path)
    write_jsonl(round_rows, round_path)
    print(f"Wrote gate dataset: {gate_path} ({len(gate_rows)} rows)")
    print(f"Gate labels: {dict(Counter(row['best_action'] for row in gate_rows))}")
    print(f"Wrote round dataset: {round_path} ({len(round_rows)} rows)")
    print(f"Round labels: {dict(Counter(row['best_action'] for row in round_rows))}")


if __name__ == "__main__":
    main()
