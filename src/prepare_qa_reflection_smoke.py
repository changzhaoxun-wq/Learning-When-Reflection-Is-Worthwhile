from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Any

from .utils import read_jsonl, save_json, write_jsonl


REQUIRED_ACTIONS = {"direct", "reflect_1", "reflect_2"}


def _candidate_map(row: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(candidate.get("action")): candidate
        for candidate in row.get("candidate_results") or []
    }


def _is_correct(candidate: dict[str, Any]) -> bool:
    return bool(candidate.get("enhanced_correct", candidate.get("correct", 0)))


def prepare_smoke_rows(
    rows: list[dict[str, Any]],
    per_group: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    complete: list[tuple[int, dict[str, Any], bool]] = []
    for index, row in enumerate(rows):
        candidates = _candidate_map(row)
        if not REQUIRED_ACTIONS.issubset(candidates):
            continue
        complete.append((index, row, _is_correct(candidates["direct"])))

    correct_rows = [item for item in complete if item[2]]
    incorrect_rows = [item for item in complete if not item[2]]
    if len(correct_rows) < per_group or len(incorrect_rows) < per_group:
        raise ValueError(
            "Not enough complete rows for a balanced smoke sample: "
            f"direct_correct={len(correct_rows)}, direct_incorrect={len(incorrect_rows)}, "
            f"requested_per_group={per_group}"
        )

    rng = random.Random(seed)
    selected = rng.sample(correct_rows, per_group) + rng.sample(incorrect_rows, per_group)
    selected.sort(key=lambda item: item[0])

    output: list[dict[str, Any]] = []
    selected_ids: dict[str, list[str]] = {
        "direct_correct": [],
        "direct_incorrect": [],
    }
    for _, source_row, direct_correct in selected:
        row = copy.deepcopy(source_row)
        candidates = _candidate_map(row)
        row["candidate_results"] = [copy.deepcopy(candidates["direct"])]
        row["best_action"] = "direct"
        row["action_space"] = ["direct", "reflect_1", "reflect_2"]
        row["reflection_smoke"] = {
            "source_has_old_reflections": True,
            "stratum": "direct_correct" if direct_correct else "direct_incorrect",
            "seed": seed,
        }
        output.append(row)
        selected_ids[row["reflection_smoke"]["stratum"]].append(str(row.get("id")))

    summary = {
        "source_rows": len(rows),
        "complete_source_rows": len(complete),
        "available_direct_correct": len(correct_rows),
        "available_direct_incorrect": len(incorrect_rows),
        "selected_rows": len(output),
        "selected_per_group": per_group,
        "seed": seed,
        "selected_ids": selected_ids,
        "base_generation_calls": 0,
        "note": "Only the frozen direct candidate is written; reflection candidates must be regenerated.",
    }
    return output, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a balanced frozen-Base QA sample for reflection prompt comparison."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary_output", required=True)
    parser.add_argument("--per_group", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows, summary = prepare_smoke_rows(
        read_jsonl(args.input),
        per_group=args.per_group,
        seed=args.seed,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(rows, output)
    summary.update(input=str(Path(args.input).resolve()), output=str(output.resolve()))
    save_json(summary, Path(args.summary_output))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
