from __future__ import annotations

import argparse
import copy
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Sequence

from .qa_dual_candidate_policy import _candidate_map, _correct
from .utils import read_jsonl, save_json, write_jsonl


def _reward(candidate: Dict[str, Any]) -> float:
    return float(candidate.get("reward", 0.0))


def filter_strict_reflection_fixes(
    input_paths: Sequence[str],
    min_reward_gain: float = 0.0,
) -> tuple[list[Dict[str, Any]], Dict[str, Any]]:
    """Keep one highest-gain exact repair per training question."""

    selected_by_id: dict[str, tuple[tuple[float, float], Dict[str, Any]]] = {}
    eligible_counts: Counter[str] = Counter()
    stats: Counter[str] = Counter()
    source_stats: dict[str, Counter[str]] = defaultdict(Counter)

    for pass_index, input_path in enumerate(input_paths, 1):
        rows = read_jsonl(input_path)
        source = str(Path(input_path))
        seen: set[str] = set()
        for row in rows:
            row_id = str(row.get("id") or "")
            if not row_id:
                raise ValueError(f"Missing id in {source}")
            if row_id in seen:
                raise ValueError(f"Duplicate id in {source}: {row_id}")
            seen.add(row_id)
            stats["input_rows"] += 1
            source_stats[source]["input_rows"] += 1

            candidates = _candidate_map(row)
            direct = candidates.get("direct")
            reflected = candidates.get("reflect_1")
            if direct is None or reflected is None:
                raise ValueError(
                    f"Row {row_id} in {source} must contain direct and reflect_1"
                )

            if _correct(direct):
                stats["rejected_direct_correct"] += 1
                source_stats[source]["rejected_direct_correct"] += 1
                continue
            if not _correct(reflected):
                stats["rejected_reflection_incorrect"] += 1
                source_stats[source]["rejected_reflection_incorrect"] += 1
                continue

            reward_gain = _reward(reflected) - _reward(direct)
            if reward_gain <= float(min_reward_gain):
                stats["rejected_nonpositive_reward_gain"] += 1
                source_stats[source]["rejected_nonpositive_reward_gain"] += 1
                continue

            stats["eligible_instances"] += 1
            source_stats[source]["eligible_instances"] += 1
            eligible_counts[row_id] += 1
            rank = (reward_gain, _reward(reflected))
            previous = selected_by_id.get(row_id)
            if previous is not None and rank <= previous[0]:
                continue

            output = copy.deepcopy(row)
            output["quality_filtered_augmentation"] = {
                "criterion": (
                    "direct incorrect, reflect_1 correct, and reflection reward "
                    "strictly greater than direct reward"
                ),
                "deduplication": "maximum reward gain per question id",
                "source_path": source,
                "source_pass": int(pass_index),
                "reward_gain": float(reward_gain),
                "test_labels_used": False,
            }
            selected_by_id[row_id] = (rank, output)

    selected = [item[1] for item in selected_by_id.values()]
    selected.sort(
        key=lambda row: (
            int((row.get("sampling_selection") or {}).get("source_index", 10**12)),
            str(row.get("id")),
        )
    )
    stability = Counter(eligible_counts.values())
    summary: Dict[str, Any] = {
        "criterion": "strict_standard_correctness_fix_with_positive_reward_gain",
        "min_reward_gain": float(min_reward_gain),
        "input_files": list(input_paths),
        **dict(stats),
        "eligible_unique_ids": len(selected_by_id),
        "duplicate_eligible_instances": int(
            stats["eligible_instances"] - len(selected_by_id)
        ),
        "eligible_pass_count_distribution": {
            str(count): int(number) for count, number in sorted(stability.items())
        },
        "output_rows": len(selected),
        "source_stats": {
            source: dict(counts) for source, counts in source_stats.items()
        },
        "test_labels_used": False,
    }
    return selected, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a deduplicated QA augmentation set containing only strict, "
            "positive-reward reflection fixes from training traces."
        )
    )
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary_output", required=True)
    parser.add_argument("--min_reward_gain", type=float, default=0.0)
    args = parser.parse_args()

    rows, summary = filter_strict_reflection_fixes(
        args.input, min_reward_gain=args.min_reward_gain
    )
    write_jsonl(rows, args.output)
    summary["output"] = args.output
    save_json(summary, args.summary_output)
    print(summary)


if __name__ == "__main__":
    main()
