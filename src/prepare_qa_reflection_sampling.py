from __future__ import annotations

import argparse
import copy
import hashlib
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence

from .prompts import QA_REFLECTION_PROMPT_ROUND_1, QA_REFLECTION_PROMPT_VERSION
from .qa_dual_candidate_policy import _candidate_map, _correct
from .utils import load_json, read_jsonl, save_json, write_jsonl


def select_unrepaired_direct_errors(
    old_rows: Sequence[Dict[str, Any]],
    new_rows: Sequence[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], Dict[str, int]]:
    if len(old_rows) != len(new_rows):
        raise ValueError(
            f"Trace lengths differ: old={len(old_rows)} new={len(new_rows)}"
        )
    selected: List[Dict[str, Any]] = []
    stats: Counter[str] = Counter()
    for index, (old_row, new_row) in enumerate(zip(old_rows, new_rows)):
        old_id = str(old_row.get("id"))
        new_id = str(new_row.get("id"))
        if old_id != new_id:
            raise ValueError(
                f"Trace IDs differ at row {index}: old={old_id} new={new_id}"
            )
        old_candidates = _candidate_map(old_row)
        new_candidates = _candidate_map(new_row)
        direct = old_candidates.get("direct")
        if direct is None:
            raise ValueError(f"Trace {old_id} has no direct candidate")
        stats["rows"] += 1
        if _correct(direct):
            stats["direct_correct"] += 1
            continue
        stats["direct_incorrect"] += 1
        existing = [
            old_candidates.get("reflect_1"),
            old_candidates.get("reflect_2"),
            new_candidates.get("reflect_1"),
        ]
        if any(_correct(candidate) for candidate in existing):
            stats["already_repaired"] += 1
            continue
        output = copy.deepcopy(old_row)
        output["candidate_results"] = [copy.deepcopy(direct)]
        output["action_space"] = ["direct", "reflect_1"]
        output["best_action"] = "direct"
        output["sampling_selection"] = {
            "criterion": (
                "direct incorrect and not repaired by v5 reflect_1, "
                "v5 reflect_2, or v7 reflect_1"
            ),
            "source_index": index,
        }
        selected.append(output)
        stats["selected_unrepaired"] += 1
    return selected, dict(stats)


def build_sampling_configs(
    base_config: Dict[str, Any],
    seeds: Sequence[int],
    temperature: float,
    top_p: float,
    output_dir: Path,
) -> List[Path]:
    config_paths = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for pass_index, seed in enumerate(seeds, 1):
        config = copy.deepcopy(base_config)
        reflector = config.get("reflector_llm") or config.get("llm")
        if reflector is None:
            raise ValueError("Base config has no reflector_llm or llm section")
        reflector["temperature"] = float(temperature)
        reflector["top_p"] = float(top_p)
        reflector["seed"] = int(seed)
        protocol = config.setdefault("experiment_protocol", {})
        protocol.update(
            {
                "reflection_sampling_augmentation": True,
                "qa_reflection_prompt_version": QA_REFLECTION_PROMPT_VERSION,
                "qa_reflection_prompt_sha256": hashlib.sha256(
                    QA_REFLECTION_PROMPT_ROUND_1.encode("utf-8")
                ).hexdigest(),
                "reflection_sampling_pass": int(pass_index),
                "reflection_sampling_seed": int(seed),
                "reflection_sampling_temperature": float(temperature),
                "reflection_sampling_top_p": float(top_p),
                "test_time_candidate": False,
            }
        )
        path = output_dir / f"sample_{pass_index}_config.json"
        save_json(config, path)
        config_paths.append(path)
    return config_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Select unresolved QA direct errors and prepare stochastic reflector "
            "sampling configs without changing the reflection prompt."
        )
    )
    parser.add_argument("--old_trace", required=True)
    parser.add_argument("--new_trace", required=True)
    parser.add_argument("--base_config", required=True)
    parser.add_argument("--output_subset", required=True)
    parser.add_argument("--config_dir", required=True)
    parser.add_argument("--summary_output", required=True)
    parser.add_argument("--seeds", default="101,202,303")
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--top_p", type=float, default=0.9)
    args = parser.parse_args()

    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    if not seeds:
        raise ValueError("At least one sampling seed is required")
    selected, stats = select_unrepaired_direct_errors(
        read_jsonl(args.old_trace), read_jsonl(args.new_trace)
    )
    write_jsonl(selected, args.output_subset)
    config_paths = build_sampling_configs(
        load_json(args.base_config),
        seeds,
        args.temperature,
        args.top_p,
        Path(args.config_dir),
    )
    summary = {
        **stats,
        "sampling_passes": len(seeds),
        "seeds": seeds,
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "reflection_prompt_changed": False,
        "config_paths": [str(path) for path in config_paths],
        "output_subset": args.output_subset,
    }
    save_json(summary, args.summary_output)
    print(summary)


if __name__ == "__main__":
    main()
