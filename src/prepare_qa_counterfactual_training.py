from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from .utils import read_jsonl, save_json, write_jsonl

ACTIONS = ("direct", "reflect_1", "reflect_2")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _candidates(row: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {
        str(candidate.get("action")): candidate
        for candidate in row.get("candidate_results", [])
    }


def prepare_sample(
    source: Path,
    output: Path,
    manifest_output: Path,
    sample_size: int,
    seed: int,
) -> Dict[str, Any]:
    rows = read_jsonl(source)
    eligible = []
    existing_complete = 0
    for row in rows:
        candidates = _candidates(row)
        if all(action in candidates for action in ACTIONS):
            existing_complete += 1
            continue
        if set(candidates) != {"direct"}:
            raise ValueError(f"Unexpected candidate space for {row.get('id')}: {sorted(candidates)}")
        if float(candidates["direct"].get("correct", 0)) < 0.5:
            raise ValueError(f"Direct-incorrect row lacks reflections: {row.get('id')}")
        eligible.append(row)
    if sample_size > len(eligible):
        raise ValueError(f"Requested {sample_size} rows, but only {len(eligible)} are eligible")

    rng = random.Random(seed)
    selected_ids = {str(row["id"]) for row in rng.sample(eligible, sample_size)}
    selected = [row for row in rows if str(row.get("id")) in selected_ids]
    if len(selected) != sample_size:
        raise AssertionError("Selected id count changed while preserving source order")
    write_jsonl(selected, output)
    selected_hash = hashlib.sha256("\n".join(sorted(selected_ids)).encode()).hexdigest()
    manifest = {
        "operation": "sample_direct_correct_counterfactuals",
        "source": str(source.resolve()),
        "source_sha256": _sha256(source),
        "output": str(output.resolve()),
        "rows_total": len(rows),
        "existing_complete_rows": existing_complete,
        "eligible_direct_correct_rows": len(eligible),
        "sample_size": sample_size,
        "seed": seed,
        "selected_ids_sha256": selected_hash,
        "sampling_order": "random.sample over source-order eligible rows; output restored to source order",
        "test_rows_used": 0,
    }
    save_json(manifest, manifest_output)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def _assert_direct_frozen(base: Dict[str, Any], replacement: Dict[str, Any]) -> None:
    fields = ("question", "initial_answer", "initial_pred", "gold_answer", "task_type")
    for field in fields:
        if base.get(field) != replacement.get(field):
            raise ValueError(f"Frozen field {field} changed for {base.get('id')}")
    base_direct = _candidates(base)["direct"]
    replacement_direct = _candidates(replacement)["direct"]
    for field in ("final_answer", "pred", "prompt_tokens", "completion_tokens", "token_cost"):
        if base_direct.get(field) != replacement_direct.get(field):
            raise ValueError(f"Frozen direct field {field} changed for {base.get('id')}")
    base_call = base.get("initial_call") or {}
    replacement_call = replacement.get("initial_call") or {}
    for field in ("prompt", "output", "prompt_tokens", "completion_tokens", "total_tokens"):
        if base_call.get(field) != replacement_call.get(field):
            raise ValueError(f"Frozen initial_call.{field} changed for {base.get('id')}")


def merge_augmented(
    source: Path,
    completed_sample: Path,
    output: Path,
    manifest_output: Path,
) -> Dict[str, Any]:
    rows = read_jsonl(source)
    replacements = {str(row["id"]): row for row in read_jsonl(completed_sample)}
    if not replacements:
        raise ValueError("Completed counterfactual sample is empty")
    source_ids = {str(row["id"]) for row in rows}
    unknown = sorted(set(replacements) - source_ids)
    if unknown:
        raise ValueError(f"Counterfactual sample contains unknown ids: {unknown[:5]}")

    output_rows: List[Dict[str, Any]] = []
    spaces: Counter[str] = Counter()
    replaced = 0
    for base in rows:
        row_id = str(base["id"])
        row = replacements.get(row_id)
        if row is not None:
            _assert_direct_frozen(base, row)
            candidates = _candidates(row)
            if not all(action in candidates for action in ACTIONS):
                raise ValueError(f"Replacement row is incomplete: {row_id}")
            if float(candidates["direct"].get("correct", 0)) < 0.5:
                raise ValueError(f"Sampled direct-correct row changed label: {row_id}")
            row = dict(row)
            row["counterfactual_training"] = {
                "sampled_direct_correct": True,
                "all_actions_observed": True,
            }
            replaced += 1
        else:
            row = dict(base)
            row["counterfactual_training"] = {
                "sampled_direct_correct": False,
                "all_actions_observed": len(_candidates(row)) == len(ACTIONS),
            }
        actions = tuple(_candidates(row))
        spaces[",".join(actions)] += 1
        output_rows.append(row)
    if replaced != len(replacements):
        raise AssertionError("Not every counterfactual row replaced its source row")
    write_jsonl(output_rows, output)
    manifest = {
        "operation": "merge_training_counterfactuals",
        "source": str(source.resolve()),
        "source_sha256": _sha256(source),
        "completed_sample": str(completed_sample.resolve()),
        "completed_sample_sha256": _sha256(completed_sample),
        "output": str(output.resolve()),
        "rows": len(output_rows),
        "replaced_direct_correct_rows": replaced,
        "candidate_spaces": dict(spaces),
        "test_rows_used": 0,
    }
    save_json(manifest, manifest_output)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare balanced QA counterfactual training data.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    sample = subparsers.add_parser("sample")
    sample.add_argument("--source", required=True)
    sample.add_argument("--output", required=True)
    sample.add_argument("--manifest_output", required=True)
    sample.add_argument("--sample_size", type=int, default=1663)
    sample.add_argument("--seed", type=int, default=42)
    merge = subparsers.add_parser("merge")
    merge.add_argument("--source", required=True)
    merge.add_argument("--completed_sample", required=True)
    merge.add_argument("--output", required=True)
    merge.add_argument("--manifest_output", required=True)
    args = parser.parse_args()
    if args.command == "sample":
        prepare_sample(
            Path(args.source), Path(args.output), Path(args.manifest_output),
            args.sample_size, args.seed,
        )
    else:
        merge_augmented(
            Path(args.source), Path(args.completed_sample), Path(args.output),
            Path(args.manifest_output),
        )


if __name__ == "__main__":
    main()
