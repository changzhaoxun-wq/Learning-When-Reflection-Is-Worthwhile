"""Prepare outcome-independent math pilot data; no generation or model download."""

import argparse
import hashlib
import json
import os
import unicodedata
from pathlib import Path

from .audit_trace_completion import audit
from .gsm8k_utils import normalize_example_item
from .prepare_qa_hard_development import file_hash, locked_write
from .prompts import MATH_DIRECT_PROMPT, MATH_REFLECTION_PROMPT_ROUND_1, MATH_REFLECTION_PROMPT_ROUND_2
from .utils import read_jsonl


PROJECT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("MATH_DATA_ROOT", PROJECT / "data"))


def question_key(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def select_rows(train, test, size, seed):
    if size < 1:
        raise ValueError("Positive sample size required")
    test_ids = {r["id"] for r in test}
    test_questions = {question_key(r["question"]) for r in test}
    seen_ids, seen_questions, eligible = set(), set(), []
    excluded = []
    for row in train:
        key = question_key(row["question"])
        if row["id"] in seen_ids:
            raise ValueError("Duplicate training ID")
        seen_ids.add(row["id"])
        if row["id"] in test_ids or key in test_questions or key in seen_questions:
            excluded.append(row["id"])
            continue
        seen_questions.add(key)
        eligible.append(row)
    if len(eligible) < size:
        raise ValueError("Not enough eligible training questions")
    ordered = sorted(eligible, key=lambda r: hashlib.sha256(f"{seed}:{r['id']}".encode()).hexdigest())
    return ordered[:size], excluded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT / "outputs/math_reflector_pilot_v1")
    parser.add_argument("--size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260910)
    args = parser.parse_args()
    root = args.root.resolve()
    train = [normalize_example_item(r, i, "train") for i, r in enumerate(read_jsonl(DATA / "train.jsonl"))]
    test = [normalize_example_item(r, i, "test") for i, r in enumerate(read_jsonl(DATA / "test.jsonl"))]
    selected, excluded = select_rows(train, test, args.size, args.seed)
    root.mkdir(parents=True, exist_ok=True)
    locked_write(root / "development.jsonl", "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in selected))
    snapshots = {"direct": MATH_DIRECT_PROMPT, "legacy_independent_round_1": MATH_REFLECTION_PROMPT_ROUND_1,
                 "legacy_independent_round_2": MATH_REFLECTION_PROMPT_ROUND_2}
    for name, text in snapshots.items():
        locked_write(root / "prompts" / f"{name}.txt", text + "\n")
    old_trace = PROJECT / "outputs/gsm8k_no_think_rebuild/direct_train.jsonl"
    result = audit(old_trace)
    locked_write(root / "old_base_audit.json", json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    manifest = {
        "role": "training-side exploratory pilot; not a new independent test",
        "sampling": "sha256(seed:id), independent of gold, correctness, confidence, and reward",
        "seed": args.seed, "size": len(selected), "selected_ids": [r["id"] for r in selected],
        "train_test_overlap_or_duplicate_excluded_ids": excluded,
        "input_sha256": {str(DATA / name): file_hash(DATA / name) for name in ("train.jsonl", "test.jsonl")},
        "development_sha256": file_hash(root / "development.jsonl"),
        "old_base_trace_sha256": file_hash(old_trace),
        "base_model": os.environ.get("BASE_MODEL", "Qwen/Qwen3-8B-Instruct"),
        "framework": ["direct", "reflect_1", "reflect_2"],
        "policy_architecture_changed": False, "global_prompts_changed": False,
        "legacy_prompt_note": "v4 hides the previous answer; label as independent re-solving, not answer-conditioned reflection",
        "next_model_candidate": "Qwen/Qwen3-14B",
        "model_choice_status": "proposed, pending user confirmation and download",
        "generation_started": False, "llm_calls": 0,
        "requirements": ["Freeze one Base response per question across all reflector arms",
                         "Run all actions on every pilot question, not only Direct errors",
                         "Compare old and new reflectors with the same answer-conditioned template",
                         "Separate independent re-solving control from reflection",
                         "Audit unresolved truncation and all prompt/completion/recovery costs",
                         "No selection or tuning on the official test outcomes"],
    }
    locked_write(root / "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"root": str(root), "rows": len(selected), "old_base_truncated": result["final_truncated_count"],
                      "legacy_prompt_uses_current_answer": "{current_answer}" in MATH_REFLECTION_PROMPT_ROUND_1,
                      "generation_started": False}, indent=2))


if __name__ == "__main__":
    main()
