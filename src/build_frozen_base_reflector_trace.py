import argparse
import copy
import hashlib
import json
import sys
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from itertools import zip_longest
from pathlib import Path
from typing import Any, Dict, Iterator, TextIO

from .answer_extractor import extract_answer
from .llm_client import BaseLLMClient, build_llm_client
from .prompts import reflection_prompt, reflection_prompt_version
from .qa_reflection_profile import load_profile, profile_hash, profile_prompt
from .reward import compute_match_metrics, compute_reward
from .strategies import execute_strategy
from .utils import load_json


REFLECTION_ACTIONS = ("reflect_1", "reflect_2")


def _progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc


def _completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {str(row["id"]) for row in _read_jsonl(path)}


def _candidate_by_action(
    row: Dict[str, Any],
    required_actions: tuple[str, ...] = ("direct", *REFLECTION_ACTIONS),
) -> Dict[str, Dict[str, Any]]:
    candidates = row.get("candidate_results") or []
    by_action = {str(item.get("action")): item for item in candidates}
    missing = [action for action in required_actions if action not in by_action]
    if missing:
        raise ValueError(f"Row {row.get('id')} is missing candidates: {missing}")
    return by_action


def _initial_usage(row: Dict[str, Any]) -> Dict[str, int]:
    call = row.get("initial_call") or {}
    required = ("prompt_tokens", "completion_tokens", "total_tokens")
    if not all(key in call for key in required):
        raise ValueError(f"Row {row.get('id')} has no complete initial_call usage")
    return {key: int(call[key]) for key in required}


def _candidate_is_reusable(
    candidate: Dict[str, Any],
    canonical_row: Dict[str, Any],
    qa_profile: Dict[str, Any] | None = None,
) -> bool:
    action = str(candidate.get("action"))
    expected_rounds = 1 if action == "reflect_1" else 2 if action == "reflect_2" else 0
    calls = candidate.get("metadata", {}).get("calls") or []
    if len(calls) != expected_rounds + 1:
        return False

    # A length-stopped reflection has no reliable final answer. Reuse complete
    # candidates, but regenerate the whole action when any reflection round was truncated.
    for call in calls[1:]:
        finish_reason = call.get("finish_reason") or call.get("response_metadata", {}).get("finish_reason")
        if finish_reason == "length":
            return False

    initial_call = canonical_row.get("initial_call") or {}
    if calls[0].get("prompt") != initial_call.get("prompt"):
        return False
    if calls[0].get("output") != canonical_row.get("initial_answer"):
        return False

    previous_answer = canonical_row.get("initial_answer", "")
    task_type = canonical_row.get("task_type", "math")
    for round_id in range(1, expected_rounds + 1):
        expected_prompt = reflection_prompt(
            question=canonical_row["question"],
            current_answer=previous_answer,
            round_id=round_id,
            task_type=task_type,
        )
        if qa_profile and task_type == "qa":
            expected_prompt = profile_prompt(qa_profile, canonical_row["question"], previous_answer, round_id)
            if calls[round_id].get("response_metadata", {}).get("qa_profile_sha256") != profile_hash(qa_profile):
                return False
        if calls[round_id].get("prompt") != expected_prompt:
            return False
        previous_answer = calls[round_id].get("output", "")
    return True


def _freeze_candidate_initial_usage(
    candidate: Dict[str, Any],
    canonical_row: Dict[str, Any],
) -> Dict[str, Any]:
    result = copy.deepcopy(candidate)
    usage = _initial_usage(canonical_row)
    initial_call = canonical_row["initial_call"]
    metadata = result.setdefault("metadata", {})
    calls = metadata.get("calls") or []
    calls[0] = {
        "round_id": 0,
        "prompt": initial_call["prompt"],
        "output": canonical_row["initial_answer"],
        **usage,
    }
    metadata["calls"] = calls
    metadata["initial_usage"] = dict(usage)

    round_usage = metadata.get("reflection_round_usage") or []
    reflection_prompt_tokens = sum(int(item.get("prompt_tokens", 0)) for item in round_usage)
    reflection_completion_tokens = sum(int(item.get("completion_tokens", 0)) for item in round_usage)
    reflection_total_tokens = sum(int(item.get("total_tokens", 0)) for item in round_usage)
    result["prompt_tokens"] = usage["prompt_tokens"] + reflection_prompt_tokens
    result["completion_tokens"] = usage["completion_tokens"] + reflection_completion_tokens
    result["token_cost"] = usage["total_tokens"] + reflection_total_tokens
    return result


def _score_candidate(
    candidate: Dict[str, Any],
    row: Dict[str, Any],
    lambda_cost: float,
    qa_f1_threshold: float,
) -> Dict[str, Any]:
    result = copy.deepcopy(candidate)
    result["pred"] = extract_answer(
        result.get("final_answer"),
        task_type=row.get("task_type", "math"),
    )
    metrics = compute_match_metrics(
        result.get("pred"),
        row["gold_answer"],
        task_type=row.get("task_type", "math"),
        gold_aliases=row.get("gold_aliases", []),
        qa_f1_threshold=qa_f1_threshold,
    )
    result["correct"] = int(metrics["correct"])
    result["reward"] = compute_reward(result["correct"], int(result["token_cost"]), lambda_cost)
    if str(row.get("task_type", "math")).lower() == "qa":
        result.update(
            exact_match=metrics.get("exact_match"),
            answer_f1=metrics.get("answer_f1"),
            matched_gold=metrics.get("matched_gold"),
        )
    return result


def _generate_candidate(
    action: str,
    row: Dict[str, Any],
    reflector: BaseLLMClient,
    lambda_cost: float,
    qa_f1_threshold: float,
    qa_profile: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    usage = _initial_usage(row)
    result = execute_strategy(
        action=action,
        llm=reflector,
        question=row["question"],
        initial_answer=row["initial_answer"],
        initial_token_cost=usage["total_tokens"],
        initial_prompt_tokens=usage["prompt_tokens"],
        initial_completion_tokens=usage["completion_tokens"],
        initial_prompt=row["initial_call"]["prompt"],
        gold_answer=row["gold_answer"],
        task_type=row.get("task_type", "math"),
        qa_profile=qa_profile,
    ).to_dict()
    result.setdefault("metadata", {})["initial_usage"] = dict(usage)
    return _score_candidate(result, row, lambda_cost, qa_f1_threshold)


def _strict_row(
    canonical_row: Dict[str, Any],
    target_row: Dict[str, Any] | None,
    reflector: BaseLLMClient | None,
    lambda_cost: float,
    qa_f1_threshold: float,
    dry_run: bool,
    incorrect_only: bool,
    reflection_actions: tuple[str, ...] = REFLECTION_ACTIONS,
    canonical_reflections_only: bool = False,
    qa_profile: Dict[str, Any] | None = None,
) -> tuple[Dict[str, Any] | None, Dict[str, str]]:
    if target_row is not None and canonical_row.get("id") != target_row.get("id"):
        raise ValueError(
            f"Trace alignment error: canonical id={canonical_row.get('id')} "
            f"target id={target_row.get('id')}"
        )
    if target_row is not None and canonical_row.get("question") != target_row.get("question"):
        raise ValueError(f"Question mismatch for id={canonical_row.get('id')}")

    canonical_candidates = _candidate_by_action(canonical_row, required_actions=("direct",))
    decisions: Dict[str, str] = {}
    selected: Dict[str, Dict[str, Any]] = {
        "direct": _score_candidate(
            copy.deepcopy(canonical_candidates["direct"]),
            canonical_row,
            lambda_cost,
            qa_f1_threshold,
        )
    }

    direct_correct = bool(selected["direct"]["correct"])
    canonical_action_names = {
        str(candidate.get("action"))
        for candidate in canonical_row.get("candidate_results", [])
    }
    skip_reason = None
    if incorrect_only and direct_correct:
        skip_reason = "skipped_direct_correct"
    elif canonical_reflections_only and not all(
        action in canonical_action_names for action in reflection_actions
    ):
        skip_reason = "skipped_not_selected"
    if skip_reason is not None:
        decisions = {action: skip_reason for action in reflection_actions}
        if dry_run:
            return None, decisions
        result = copy.deepcopy(canonical_row)
        result["candidate_results"] = [selected["direct"]]
        result["action_space"] = ["direct", *reflection_actions]
        result["best_action"] = "direct"
        result["reflection_prompt_version"] = qa_profile["version"] if qa_profile else reflection_prompt_version(
            result.get("task_type", "math")
        )
        result["selective_reflection"] = {
            "enabled": True,
            "criterion": (
                "direct_correct == 0"
                if incorrect_only
                else "canonical_has_requested_reflections"
            ),
            "direct_correct": direct_correct,
            "reflection_skipped": True,
            "reason": skip_reason,
        }
        result["strict_decoupling"] = {
            "base_frozen": True,
            "base_prompt_sha256": hashlib.sha256(
                result["initial_call"]["prompt"].encode("utf-8")
            ).hexdigest(),
            "reflection_candidates": decisions,
        }
        if qa_profile:
            result["qa_profile_sha256"] = profile_hash(qa_profile)
        return result, decisions

    target_candidates = (
        _candidate_by_action(
            target_row,
            required_actions=("direct", *reflection_actions),
        )
        if target_row is not None
        else {}
    )
    for action in reflection_actions:
        if action in target_candidates and _candidate_is_reusable(target_candidates[action], canonical_row, qa_profile):
            decisions[action] = "reused"
            if not dry_run:
                candidate = _freeze_candidate_initial_usage(target_candidates[action], canonical_row)
                selected[action] = _score_candidate(
                    candidate,
                    canonical_row,
                    lambda_cost,
                    qa_f1_threshold,
                )
        else:
            decisions[action] = "regenerated"
            if not dry_run:
                if reflector is None:
                    raise RuntimeError("Reflector client is required to regenerate candidates")
                selected[action] = _generate_candidate(
                    action,
                    canonical_row,
                    reflector,
                    lambda_cost,
                    qa_f1_threshold,
                    qa_profile,
                )

    if dry_run:
        return None, decisions

    result = copy.deepcopy(canonical_row)
    result["candidate_results"] = [
        selected[action] for action in ("direct", *reflection_actions)
    ]
    result["action_space"] = ["direct", *reflection_actions]
    result["reflection_prompt_version"] = qa_profile["version"] if qa_profile else reflection_prompt_version(
        result.get("task_type", "math")
    )
    result["best_action"] = max(
        result["candidate_results"],
        key=lambda item: (item["reward"], -item["token_cost"]),
    )["action"]
    result["selective_reflection"] = {
        "enabled": bool(incorrect_only or canonical_reflections_only),
        "criterion": (
            "direct_correct == 0"
            if incorrect_only
            else (
                "canonical_has_requested_reflections"
                if canonical_reflections_only
                else "all_examples"
            )
        ),
        "direct_correct": direct_correct,
        "reflection_skipped": False,
    }
    result["strict_decoupling"] = {
        "base_frozen": True,
        "base_prompt_sha256": hashlib.sha256(
            result["initial_call"]["prompt"].encode("utf-8")
        ).hexdigest(),
        "reflection_candidates": decisions,
    }
    if qa_profile:
        result["qa_profile_sha256"] = profile_hash(qa_profile)
    return result, decisions


def _process_trace_pair(
    index: int,
    canonical_row: Dict[str, Any],
    target_row: Dict[str, Any] | None,
    reflector: BaseLLMClient | None,
    lambda_cost: float,
    qa_f1_threshold: float,
    dry_run: bool,
    incorrect_only: bool,
    reflection_actions: tuple[str, ...] = REFLECTION_ACTIONS,
    canonical_reflections_only: bool = False,
    qa_profile: Dict[str, Any] | None = None,
) -> tuple[int, str, Dict[str, Any] | None, Dict[str, str], bool, bool]:
    probe_row, decisions = _strict_row(
        canonical_row,
        target_row,
        reflector=None,
        lambda_cost=lambda_cost,
        qa_f1_threshold=qa_f1_threshold,
        dry_run=True,
        incorrect_only=incorrect_only,
        reflection_actions=reflection_actions,
        canonical_reflections_only=canonical_reflections_only,
        qa_profile=qa_profile,
    )
    del probe_row
    direct_candidate = _candidate_by_action(
        canonical_row, required_actions=("direct",)
    )["direct"]
    direct_scored = _score_candidate(
        direct_candidate, canonical_row, lambda_cost, qa_f1_threshold
    )
    needs_generation = "regenerated" in decisions.values()
    row = None
    if not dry_run:
        row, _ = _strict_row(
            canonical_row,
            target_row,
            reflector=reflector,
            lambda_cost=lambda_cost,
            qa_f1_threshold=qa_f1_threshold,
            dry_run=False,
            incorrect_only=incorrect_only,
            reflection_actions=reflection_actions,
            canonical_reflections_only=canonical_reflections_only,
            qa_profile=qa_profile,
        )
    return (
        index,
        str(canonical_row.get("id")),
        row,
        decisions,
        bool(direct_scored["correct"]),
        needs_generation,
    )


def build_trace(
    canonical_path: Path,
    target_path: Path | None,
    output_path: Path,
    config: Dict[str, Any],
    dry_run: bool,
    resume: bool,
    limit: int | None,
    incorrect_only: bool,
    max_workers: int = 1,
    reflection_actions: tuple[str, ...] = REFLECTION_ACTIONS,
    canonical_reflections_only: bool = False,
) -> Dict[str, Any]:
    lambda_cost = float(config.get("lambda_cost", 0.0001))
    qa_f1_threshold = float(config.get("qa_f1_threshold", 0.8))
    qa_profile = load_profile(config.get("qa_reflection_profile"))
    if qa_profile and resume and output_path.exists():
        for row in _read_jsonl(output_path):
            if row.get("qa_profile_sha256") != profile_hash(qa_profile):
                raise ValueError("Resume would mix QA profiles; use a new output path")
    max_workers = max(1, int(max_workers))
    reflection_actions = tuple(reflection_actions)
    invalid_actions = [
        action for action in reflection_actions if action not in REFLECTION_ACTIONS
    ]
    if not reflection_actions or invalid_actions:
        raise ValueError(
            f"reflection_actions must be a non-empty subset of {REFLECTION_ACTIONS}; "
            f"got {reflection_actions}"
        )
    if incorrect_only and canonical_reflections_only:
        raise ValueError(
            "--incorrect-only and --canonical-reflections-only are mutually exclusive"
        )
    completed = _completed_ids(output_path) if resume and not dry_run else set()
    mode = "a" if completed else "w"
    reflector: BaseLLMClient | None = None
    if not dry_run:
        reflector_config = config.get("reflector_llm") or config.get("llm")
        reflector = build_llm_client(reflector_config)
    counts = {
        "rows_seen": 0,
        "rows_written": len(completed),
        "rows_skipped": 0,
        "max_workers": max_workers,
        "direct_correct_rows": 0,
        "direct_incorrect_rows": 0,
    }
    for action in reflection_actions:
        for decision in (
            "reused",
            "regenerated",
            "skipped_direct_correct",
            "skipped_not_selected",
        ):
            counts[f"{action}_{decision}"] = 0
    started_at = time.time()
    output_handle: TextIO | None = None
    if not dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_handle = output_path.open(mode, encoding="utf-8")

    def consume(future: Future) -> None:
        index, row_id, row, decisions, direct_correct, needs_generation = future.result()
        direct_key = "direct_correct_rows" if direct_correct else "direct_incorrect_rows"
        counts[direct_key] += 1
        for action, decision in decisions.items():
            counts.setdefault(f"{action}_{decision}", 0)
            counts[f"{action}_{decision}"] += 1
        if not dry_run:
            assert row is not None and output_handle is not None
            output_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            output_handle.flush()
            counts["rows_written"] += 1
        if index == 1 or index % 100 == 0 or needs_generation:
            elapsed = max(time.time() - started_at, 1e-9)
            action_status = " ".join(
                f"{action}={decisions[action]}" for action in reflection_actions
            )
            _progress(
                f"[frozen-base] {index} id={row_id} "
                f"{action_status} "
                f"elapsed={elapsed:.0f}s workers={max_workers}"
            )

    try:
        if target_path is None:
            pairs = ((row, None) for row in _read_jsonl(canonical_path))
        else:
            pairs = zip_longest(_read_jsonl(canonical_path), _read_jsonl(target_path))
        pending: deque[Future] = deque()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for index, pair in enumerate(pairs, start=1):
                if limit is not None and index > limit:
                    break
                canonical_row, target_row = pair
                if canonical_row is None or (target_path is not None and target_row is None):
                    raise ValueError("Canonical and target traces have different row counts")
                counts["rows_seen"] += 1
                if str(canonical_row.get("id")) in completed:
                    counts["rows_skipped"] += 1
                    continue
                pending.append(
                    executor.submit(
                        _process_trace_pair,
                        index,
                        canonical_row,
                        target_row,
                        reflector,
                        lambda_cost,
                        qa_f1_threshold,
                        dry_run,
                        incorrect_only,
                        reflection_actions,
                        canonical_reflections_only,
                        qa_profile,
                    )
                )
                if len(pending) >= max_workers * 2:
                    consume(pending.popleft())
            while pending:
                consume(pending.popleft())
    finally:
        if output_handle is not None:
            output_handle.close()

    return counts

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild reflector candidates while freezing Base and verifier state."
    )
    parser.add_argument("--canonical", required=True, help="Trace providing frozen Base/state/direct data")
    parser.add_argument("--target", default=None, help="Optional existing trace providing reusable reflector candidates")
    parser.add_argument("--config", required=True, help="Target reflector configuration")
    parser.add_argument("--output", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--incorrect-only",
        action="store_true",
        help="Generate reflection candidates only when the rescored direct answer is incorrect.",
    )
    parser.add_argument(
        "--reflection-actions",
        default="reflect_1,reflect_2",
        help="Comma-separated reflection candidates to build.",
    )
    parser.add_argument(
        "--canonical-reflections-only",
        action="store_true",
        help=(
            "Generate requested reflections only for rows that already contain "
            "those candidates in the canonical trace; preserve other rows as direct-only."
        ),
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Concurrent reflector requests; output remains in canonical input order.",
    )
    args = parser.parse_args()

    counts = build_trace(
        canonical_path=Path(args.canonical),
        target_path=Path(args.target) if args.target else None,
        output_path=Path(args.output),
        config=load_json(args.config),
        dry_run=args.dry_run,
        resume=args.resume,
        limit=args.limit,
        incorrect_only=args.incorrect_only,
        max_workers=args.max_workers,
        reflection_actions=tuple(
            action.strip()
            for action in args.reflection_actions.split(",")
            if action.strip()
        ),
        canonical_reflections_only=args.canonical_reflections_only,
    )
    print(json.dumps(counts, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
