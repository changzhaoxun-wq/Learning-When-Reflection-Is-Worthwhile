from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterator


def _rows(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc


def _finish_reason(call: dict[str, Any]) -> Any:
    return call.get("finish_reason") or (call.get("response_metadata") or {}).get("finish_reason")


def _logical_calls(row: dict[str, Any]) -> Iterator[tuple[str, dict[str, Any]]]:
    initial = row.get("initial_call") or {}
    yield "direct", initial
    for candidate in row.get("candidate_results") or []:
        action = str(candidate.get("action"))
        if not action.startswith("reflect_"):
            continue
        calls = (candidate.get("metadata") or {}).get("calls") or []
        for call in calls[1:]:
            yield f"{action}.round_{call.get('round_id', '?')}", call


def audit(path: Path) -> dict[str, Any]:
    finish_reasons: Counter[str] = Counter()
    call_types: Counter[str] = Counter()
    rows = 0
    calls = 0
    final_truncated = []
    missing_finish_reason = []
    missing_final_answer = []
    usage_mismatch = []
    retried_after_length = 0
    context_window_retries = 0

    for row in _rows(path):
        rows += 1
        row_id = str(row.get("id"))
        for call_type, call in _logical_calls(row):
            calls += 1
            call_types[call_type] += 1
            reason = _finish_reason(call)
            finish_reasons[str(reason or "missing")] += 1
            if reason == "length":
                final_truncated.append({"id": row_id, "call": call_type})
            if reason is None:
                missing_finish_reason.append({"id": row_id, "call": call_type})
            output = str(call.get("output") or "")
            if "final answer:" not in output.lower():
                missing_final_answer.append({"id": row_id, "call": call_type})
            prompt_tokens = int(call.get("prompt_tokens", 0))
            completion_tokens = int(call.get("completion_tokens", 0))
            total_tokens = int(call.get("total_tokens", 0))
            if total_tokens != prompt_tokens + completion_tokens:
                usage_mismatch.append({
                    "id": row_id,
                    "call": call_type,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                })
            metadata = call.get("response_metadata") or {}
            attempts = metadata.get("generation_attempts") or []
            if any(attempt.get("finish_reason") == "length" for attempt in attempts[:-1]):
                retried_after_length += 1
            context_window_retries += sum(
                attempt.get("context_window_retry") is not None for attempt in attempts
            )

    return {
        "trace": str(path.resolve()),
        "rows": rows,
        "logical_calls": calls,
        "call_types": dict(call_types),
        "finish_reasons": dict(finish_reasons),
        "final_truncated_count": len(final_truncated),
        "final_truncated_examples": final_truncated[:50],
        "missing_finish_reason_count": len(missing_finish_reason),
        "missing_finish_reason_examples": missing_finish_reason[:50],
        "missing_final_answer_count": len(missing_final_answer),
        "missing_final_answer_examples": missing_final_answer[:50],
        "usage_mismatch_count": len(usage_mismatch),
        "usage_mismatch_examples": usage_mismatch[:50],
        "retried_after_length_count": retried_after_length,
        "context_window_retry_count": context_window_retries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit trace completion and token accounting.")
    parser.add_argument("--trace", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--require_finish_reason", action="store_true")
    parser.add_argument("--require_final_answer", action="store_true")
    args = parser.parse_args()

    result = audit(Path(args.trace))
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")

    failed = result["final_truncated_count"] > 0 or result["usage_mismatch_count"] > 0
    if args.require_finish_reason:
        failed = failed or result["missing_finish_reason_count"] > 0
    if args.require_final_answer:
        failed = failed or result["missing_final_answer_count"] > 0
    if failed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
