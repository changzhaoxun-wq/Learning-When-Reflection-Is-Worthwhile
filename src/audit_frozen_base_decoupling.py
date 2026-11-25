import argparse
import json
from itertools import zip_longest
from pathlib import Path
from typing import Any, Dict, Iterator


def _read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc


def _direct_candidate(row: Dict[str, Any]) -> Dict[str, Any] | None:
    for candidate in row.get("candidate_results") or []:
        if candidate.get("action") == "direct":
            return candidate
    return None


def audit(canonical_path: Path, candidate_path: Path) -> Dict[str, Any]:
    fields = (
        "id",
        "question",
        "gold_answer",
        "prompt_version",
        "initial_call",
        "initial_answer",
        "initial_pred",
        "state_features",
    )
    mismatches = {field: 0 for field in fields}
    mismatches["direct_candidate"] = 0
    mismatches["strict_metadata"] = 0
    row_count = 0

    pairs = zip_longest(_read_jsonl(canonical_path), _read_jsonl(candidate_path))
    for canonical, candidate in pairs:
        if canonical is None or candidate is None:
            mismatches["row_count"] = 1
            break
        row_count += 1
        for field in fields:
            if canonical.get(field) != candidate.get(field):
                mismatches[field] += 1
        if _direct_candidate(canonical) != _direct_candidate(candidate):
            mismatches["direct_candidate"] += 1
        if not candidate.get("strict_decoupling", {}).get("base_frozen"):
            mismatches["strict_metadata"] += 1

    passed = all(value == 0 for value in mismatches.values())
    return {
        "canonical": str(canonical_path),
        "candidate": str(candidate_path),
        "rows": row_count,
        "mismatches": mismatches,
        "passed": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit frozen-Base decoupling traces.")
    parser.add_argument("--canonical", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    report = audit(Path(args.canonical), Path(args.candidate))
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
