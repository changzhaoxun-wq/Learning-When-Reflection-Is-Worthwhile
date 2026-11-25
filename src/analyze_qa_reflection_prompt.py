from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from .answer_extractor import normalize_text_answer
from .utils import read_jsonl, save_json


ACTIONS = ("reflect_1", "reflect_2")


def _candidate_map(row: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(candidate.get("action")): candidate
        for candidate in row.get("candidate_results") or []
    }


def _correct(candidate: dict[str, Any]) -> int:
    return int(candidate.get("enhanced_correct", candidate.get("correct", 0)))


def _f1(candidate: dict[str, Any]) -> float:
    return float(candidate.get("enhanced_answer_f1", candidate.get("answer_f1", 0.0)))


def _decision(candidate: dict[str, Any]) -> str:
    calls = (candidate.get("metadata") or {}).get("calls") or []
    if len(calls) < 2:
        return "missing"
    match = re.search(r"(?im)^Decision:\s*(KEEP|REVISE)\s*$", str(calls[-1].get("output") or ""))
    return match.group(1).lower() if match else "missing"


def _changed(direct: dict[str, Any], reflected: dict[str, Any]) -> bool:
    return normalize_text_answer(direct.get("pred")) != normalize_text_answer(reflected.get("pred"))


def _action_summary(rows: list[dict[str, Any]], action: str) -> dict[str, Any]:
    direct_correct = 0
    direct_incorrect = 0
    preserved = 0
    harmed = 0
    repaired = 0
    unchanged_wrong = 0
    f1_delta = 0.0
    token_cost = 0.0
    reflected_correct = 0
    decisions: Counter[str] = Counter()
    decision_inconsistency: Counter[str] = Counter()

    for row in rows:
        candidates = _candidate_map(row)
        direct = candidates["direct"]
        reflected = candidates[action]
        is_direct_correct = _correct(direct)
        is_reflected_correct = _correct(reflected)
        changed = _changed(direct, reflected)
        decision = _decision(reflected)
        decisions[decision] += 1
        if decision == "keep" and changed:
            decision_inconsistency["keep_but_changed"] += 1
        if decision == "revise" and not changed:
            decision_inconsistency["revise_but_unchanged"] += 1

        direct_correct += is_direct_correct
        direct_incorrect += 1 - is_direct_correct
        reflected_correct += is_reflected_correct
        f1_delta += _f1(reflected) - _f1(direct)
        token_cost += float(reflected.get("token_cost", 0))
        if is_direct_correct:
            preserved += is_reflected_correct
            harmed += 1 - is_reflected_correct
        else:
            repaired += is_reflected_correct
            unchanged_wrong += 1 - is_reflected_correct

    n = len(rows)
    return {
        "n": n,
        "accuracy": reflected_correct / n,
        "avg_f1_delta_vs_direct": f1_delta / n,
        "avg_token_cost": token_cost / n,
        "direct_correct_n": direct_correct,
        "preserved_n": preserved,
        "preservation_rate": preserved / direct_correct if direct_correct else 0.0,
        "harmed_n": harmed,
        "harm_rate": harmed / direct_correct if direct_correct else 0.0,
        "direct_incorrect_n": direct_incorrect,
        "repaired_n": repaired,
        "repair_rate": repaired / direct_incorrect if direct_incorrect else 0.0,
        "still_wrong_n": unchanged_wrong,
        "net_correct_change": repaired - harmed,
        "decision_distribution": dict(decisions),
        "decision_inconsistency": dict(decision_inconsistency),
    }


def _candidate_ceiling(rows: list[dict[str, Any]]) -> dict[str, Any]:
    direct_correct = 0
    repaired_by_any = 0
    for row in rows:
        candidates = _candidate_map(row)
        direct = _correct(candidates["direct"])
        direct_correct += direct
        if not direct and any(_correct(candidates[action]) for action in ACTIONS):
            repaired_by_any += 1
    return {
        "accuracy": (direct_correct + repaired_by_any) / len(rows),
        "direct_correct_n": direct_correct,
        "direct_wrong_repaired_by_any_n": repaired_by_any,
    }


def compare(old_rows: list[dict[str, Any]], new_rows: list[dict[str, Any]]) -> dict[str, Any]:
    old_by_id = {str(row.get("id")): row for row in old_rows}
    aligned_old: list[dict[str, Any]] = []
    aligned_new: list[dict[str, Any]] = []
    direct_mismatches: list[str] = []
    for row in new_rows:
        row_id = str(row.get("id"))
        old = old_by_id.get(row_id)
        if old is None:
            raise ValueError(f"New smoke row is absent from old trace: {row_id}")
        old_candidates = _candidate_map(old)
        new_candidates = _candidate_map(row)
        missing_old = [action for action in ("direct", *ACTIONS) if action not in old_candidates]
        missing_new = [action for action in ("direct", *ACTIONS) if action not in new_candidates]
        if missing_old or missing_new:
            raise ValueError(f"Incomplete candidates for {row_id}: old={missing_old}, new={missing_new}")
        if (
            old_candidates["direct"].get("pred") != new_candidates["direct"].get("pred")
            or int(old_candidates["direct"].get("token_cost", 0))
            != int(new_candidates["direct"].get("token_cost", 0))
        ):
            direct_mismatches.append(row_id)
        aligned_old.append(old)
        aligned_new.append(row)

    old_action = {action: _action_summary(aligned_old, action) for action in ACTIONS}
    new_action = {action: _action_summary(aligned_new, action) for action in ACTIONS}
    deltas = {
        action: {
            key: new_action[action][key] - old_action[action][key]
            for key in (
                "accuracy",
                "preservation_rate",
                "harm_rate",
                "repair_rate",
                "net_correct_change",
                "avg_f1_delta_vs_direct",
                "avg_token_cost",
            )
        }
        for action in ACTIONS
    }
    return {
        "rows": len(aligned_new),
        "direct_frozen": len(direct_mismatches) == 0,
        "direct_mismatch_count": len(direct_mismatches),
        "direct_mismatch_examples": direct_mismatches[:20],
        "old_reflection_prompt_version": aligned_old[0].get("reflection_prompt_version"),
        "new_reflection_prompt_version": aligned_new[0].get("reflection_prompt_version"),
        "old": {
            "actions": old_action,
            "candidate_ceiling": _candidate_ceiling(aligned_old),
        },
        "new": {
            "actions": new_action,
            "candidate_ceiling": _candidate_ceiling(aligned_new),
        },
        "new_minus_old": deltas,
        "selection_rule": (
            "Prefer the new prompt only when direct is frozen, harm decreases without "
            "a larger loss in repaired examples, and the candidate ceiling does not regress."
        ),
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# QA Reflection Prompt Smoke Report",
        "",
        f"- Rows: {report['rows']}",
        f"- Direct frozen: {report['direct_frozen']}",
        f"- Old prompt: `{report['old_reflection_prompt_version']}`",
        f"- New prompt: `{report['new_reflection_prompt_version']}`",
        "",
        "| protocol | action | accuracy | preserve | harm | repair | net correct | avg F1 delta | avg tokens |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for protocol in ("old", "new"):
        for action in ACTIONS:
            item = report[protocol]["actions"][action]
            lines.append(
                f"| {protocol} | {action} | {item['accuracy']:.4f} | "
                f"{item['preservation_rate']:.4f} | {item['harm_rate']:.4f} | "
                f"{item['repair_rate']:.4f} | {item['net_correct_change']:+d} | "
                f"{item['avg_f1_delta_vs_direct']:+.4f} | {item['avg_token_cost']:.1f} |"
            )
    lines.extend(["", "## Candidate Ceiling", ""])
    for protocol in ("old", "new"):
        ceiling = report[protocol]["candidate_ceiling"]
        lines.append(
            f"- {protocol}: accuracy={ceiling['accuracy']:.4f}, "
            f"direct-wrong repaired by either reflection={ceiling['direct_wrong_repaired_by_any_n']}"
        )
    lines.extend(["", "## Decision Format", ""])
    for action in ACTIONS:
        item = report["new"]["actions"][action]
        lines.append(
            f"- {action}: decisions={item['decision_distribution']}, "
            f"inconsistency={item['decision_inconsistency']}"
        )
    lines.extend(["", "## Rule", "", report["selection_rule"], ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare old and new QA reflection prompts on fixed Base rows.")
    parser.add_argument("--old", required=True)
    parser.add_argument("--new", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown_output", required=True)
    args = parser.parse_args()

    report = compare(read_jsonl(args.old), read_jsonl(args.new))
    output = Path(args.output)
    markdown_output = Path(args.markdown_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    save_json(report, output)
    markdown_output.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
