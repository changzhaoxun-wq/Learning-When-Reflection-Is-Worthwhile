import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from .answer_extractor import extract_answer
from .reward import compute_match_metrics


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
    return rows


def candidate(row: Dict[str, Any], action: str) -> Dict[str, Any] | None:
    return next((item for item in row.get("candidate_results", []) if item.get("action") == action), None)


def call(row: Dict[str, Any]) -> Dict[str, Any]:
    return row.get("initial_call") or {}


def stored_correct(row: Dict[str, Any], action: str = "direct") -> int:
    return int((candidate(row, action) or {}).get("correct", 0))


def action_correct(row: Dict[str, Any], action: str = "direct") -> int:
    item = candidate(row, action) or {}
    output = item.get("final_answer")
    if output is None and action == "direct":
        output = call(row).get("output", row.get("initial_answer", ""))
    pred = extract_answer(output, task_type=row.get("task_type"))
    metrics = compute_match_metrics(
        pred,
        row.get("gold_answer"),
        task_type=row.get("task_type"),
        gold_aliases=row.get("gold_aliases"),
    )
    return int(metrics["correct"])


def correct(row: Dict[str, Any]) -> int:
    return action_correct(row, "direct")


def rate(value: int | float, n: int) -> float:
    return float(value) / n if n else 0.0


def at_limit(row: Dict[str, Any], max_tokens: int) -> bool:
    initial = call(row)
    finish = initial.get("finish_reason") or (initial.get("response_metadata") or {}).get("finish_reason")
    return finish == "length" if finish is not None else int(initial.get("completion_tokens", 0)) >= max_tokens


def summarize(rows: List[Dict[str, Any]], max_tokens: int) -> Dict[str, Any]:
    n = len(rows)
    completions = [int(call(row).get("completion_tokens", 0)) for row in rows]
    outputs = [str(call(row).get("output", row.get("initial_answer", ""))) for row in rows]
    correctness = [correct(row) for row in rows]
    stored_correctness = [stored_correct(row) for row in rows]
    capped = [at_limit(row, max_tokens) for row in rows]
    final = ["Final answer:" in output for output in outputs]
    unclosed = ["<think>" in output and "</think>" not in output for output in outputs]
    recorded_finish = any(
        call(row).get("finish_reason") is not None
        or (call(row).get("response_metadata") or {}).get("finish_reason") is not None
        for row in rows
    )

    def subset(mask: List[bool]) -> float:
        values = [ok for ok, keep in zip(correctness, mask) if keep]
        return rate(sum(values), len(values))

    ordered = sorted(completions)
    percentile = lambda q: float(ordered[round((n - 1) * q)]) if n else 0.0
    return {
        "n": n,
        "correct": sum(correctness),
        "accuracy": rate(sum(correctness), n),
        "stored_correct": sum(stored_correctness),
        "stored_accuracy": rate(sum(stored_correctness), n),
        "rescored_changes": sum(a != b for a, b in zip(correctness, stored_correctness)),
        "avg_prompt_tokens": rate(sum(int(call(row).get("prompt_tokens", 0)) for row in rows), n),
        "avg_completion_tokens": rate(sum(completions), n),
        "median_completion_tokens": percentile(0.5),
        "p90_completion_tokens": percentile(0.9),
        "max_completion_tokens": max(completions, default=0),
        "finish_reason_recorded": recorded_finish,
        "at_limit_definition": "finish_reason=length" if recorded_finish else f"completion_tokens>={max_tokens} proxy",
        "at_limit": sum(capped),
        "at_limit_rate": rate(sum(capped), n),
        "at_limit_accuracy": subset(capped),
        "below_limit_accuracy": subset([not value for value in capped]),
        "has_final_marker": sum(final),
        "has_final_marker_rate": rate(sum(final), n),
        "has_final_marker_accuracy": subset(final),
        "no_final_marker_accuracy": subset([not value for value in final]),
        "has_think_open": sum("<think>" in output for output in outputs),
        "has_think_close": sum("</think>" in output for output in outputs),
        "unclosed_think": sum(unclosed),
        "unclosed_think_rate": rate(sum(unclosed), n),
        "unclosed_think_accuracy": subset(unclosed),
    }


def paired(left: List[Dict[str, Any]], right: List[Dict[str, Any]], max_tokens: int) -> Dict[str, Any]:
    if len(left) != len(right):
        raise ValueError(f"Row count differs: {len(left)} != {len(right)}")
    alignment = {key: 0 for key in (
        "id_mismatches", "question_mismatches", "gold_answer_mismatches",
        "prompt_version_mismatches", "literal_prompt_mismatches", "prompt_token_count_mismatches",
    )}
    table = {"both_correct": 0, "left_only_correct": 0, "right_only_correct": 0, "both_wrong": 0}
    table.update({
        "right_only_with_left_at_limit": 0,
        "right_only_with_left_no_final_marker": 0,
        "right_only_with_left_unclosed_think": 0,
        "left_below_limit_n": 0,
        "left_below_limit_left_correct": 0,
        "left_below_limit_right_correct": 0,
        "left_at_limit_n": 0,
        "left_at_limit_left_correct": 0,
        "left_at_limit_right_correct": 0,
    })
    for a, b in zip(left, right):
        for field in ("id", "question", "gold_answer", "prompt_version"):
            alignment[f"{field}_mismatches"] += int(a.get(field) != b.get(field))
        alignment["literal_prompt_mismatches"] += int(call(a).get("prompt") != call(b).get("prompt"))
        alignment["prompt_token_count_mismatches"] += int(call(a).get("prompt_tokens") != call(b).get("prompt_tokens"))
        ca, cb = correct(a), correct(b)
        left_capped = at_limit(a, max_tokens)
        subset = "left_at_limit" if left_capped else "left_below_limit"
        table[f"{subset}_n"] += 1
        table[f"{subset}_left_correct"] += ca
        table[f"{subset}_right_correct"] += cb
        key = "both_correct" if ca and cb else "left_only_correct" if ca else "right_only_correct" if cb else "both_wrong"
        table[key] += 1
        if key == "right_only_correct":
            output = str(call(a).get("output", a.get("initial_answer", "")))
            table["right_only_with_left_at_limit"] += int(left_capped)
            table["right_only_with_left_no_final_marker"] += int("Final answer:" not in output)
            table["right_only_with_left_unclosed_think"] += int("<think>" in output and "</think>" not in output)
    content_keys = [key for key in alignment if key != "prompt_token_count_mismatches"]
    alignment["content_parity_passed"] = not any(alignment[key] for key in content_keys)
    alignment["model_token_count_parity"] = alignment["prompt_token_count_mismatches"] == 0
    alignment["passed"] = alignment["content_parity_passed"]
    return {"alignment": alignment, "paired_correctness": table}


def reflection_check(rows: List[Dict[str, Any]], max_tokens: int) -> Dict[str, Any]:
    counts = {"eligible_rows": 0, "direct_wrong": 0, "reflect_1_repairs": 0,
              "repairs_with_direct_at_limit": 0, "repairs_with_direct_below_limit": 0,
              "direct_correct": 0, "reflect_1_harms": 0}
    for row in rows:
        direct, reflected = candidate(row, "direct"), candidate(row, "reflect_1")
        if not direct or not reflected:
            continue
        counts["eligible_rows"] += 1
        d, r = action_correct(row, "direct"), action_correct(row, "reflect_1")
        if d:
            counts["direct_correct"] += 1
            counts["reflect_1_harms"] += int(not r)
        else:
            counts["direct_wrong"] += 1
            if r:
                counts["reflect_1_repairs"] += 1
                key = "repairs_with_direct_at_limit" if at_limit(row, max_tokens) else "repairs_with_direct_below_limit"
                counts[key] += 1
    counts["repair_rate"] = rate(counts["reflect_1_repairs"], counts["direct_wrong"])
    counts["harm_rate"] = rate(counts["reflect_1_harms"], counts["direct_correct"])
    return counts


def markdown(report: Dict[str, Any]) -> str:
    ln, rn = report["left_name"], report["right_name"]
    left, right = report["models"][ln], report["models"][rn]
    align = report["paired"]["alignment"]
    pair = report["paired"]["paired_correctness"]
    ref = report["left_reflection_contamination"]
    controlled = (
        left["has_think_open"] == 0
        and right["has_think_open"] == 0
        and left["at_limit"] == 0
        and right["at_limit"] == 0
    )
    if controlled:
        interpretation = (
            "Both traces use completed non-thinking outputs, so thinking and truncation no longer explain "
            "the remaining accuracy difference. The checkpoints still belong to different release/training "
            "generations; use a larger paired sample before attributing a residual difference to parameter size."
        )
    else:
        interpretation = (
            "This is not a clean parameter-size ablation when one checkpoint emits long thinking traces and "
            "the other is non-thinking-only. Equal `max_tokens` is not an equal generation protocol: one model "
            "spends most of the budget inside thinking. Re-run a common explicit non-thinking mode, and evaluate "
            "thinking mode separately with recommended sampling and sufficient output budget."
        )
    return f'''# Direct Generation Audit

## Input parity

- Rows: {left['n']}; literal content parity passed: {align['content_parity_passed']}
- ID/question/gold/prompt-version mismatches: {align['id_mismatches']}/{align['question_mismatches']}/{align['gold_answer_mismatches']}/{align['prompt_version_mismatches']}
- Literal prompt mismatches: {align['literal_prompt_mismatches']}
- Model-template prompt-token-count mismatches: {align['prompt_token_count_mismatches']} (token-count parity: {align['model_token_count_parity']})

## Direct results

| model | rescored accuracy | stored accuracy | rescored changes | avg completion | at limit | final marker | unclosed think |
|---|---:|---:|---:|---:|---:|---:|---:|
| {ln} | {left['accuracy']:.4f} | {left['stored_accuracy']:.4f} | {left['rescored_changes']} | {left['avg_completion_tokens']:.1f} | {left['at_limit_rate']:.4f} | {left['has_final_marker_rate']:.4f} | {left['unclosed_think_rate']:.4f} |
| {rn} | {right['accuracy']:.4f} | {right['stored_accuracy']:.4f} | {right['rescored_changes']} | {right['avg_completion_tokens']:.1f} | {right['at_limit_rate']:.4f} | {right['has_final_marker_rate']:.4f} | {right['unclosed_think_rate']:.4f} |

Old traces use `{left['at_limit_definition']}` because finish reasons were not saved.

## Paired correctness

- Both correct: {pair['both_correct']}
- {ln} only: {pair['left_only_correct']}
- {rn} only: {pair['right_only_correct']}
- Both wrong: {pair['both_wrong']}
- In {rn}-only wins, left hit limit / lacked final marker / had unclosed think: {pair['right_only_with_left_at_limit']} / {pair['right_only_with_left_no_final_marker']} / {pair['right_only_with_left_unclosed_think']}
- On the {pair['left_below_limit_n']} rows where {ln} stayed below the limit, left/right accuracy: {rate(pair['left_below_limit_left_correct'], pair['left_below_limit_n']):.4f} / {rate(pair['left_below_limit_right_correct'], pair['left_below_limit_n']):.4f}
- On the {pair['left_at_limit_n']} rows where {ln} hit the limit, left/right accuracy: {rate(pair['left_at_limit_left_correct'], pair['left_at_limit_n']):.4f} / {rate(pair['left_at_limit_right_correct'], pair['left_at_limit_n']):.4f}

## Reflection contamination

- Direct wrong: {ref['direct_wrong']}
- Reflect-1 repairs: {ref['reflect_1_repairs']} ({ref['repair_rate']:.4f})
- Repairs with direct at/below limit: {ref['repairs_with_direct_at_limit']} / {ref['repairs_with_direct_below_limit']}
- Reflect-1 harms: {ref['reflect_1_harms']} ({ref['harm_rate']:.4f} of direct-correct rows)

## Interpretation

{interpretation}
'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--left_name", default="left")
    parser.add_argument("--right_name", default="right")
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--json_output", required=True)
    parser.add_argument("--markdown_output", required=True)
    args = parser.parse_args()
    left, right = read_jsonl(Path(args.left)), read_jsonl(Path(args.right))
    report = {
        "left_name": args.left_name,
        "right_name": args.right_name,
        "max_tokens": args.max_tokens,
        "models": {args.left_name: summarize(left, args.max_tokens), args.right_name: summarize(right, args.max_tokens)},
        "paired": paired(left, right, args.max_tokens),
        "left_reflection_contamination": reflection_check(left, args.max_tokens),
    }
    json_path, md_path = Path(args.json_output), Path(args.markdown_output)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Wrote {json_path}\nWrote {md_path}")


if __name__ == "__main__":
    main()
