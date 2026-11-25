import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

from src.answer_extractor import answers_equal


DEFAULT_INPUTS = [
    "data/reflectionevo/Meta-Llama-3-8B-Instruct_math.jsonl",
    "data/reflectionevo/gemma-2-9b-it_math.jsonl",
]

SYSTEM_PROMPT = """/no_think
You are a math reflection and correction model.
Given a math problem and a previous incorrect solution, diagnose the mistake briefly, then produce a corrected solution.
If the previous solution has formatting issues, still solve the problem from first principles.
End with exactly this format:
Final answer: <answer>"""


def build_messages(row: Dict[str, Any], drop_equivalent_answers: bool = True) -> Dict[str, Any] | None:
    question = str(row.get("question", "")).strip()
    wrong_solution = str(row.get("first_trial_reasoning", "")).strip()
    reflection = str(row.get("reflections", "")).strip()
    corrected = str(row.get("second_trial_reasoning", "")).strip()
    final_answer = str(row.get("second_trial_answer", row.get("answer", ""))).strip()

    if not question or not wrong_solution or not corrected or not final_answer:
        return None
    if "Answer is CORRECT" not in corrected:
        return None
    if drop_equivalent_answers and answers_equal(row.get("first_trial_answer"), final_answer):
        return None

    corrected_clean = _clean_trial_reasoning(corrected)
    reflection_clean = _clean_reflection(reflection)
    final_answer_clean = _clean_final_answer(final_answer)

    user = f"""{SYSTEM_PROMPT}

Question:
{question}

Previous incorrect solution:
{wrong_solution}

Now provide a brief reflection and a corrected solution."""
    assistant = f"""Reflection:
{reflection_clean}

Corrected solution:
{corrected_clean}

Final answer: {final_answer_clean}"""

    return {
        "id": str(row.get("id", "")),
        "dataset": "ReflectionEvo-D+",
        "source_model": _source_model(row),
        "task": "math_reflection_sft",
        "question": question,
        "answer": final_answer_clean,
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "metadata": {
            "first_trial_answer": row.get("first_trial_answer"),
            "second_trial_answer": row.get("second_trial_answer"),
        },
    }


def convert(inputs: Iterable[str], output: str, limit: int | None = None, dedupe: bool = True, drop_equivalent_answers: bool = True) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen = set()
    for input_path in inputs:
        source_name = Path(input_path).name
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                raw = json.loads(line)
                raw["_source_file"] = source_name
                item = build_messages(raw, drop_equivalent_answers=drop_equivalent_answers)
                if item is None:
                    continue
                key = _dedupe_key(item)
                if dedupe and key in seen:
                    continue
                seen.add(key)
                rows.append(item)
                if limit is not None and len(rows) >= limit:
                    break
        if limit is not None and len(rows) >= limit:
            break

    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return rows


def _source_model(row: Dict[str, Any]) -> str:
    source = str(row.get("_source_file", "unknown"))
    return source.replace("_math.jsonl", "")


def _dedupe_key(item: Dict[str, Any]) -> str:
    question = re.sub(r"\s+", " ", item.get("question", "")).strip().lower()
    answer = re.sub(r"\s+", " ", item.get("answer", "")).strip().lower()
    return f"{question}\t{answer}"


def _clean_reflection(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text or "The previous solution contains an error. Re-check the problem conditions and recompute carefully."


def _clean_trial_reasoning(text: str) -> str:
    text = text or ""
    text = re.sub(r"\n?Observation:\s*Answer is (?:CORRECT|INCORRECT).*", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"Action:\s*Finish<<<(.*?)>>>", r"Final answer: \1", text, flags=re.DOTALL)
    text = re.sub(r"Finish<<<(.*?)>>>", r"Final answer: \1", text, flags=re.DOTALL)
    text = re.sub(r"<<<(.*?)>>>", r"\1", text, flags=re.DOTALL)
    text = text.replace("Thought:", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _clean_final_answer(answer: str) -> str:
    answer = str(answer or "").strip()
    answer = re.sub(r"^<<<|>>>$", "", answer).strip()
    answer = answer.strip("$ ")
    answer = re.sub(r"}\$?\.?$", "}", answer) if answer.count("{") >= answer.count("}") else answer
    return answer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", default=DEFAULT_INPUTS)
    parser.add_argument("--output", default="outputs/reflector_lora/reflectionevo_dplus_math_sft.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no_dedupe", action="store_true")
    parser.add_argument("--keep_equivalent_answers", action="store_true")
    args = parser.parse_args()

    rows = convert(
        args.inputs,
        args.output,
        limit=args.limit,
        dedupe=not args.no_dedupe,
        drop_equivalent_answers=not args.keep_equivalent_answers,
    )
    print(f"Wrote {len(rows)} reflector SFT rows to {args.output}")


if __name__ == "__main__":
    main()
