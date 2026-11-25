import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from src.answer_extractor import answers_equal, extract_final_answer
from src.gsm8k_utils import extract_gold_answer, normalize_gsm8k_item
from src.llm_client import build_llm_client
from src.prompts import direct_prompt, reflection_prompt
from src.utils import load_json, read_jsonl


DEFAULT_DATA_PATH = os.environ.get("REFLECTOR_SFT_DATA", "data/hendrycks_math_train.jsonl")
DEFAULT_OUTPUT = os.environ.get("REFLECTOR_SFT_OUTPUT", "outputs/reflector_lora/qwen3_error_reflector_sft.jsonl")


def build_dataset(
    config: Dict[str, Any],
    data_path: str,
    output: str,
    limit: Optional[int] = None,
    correct_keep_prob: float = 0.25,
    resume: bool = True,
    progress_every: int = 10,
    seed: int = 42,
    shuffle_before_limit: bool = False,
) -> Dict[str, Any]:
    rows = _load_math_sft_rows(
        data_path,
        limit=limit,
        shuffle_before_limit=shuffle_before_limit,
        seed=seed,
    )
    llm = build_llm_client(config.get("base_llm", config.get("llm", {"mode": "fake"})))

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written_ids = _read_written_ids(out_path) if resume else set()
    mode = "a" if resume and out_path.exists() else "w"

    stats = {
        "seen": 0,
        "skipped_existing": 0,
        "base_correct": 0,
        "base_wrong": 0,
        "kept_correct": 0,
        "written": 0,
        "no_pred": 0,
        "started_at": time.time(),
    }

    rng = random.Random(seed)
    start_time = time.time()
    with out_path.open(mode, encoding="utf-8") as f:
        for idx, row in enumerate(rows, start=1):
            stats["seen"] += 1
            item_id = str(row["id"])
            if item_id in written_ids:
                stats["skipped_existing"] += 1
                _print_progress(stats, idx, len(rows), start_time, progress_every, item_id, skipped=True)
                continue

            question = row["question"]
            gold_answer = row["gold_answer"]
            gold_solution = row["gold_solution"]
            response = llm.generate(direct_prompt(question), gold_answer=gold_answer)
            initial_answer = response.text.strip()
            initial_pred = extract_final_answer(initial_answer)
            if initial_pred is None:
                stats["no_pred"] += 1

            is_correct = answers_equal(initial_pred, gold_answer)
            if is_correct:
                stats["base_correct"] += 1
            else:
                stats["base_wrong"] += 1

            keep_correct = is_correct and _stable_keep(item_id, correct_keep_prob, rng)
            if is_correct and not keep_correct:
                _print_progress(stats, idx, len(rows), start_time, progress_every, item_id)
                continue
            if keep_correct:
                stats["kept_correct"] += 1

            sft_row = _make_reflector_row(
                row=row,
                initial_answer=initial_answer,
                initial_pred=initial_pred,
                is_correct=is_correct,
                response_usage={
                    "prompt_tokens": response.prompt_tokens,
                    "completion_tokens": response.completion_tokens,
                    "total_tokens": response.total_tokens,
                },
                gold_solution=gold_solution,
                gold_answer=gold_answer,
            )
            f.write(json.dumps(sft_row, ensure_ascii=False) + "\n")
            f.flush()
            written_ids.add(item_id)
            stats["written"] += 1
            _print_progress(stats, idx, len(rows), start_time, progress_every, item_id)

    stats["output"] = str(out_path)
    stats["elapsed_seconds"] = round(time.time() - start_time, 2)
    return stats


def _load_math_sft_rows(
    data_path: str,
    limit: Optional[int],
    shuffle_before_limit: bool = False,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    raw_rows = read_jsonl(data_path)
    rows = []
    for idx, raw in enumerate(raw_rows):
        normalized = normalize_gsm8k_item(raw, idx=idx, split=str(raw.get("split", "train")))
        gold_solution = _extract_assistant_text(raw)
        if not normalized.get("question") or not normalized.get("gold_answer") or not gold_solution:
            continue
        normalized["gold_solution"] = gold_solution
        normalized["metadata"] = raw.get("metadata", {})
        rows.append(normalized)

    if shuffle_before_limit:
        random.Random(seed).shuffle(rows)
    if limit is not None:
        rows = rows[: int(limit)]
    return rows


def _extract_assistant_text(row: Dict[str, Any]) -> str:
    for message in row.get("messages", []):
        if message.get("role") == "assistant":
            return str(message.get("content", "")).strip()
    answer = row.get("answer")
    return str(answer or "").strip()


def _make_reflector_row(
    row: Dict[str, Any],
    initial_answer: str,
    initial_pred: Optional[str],
    is_correct: bool,
    response_usage: Dict[str, int],
    gold_solution: str,
    gold_answer: str,
) -> Dict[str, Any]:
    question = row["question"]
    prompt = reflection_prompt(question=question, current_answer=initial_answer, round_id=1)
    if is_correct:
        assistant = (
            "Reflection:\n"
            "The previous solution already reaches the correct final answer. Preserve the answer and avoid unnecessary changes.\n\n"
            "Corrected solution:\n"
            f"{initial_answer}\n\n"
            f"Final answer: {gold_answer}"
        )
        sample_type = "preserve_correct"
    else:
        pred_text = initial_pred if initial_pred is not None else "no extractable answer"
        assistant = (
            "Reflection:\n"
            f"The previous solution's final answer ({pred_text}) does not match the reference answer. "
            "Re-solve the problem from the given conditions and correct the final answer.\n\n"
            "Corrected solution:\n"
            f"{gold_solution}\n\n"
            f"Final answer: {gold_answer}"
        )
        sample_type = "correct_error"

    return {
        "id": row["id"],
        "dataset": row.get("dataset", "hendrycks_math"),
        "source": row.get("source", "unknown"),
        "task": "qwen3_error_reflector_sft",
        "sample_type": sample_type,
        "question": question,
        "answer": gold_answer,
        "initial_pred": initial_pred,
        "initial_correct": is_correct,
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": assistant},
        ],
        "metadata": {
            "difficulty": row.get("difficulty"),
            "difficulty_score": row.get("difficulty_score"),
            "base_usage": response_usage,
            "gold_solution": gold_solution,
        },
    }


def _stable_keep(item_id: str, keep_prob: float, rng: random.Random) -> bool:
    if keep_prob <= 0:
        return False
    if keep_prob >= 1:
        return True
    digest = hashlib.md5(item_id.encode("utf-8")).hexdigest()
    value = int(digest[:8], 16) / 0xFFFFFFFF
    return value < keep_prob


def _read_written_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                ids.add(str(json.loads(line).get("id", "")))
            except json.JSONDecodeError:
                continue
    return ids


def _print_progress(stats: Dict[str, Any], idx: int, total: int, start_time: float, progress_every: int, item_id: str, skipped: bool = False) -> None:
    if idx == 1 or idx == total or idx % max(1, progress_every) == 0:
        elapsed = time.time() - start_time
        avg = elapsed / max(1, idx)
        remaining = max(0, total - idx)
        eta = remaining * avg
        status = "skipped" if skipped else "done"
        print(
            f"[build-qwen-error-reflector] {idx}/{total} id={item_id} {status} "
            f"written={stats['written']} wrong={stats['base_wrong']} correct={stats['base_correct']} "
            f"elapsed={_fmt(elapsed)} avg={avg:.1f}s eta={_fmt(eta)}",
            flush=True,
        )


def _fmt(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/math500_eval_config.json")
    parser.add_argument("--data_path", default=DEFAULT_DATA_PATH)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--correct_keep_prob", type=float, default=0.25)
    parser.add_argument("--no_resume", action="store_true")
    parser.add_argument("--shuffle_before_limit", action="store_true")
    parser.add_argument("--progress_every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = load_json(args.config)
    stats = build_dataset(
        config=config,
        data_path=args.data_path,
        output=args.output,
        limit=args.limit,
        correct_keep_prob=args.correct_keep_prob,
        resume=not args.no_resume,
        progress_every=args.progress_every,
        seed=args.seed,
        shuffle_before_limit=args.shuffle_before_limit,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
