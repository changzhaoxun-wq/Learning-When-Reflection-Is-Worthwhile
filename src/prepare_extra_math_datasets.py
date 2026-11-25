import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from datasets import Dataset, load_from_disk
import pyarrow.parquet as pq

from src.answer_extractor import extract_final_answer, extract_final_number, normalize_math_answer
from src.difficulty import classify_gsm8k_difficulty
from src.utils import ensure_dir, write_jsonl, save_json


DEFAULT_INPUT_ROOT = os.environ.get("EXTRA_DATA_ROOT", "data/extra_raw")
DEFAULT_OUTPUT_ROOT = os.environ.get("PREPARED_DATA_ROOT", "data/prepared")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output_root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--datasets", nargs="*", default=["gsm1k", "svamp", "gsm_symbolic", "asdiv", "multiarith", "mawps"])
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    summaries = []
    converters = {
        "gsm1k": convert_gsm1k,
        "svamp": convert_svamp,
        "gsm_symbolic": convert_gsm_symbolic,
        "asdiv": convert_asdiv,
        "multiarith": convert_multiarith,
        "mawps": convert_mawps,
    }
    for name in args.datasets:
        if name not in converters:
            raise ValueError(f"Unknown dataset {name}. Choices: {sorted(converters)}")
        summaries.extend(converters[name](input_root, output_root))

    summary_path = output_root / "extra_dataset_summary.json"
    save_json(summaries, summary_path)
    print(f"Wrote summary to {summary_path}")
    for row in summaries:
        print(f"{row['dataset']}/{row['split']}: {row['n']} rows -> {row['sft_path']}")


def convert_gsm1k(input_root: Path, output_root: Path) -> List[Dict[str, Any]]:
    ds = load_from_disk(str(input_root / "GSM1K" / "data"))
    rows = []
    for idx, item in enumerate(ds["test"]):
        rows.append(make_row(
            dataset="gsm1k",
            split="test",
            idx=idx,
            question=str(item["question"]),
            final_answer=item["answer"],
            source="gsm1k",
            raw=dict(item),
        ))
    return write_dataset(output_root, "gsm1k", {"test": rows})


def convert_svamp(input_root: Path, output_root: Path) -> List[Dict[str, Any]]:
    base = input_root / "SVAMP"
    splits = {}
    for split in ("train", "test"):
        data = json.loads((base / f"{split}.json").read_text(encoding="utf-8"))
        rows = []
        for idx, item in enumerate(data):
            body = str(item.get("Body", "")).strip()
            q = str(item.get("Question", "")).strip()
            rows.append(make_row(
                dataset="svamp",
                split=split,
                idx=idx,
                question=_join_question_parts(body, q),
                final_answer=item.get("Answer"),
                source=str(item.get("Type", "unknown")),
                raw=item,
                item_id=str(item.get("ID") or f"{split}-{idx}"),
                solution=str(item.get("Equation", "")).strip(),
            ))
        splits[split] = rows
    return write_dataset(output_root, "svamp", splits)


def convert_gsm_symbolic(input_root: Path, output_root: Path) -> List[Dict[str, Any]]:
    base = input_root / "GSM-Symbolic"
    splits = {}
    for variant in ("main", "p1", "p2"):
        path = base / variant / "test.jsonl"
        rows = []
        for idx, item in enumerate(read_jsonl(path)):
            original_id = item.get("original_id", "unknown")
            instance = item.get("instance", idx)
            rows.append(make_row(
                dataset="gsm_symbolic",
                split=f"{variant}_test",
                idx=idx,
                question=str(item.get("question", "")),
                final_answer=extract_final_answer(item.get("answer")),
                source=variant,
                raw=item,
                item_id=f"{variant}-{original_id}-{instance}",
                solution=str(item.get("answer", "")).split("####")[0].strip(),
                metadata={
                    "variant": variant,
                    "original_id": original_id,
                    "instance": instance,
                    "original_question": item.get("original_question"),
                },
            ))
        splits[f"{variant}_test"] = rows
    all_rows = []
    for rows in splits.values():
        all_rows.extend(rows)
    splits["test"] = all_rows
    return write_dataset(output_root, "gsm_symbolic", splits)


def convert_asdiv(input_root: Path, output_root: Path) -> List[Dict[str, Any]]:
    path = input_root / "ASDiv" / "asdiv" / "validation-00000-of-00001.parquet"
    ds = Dataset(pq.read_table(path))
    rows = []
    for idx, item in enumerate(ds):
        formula = str(item.get("formula", "")).strip()
        rows.append(make_row(
            dataset="asdiv",
            split="test",
            idx=idx,
            question=_join_question_parts(item.get("body", ""), item.get("question", "")),
            final_answer=item.get("answer"),
            source=str(item.get("solution_type", "unknown")),
            raw=dict(item),
            solution=formula,
        ))
    return write_dataset(output_root, "asdiv", {"test": rows})


def convert_multiarith(input_root: Path, output_root: Path) -> List[Dict[str, Any]]:
    base = input_root / "MultiArith"
    splits = {}
    for split in ("train", "test"):
        data = json.loads((base / f"{split}.json").read_text(encoding="utf-8"))
        rows = []
        for idx, item in enumerate(data):
            rows.append(make_row(
                dataset="multiarith",
                split=split,
                idx=idx,
                question=str(item.get("question", "")).strip(),
                final_answer=item.get("final_ans"),
                source="multiarith",
                raw=item,
            ))
        splits[split] = rows
    return write_dataset(output_root, "multiarith", splits)


def convert_mawps(input_root: Path, output_root: Path) -> List[Dict[str, Any]]:
    base = input_root / "MAWPS" / "data"
    files = {
        "train": next(base.glob("train-*.parquet")),
        "validation": next(base.glob("validation-*.parquet")),
        "test": next(base.glob("test-*.parquet")),
    }
    splits = {}
    for split, path in files.items():
        ds = Dataset(pq.read_table(path))
        rows = []
        for idx, item in enumerate(ds):
            rows.append(make_row(
                dataset="mawps",
                split=split,
                idx=idx,
                question=str(item.get("question", "")).strip(),
                final_answer=item.get("result"),
                source="mawps",
                raw=dict(item),
                item_id=str(item.get("id") or f"{split}-{idx}"),
                solution=str(item.get("chain") or item.get("equation") or "").strip(),
            ))
        splits[split] = rows
    return write_dataset(output_root, "mawps", splits)


def make_row(dataset: str, split: str, idx: int, question: str, final_answer: Any, source: str, raw: Dict[str, Any], item_id: Optional[str] = None, solution: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    question = _clean_text(question)
    gold = extract_final_number(str(final_answer))
    if gold is None:
        gold = extract_final_answer(str(final_answer))
    if gold is None:
        gold = normalize_math_answer(str(final_answer))
    gold = str(gold) if gold is not None else ""
    solution_text = _clean_solution(solution)
    answer = f"{solution_text}\n#### {gold}" if solution_text else f"#### {gold}"
    row_id = _safe_id(dataset, split, item_id or str(idx))
    difficulty = classify_gsm8k_difficulty(question)
    meta = dict(metadata or {})
    meta.update({"raw": _jsonable(raw)})
    return {
        "id": row_id,
        "dataset": dataset,
        "source": source or dataset,
        "split": split,
        "question": question,
        "answer": answer,
        "gold_answer": gold,
        "difficulty": difficulty["difficulty"],
        "difficulty_id": difficulty["difficulty_id"],
        "difficulty_score": difficulty["difficulty_score"],
        "difficulty_features": difficulty["difficulty_features"],
        "messages": [
            {"role": "user", "content": f"Solve the math problem step by step and provide the final answer.\n\nProblem:\n{question}"},
            {"role": "assistant", "content": answer},
        ],
        "metadata": meta,
    }


def write_dataset(output_root: Path, dataset: str, splits: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    summaries = []
    for mode in ("normalized_extra", "sft_extra"):
        for split, rows in splits.items():
            out_dir = output_root / mode / dataset
            ensure_dir(out_dir)
            out_path = out_dir / f"{split}.jsonl"
            write_jsonl([_normalized_view(row) for row in rows] if mode == "normalized_extra" else rows, out_path)
    for split, rows in splits.items():
        summaries.append({
            "dataset": dataset,
            "split": split,
            "n": len(rows),
            "sft_path": str(output_root / "sft_extra" / dataset / f"{split}.jsonl"),
            "normalized_path": str(output_root / "normalized_extra" / dataset / f"{split}.jsonl"),
            "sources": _counts(row.get("source", dataset) for row in rows),
            "difficulties": _counts(row.get("difficulty", "unknown") for row in rows),
        })
    return summaries


def _normalized_view(row: Dict[str, Any]) -> Dict[str, Any]:
    keys = ["id", "dataset", "source", "split", "question", "answer", "gold_answer", "difficulty", "difficulty_id", "difficulty_score", "difficulty_features", "metadata"]
    return {key: row[key] for key in keys if key in row}


def _counts(values: Iterable[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for value in values:
        out[str(value)] = out.get(str(value), 0) + 1
    return dict(sorted(out.items()))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _join_question_parts(*parts: Any) -> str:
    return _clean_text(" ".join(str(part).strip() for part in parts if str(part).strip()))


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _clean_solution(solution: Optional[str]) -> str:
    if solution is None:
        return ""
    return re.sub(r"\s+", " ", str(solution).strip())


def _safe_id(dataset: str, split: str, value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value).strip()).strip("-")
    return f"{dataset}-{split}-{value}"


def _jsonable(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        if isinstance(value, dict):
            return {k: _jsonable(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_jsonable(v) for v in value]
        return str(value)


if __name__ == "__main__":
    main()
