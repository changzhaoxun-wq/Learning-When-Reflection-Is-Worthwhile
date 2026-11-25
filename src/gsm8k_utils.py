from pathlib import Path
from typing import Any, Dict, List, Optional

from .answer_extractor import extract_answer, extract_final_answer, extract_final_number
from .difficulty import classify_gsm8k_difficulty
from .utils import project_root, read_jsonl, write_jsonl


FALLBACK_GSM8K = [
    {
        "id": "toy_0",
        "question": "Mia has 12 apples and gives 5 to her friend. How many apples does Mia have left?",
        "answer": "Mia has 12 - 5 = 7 apples left. #### 7",
    },
    {
        "id": "toy_1",
        "question": "A box has 6 packs of pencils with 4 pencils in each pack. How many pencils are there?",
        "answer": "There are 6 * 4 = 24 pencils. #### 24",
    },
]


def extract_gold_answer(answer_text: str, dataset: str | None = None, task_type: str | None = None) -> Optional[str]:
    task = str(task_type or "math").lower()
    if task == "qa":
        return str(answer_text).strip() if str(answer_text).strip() else None
    if dataset and dataset.upper().startswith("MATH"):
        return extract_final_answer(answer_text)
    if "####" in str(answer_text):
        return extract_final_number(answer_text)
    return extract_final_answer(answer_text)


def normalize_gsm8k_item(item: Dict[str, Any], idx: int, split: str) -> Dict[str, Any]:
    return normalize_example_item(item, idx=idx, split=split)


def normalize_example_item(item: Dict[str, Any], idx: int, split: str) -> Dict[str, Any]:
    task_type = str(item.get("task_type") or _infer_task_type(item)).lower()
    question = item.get("question") or item.get("problem") or item.get("input")
    answer_text = item.get("answer") or item.get("solution") or item.get("target")

    if (question is None or answer_text is None) and item.get("messages"):
        for message in item["messages"]:
            if message.get("role") == "user" and question is None:
                question = _clean_question(str(message.get("content", "")))
            if message.get("role") == "assistant":
                content = str(message.get("content", ""))
                if "####" in content or answer_text is None:
                    answer_text = content

    question = _clean_question(str(question or ""))
    context = str(item.get("context") or "").strip()
    if task_type == "qa" and context and "Context:" not in question:
        question_for_model = f"Context:\n{context}\n\nQuestion:\n{question}"
    else:
        question_for_model = question

    answer_text = str(answer_text or item.get("gold_answer") or "")
    dataset = str(item.get("dataset") or ("qa" if task_type == "qa" else "gsm8k"))
    item_id = str(item.get("id", f"{split}_{idx}"))
    source = _extract_source(item, dataset=dataset, item_id=item_id)
    gold_answer = item.get("gold_answer") or extract_gold_answer(answer_text, dataset=dataset, task_type=task_type)
    gold_aliases = _extract_gold_aliases(item)
    difficulty = _difficulty_for_item(item, question_for_model, task_type=task_type)

    return {
        "id": item_id,
        "dataset": dataset,
        "source": source,
        "task_type": task_type,
        "question": question_for_model,
        "raw_question": question,
        "context": context,
        "answer": answer_text,
        "gold_answer": gold_answer,
        "gold_aliases": gold_aliases,
        "difficulty": difficulty["difficulty"],
        "difficulty_id": difficulty["difficulty_id"],
        "difficulty_score": difficulty["difficulty_score"],
        "difficulty_features": difficulty.get("difficulty_features", {}),
        "metadata": item.get("metadata", {}),
    }



def _extract_gold_aliases(item: Dict[str, Any]) -> List[str]:
    aliases: List[str] = []
    for key in ("gold_aliases", "answer_aliases", "aliases", "answers"):
        value = item.get(key)
        if value:
            aliases.extend(_flatten_aliases(value))
    metadata = item.get("metadata") or {}
    if isinstance(metadata, dict):
        for key in ("gold_aliases", "answer_aliases", "aliases", "answers"):
            value = metadata.get(key)
            if value:
                aliases.extend(_flatten_aliases(value))
    seen = set()
    unique = []
    for alias in aliases:
        alias = str(alias).strip()
        if alias and alias not in seen:
            seen.add(alias)
            unique.append(alias)
    return unique


def _flatten_aliases(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        out: List[str] = []
        for nested in value.values():
            out.extend(_flatten_aliases(nested))
        return out
    if isinstance(value, (list, tuple, set)):
        out: List[str] = []
        for nested in value:
            out.extend(_flatten_aliases(nested))
        return out
    return [str(value)]

def _infer_task_type(item: Dict[str, Any]) -> str:
    dataset = str(item.get("dataset") or "").lower()
    if dataset in {"hotpotqa", "2wikimultihopqa", "musique"}:
        return "qa"
    if item.get("context") and not ("####" in str(item.get("answer", ""))):
        return "qa"
    return "math"


def _difficulty_for_item(item: Dict[str, Any], question: str, task_type: str) -> Dict[str, Any]:
    existing = item.get("difficulty") or item.get("level")
    if task_type == "qa" and existing:
        label = _normalize_difficulty_label(str(existing))
        return {
            "difficulty": label,
            "difficulty_id": {"easy": 0, "medium": 1, "hard": 2}.get(label, 3),
            "difficulty_score": {"easy": 0.2, "medium": 0.5, "hard": 0.8}.get(label, 0.5),
            "difficulty_features": {"source_label": str(existing)},
        }
    if task_type == "math" and existing and str(existing).lower().startswith("level"):
        label = _math_level_to_difficulty(str(existing))
        return {
            "difficulty": label,
            "difficulty_id": {"easy": 0, "medium": 1, "hard": 2}.get(label, 3),
            "difficulty_score": {"easy": 0.25, "medium": 0.5, "hard": 0.8}.get(label, 0.5),
            "difficulty_features": {"source_level": str(existing)},
        }
    return classify_gsm8k_difficulty(question)


def _normalize_difficulty_label(value: str) -> str:
    lowered = value.lower()
    if "easy" in lowered:
        return "easy"
    if "medium" in lowered or "moderate" in lowered:
        return "medium"
    if "hard" in lowered or "challeng" in lowered:
        return "hard"
    return "medium"


def _math_level_to_difficulty(level: str) -> str:
    numbers = [int(ch) for ch in level if ch.isdigit()]
    if not numbers:
        return "medium"
    value = numbers[0]
    if value <= 1:
        return "easy"
    if value <= 3:
        return "medium"
    return "hard"


def _extract_source(item: Dict[str, Any], dataset: str, item_id: str) -> str:
    for key in ("source", "subject", "category", "domain", "type", "level"):
        value = item.get(key)
        if value:
            return str(value)

    metadata = item.get("metadata") or {}
    if isinstance(metadata, dict):
        for key in ("source", "subject", "category", "domain", "type", "level"):
            value = metadata.get(key)
            if value:
                return str(value)
        unique_id = str(metadata.get("unique_id", ""))
        source = _source_from_slash_id(unique_id)
        if source:
            return source

    source = _source_from_slash_id(item_id)
    if source:
        return source
    return dataset or "unknown"


def _source_from_slash_id(item_id: str) -> Optional[str]:
    parts = [part for part in str(item_id).split("/") if part]
    if len(parts) >= 3 and parts[0] in {"train", "test", "validation", "val"}:
        return parts[1]
    if len(parts) >= 2 and parts[0].lower().startswith("math"):
        return parts[1]
    return None


def load_gsm8k(
    split: str = "train",
    limit: Optional[int] = None,
    data_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    raw_items: List[Dict[str, Any]]
    if data_path:
        path = Path(data_path)
        if not path.is_absolute():
            path = project_root() / path
        raw_items = _load_local(path)
    else:
        local_path = project_root() / "data" / f"gsm8k_{split}.jsonl"
        if local_path.exists():
            raw_items = _load_local(local_path)
        else:
            raw_items = _load_from_huggingface(split)

    rows = [normalize_example_item(item, idx, split) for idx, item in enumerate(raw_items)]
    rows = [row for row in rows if row["gold_answer"] is not None]
    if limit is not None:
        rows = rows[: int(limit)]
    return rows


def save_gsm8k_jsonl(rows: List[Dict[str, Any]], path: str) -> None:
    write_jsonl(rows, path)


def _clean_question(text: str) -> str:
    marker = "Problem:"
    if marker in text:
        return text.split(marker, 1)[1].strip()
    return text.strip()


def _load_local(path: Path) -> List[Dict[str, Any]]:
    if path.suffix == ".jsonl":
        return read_jsonl(path)
    if path.suffix == ".json":
        import json

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = data.get("data", data.get("examples", []))
        return list(data)
    raise ValueError(f"Unsupported file format: {path}")


def _load_from_huggingface(split: str) -> List[Dict[str, Any]]:
    try:
        from datasets import load_dataset

        ds = load_dataset("gsm8k", "main", split=split)
        return [dict(item) for item in ds]
    except Exception as exc:
        print(
            f"[gsm8k_utils] Could not load GSM8K via datasets ({exc}). "
            "Using the built-in toy GSM8K-like fallback set."
        )
        return FALLBACK_GSM8K
