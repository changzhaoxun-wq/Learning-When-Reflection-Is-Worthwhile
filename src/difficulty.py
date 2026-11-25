import re
from typing import Any, Dict


HARD_KEYWORDS = {
    "percent", "percentage", "ratio", "average", "rate", "per", "profit",
    "discount", "interest", "fraction", "remaining", "left", "twice",
    "half", "third", "quarter", "difference", "combined", "altogether",
    "each", "every", "total", "more than", "less than", "at least",
    "at most", "cost", "earn", "spend", "change", "times",
}

OPERATION_HINTS = {
    "add": {"total", "altogether", "combined", "sum", "in all", "more", "plus"},
    "subtract": {"left", "remaining", "remain", "less", "difference", "after", "change"},
    "multiply": {"each", "every", "per", "times", "twice", "double", "product"},
    "divide": {"half", "third", "quarter", "ratio", "average", "split", "share", "per"},
}


def classify_gsm8k_difficulty(question: str) -> Dict[str, Any]:
    """Question-only heuristic difficulty split for GSM8K.

    The split intentionally avoids using the gold answer or model outputs, so it can
    be used both for analysis and as a policy feature without label leakage.
    """
    text = question or ""
    lowered = text.lower()
    words = re.findall(r"[a-zA-Z]+", lowered)
    numbers = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text)

    question_length = len(words)
    number_count = len(numbers)
    sentence_count = max(1, len(re.findall(r"[.!?]", text)) or 1)
    keyword_count = sum(1 for kw in HARD_KEYWORDS if kw in lowered)
    operation_types = sum(1 for hints in OPERATION_HINTS.values() if any(h in lowered for h in hints))

    length_score = min(question_length / 90.0, 1.0)
    number_score = min(number_count / 8.0, 1.0)
    sentence_score = min(sentence_count / 4.0, 1.0)
    keyword_score = min(keyword_count / 6.0, 1.0)
    operation_score = min(operation_types / 4.0, 1.0)

    score = (
        0.25 * length_score
        + 0.25 * number_score
        + 0.15 * sentence_score
        + 0.20 * keyword_score
        + 0.15 * operation_score
    )

    if score < 0.34:
        label = "easy"
        label_id = 0
    elif score < 0.58:
        label = "medium"
        label_id = 1
    else:
        label = "hard"
        label_id = 2

    return {
        "difficulty": label,
        "difficulty_id": label_id,
        "difficulty_score": round(score, 6),
        "difficulty_features": {
            "question_length": question_length,
            "number_count": number_count,
            "sentence_count": sentence_count,
            "hard_keyword_count": keyword_count,
            "operation_type_count": operation_types,
            "length_score": round(length_score, 6),
            "number_score": round(number_score, 6),
            "sentence_score": round(sentence_score, 6),
            "keyword_score": round(keyword_score, 6),
            "operation_score": round(operation_score, 6),
        },
    }


def difficulty_one_hot(label: str) -> Dict[str, float]:
    return {
        "difficulty_easy": 1.0 if label == "easy" else 0.0,
        "difficulty_medium": 1.0 if label == "medium" else 0.0,
        "difficulty_hard": 1.0 if label == "hard" else 0.0,
    }
