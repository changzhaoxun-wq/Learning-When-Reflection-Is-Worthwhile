import math
import re
import string
import unicodedata
from fractions import Fraction
from collections import Counter
from typing import Any, Iterable, Optional


NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")
FINAL_LINE_RE = re.compile(
    r"(?:final\s+answer|answer\s+is|answer)\s*[:=]\s*(.+)",
    flags=re.IGNORECASE,
)


def normalize_answer(value: Optional[str]) -> Optional[str]:
    """Normalize a numeric answer when possible, preserving strings otherwise."""
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    numeric_candidate = value.replace(",", "")
    try:
        number = float(numeric_candidate)
    except ValueError:
        return value
    if math.isfinite(number) and abs(number - round(number)) < 1e-9:
        return str(int(round(number)))
    return ("%f" % number).rstrip("0").rstrip(".")


def normalize_math_answer(value: Optional[str]) -> Optional[str]:
    """Canonicalize a GSM8K/MATH-style final answer string."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None

    if "####" in text:
        text = text.split("####")[-1].strip()

    boxed = _extract_boxed_candidates(text)
    if boxed:
        text = boxed[-1]

    text = text.strip().strip("*_` ").strip("$ ")
    text = re.sub(r"^```(?:latex|tex|math)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()
    text = _strip_trailing_sentence_punctuation(text)

    replacements = {
        "π": "pi",
        r"\left": "",
        r"\right": "",
        r"\!": "",
        r"\,": "",
        r"\;": "",
        r"\:": "",
        r"\cdot": "*",
        r"\times": "*",
        r"\pi": "pi",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)

    text = re.sub(r"\\(?:dfrac|tfrac|frac)\{([^{}]+)\}\{([^{}]+)\}", r"\1/\2", text)
    text = re.sub(r"\\text\{([^{}]*)\}", r"\1", text)
    text = text.replace(" ", "")
    text = text.replace("{", "").replace("}", "")
    text = text.strip().strip("*_` ").strip("$ ")
    text = _strip_trailing_sentence_punctuation(text)

    numeric = normalize_answer(text)
    return numeric if numeric is not None else text or None


def normalize_text_answer(value: Optional[str]) -> Optional[str]:
    """SQuAD/HotpotQA-style normalization for short text answers."""
    if value is None:
        return None
    text = str(value).strip().lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    if not text:
        return None
    text = re.sub(r"^```(?:text)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip().strip('"\'` ')
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch if ch not in string.punctuation else " " for ch in text)
    text = " ".join(text.split())
    return text or None


def extract_final_number(text: Optional[str]) -> Optional[str]:
    """Extract the final numeric answer from an LLM response or GSM8K answer."""
    if text is None:
        return None
    text = str(text)

    if "####" in text:
        tail = text.split("####")[-1]
        match = NUMBER_RE.search(tail)
        if match:
            return normalize_answer(match.group(0))

    patterns = [
        r"(?:final answer|answer is|answer|therefore|so)\s*[:=]?\s*([-+]?\d[\d,]*(?:\.\d+)?)",
        r"boxed\{\s*([-+]?\d[\d,]*(?:\.\d+)?)\s*\}",
    ]
    lowered = text.lower()
    for pattern in patterns:
        matches = re.findall(pattern, lowered, flags=re.IGNORECASE)
        if matches:
            return normalize_answer(matches[-1])

    matches = NUMBER_RE.findall(text)
    if not matches:
        return None
    return normalize_answer(matches[-1])


def extract_final_answer(text: Optional[str]) -> Optional[str]:
    """Extract a final answer for GSM8K or MATH-style outputs."""
    if text is None:
        return None
    raw = str(text).strip()
    if not raw:
        return None

    if "####" in raw:
        return normalize_math_answer(raw.split("####")[-1])

    boxed = _extract_boxed_candidates(raw)
    if boxed:
        return normalize_math_answer(boxed[-1])

    final_matches = FINAL_LINE_RE.findall(raw)
    if final_matches:
        candidate = final_matches[-1].strip().splitlines()[0]
        return normalize_math_answer(candidate)

    if _looks_like_compact_math_answer(raw):
        return normalize_math_answer(raw)

    numeric = extract_final_number(raw)
    if numeric is not None:
        return numeric
    return normalize_math_answer(raw)


def extract_final_text_answer(text: Optional[str]) -> Optional[str]:
    """Extract a short final answer for open-domain QA outputs."""
    if text is None:
        return None
    raw = str(text).strip()
    if not raw:
        return None
    final_matches = FINAL_LINE_RE.findall(raw)
    if final_matches:
        candidate = final_matches[-1].strip().splitlines()[0]
        return candidate.strip().strip('"\'` ')
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    for line in reversed(lines):
        lowered = line.lower()
        if lowered.startswith(("therefore", "so ", "answer:")) or len(line.split()) <= 12:
            line = re.sub(r"^(therefore|so|answer)\s*[:=,]?\s*", "", line, flags=re.IGNORECASE)
            return line.strip().strip('"\'` ')
    return lines[-1].strip().strip('"\'` ') if lines else None


def extract_answer(text: Optional[str], task_type: str | None = None) -> Optional[str]:
    if str(task_type or "math").lower() == "qa":
        return extract_final_text_answer(text)
    return extract_final_answer(text)


def answers_equal(pred: Optional[str], gold: Optional[str], tolerance: float = 1e-6) -> bool:
    pred_norm = normalize_math_answer(pred)
    gold_norm = normalize_math_answer(gold)
    if pred_norm is None or gold_norm is None:
        return False

    pred_float = _to_float(pred_norm)
    gold_float = _to_float(gold_norm)
    if pred_float is not None and gold_float is not None:
        return abs(pred_float - gold_float) <= tolerance
    return pred_norm == gold_norm


_QA_NUMBER_WORDS = {
    "zero": "0", "one": "1", "once": "1", "two": "2", "twice": "2",
    "three": "3", "thrice": "3", "four": "4", "five": "5", "six": "6",
    "seven": "7", "eight": "8", "nine": "9", "ten": "10", "eleven": "11",
    "twelve": "12", "thirteen": "13", "fourteen": "14", "fifteen": "15",
    "sixteen": "16", "seventeen": "17", "eighteen": "18", "nineteen": "19",
    "twenty": "20",
}
_QA_SINGULAR_UNITS = {
    "years": "year", "weeks": "week", "days": "day", "hours": "hour",
    "minutes": "minute", "seconds": "second", "miles": "mile",
    "kilometers": "kilometer", "metres": "metre", "meters": "meter",
    "episodes": "episode", "times": "time", "months": "month",
}
_QA_SPLIT_INITIALISMS = {("u", "s"): "us", ("u", "k"): "uk"}


def normalize_qa_equivalence_answer(value: Optional[str]) -> Optional[str]:
    """Apply conservative QA equivalences in addition to standard normalization."""
    normalized = normalize_text_answer(value)
    if normalized is None:
        return None
    tokens = normalized.split()
    output: list[str] = []
    index = 0
    while index < len(tokens):
        pair = tuple(tokens[index:index + 2])
        if pair in _QA_SPLIT_INITIALISMS:
            output.append(_QA_SPLIT_INITIALISMS[pair])
            index += 2
            continue
        token = _QA_NUMBER_WORDS.get(tokens[index], tokens[index])
        output.append(_QA_SINGULAR_UNITS.get(token, token))
        index += 1
    return " ".join(output) or None


def qa_equivalence_answers_equal(pred: Optional[str], gold: Optional[str]) -> bool:
    pred_norm = normalize_qa_equivalence_answer(pred)
    gold_norm = normalize_qa_equivalence_answer(gold)
    return pred_norm is not None and gold_norm is not None and pred_norm == gold_norm


def qa_equivalence_answer_f1(pred: Optional[str], gold: Optional[str]) -> float:
    pred_norm = normalize_qa_equivalence_answer(pred)
    gold_norm = normalize_qa_equivalence_answer(gold)
    if pred_norm is None or gold_norm is None:
        return 0.0
    pred_tokens = pred_norm.split()
    gold_tokens = gold_norm.split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def text_answers_equal(pred: Optional[str], gold: Optional[str]) -> bool:
    pred_norm = normalize_text_answer(pred)
    gold_norm = normalize_text_answer(gold)
    if pred_norm is None or gold_norm is None:
        return False
    return pred_norm == gold_norm


def text_answer_f1(pred: Optional[str], gold: Optional[str]) -> float:
    pred_norm = normalize_text_answer(pred)
    gold_norm = normalize_text_answer(gold)
    if pred_norm is None or gold_norm is None:
        return 0.0
    pred_tokens = pred_norm.split()
    gold_tokens = gold_norm.split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def coerce_answer_aliases(gold: Optional[str], aliases: Iterable[Any] | None = None) -> list[str]:
    candidates: list[str] = []
    if gold is not None and str(gold).strip():
        candidates.append(str(gold).strip())
    for alias in aliases or []:
        if alias is None:
            continue
        if isinstance(alias, (list, tuple, set)):
            for item in alias:
                if item is not None and str(item).strip():
                    candidates.append(str(item).strip())
        elif str(alias).strip():
            candidates.append(str(alias).strip())
    seen = set()
    unique = []
    for candidate in candidates:
        key = normalize_text_answer(candidate) or candidate
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def best_text_answer_match(pred: Optional[str], gold: Optional[str], aliases: Iterable[Any] | None = None) -> dict[str, Any]:
    golds = coerce_answer_aliases(gold, aliases)
    best = {"exact_match": 0.0, "f1": 0.0, "matched_gold": None}
    for candidate in golds:
        exact = float(text_answers_equal(pred, candidate))
        f1 = text_answer_f1(pred, candidate)
        if (exact, f1) > (best["exact_match"], best["f1"]):
            best = {"exact_match": exact, "f1": f1, "matched_gold": candidate}
    return best


def best_qa_equivalence_match(
    pred: Optional[str],
    gold: Optional[str],
    aliases: Iterable[Any] | None = None,
) -> dict[str, Any]:
    golds = coerce_answer_aliases(gold, aliases)
    best = {"exact_match": 0.0, "f1": 0.0, "matched_gold": None}
    for candidate in golds:
        exact = float(qa_equivalence_answers_equal(pred, candidate))
        f1 = qa_equivalence_answer_f1(pred, candidate)
        if (exact, f1) > (best["exact_match"], best["f1"]):
            best = {"exact_match": exact, "f1": f1, "matched_gold": candidate}
    return best


def answers_equal_for_task(
    pred: Optional[str],
    gold: Optional[str],
    task_type: str | None = None,
    aliases: Iterable[Any] | None = None,
    qa_f1_threshold: float = 0.8,
) -> bool:
    if str(task_type or "math").lower() == "qa":
        match = best_text_answer_match(pred, gold, aliases=aliases)
        return bool(match["exact_match"] or match["f1"] >= qa_f1_threshold)
    return answers_equal(pred, gold)


def _looks_like_compact_math_answer(text: str) -> bool:
    stripped = text.strip()
    if not stripped or "\n" in stripped or len(stripped) > 200:
        return False
    math_markers = ["\\", "^", "_", "=", "/", "(", ")", "[", "]"]
    return any(marker in stripped for marker in math_markers)


def _to_float(value: str) -> Optional[float]:
    value = value.replace(",", "")
    try:
        return float(value)
    except ValueError:
        pass
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?/[-+]?\d+(?:\.\d+)?", value):
        try:
            return float(Fraction(value))
        except (ValueError, ZeroDivisionError):
            return None
    return None


def _strip_trailing_sentence_punctuation(text: str) -> str:
    while text and text[-1] in ".。":
        text = text[:-1].rstrip()
    return text.strip()


def _extract_boxed_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    for marker in (r"\boxed{", "boxed{"):
        start = 0
        while True:
            idx = text.find(marker, start)
            if idx < 0:
                break
            brace_start = idx + len(marker) - 1
            extracted = _extract_balanced_braces(text, brace_start)
            if extracted is not None:
                candidates.append(extracted)
                start = brace_start + len(extracted) + 2
            else:
                start = idx + len(marker)
    return candidates


def _extract_balanced_braces(text: str, brace_start: int) -> Optional[str]:
    if brace_start >= len(text) or text[brace_start] != "{":
        return None
    depth = 0
    content_start = brace_start + 1
    for pos in range(brace_start, len(text)):
        char = text[pos]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[content_start:pos]
    return None
