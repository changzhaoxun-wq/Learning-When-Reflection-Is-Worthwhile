import re
from typing import Any, Dict

from .difficulty import classify_gsm8k_difficulty, difficulty_one_hot
from .utils import count_tokens
from .verifier import BaseVerifier, rule_based_verify


WH_WORDS = {"who", "what", "when", "where", "which", "why", "how", "whose", "whom"}


def estimate_confidence(answer_text: str, verifier_score: float) -> float:
    lowered = (answer_text or "").lower()
    score = verifier_score
    if "final answer" in lowered or "answer:" in lowered:
        score += 0.2
    if any(marker in lowered for marker in ["not sure", "maybe", "uncertain", "cannot determine"]):
        score -= 0.2
    return min(1.0, max(0.0, score))


def extract_state_features(
    question: str,
    initial_answer: str,
    round_id: int,
    token_cost: int,
    verifier: BaseVerifier | None = None,
    task_type: str | None = None,
) -> Dict[str, Any]:
    task = str(task_type or "math").lower()
    verify = verifier.verify(question, initial_answer, task_type=task) if verifier else rule_based_verify(initial_answer, task_type=task)
    difficulty = _classify_difficulty(question, task)
    one_hot = difficulty_one_hot(difficulty["difficulty"])
    question_without_context = _strip_context_prefix(question)
    return {
        "question": question,
        "initial_answer": initial_answer,
        "task_type": task,
        "task_math": 1.0 if task == "math" else 0.0,
        "task_qa": 1.0 if task == "qa" else 0.0,
        "confidence_score": estimate_confidence(initial_answer, verify.verifier_score),
        "verifier_score": verify.verifier_score,
        "verifier_mode": verify.mode,
        "verifier_reason": verify.reason,
        "verifier_prompt_tokens": verify.prompt_tokens,
        "verifier_completion_tokens": verify.completion_tokens,
        "verifier_token_cost": verify.token_cost,
        "answer_length": count_tokens(initial_answer),
        "round_id": round_id,
        "token_cost": token_cost,
        "difficulty": difficulty["difficulty"],
        "difficulty_id": difficulty["difficulty_id"],
        "difficulty_score": difficulty["difficulty_score"],
        "question_length": count_tokens(question_without_context),
        "context_length": _context_length(question),
        "question_num_count": _count_numbers(question_without_context),
        "question_entity_count": _count_capitalized_phrases(question_without_context),
        "wh_word_count": _count_wh_words(question_without_context),
        "initial_answer_has_final_marker": 1.0 if _has_final_marker(initial_answer) else 0.0,
        **one_hot,
    }


def _classify_difficulty(question: str, task: str) -> Dict[str, Any]:
    if task != "qa":
        return classify_gsm8k_difficulty(question)
    q = _strip_context_prefix(question)
    context_len = _context_length(question)
    wh_count = _count_wh_words(q)
    entity_count = _count_capitalized_phrases(q)
    sentence_count = max(1, len(re.findall(r"[.!?]", q)) or 1)
    comparison = 1.0 if any(w in q.lower() for w in ["same", "both", "larger", "older", "first", "more", "before", "after"]) else 0.0
    length_score = min(count_tokens(q) / 35.0, 1.0)
    context_score = min(context_len / 900.0, 1.0)
    entity_score = min(entity_count / 6.0, 1.0)
    wh_score = min(wh_count / 3.0, 1.0)
    sentence_score = min(sentence_count / 3.0, 1.0)
    score = 0.20 * length_score + 0.30 * context_score + 0.20 * entity_score + 0.15 * wh_score + 0.10 * comparison + 0.05 * sentence_score
    if score < 0.34:
        label, label_id = "easy", 0
    elif score < 0.62:
        label, label_id = "medium", 1
    else:
        label, label_id = "hard", 2
    return {
        "difficulty": label,
        "difficulty_id": label_id,
        "difficulty_score": round(score, 6),
        "difficulty_features": {
            "question_length": count_tokens(q),
            "context_length": context_len,
            "wh_word_count": wh_count,
            "entity_count": entity_count,
            "comparison_hint": comparison,
            "sentence_count": sentence_count,
        },
    }


def _strip_context_prefix(question: str) -> str:
    text = str(question or "")
    marker = "Question:"
    if marker in text:
        return text.rsplit(marker, 1)[-1].strip()
    return text.strip()


def _context_length(question: str) -> int:
    text = str(question or "")
    marker = "Question:"
    if marker not in text:
        return 0
    return count_tokens(text.split(marker, 1)[0])


def _count_numbers(text: str) -> int:
    return len(re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", str(text)))


def _count_capitalized_phrases(text: str) -> int:
    return len(re.findall(r"\b[A-Z][\w'’.-]*(?:\s+[A-Z][\w'’.-]*)*", str(text)))


def _count_wh_words(text: str) -> int:
    words = re.findall(r"[a-zA-Z]+", str(text).lower())
    return sum(1 for word in words if word in WH_WORDS)


def _has_final_marker(text: str) -> bool:
    lowered = str(text or "").lower()
    return "final answer" in lowered or "answer:" in lowered
