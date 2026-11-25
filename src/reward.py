from typing import Any, Iterable, Optional

from .answer_extractor import answers_equal_for_task, best_qa_equivalence_match, best_text_answer_match


def compute_match_metrics(
    pred: Optional[str],
    gold_answer: Optional[str],
    task_type: str | None = None,
    gold_aliases: Iterable[Any] | None = None,
    qa_f1_threshold: float = 0.8,
) -> dict[str, Any]:
    task = str(task_type or "math").lower()
    if task == "qa":
        match = best_text_answer_match(pred, gold_answer, aliases=gold_aliases)
        enhanced = best_qa_equivalence_match(pred, gold_answer, aliases=gold_aliases)
        correct = int(match["exact_match"] or match["f1"] >= qa_f1_threshold)
        enhanced_correct = int(
            enhanced["exact_match"] or enhanced["f1"] >= qa_f1_threshold
        )
        return {
            "correct": correct,
            "exact_match": float(match["exact_match"]),
            "answer_f1": float(match["f1"]),
            "matched_gold": match.get("matched_gold"),
            "enhanced_correct": enhanced_correct,
            "enhanced_exact_match": float(enhanced["exact_match"]),
            "enhanced_answer_f1": float(enhanced["f1"]),
            "enhanced_matched_gold": enhanced.get("matched_gold"),
        }
    correct = int(answers_equal_for_task(pred, gold_answer, task_type=task))
    return {
        "correct": correct,
        "exact_match": float(correct),
        "answer_f1": float(correct),
        "matched_gold": gold_answer,
    }


def compute_correct(
    pred: Optional[str],
    gold_answer: Optional[str],
    task_type: str | None = None,
    gold_aliases: Iterable[Any] | None = None,
    qa_f1_threshold: float = 0.8,
) -> int:
    return compute_match_metrics(
        pred,
        gold_answer,
        task_type=task_type,
        gold_aliases=gold_aliases,
        qa_f1_threshold=qa_f1_threshold,
    )["correct"]


def compute_utility_reward(utility_score: float, token_cost: int, lambda_cost: float) -> float:
    return float(utility_score) - float(lambda_cost) * float(token_cost)


def compute_reward(correct: int, token_cost: int, lambda_cost: float) -> float:
    return compute_utility_reward(float(correct), token_cost, lambda_cost)
