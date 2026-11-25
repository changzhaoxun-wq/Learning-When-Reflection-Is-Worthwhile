from __future__ import annotations

from src.prepare_qa_reflection_sampling import select_unrepaired_direct_errors
from src.qa_top_layer_policy import attach_sampled_augmentations


def _candidate(action: str, correct: int, answer: str = "answer") -> dict:
    return {
        "action": action,
        "correct": correct,
        "pred": answer,
        "final_answer": f"Final answer: {answer}",
    }


def _row(row_id: str, direct: int, r1: int, r2: int | None = None) -> dict:
    candidates = [_candidate("direct", direct), _candidate("reflect_1", r1)]
    if r2 is not None:
        candidates.append(_candidate("reflect_2", r2))
    return {
        "id": row_id,
        "question": "Context:\nEvidence.\n\nQuestion:\nWho?",
        "initial_answer": "Final answer: answer",
        "candidate_results": candidates,
    }


def test_selector_keeps_only_unrepaired_direct_errors():
    old = [
        _row("correct", 1, 0, 0),
        _row("old-fix", 0, 1, 0),
        _row("r2-fix", 0, 0, 1),
        _row("unrepaired", 0, 0, 0),
    ]
    new = [
        _row("correct", 1, 0),
        _row("old-fix", 0, 0),
        _row("r2-fix", 0, 0),
        _row("unrepaired", 0, 0),
    ]
    selected, stats = select_unrepaired_direct_errors(old, new)
    assert [row["id"] for row in selected] == ["unrepaired"]
    assert stats["selected_unrepaired"] == 1
    assert [item["action"] for item in selected[0]["candidate_results"]] == [
        "direct"
    ]


def test_sampled_candidates_attach_by_id():
    old = _row("x", 0, 0, 0)
    new = _row("x", 0, 0)
    pair = {
        "id": "x",
        "new_candidates": {"direct": _candidate("direct", 0)},
    }
    sampled = _row("x", 0, 1)
    stats = attach_sampled_augmentations([pair], [[sampled]])
    assert stats["rows_attached"] == 1
    assert stats["correct_candidates"] == 1
    assert pair["sampled_augmentation_candidates"][0]["candidate"]["correct"] == 1
