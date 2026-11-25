from __future__ import annotations

import numpy as np

from src.qa_top_layer_policy import (
    TASK_GATE,
    TASK_RANKER,
    _build_examples,
    _gate_training_label,
    _ranker_training_label,
    _split_rows,
    configure_trainable_top_layers,
)


class _Parameter:
    def __init__(self) -> None:
        self.requires_grad = True

    def numel(self) -> int:
        return 1


class _Layer:
    def __init__(self) -> None:
        self.values = [_Parameter()]

    def parameters(self):
        return iter(self.values)


class _Encoder:
    def __init__(self) -> None:
        self.layers = [_Layer() for _ in range(4)]
        self.norm = _Layer()

    def parameters(self):
        for layer in [*self.layers, self.norm]:
            yield from layer.parameters()


def _row(identifier: str, direct: int, old_reflect: int, new_reflect: int):
    base = {
        "id": identifier,
        "task_type": "qa",
        "raw_question": "Who won?",
        "context": "Evidence without a gold label.",
        "initial_answer": "Initial response",
        "candidate_results": [
            {"action": "direct", "correct": direct, "final_answer": "Initial response"},
            {"action": "reflect_1", "correct": old_reflect, "final_answer": "Old candidate"},
        ],
    }
    newer = dict(base)
    newer["candidate_results"] = [
        {"action": "direct", "correct": direct, "final_answer": "Initial response"},
        {"action": "reflect_1", "correct": new_reflect, "final_answer": "New candidate"},
    ]
    return base, newer


def test_only_final_blocks_and_norm_are_trainable():
    encoder = _Encoder()
    info = configure_trainable_top_layers(encoder, 2)
    assert info["frozen_blocks"] == 2
    assert [next(layer.parameters()).requires_grad for layer in encoder.layers] == [
        False,
        False,
        True,
        True,
    ]
    assert next(encoder.norm.parameters()).requires_grad


def test_examples_are_pure_text_and_tasks_are_separate():
    from src.qa_dual_candidate_policy import align_dual_rows, build_dual_targets

    old, new = _row("x", 0, 1, 0)
    pairs = align_dual_rows([old], [new])
    examples = _build_examples(
        pairs, build_dual_targets(pairs), [0], 8000, None, 42
    )
    assert [item["task"] for item in examples] == [TASK_GATE, TASK_RANKER, TASK_RANKER]
    assert [item["label"] for item in examples] == [1, 1, 0]
    text = "\n".join(item["text"] for item in examples).lower()
    assert "gold answer" not in text
    assert "reward" not in text
    assert "verifier score" not in text


def test_reflect_2_is_train_only_augmentation():
    from src.qa_dual_candidate_policy import align_dual_rows, build_dual_targets

    old, new = _row("x", 0, 1, 0)
    old["candidate_results"].append(
        {
            "action": "reflect_2",
            "correct": 1,
            "final_answer": "Second-round candidate",
        }
    )
    pairs = align_dual_rows([old], [new])
    examples = _build_examples(
        pairs,
        build_dual_targets(pairs),
        [0],
        8000,
        None,
        42,
        include_reflect_2_augmentation=True,
    )
    assert {item["source_index"] for item in examples} == {0, 1, 2, 3}
    augmented = next(item for item in examples if item["source_index"] == 3)
    assert augmented["label"] == 1
    assert "v5_reflect_2_train_only" in augmented["text"]


def test_split_keeps_row_level_strata():
    targets = {
        "complete": np.asarray([0, 0, 1, 1] * 10, dtype=bool),
        "direct_error_target": np.asarray([0, 0, 1, 1] * 10),
        "gate_target": np.asarray([0, 0, 0, 1] * 10),
    }
    train, validation = _split_rows(targets, 0.2, 42)
    assert len(train) == 32
    assert len(validation) == 8
    assert not set(train).intersection(validation)


def test_utility_targets_measure_gain_over_direct():
    from src.qa_dual_candidate_policy import align_dual_rows, build_dual_targets

    old, new = _row("gain", 1, 0, 0)
    direct = new["candidate_results"][0]
    old_candidate = old["candidate_results"][1]
    new_candidate = new["candidate_results"][1]
    direct["qa_utility_score"] = 0.40
    old["candidate_results"][0]["qa_utility_score"] = 0.40
    old_candidate["qa_utility_score"] = 0.60
    new_candidate["qa_utility_score"] = 0.44
    pairs = align_dual_rows([old], [new])
    targets = build_dual_targets(pairs)

    assert _ranker_training_label(
        direct, old_candidate, "utility_improvement", 0.05
    ) == 1
    assert _ranker_training_label(
        direct, new_candidate, "utility_improvement", 0.05
    ) == 0
    assert _gate_training_label(
        pairs[0], targets, 0, "union_utility_gain", False, 0.05
    ) == 1
