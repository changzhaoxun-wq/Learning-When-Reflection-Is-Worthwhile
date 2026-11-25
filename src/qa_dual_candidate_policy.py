from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
from sklearn.model_selection import StratifiedKFold

from .accuracy_first_qa_policy import (
    _fit_fold,
    _fold_count,
    _predict_probability,
    encode_text_features,
    ensemble_probability,
)
from .evaluate_policy_offline import _candidate_usage, _candidate_utility
from .neural_policy import _clip_context, _qa_question_and_context, build_policy_text
from .utils import read_jsonl, save_json, save_pickle


CANDIDATE_SOURCES = ("direct", "v5_reflect_1", "v7_reflect_1")


def _candidate_map(row: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {
        str(candidate.get("action")): candidate
        for candidate in row.get("candidate_results", [])
    }


def _correct(candidate: Dict[str, Any] | None) -> int:
    return int(float((candidate or {}).get("correct", 0.0)) >= 0.5)


def align_dual_rows(
    old_rows: Sequence[Dict[str, Any]],
    new_rows: Sequence[Dict[str, Any]],
    require_complete: bool = False,
) -> List[Dict[str, Any]]:
    """Align old/new reflection traces without copying their large contexts."""

    if len(old_rows) != len(new_rows):
        raise ValueError(
            f"Trace lengths differ: old={len(old_rows)} new={len(new_rows)}"
        )
    pairs: List[Dict[str, Any]] = []
    for index, (old_row, new_row) in enumerate(zip(old_rows, new_rows)):
        old_id = str(old_row.get("id"))
        new_id = str(new_row.get("id"))
        if old_id != new_id:
            raise ValueError(
                f"Trace IDs differ at row {index}: old={old_id} new={new_id}"
            )
        old_candidates = _candidate_map(old_row)
        new_candidates = _candidate_map(new_row)
        if "direct" not in old_candidates or "direct" not in new_candidates:
            raise ValueError(f"Trace {old_id} has no direct candidate")
        old_complete = "reflect_1" in old_candidates
        new_complete = "reflect_1" in new_candidates
        if old_complete != new_complete:
            raise ValueError(
                f"Trace {old_id} has mismatched reflection coverage: "
                f"old={old_complete} new={new_complete}"
            )
        if require_complete and not old_complete:
            raise ValueError(f"Trace {old_id} has no paired reflection candidates")
        if _correct(old_candidates["direct"]) != _correct(
            new_candidates["direct"]
        ):
            raise ValueError(f"Trace {old_id} has inconsistent direct labels")
        pairs.append(
            {
                "id": old_id,
                "old_row": old_row,
                "new_row": new_row,
                "old_candidates": old_candidates,
                "new_candidates": new_candidates,
                "complete": old_complete,
            }
        )
    return pairs


def pair_candidates(pair: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not pair["complete"]:
        raise ValueError(f"Trace {pair['id']} has no paired reflection candidates")
    return [
        pair["new_candidates"]["direct"],
        pair["old_candidates"]["reflect_1"],
        pair["new_candidates"]["reflect_1"],
    ]


def build_dual_targets(pairs: Sequence[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    n = len(pairs)
    complete = np.asarray([bool(pair["complete"]) for pair in pairs], dtype=bool)
    correctness = np.zeros((n, len(CANDIDATE_SOURCES)), dtype="int64")
    utility = np.zeros((n, len(CANDIDATE_SOURCES)), dtype="float32")
    for index, pair in enumerate(pairs):
        direct = pair["new_candidates"]["direct"]
        correctness[index, 0] = _correct(direct)
        utility[index, 0] = _candidate_utility(direct)
        if not pair["complete"]:
            continue
        for source_index, candidate in enumerate(pair_candidates(pair)[1:], 1):
            correctness[index, source_index] = _correct(candidate)
            utility[index, source_index] = _candidate_utility(candidate)
    gate_target = (
        (correctness[:, 0] == 0)
        & ((correctness[:, 1] == 1) | (correctness[:, 2] == 1))
        & complete
    ).astype("int64")
    direct_error_target = (correctness[:, 0] == 0).astype("int64")
    return {
        "complete": complete,
        "correctness": correctness,
        "utility": utility,
        "gate_target": gate_target,
        "direct_error_target": direct_error_target,
    }


def build_candidate_text(
    pair: Dict[str, Any],
    source_index: int,
    max_text_chars: int = 8000,
) -> str:
    """Build a candidate-scoring input with no gold answer or reward fields."""

    if source_index not in range(len(CANDIDATE_SOURCES)):
        raise ValueError(f"Unknown candidate source index: {source_index}")
    row = pair["new_row"]
    candidate = pair_candidates(pair)[source_index]
    state = row.get("state_features", {}) or {}
    question, context = _qa_question_and_context(row, state)
    initial_answer = str(
        row.get("initial_answer") or state.get("initial_answer") or ""
    )
    candidate_answer = str(
        candidate.get("final_answer") or candidate.get("pred") or ""
    )
    source_tag = CANDIDATE_SOURCES[source_index]
    fixed = "\n".join(
        [
            "Question:",
            question,
            "Initial response:",
            initial_answer,
            f"Candidate source: {source_tag}",
            "Candidate response:",
            candidate_answer,
        ]
    )
    if not context:
        return fixed[:max_text_chars] if max_text_chars else fixed
    header = "\nContext:\n"
    budget = len(context)
    if max_text_chars:
        budget = max(0, max_text_chars - len(fixed) - len(header))
    text = fixed + header + _clip_context(context, budget)
    return text[:max_text_chars] if max_text_chars else text


def dual_usage(pair: Dict[str, Any]) -> tuple[float, float, float]:
    """Count Base once and each of the two reflection calls once."""

    direct_new, old_reflect, new_reflect = pair_candidates(pair)
    direct_old = pair["old_candidates"]["direct"]
    base_prompt, base_completion, base_total = _candidate_usage(direct_new)
    old_prompt, old_completion, old_total = _candidate_usage(old_reflect)
    old_base_prompt, old_base_completion, old_base_total = _candidate_usage(
        direct_old
    )
    new_prompt, new_completion, new_total = _candidate_usage(new_reflect)
    prompt = base_prompt + (old_prompt - old_base_prompt) + (
        new_prompt - base_prompt
    )
    completion = base_completion + (old_completion - old_base_completion) + (
        new_completion - base_completion
    )
    total = base_total + (old_total - old_base_total) + (new_total - base_total)
    return max(prompt, 0.0), max(completion, 0.0), max(total, 0.0)


def build_usage_arrays(
    pairs: Sequence[Dict[str, Any]],
) -> Dict[str, np.ndarray]:
    direct = np.asarray(
        [_candidate_usage(pair["new_candidates"]["direct"]) for pair in pairs],
        dtype="float64",
    )
    dual = direct.copy()
    complete = np.asarray([pair["complete"] for pair in pairs], dtype=bool)
    for index in np.flatnonzero(complete):
        dual[index] = dual_usage(pairs[int(index)])
    if np.any(complete):
        mean_increment = np.mean(dual[complete] - direct[complete], axis=0)
        dual[~complete] = direct[~complete] + mean_increment
    return {
        "direct_prompt": direct[:, 0],
        "direct_completion": direct[:, 1],
        "direct_total": direct[:, 2],
        "dual_prompt": dual[:, 0],
        "dual_completion": dual[:, 1],
        "dual_total": dual[:, 2],
    }


def _flat_candidate_indices(
    local_rows: np.ndarray, width: int = len(CANDIDATE_SOURCES)
) -> np.ndarray:
    if not len(local_rows):
        return np.asarray([], dtype="int64")
    return (
        local_rows[:, None] * int(width)
        + np.arange(int(width))[None, :]
    ).reshape(-1)


def _threshold_grid(values: np.ndarray, points: int = 81) -> np.ndarray:
    clean = np.asarray(values, dtype="float64")
    clean = clean[np.isfinite(clean)]
    if not len(clean):
        return np.asarray([0.0, 1.000001], dtype="float64")
    quantiles = np.quantile(clean, np.linspace(0.0, 1.0, int(points)))
    return np.unique(
        np.concatenate(
            [quantiles, [float(clean.min()) - 1e-6, float(clean.max()) + 1e-6]]
        )
    )


def reflection_decision_score(
    candidate_probability: np.ndarray,
    selection_score_mode: str = "correctness_margin",
) -> tuple[np.ndarray, np.ndarray]:
    rows = np.arange(len(candidate_probability))
    best_reflection = np.argmax(candidate_probability[:, 1:], axis=1) + 1
    best_score = candidate_probability[rows, best_reflection]
    if selection_score_mode == "correctness_margin":
        score = best_score - candidate_probability[:, 0]
    elif selection_score_mode == "repair_probability":
        score = best_score
    else:
        raise ValueError(
            f"Unknown selection score mode: {selection_score_mode}"
        )
    return best_reflection, score


def dual_selection_metrics(
    gate_probability: np.ndarray,
    candidate_probability: np.ndarray,
    targets: Dict[str, np.ndarray],
    usage: Dict[str, np.ndarray],
    gate_threshold: float,
    margin_threshold: float,
    lambda_cost: float = 0.0001,
    selection_score_mode: str = "correctness_margin",
) -> Dict[str, Any]:
    n = len(gate_probability)
    rows = np.arange(n)
    complete = targets["complete"]
    best_reflection, decision_score = reflection_decision_score(
        candidate_probability, selection_score_mode
    )
    invoked = gate_probability >= float(gate_threshold)
    adopted = invoked & complete & (
        decision_score >= float(margin_threshold)
    )
    selected = np.where(adopted, best_reflection, 0)
    selected_correct = targets["correctness"][rows, selected]
    selected_utility = targets["utility"][rows, selected]
    direct_correct = targets["correctness"][:, 0]
    fixes = int(np.sum((direct_correct == 0) & (selected_correct == 1)))
    harms = int(np.sum((direct_correct == 1) & (selected_correct == 0)))
    token_cost = np.where(
        invoked, usage["dual_total"], usage["direct_total"]
    )
    reward = selected_utility - float(lambda_cost) * token_cost
    return {
        "n": n,
        "accuracy": float(np.mean(selected_correct)),
        "direct_accuracy": float(np.mean(direct_correct)),
        "avg_reward": float(np.mean(reward)),
        "avg_token_cost": float(np.mean(token_cost)),
        "gate_invocation_rate": float(np.mean(invoked)),
        "candidate_adoption_rate": float(np.mean(adopted)),
        "fixes": fixes,
        "harms": harms,
        "net_fixes": fixes - harms,
        "gate_threshold": float(gate_threshold),
        "margin_threshold": float(margin_threshold),
        "selection_score_mode": selection_score_mode,
    }


def select_dual_thresholds(
    gate_probability: np.ndarray,
    candidate_probability: np.ndarray,
    targets: Dict[str, np.ndarray],
    usage: Dict[str, np.ndarray],
    max_invocation_rate: float = 0.15,
    lambda_cost: float = 0.0001,
    grid_points: int = 81,
    selection_score_mode: str = "correctness_margin",
) -> tuple[float, float, Dict[str, Any]]:
    complete = targets["complete"]
    _, decision_score = reflection_decision_score(
        candidate_probability, selection_score_mode
    )
    gate_grid = _threshold_grid(gate_probability, grid_points)
    margin_grid = _threshold_grid(decision_score[complete], grid_points)
    best_key = None
    best = None
    for gate_threshold in gate_grid:
        invocation_rate = float(np.mean(gate_probability >= gate_threshold))
        if invocation_rate > float(max_invocation_rate) + 1e-12:
            continue
        for margin_threshold in margin_grid:
            metrics = dual_selection_metrics(
                gate_probability,
                candidate_probability,
                targets,
                usage,
                float(gate_threshold),
                float(margin_threshold),
                lambda_cost=lambda_cost,
                selection_score_mode=selection_score_mode,
            )
            key = (
                metrics["accuracy"],
                metrics["avg_reward"],
                metrics["net_fixes"],
                -metrics["gate_invocation_rate"],
                -metrics["candidate_adoption_rate"],
                float(gate_threshold),
                float(margin_threshold),
            )
            if best_key is None or key > best_key:
                best_key = key
                best = (float(gate_threshold), float(margin_threshold), metrics)
    if best is None:
        raise ValueError("No dual-candidate thresholds satisfy max_invocation_rate")
    return best


def _plot_histories(
    gate_histories: Sequence[Dict[str, List[float]]],
    ranker_histories: Sequence[Dict[str, List[float]]],
    output: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for axis, title, histories in (
        (axes[0], "Pre-reflection gate", gate_histories),
        (axes[1], "Dual-candidate correctness ranker", ranker_histories),
    ):
        train = np.mean([item["train_loss"] for item in histories], axis=0)
        val = np.mean([item["val_loss"] for item in histories], axis=0)
        epochs = range(1, len(train) + 1)
        axis.plot(epochs, train, label="OOF mean train")
        axis.plot(epochs, val, label="OOF mean validation")
        axis.set_title(title)
        axis.set_xlabel("epoch")
        axis.set_ylabel("weighted BCE loss")
        axis.grid(alpha=0.2)
        axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def train_dual_candidate_policy(
    old_rows: List[Dict[str, Any]],
    new_rows: List[Dict[str, Any]],
    output: str,
    encoder_name: str,
    gate_cache: str,
    candidate_cache: str,
    curve_output: str,
    history_output: str,
    layers: Sequence[int] = (-4, -3, -2, -1),
    pooling: str = "mean",
    max_length: int = 2048,
    max_text_chars: int = 8000,
    encoder_batch_size: int = 4,
    hidden_sizes: Sequence[int] = (128, 32),
    dropout: float = 0.3,
    epochs: int = 30,
    train_batch_size: int = 128,
    learning_rate: float = 5e-4,
    weight_decay: float = 1e-3,
    folds: int = 5,
    random_state: int = 42,
    device: str = "cuda",
    max_invocation_rate: float = 0.15,
    lambda_cost: float = 0.0001,
    threshold_grid_points: int = 81,
    gate_target_name: str = "direct_error",
    ranker_target_name: str = "repair_advantage",
) -> Dict[str, Any]:
    pairs = align_dual_rows(old_rows, new_rows)
    if not pairs:
        raise ValueError("Cannot train on empty dual traces")
    targets = build_dual_targets(pairs)
    complete_idx = np.flatnonzero(targets["complete"])
    if not len(complete_idx):
        raise ValueError("No rows contain both reflection candidates")

    gate_texts = [
        build_policy_text(
            pair["new_row"], max_text_chars=max_text_chars, include_state=False
        )
        for pair in pairs
    ]
    gate_features, encoder_info = encode_text_features(
        gate_texts,
        encoder_name,
        layers,
        pooling,
        max_length,
        encoder_batch_size,
        device,
        gate_cache,
        "qa_dual_candidate_gate",
    )
    candidate_texts = [
        build_candidate_text(pairs[int(global_index)], source_index, max_text_chars)
        for global_index in complete_idx
        for source_index in range(len(CANDIDATE_SOURCES))
    ]
    all_candidate_features, _ = encode_text_features(
        candidate_texts,
        encoder_name,
        layers,
        pooling,
        max_length,
        encoder_batch_size,
        device,
        candidate_cache,
        "qa_dual_candidate_ranker",
    )
    if gate_target_name == "direct_error":
        gate_target = targets["direct_error_target"]
    elif gate_target_name == "union_repair":
        gate_target = targets["gate_target"]
    else:
        raise ValueError(f"Unknown gate target: {gate_target_name}")

    if ranker_target_name == "repair_advantage":
        candidate_features = all_candidate_features.reshape(
            len(complete_idx), len(CANDIDATE_SOURCES), -1
        )[:, 1:, :].reshape(-1, all_candidate_features.shape[1])
        candidate_target = (
            (targets["correctness"][complete_idx, 0:1] == 0)
            & (targets["correctness"][complete_idx, 1:] == 1)
        ).astype("int64").reshape(-1)
        ranker_width = 2
    elif ranker_target_name == "candidate_correctness":
        candidate_features = all_candidate_features
        candidate_target = targets["correctness"][complete_idx].reshape(-1)
        ranker_width = len(CANDIDATE_SOURCES)
    else:
        raise ValueError(f"Unknown ranker target: {ranker_target_name}")

    fold_count = _fold_count(gate_target, folds)
    splitter = StratifiedKFold(
        fold_count, shuffle=True, random_state=int(random_state)
    )
    splits = list(splitter.split(gate_features, gate_target))
    gate_oof = np.zeros(len(pairs), dtype="float32")
    candidate_oof_complete = np.zeros(
        (len(complete_idx), len(CANDIDATE_SOURCES)), dtype="float32"
    )
    global_to_local = np.full(len(pairs), -1, dtype="int64")
    global_to_local[complete_idx] = np.arange(len(complete_idx))
    gate_models = []
    ranker_models = []
    gate_histories = []
    ranker_histories = []
    gate_best_epochs = []
    ranker_best_epochs = []

    for fold, (train_idx, val_idx) in enumerate(splits, 1):
        model, scaler, history, best_epoch = _fit_fold(
            gate_features,
            gate_target,
            train_idx,
            val_idx,
            hidden_sizes,
            dropout,
            epochs,
            train_batch_size,
            learning_rate,
            weight_decay,
            random_state + fold,
            device,
            "dual_gate",
            fold,
        )
        gate_oof[val_idx] = _predict_probability(
            model, scaler, gate_features[val_idx]
        )
        gate_models.append({"model": model, "scaler": scaler})
        gate_histories.append(history)
        gate_best_epochs.append(best_epoch)

        train_local = global_to_local[train_idx]
        val_local = global_to_local[val_idx]
        train_local = train_local[train_local >= 0]
        val_local = val_local[val_local >= 0]
        train_flat = _flat_candidate_indices(train_local, ranker_width)
        val_flat = _flat_candidate_indices(val_local, ranker_width)
        ranker, ranker_scaler, ranker_history, ranker_best_epoch = _fit_fold(
            candidate_features,
            candidate_target,
            train_flat,
            val_flat,
            hidden_sizes,
            dropout,
            epochs,
            train_batch_size,
            learning_rate,
            weight_decay,
            random_state + 1000 + fold,
            device,
            "dual_ranker",
            fold,
        )
        fold_probability = _predict_probability(
            ranker, ranker_scaler, candidate_features[val_flat]
        ).reshape(-1, ranker_width)
        if ranker_target_name == "repair_advantage":
            candidate_oof_complete[val_local, 1:] = fold_probability
        else:
            candidate_oof_complete[val_local] = fold_probability
        ranker_models.append({"model": ranker, "scaler": ranker_scaler})
        ranker_histories.append(ranker_history)
        ranker_best_epochs.append(ranker_best_epoch)

    candidate_oof = np.zeros(
        (len(pairs), len(CANDIDATE_SOURCES)), dtype="float32"
    )
    if ranker_target_name == "candidate_correctness":
        candidate_oof[:, 0] = 1.0
    candidate_oof[complete_idx] = candidate_oof_complete
    usage = build_usage_arrays(pairs)
    gate_threshold, margin_threshold, oof_metrics = select_dual_thresholds(
        gate_oof,
        candidate_oof,
        targets,
        usage,
        max_invocation_rate=max_invocation_rate,
        lambda_cost=lambda_cost,
        grid_points=threshold_grid_points,
    )

    direct_correct = targets["correctness"][:, 0]
    union_fix = (
        (direct_correct == 0)
        & targets["complete"]
        & (
            (targets["correctness"][:, 1] == 1)
            | (targets["correctness"][:, 2] == 1)
        )
    )
    union_oracle = {
        "n": len(pairs),
        "direct_accuracy": float(np.mean(direct_correct)),
        "accuracy": float(np.mean(direct_correct | union_fix)),
        "fixes": int(np.sum(union_fix)),
        "reflection_rate": float(np.mean(union_fix)),
    }
    _plot_histories(gate_histories, ranker_histories, curve_output)
    history = {
        "gate_fold_histories": gate_histories,
        "ranker_fold_histories": ranker_histories,
        "gate_best_epochs": gate_best_epochs,
        "ranker_best_epochs": ranker_best_epochs,
        "gate_target_name": gate_target_name,
        "ranker_target_name": ranker_target_name,
        "gate_threshold": gate_threshold,
        "margin_threshold": margin_threshold,
        "oof_metrics": oof_metrics,
        "union_accuracy_oracle": union_oracle,
        "selection_rule": (
            "maximize OOF selected-answer accuracy, then utility reward, "
            "then minimize dual-candidate invocation"
        ),
        "test_labels_used": False,
    }
    save_json(history, history_output)
    bundle = {
        "policy_type": "qa_dual_candidate_accuracy_first_ensemble",
        "candidate_sources": list(CANDIDATE_SOURCES),
        "encoder_name": encoder_name,
        "encoder_info": encoder_info,
        "layers": list(layers),
        "pooling": pooling,
        "max_length": int(max_length),
        "max_text_chars": int(max_text_chars),
        "encoder_batch_size": int(encoder_batch_size),
        "device": device,
        "hidden_sizes": list(hidden_sizes),
        "gate_target_name": gate_target_name,
        "ranker_target_name": ranker_target_name,
        "gate_models": gate_models,
        "ranker_models": ranker_models,
        "gate_threshold": float(gate_threshold),
        "margin_threshold": float(margin_threshold),
        "max_invocation_rate": float(max_invocation_rate),
        "lambda_cost": float(lambda_cost),
        "oof_metrics": oof_metrics,
        "union_accuracy_oracle": union_oracle,
        "gate_best_epochs": gate_best_epochs,
        "ranker_best_epochs": ranker_best_epochs,
        "n_rows": len(pairs),
        "n_complete": int(np.sum(targets["complete"])),
        "n_positive": int(np.sum(gate_target)),
        "n_union_repair": int(np.sum(targets["gate_target"])),
        "n_ranker_positive": int(np.sum(candidate_target)),
        "training_objective": (
            "stage 1 predicts whether direct is incorrect; stage 2 predicts "
            "whether each saved reflection repairs that direct error"
        ),
        "token_accounting": (
            "on invocation: Base once + v5 reflection once + v7 reflection once"
        ),
        "history_output": history_output,
        "curve_output": curve_output,
    }
    save_pickle(bundle, output)
    return bundle


def predict_dual_candidate(
    bundle: Dict[str, Any],
    old_rows: List[Dict[str, Any]],
    new_rows: List[Dict[str, Any]],
    gate_cache: str | None = None,
    candidate_cache: str | None = None,
) -> tuple[List[Dict[str, Any]], Dict[str, np.ndarray]]:
    if bundle.get("policy_type") == "qa_top_layer_dual_candidate":
        from .qa_top_layer_policy import predict_top_layer_dual_candidate

        return predict_top_layer_dual_candidate(bundle, old_rows, new_rows)
    if bundle.get("policy_type") != "qa_dual_candidate_accuracy_first_ensemble":
        raise ValueError(f"Unexpected policy_type={bundle.get('policy_type')}")
    pairs = align_dual_rows(old_rows, new_rows, require_complete=True)
    gate_texts = [
        build_policy_text(
            pair["new_row"],
            max_text_chars=int(bundle["max_text_chars"]),
            include_state=False,
        )
        for pair in pairs
    ]
    gate_features, _ = encode_text_features(
        gate_texts,
        bundle["encoder_name"],
        bundle["layers"],
        bundle["pooling"],
        int(bundle["max_length"]),
        int(bundle["encoder_batch_size"]),
        bundle.get("device", "auto"),
        gate_cache,
        "qa_dual_candidate_gate",
    )
    candidate_texts = [
        build_candidate_text(
            pair, source_index, int(bundle["max_text_chars"])
        )
        for pair in pairs
        for source_index in range(len(CANDIDATE_SOURCES))
    ]
    candidate_features, _ = encode_text_features(
        candidate_texts,
        bundle["encoder_name"],
        bundle["layers"],
        bundle["pooling"],
        int(bundle["max_length"]),
        int(bundle["encoder_batch_size"]),
        bundle.get("device", "auto"),
        candidate_cache,
        "qa_dual_candidate_ranker",
    )
    gate_probability = ensemble_probability(bundle["gate_models"], gate_features)
    ranker_target_name = bundle.get(
        "ranker_target_name", "candidate_correctness"
    )
    if ranker_target_name == "repair_advantage":
        ranker_features = candidate_features.reshape(
            len(pairs), len(CANDIDATE_SOURCES), -1
        )[:, 1:, :].reshape(-1, candidate_features.shape[1])
        reflection_probability = ensemble_probability(
            bundle["ranker_models"], ranker_features
        ).reshape(-1, 2)
        candidate_probability = np.zeros(
            (len(pairs), len(CANDIDATE_SOURCES)), dtype="float32"
        )
        candidate_probability[:, 1:] = reflection_probability
    else:
        candidate_probability = ensemble_probability(
            bundle["ranker_models"], candidate_features
        ).reshape(-1, len(CANDIDATE_SOURCES))
    rows = np.arange(len(pairs))
    best_reflection = np.argmax(candidate_probability[:, 1:], axis=1) + 1
    margin = (
        candidate_probability[rows, best_reflection]
        - candidate_probability[:, 0]
    )
    invoked = gate_probability >= float(bundle["gate_threshold"])
    adopted = invoked & (margin >= float(bundle["margin_threshold"]))
    selected_source = np.where(adopted, best_reflection, 0)
    return pairs, {
        "gate_probability": gate_probability,
        "candidate_probability": candidate_probability,
        "best_reflection": best_reflection,
        "margin": margin,
        "invoked": invoked,
        "adopted": adopted,
        "selected_source": selected_source,
    }


def _integers(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a two-stage QA policy over saved v5/v7 candidates."
    )
    parser.add_argument("--old_dataset", required=True)
    parser.add_argument("--new_dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--encoder_name", required=True)
    parser.add_argument("--gate_cache", required=True)
    parser.add_argument("--candidate_cache", required=True)
    parser.add_argument("--curve_output", required=True)
    parser.add_argument("--history_output", required=True)
    parser.add_argument("--layers", default="-4,-3,-2,-1")
    parser.add_argument("--pooling", default="mean")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--max_text_chars", type=int, default=8000)
    parser.add_argument("--encoder_batch_size", type=int, default=4)
    parser.add_argument("--hidden_sizes", default="128,32")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--train_batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_invocation_rate", type=float, default=0.15)
    parser.add_argument("--lambda_cost", type=float, default=0.0001)
    parser.add_argument("--threshold_grid_points", type=int, default=81)
    parser.add_argument(
        "--gate_target",
        choices=("direct_error", "union_repair"),
        default="direct_error",
    )
    parser.add_argument(
        "--ranker_target",
        choices=("repair_advantage", "candidate_correctness"),
        default="repair_advantage",
    )
    args = parser.parse_args()
    bundle = train_dual_candidate_policy(
        old_rows=read_jsonl(args.old_dataset),
        new_rows=read_jsonl(args.new_dataset),
        output=args.output,
        encoder_name=args.encoder_name,
        gate_cache=args.gate_cache,
        candidate_cache=args.candidate_cache,
        curve_output=args.curve_output,
        history_output=args.history_output,
        layers=_integers(args.layers),
        pooling=args.pooling,
        max_length=args.max_length,
        max_text_chars=args.max_text_chars,
        encoder_batch_size=args.encoder_batch_size,
        hidden_sizes=_integers(args.hidden_sizes),
        dropout=args.dropout,
        epochs=args.epochs,
        train_batch_size=args.train_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        folds=args.folds,
        random_state=args.random_state,
        device=args.device,
        max_invocation_rate=args.max_invocation_rate,
        lambda_cost=args.lambda_cost,
        threshold_grid_points=args.threshold_grid_points,
        gate_target_name=args.gate_target,
        ranker_target_name=args.ranker_target,
    )
    print(f"Saved dual-candidate policy to {args.output}")
    print(
        f"Rows={bundle['n_rows']} complete={bundle['n_complete']} "
        f"gate_positive={bundle['n_positive']} "
        f"ranker_positive={bundle['n_ranker_positive']}"
    )
    print(f"OOF: {bundle['oof_metrics']}")
    print(f"Union oracle: {bundle['union_accuracy_oracle']}")
    print(f"Curve: {args.curve_output}")


if __name__ == "__main__":
    main()
