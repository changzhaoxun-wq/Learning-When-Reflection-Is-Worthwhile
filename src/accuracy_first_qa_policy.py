from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch import nn

from .neural_policy import (
    _clip_context,
    _encode_hf_hidden,
    _qa_question_and_context,
    build_policy_text,
)
from .utils import read_jsonl, save_json, save_pickle


class AccuracyFirstBinaryMLP(nn.Module):
    """Small binary head over frozen Transformer hidden states."""

    def __init__(
        self,
        input_dim: int,
        hidden_sizes: Sequence[int],
        dropout: float,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        previous = int(input_dim)
        for size in hidden_sizes:
            layers.extend(
                [
                    nn.Linear(previous, int(size)),
                    nn.LayerNorm(int(size)),
                    nn.GELU(),
                    nn.Dropout(float(dropout)),
                ]
            )
            previous = int(size)
        self.network = nn.Sequential(*layers) if layers else nn.Identity()
        self.output = nn.Linear(previous, 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.output(self.network(values)).squeeze(-1)


def _candidates(row: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {
        str(candidate.get("action")): candidate
        for candidate in row.get("candidate_results", [])
    }


def _is_correct(candidate: Dict[str, Any] | None) -> int:
    return int(float((candidate or {}).get("correct", 0)) >= 0.5)


def build_accuracy_targets(rows: Sequence[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    """Create automatic accuracy-first labels without gold text in model inputs."""

    direct_correct = np.zeros(len(rows), dtype="int64")
    reflect_correct = np.zeros(len(rows), dtype="int64")
    complete = np.zeros(len(rows), dtype=bool)
    target = np.zeros(len(rows), dtype="int64")
    for index, row in enumerate(rows):
        candidates = _candidates(row)
        if "direct" not in candidates:
            raise ValueError(f"Row {row.get('id')} has no direct candidate")
        direct_correct[index] = _is_correct(candidates["direct"])
        if "reflect_1" not in candidates:
            if direct_correct[index] == 0:
                raise ValueError(
                    f"Direct-wrong row {row.get('id')} has no reflect_1 candidate"
                )
            continue
        complete[index] = True
        reflect_correct[index] = _is_correct(candidates["reflect_1"])
        target[index] = int(
            direct_correct[index] == 0 and reflect_correct[index] == 1
        )
    return {
        "direct_correct": direct_correct,
        "reflect_correct": reflect_correct,
        "complete": complete,
        "target": target,
    }


def build_acceptance_text(
    row: Dict[str, Any],
    max_text_chars: int = 8000,
) -> str:
    """Build post-reflection input without exposing gold labels or rewards."""

    state = row.get("state_features", {}) or {}
    raw_question, context = _qa_question_and_context(row, state)
    candidates = _candidates(row)
    direct = candidates.get("direct", {})
    reflected = candidates.get("reflect_1", {})
    direct_response = str(
        direct.get("final_answer")
        or row.get("initial_answer")
        or row.get("initial_pred")
        or ""
    )
    reflected_response = str(
        reflected.get("final_answer") or reflected.get("pred") or ""
    )
    fixed = "\n".join(
        [
            "Question:",
            raw_question,
            "Direct response:",
            direct_response,
            "Reflected response:",
            reflected_response,
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


def _digest_texts(texts: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for text in texts:
        digest.update(text.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def encode_text_features(
    texts: List[str],
    encoder_name: str,
    layers: Sequence[int],
    pooling: str,
    max_length: int,
    batch_size: int,
    device: str,
    cache_path: str | None,
    feature_kind: str,
) -> tuple[np.ndarray, Dict[str, Any]]:
    expected = {
        "feature_kind": feature_kind,
        "text_digest": _digest_texts(texts),
        "encoder_name": str(encoder_name),
        "layers": list(layers),
        "pooling": pooling,
        "max_length": int(max_length),
    }
    if cache_path and Path(cache_path).exists():
        cached = np.load(cache_path, allow_pickle=False)
        metadata = json.loads(str(cached["metadata"].item()))
        if all(metadata.get(key) == value for key, value in expected.items()):
            features = cached["features"].astype("float32")
            if len(features) == len(texts):
                print(
                    f"[accuracy-first] loaded {feature_kind} cache "
                    f"{cache_path} shape={features.shape}",
                    flush=True,
                )
                return features, metadata["encoder_info"]

    features, encoder_info = _encode_hf_hidden(
        texts,
        encoder_name=encoder_name,
        max_length=max_length,
        batch_size=batch_size,
        device=device,
        pooling=pooling,
        layer_indices=list(layers),
    )
    features = np.clip(
        np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0),
        -1e6,
        1e6,
    ).astype("float32")
    if cache_path:
        path = Path(cache_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            features=features.astype("float16"),
            metadata=json.dumps({**expected, "encoder_info": encoder_info}),
        )
        print(
            f"[accuracy-first] wrote {feature_kind} cache "
            f"{path} shape={features.shape}",
            flush=True,
        )
    return features, encoder_info


def _predict_probability(
    model: AccuracyFirstBinaryMLP,
    scaler: StandardScaler,
    features: np.ndarray,
) -> np.ndarray:
    scaled = scaler.transform(features).astype("float32")
    values: List[np.ndarray] = []
    model = model.to("cpu").eval()
    with torch.no_grad():
        for start in range(0, len(scaled), 512):
            logits = model(torch.from_numpy(scaled[start : start + 512]))
            values.append(torch.sigmoid(logits).numpy())
    return np.concatenate(values).astype("float32")


def ensemble_probability(
    models: Sequence[Dict[str, Any]],
    features: np.ndarray,
) -> np.ndarray:
    if not models:
        raise ValueError("The policy bundle has no fold models")
    predictions = [
        _predict_probability(item["model"], item["scaler"], features)
        for item in models
    ]
    return np.mean(predictions, axis=0).astype("float32")


def _fit_fold(
    features: np.ndarray,
    targets: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    hidden_sizes: Sequence[int],
    dropout: float,
    epochs: int,
    train_batch_size: int,
    learning_rate: float,
    weight_decay: float,
    random_state: int,
    device: str,
    stage: str,
    fold: int,
) -> tuple[AccuracyFirstBinaryMLP, StandardScaler, Dict[str, List[float]], int]:
    scaler = StandardScaler()
    scaler.fit(features[train_idx])
    scaled = scaler.transform(features).astype("float32")

    resolved = device
    if resolved in {"", "auto", None}:
        resolved = "cuda" if torch.cuda.is_available() else "cpu"
    if str(resolved).startswith("cuda") and not torch.cuda.is_available():
        resolved = "cpu"
    train_device = torch.device(str(resolved))

    torch.manual_seed(int(random_state))
    model = AccuracyFirstBinaryMLP(
        scaled.shape[1], hidden_sizes, dropout
    ).to(train_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    positive = max(int(targets[train_idx].sum()), 1)
    negative = max(len(train_idx) - positive, 1)
    positive_weight = min(5.0, max(1.0, float(np.sqrt(negative / positive))))
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(positive_weight, device=train_device)
    )

    x = torch.from_numpy(scaled)
    y = torch.from_numpy(targets.astype("float32"))
    rng = np.random.default_rng(int(random_state))
    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
    }
    best_epoch = 1
    best_loss = float("inf")
    best_state: Dict[str, torch.Tensor] | None = None

    for epoch in range(1, int(epochs) + 1):
        model.train()
        order = rng.permutation(train_idx)
        for start in range(0, len(order), max(1, int(train_batch_size))):
            batch_idx = order[start : start + max(1, int(train_batch_size))]
            logits = model(x[batch_idx].to(train_device))
            loss = criterion(logits, y[batch_idx].to(train_device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            train_logits = model(x[train_idx].to(train_device))
            val_logits = model(x[val_idx].to(train_device))
            train_loss = float(
                criterion(train_logits, y[train_idx].to(train_device)).item()
            )
            val_loss = float(
                criterion(val_logits, y[val_idx].to(train_device)).item()
            )
            train_prediction = (
                torch.sigmoid(train_logits) >= 0.5
            ).cpu().numpy()
            val_prediction = (
                torch.sigmoid(val_logits) >= 0.5
            ).cpu().numpy()
        train_accuracy = float(
            np.mean(train_prediction == targets[train_idx])
        )
        val_accuracy = float(np.mean(val_prediction == targets[val_idx]))
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_accuracy"].append(val_accuracy)
        print(
            f"[accuracy-first] stage={stage} fold={fold} "
            f"epoch={epoch:03d}/{epochs} train_loss={train_loss:.5f} "
            f"val_loss={val_loss:.5f} train_acc={train_accuracy:.4f} "
            f"val_acc={val_accuracy:.4f}",
            flush=True,
        )
        if val_loss < best_loss - 1e-12:
            best_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to("cpu").eval()
    return model, scaler, history, best_epoch


def _threshold_metrics(
    probability: np.ndarray,
    direct_correct: np.ndarray,
    reflect_correct: np.ndarray,
    mask: np.ndarray,
    threshold: float,
    active: np.ndarray | None = None,
) -> Dict[str, float]:
    reflected = probability >= float(threshold)
    if active is not None:
        reflected &= active
    reflected &= mask
    selected_correct = np.where(
        reflected, reflect_correct, direct_correct
    )
    evaluated = mask
    beneficial = (
        (direct_correct == 0) & (reflect_correct == 1) & evaluated
    )
    fixes = int(np.sum(reflected & beneficial))
    harms = int(
        np.sum(
            reflected
            & (direct_correct == 1)
            & (reflect_correct == 0)
            & evaluated
        )
    )
    selected_count = int(np.sum(reflected & evaluated))
    positives = int(np.sum(beneficial))
    n = int(np.sum(evaluated))
    return {
        "n": n,
        "accuracy": float(np.mean(selected_correct[evaluated])) if n else 0.0,
        "direct_accuracy": float(np.mean(direct_correct[evaluated])) if n else 0.0,
        "reflection_rate": selected_count / n if n else 0.0,
        "fixes": fixes,
        "harms": harms,
        "net_fixes": fixes - harms,
        "precision": fixes / selected_count if selected_count else 0.0,
        "recall": fixes / positives if positives else 0.0,
        "positive_count": positives,
    }


def select_accuracy_threshold(
    probability: np.ndarray,
    direct_correct: np.ndarray,
    reflect_correct: np.ndarray,
    mask: np.ndarray,
    max_reflection_rate: float = 0.15,
    active: np.ndarray | None = None,
) -> tuple[float, Dict[str, float]]:
    values = probability[mask]
    thresholds = np.unique(
        np.concatenate(
            [
                np.linspace(0.0, 1.0, 501),
                values,
                np.asarray([1.000001], dtype="float32"),
            ]
        )
    )
    best_key = None
    best = None
    for threshold in thresholds:
        metrics = _threshold_metrics(
            probability,
            direct_correct,
            reflect_correct,
            mask,
            float(threshold),
            active=active,
        )
        if metrics["reflection_rate"] > float(max_reflection_rate) + 1e-12:
            continue
        key = (
            metrics["accuracy"],
            metrics["net_fixes"],
            -metrics["reflection_rate"],
            float(threshold),
        )
        if best_key is None or key > best_key:
            best_key = key
            best = (float(threshold), metrics)
    if best is None:
        raise ValueError("No threshold satisfies max_reflection_rate")
    return best


def _joint_key(
    metrics: Dict[str, float],
    gate_threshold: float,
    acceptance_threshold: float,
) -> tuple[float, ...]:
    return (
        metrics["accuracy"],
        metrics["net_fixes"],
        -metrics["gate_invocation_rate"],
        -metrics["reflection_rate"],
        float(gate_threshold),
        float(acceptance_threshold),
    )


def _scan_gate_threshold(
    gate_probability: np.ndarray,
    acceptance_active: np.ndarray,
    direct_correct: np.ndarray,
    reflect_correct: np.ndarray,
    mask: np.ndarray,
    acceptance_threshold: float,
    max_reflection_rate: float,
) -> tuple[float, Dict[str, float]]:
    values = gate_probability[mask]
    thresholds = np.unique(
        np.concatenate(
            [
                np.linspace(0.0, 1.0, 501),
                values,
                np.asarray([1.000001], dtype="float32"),
            ]
        )
    )
    n = max(int(mask.sum()), 1)
    best_key = None
    best = None
    for gate_threshold in thresholds:
        gate_active = (gate_probability >= float(gate_threshold)) & mask
        invocation_rate = float(gate_active.sum() / n)
        if invocation_rate > float(max_reflection_rate) + 1e-12:
            continue
        metrics = _threshold_metrics(
            gate_probability,
            direct_correct,
            reflect_correct,
            mask,
            float(gate_threshold),
            active=acceptance_active,
        )
        metrics["gate_invocation_rate"] = invocation_rate
        key = _joint_key(
            metrics, float(gate_threshold), acceptance_threshold
        )
        if best_key is None or key > best_key:
            best_key = key
            best = (float(gate_threshold), metrics)
    if best is None:
        raise ValueError("No gate threshold satisfies max_reflection_rate")
    return best


def select_joint_accuracy_thresholds(
    gate_probability: np.ndarray,
    acceptance_probability: np.ndarray,
    direct_correct: np.ndarray,
    reflect_correct: np.ndarray,
    mask: np.ndarray,
    max_reflection_rate: float = 0.15,
) -> tuple[float, float, Dict[str, float]]:
    """Coordinate-search exact OOF thresholds without test labels."""

    gate_threshold, _ = select_accuracy_threshold(
        gate_probability,
        direct_correct,
        reflect_correct,
        mask,
        max_reflection_rate=max_reflection_rate,
    )
    gate_active = (gate_probability >= gate_threshold) & mask
    acceptance_threshold, metrics = select_accuracy_threshold(
        acceptance_probability,
        direct_correct,
        reflect_correct,
        mask,
        max_reflection_rate=max_reflection_rate,
        active=gate_active,
    )
    metrics["gate_invocation_rate"] = float(
        gate_active.sum() / max(int(mask.sum()), 1)
    )
    best = (gate_threshold, acceptance_threshold, metrics)
    best_key = _joint_key(metrics, gate_threshold, acceptance_threshold)

    for _ in range(4):
        acceptance_active = (
            acceptance_probability >= acceptance_threshold
        ) & mask
        gate_threshold, _ = _scan_gate_threshold(
            gate_probability,
            acceptance_active,
            direct_correct,
            reflect_correct,
            mask,
            acceptance_threshold,
            max_reflection_rate,
        )
        gate_active = (gate_probability >= gate_threshold) & mask
        acceptance_threshold, metrics = select_accuracy_threshold(
            acceptance_probability,
            direct_correct,
            reflect_correct,
            mask,
            max_reflection_rate=max_reflection_rate,
            active=gate_active,
        )
        metrics["gate_invocation_rate"] = float(
            gate_active.sum() / max(int(mask.sum()), 1)
        )
        key = _joint_key(metrics, gate_threshold, acceptance_threshold)
        if key > best_key:
            best_key = key
            best = (gate_threshold, acceptance_threshold, metrics)

    return best


def _plot_histories(
    gate_histories: Sequence[Dict[str, List[float]]],
    acceptance_histories: Sequence[Dict[str, List[float]]],
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
        (axes[1], "Post-reflection acceptance", acceptance_histories),
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


def _fold_count(target: np.ndarray, requested: int) -> int:
    counts = np.bincount(target.astype("int64"), minlength=2)
    available = int(counts[counts > 0].min()) if np.any(counts > 0) else 0
    folds = min(int(requested), available)
    if folds < 2:
        raise ValueError(
            f"Need at least two examples in each class for OOF; counts={counts.tolist()}"
        )
    return folds


def train_accuracy_first_policy(
    rows: List[Dict[str, Any]],
    output: str,
    encoder_name: str,
    gate_cache: str,
    acceptance_cache: str,
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
    max_reflection_rate: float = 0.15,
) -> Dict[str, Any]:
    if not rows:
        raise ValueError("Cannot train on an empty dataset")
    targets = build_accuracy_targets(rows)
    gate_texts = [
        build_policy_text(
            row, max_text_chars=max_text_chars, include_state=False
        )
        for row in rows
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
        "qa_accuracy_gate",
    )

    gate_oof = np.zeros(len(rows), dtype="float32")
    gate_models = []
    gate_histories = []
    gate_best_epochs = []
    gate_folds = _fold_count(targets["target"], folds)
    splitter = StratifiedKFold(
        gate_folds, shuffle=True, random_state=int(random_state)
    )
    for fold, (train_idx, val_idx) in enumerate(
        splitter.split(gate_features, targets["target"]), 1
    ):
        model, scaler, history, best_epoch = _fit_fold(
            gate_features,
            targets["target"],
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
            "gate",
            fold,
        )
        gate_oof[val_idx] = _predict_probability(
            model, scaler, gate_features[val_idx]
        )
        gate_models.append({"model": model, "scaler": scaler})
        gate_histories.append(history)
        gate_best_epochs.append(best_epoch)

    complete_idx = np.flatnonzero(targets["complete"])
    complete_rows = [rows[int(index)] for index in complete_idx]
    acceptance_texts = [
        build_acceptance_text(row, max_text_chars=max_text_chars)
        for row in complete_rows
    ]
    acceptance_features, _ = encode_text_features(
        acceptance_texts,
        encoder_name,
        layers,
        pooling,
        max_length,
        encoder_batch_size,
        device,
        acceptance_cache,
        "qa_reflection_acceptance",
    )
    acceptance_target = targets["target"][complete_idx]
    acceptance_oof_complete = np.zeros(len(complete_idx), dtype="float32")
    acceptance_models = []
    acceptance_histories = []
    acceptance_best_epochs = []
    acceptance_folds = _fold_count(acceptance_target, folds)
    acceptance_splitter = StratifiedKFold(
        acceptance_folds, shuffle=True, random_state=int(random_state) + 1000
    )
    for fold, (train_local, val_local) in enumerate(
        acceptance_splitter.split(acceptance_features, acceptance_target), 1
    ):
        model, scaler, history, best_epoch = _fit_fold(
            acceptance_features,
            acceptance_target,
            train_local,
            val_local,
            hidden_sizes,
            dropout,
            epochs,
            train_batch_size,
            learning_rate,
            weight_decay,
            random_state + 1000 + fold,
            device,
            "acceptance",
            fold,
        )
        acceptance_oof_complete[val_local] = _predict_probability(
            model, scaler, acceptance_features[val_local]
        )
        acceptance_models.append({"model": model, "scaler": scaler})
        acceptance_histories.append(history)
        acceptance_best_epochs.append(best_epoch)

    gate_threshold, gate_metrics = select_accuracy_threshold(
        gate_oof,
        targets["direct_correct"],
        targets["reflect_correct"],
        targets["complete"],
        max_reflection_rate=max_reflection_rate,
    )
    acceptance_oof = np.zeros(len(rows), dtype="float32")
    acceptance_oof[complete_idx] = acceptance_oof_complete
    joint_gate_threshold, acceptance_threshold, combined_metrics = (
        select_joint_accuracy_thresholds(
            gate_oof,
            acceptance_oof,
            targets["direct_correct"],
            targets["reflect_correct"],
            targets["complete"],
            max_reflection_rate=max_reflection_rate,
        )
    )
    oracle_metrics = _threshold_metrics(
        targets["target"].astype("float32"),
        targets["direct_correct"],
        targets["reflect_correct"],
        targets["complete"],
        0.5,
    )

    _plot_histories(
        gate_histories, acceptance_histories, curve_output
    )
    history = {
        "gate_fold_histories": gate_histories,
        "acceptance_fold_histories": acceptance_histories,
        "gate_best_epochs": gate_best_epochs,
        "acceptance_best_epochs": acceptance_best_epochs,
        "gate_threshold": gate_threshold,
        "joint_gate_threshold": joint_gate_threshold,
        "acceptance_threshold": acceptance_threshold,
        "gate_oof_metrics": gate_metrics,
        "combined_oof_metrics": combined_metrics,
        "binary_accuracy_oracle": oracle_metrics,
        "selection_rule": (
            "maximize OOF selected-answer accuracy on complete training "
            "counterfactuals; tie-break by fewer reflections"
        ),
        "test_labels_used": False,
    }
    save_json(history, history_output)

    bundle = {
        "policy_type": "qa_accuracy_first_binary_ensemble",
        "encoder_name": encoder_name,
        "encoder_info": encoder_info,
        "layers": list(layers),
        "pooling": pooling,
        "max_length": int(max_length),
        "max_text_chars": int(max_text_chars),
        "encoder_batch_size": int(encoder_batch_size),
        "device": device,
        "hidden_sizes": list(hidden_sizes),
        "gate_models": gate_models,
        "acceptance_models": acceptance_models,
        "gate_threshold": float(gate_threshold),
        "joint_gate_threshold": float(joint_gate_threshold),
        "acceptance_threshold": float(acceptance_threshold),
        "max_reflection_rate": float(max_reflection_rate),
        "gate_oof_metrics": gate_metrics,
        "combined_oof_metrics": combined_metrics,
        "binary_accuracy_oracle": oracle_metrics,
        "gate_best_epochs": gate_best_epochs,
        "acceptance_best_epochs": acceptance_best_epochs,
        "n_rows": len(rows),
        "n_complete": int(targets["complete"].sum()),
        "n_positive": int(targets["target"].sum()),
        "training_objective": (
            "reflect_1 only when direct is incorrect and reflect_1 is correct; "
            "otherwise keep direct"
        ),
        "history_output": history_output,
        "curve_output": curve_output,
    }
    save_pickle(bundle, output)
    return bundle


def predict_accuracy_first(
    bundle: Dict[str, Any],
    rows: List[Dict[str, Any]],
    gate_cache: str | None = None,
    acceptance_cache: str | None = None,
) -> Dict[str, np.ndarray]:
    if bundle.get("policy_type") != "qa_accuracy_first_binary_ensemble":
        raise ValueError(
            f"Unexpected policy_type={bundle.get('policy_type')}"
        )
    gate_texts = [
        build_policy_text(
            row,
            max_text_chars=int(bundle["max_text_chars"]),
            include_state=False,
        )
        for row in rows
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
        "qa_accuracy_gate",
    )
    acceptance_texts = [
        build_acceptance_text(
            row, max_text_chars=int(bundle["max_text_chars"])
        )
        for row in rows
    ]
    acceptance_features, _ = encode_text_features(
        acceptance_texts,
        bundle["encoder_name"],
        bundle["layers"],
        bundle["pooling"],
        int(bundle["max_length"]),
        int(bundle["encoder_batch_size"]),
        bundle.get("device", "auto"),
        acceptance_cache,
        "qa_reflection_acceptance",
    )
    gate_probability = ensemble_probability(
        bundle["gate_models"], gate_features
    )
    acceptance_probability = ensemble_probability(
        bundle["acceptance_models"], acceptance_features
    )
    gate_reflect = gate_probability >= float(bundle["gate_threshold"])
    joint_gate_reflect = gate_probability >= float(
        bundle.get("joint_gate_threshold", bundle["gate_threshold"])
    )
    accepted_reflect = joint_gate_reflect & (
        acceptance_probability >= float(bundle["acceptance_threshold"])
    )
    return {
        "gate_probability": gate_probability,
        "acceptance_probability": acceptance_probability,
        "gate_reflect": gate_reflect,
        "joint_gate_reflect": joint_gate_reflect,
        "accepted_reflect": accepted_reflect,
    }


def _integers(value: str) -> tuple[int, ...]:
    return tuple(
        int(item.strip()) for item in value.split(",") if item.strip()
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train an accuracy-first QA reflection policy."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--encoder_name", required=True)
    parser.add_argument("--gate_cache", required=True)
    parser.add_argument("--acceptance_cache", required=True)
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
    parser.add_argument("--max_reflection_rate", type=float, default=0.15)
    args = parser.parse_args()

    bundle = train_accuracy_first_policy(
        rows=read_jsonl(args.dataset),
        output=args.output,
        encoder_name=args.encoder_name,
        gate_cache=args.gate_cache,
        acceptance_cache=args.acceptance_cache,
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
        max_reflection_rate=args.max_reflection_rate,
    )
    print(f"Saved accuracy-first policy to {args.output}")
    print(
        f"Rows={bundle['n_rows']} complete={bundle['n_complete']} "
        f"positive={bundle['n_positive']} max_length={bundle['max_length']}"
    )
    print(f"Gate OOF: {bundle['gate_oof_metrics']}")
    print(f"Gate + acceptance OOF: {bundle['combined_oof_metrics']}")
    print(f"Curve: {args.curve_output}")


if __name__ == "__main__":
    main()

