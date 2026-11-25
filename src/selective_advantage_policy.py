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

from .neural_policy import _encode_hf_hidden, build_policy_text
from .reward import compute_utility_reward
from .utils import read_jsonl, save_json, save_pickle

ACTIONS = ("direct", "reflect_1", "reflect_2")


class SelectiveAdvantageMLP(nn.Module):
    """Shared frozen-embedding head for error, advantage, and round prediction."""

    def __init__(self, input_dim: int, hidden_sizes: Sequence[int], dropout: float) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        previous = input_dim
        for size in hidden_sizes:
            layers.extend([
                nn.Linear(previous, int(size)), nn.LayerNorm(int(size)),
                nn.GELU(), nn.Dropout(dropout),
            ])
            previous = int(size)
        self.shared = nn.Sequential(*layers) if layers else nn.Identity()
        self.error_head = nn.Linear(previous, 1)
        self.advantage_head = nn.Linear(previous, 1)
        self.round_head = nn.Linear(previous, 1)

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.shared(values)
        return tuple(head(hidden).squeeze(-1) for head in (
            self.error_head, self.advantage_head, self.round_head,
        ))  # type: ignore[return-value]


def _utility(candidate: Dict[str, Any]) -> float:
    for key in ("qa_utility_score", "enhanced_answer_f1", "answer_f1", "correct"):
        if candidate.get(key) is not None:
            return float(candidate[key])
    return 0.0


def _reward(candidate: Dict[str, Any], lambda_cost: float) -> float:
    return compute_utility_reward(
        _utility(candidate), int(float(candidate.get("token_cost", 0))), lambda_cost
    )


def build_targets(rows: Sequence[Dict[str, Any]], lambda_cost: float) -> Dict[str, np.ndarray]:
    targets = {
        "direct_error": np.zeros(len(rows), dtype="float32"),
        "complete": np.zeros(len(rows), dtype=bool),
        "advantage": np.zeros(len(rows), dtype="float32"),
        "round_delta": np.zeros(len(rows), dtype="float32"),
        "gate_target": np.zeros(len(rows), dtype="int64"),
    }
    for index, row in enumerate(rows):
        candidates = {str(item.get("action")): item for item in row.get("candidate_results", [])}
        if "direct" not in candidates:
            raise ValueError(f"Row {row.get('id')} has no direct candidate")
        direct = candidates["direct"]
        targets["direct_error"][index] = float(float(direct.get("correct", 0)) < 0.5)
        if not all(action in candidates for action in ACTIONS):
            if targets["direct_error"][index] > 0.5:
                raise ValueError(f"Incorrect-direct row {row.get('id')} lacks reflections")
            continue
        targets["complete"][index] = True
        direct_reward = _reward(direct, lambda_cost)
        r1_reward = _reward(candidates["reflect_1"], lambda_cost)
        r2_reward = _reward(candidates["reflect_2"], lambda_cost)
        targets["advantage"][index] = max(r1_reward, r2_reward) - direct_reward
        targets["round_delta"][index] = r2_reward - r1_reward
        targets["gate_target"][index] = int(targets["advantage"][index] > 0)
    return targets


def _row_digest(rows: Sequence[Dict[str, Any]]) -> str:
    value = "\n".join(str(row.get("id", i)) for i, row in enumerate(rows))
    return hashlib.sha256(value.encode()).hexdigest()


def encode_features(
    rows: Sequence[Dict[str, Any]], encoder_name: str, layers: Sequence[int],
    pooling: str, max_length: int, max_text_chars: int, batch_size: int,
    device: str, cache_path: str | None = None,
) -> tuple[np.ndarray, Dict[str, Any]]:
    expected = {
        "row_digest": _row_digest(rows), "encoder_name": str(encoder_name),
        "layers": list(layers), "pooling": pooling, "max_length": max_length,
        "max_text_chars": max_text_chars, "pure_text": True,
    }
    if cache_path and Path(cache_path).exists():
        cached = np.load(cache_path, allow_pickle=False)
        metadata = json.loads(str(cached["metadata"].item()))
        if all(metadata.get(key) == value for key, value in expected.items()):
            features = cached["features"].astype("float32")
            if len(features) == len(rows):
                print(f"[advantage] loaded cache {cache_path} shape={features.shape}", flush=True)
                return features, metadata["encoder_info"]
    texts = [build_policy_text(row, max_text_chars=max_text_chars, include_state=False) for row in rows]
    features, encoder_info = _encode_hf_hidden(
        texts, encoder_name=encoder_name, max_length=max_length,
        batch_size=batch_size, device=device, pooling=pooling,
        layer_indices=list(layers),
    )
    features = np.clip(np.nan_to_num(features), -1e6, 1e6).astype("float32")
    if cache_path:
        path = Path(cache_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path, features=features.astype("float16"),
            metadata=json.dumps({**expected, "encoder_info": encoder_info}),
        )
        print(f"[advantage] wrote cache {path} shape={features.shape}", flush=True)
    return features, encoder_info


def _weights(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    result = np.ones(len(values), dtype="float32")
    observed = np.abs(values[mask])
    scale = float(np.median(observed[observed > 0])) if np.any(observed > 0) else 1.0
    result[mask] = 1 + np.minimum(observed / max(scale, 1e-6), 4)
    return result


def _target_tensors(targets: Dict[str, np.ndarray], device: torch.device) -> Dict[str, torch.Tensor]:
    output = {key: torch.from_numpy(value).to(device) for key, value in targets.items()}
    output["advantage_weight"] = torch.from_numpy(
        _weights(targets["advantage"], targets["complete"])
    ).to(device)
    output["round_weight"] = torch.from_numpy(
        _weights(targets["round_delta"], targets["complete"])
    ).to(device)
    return output


def _loss(
    output: tuple[torch.Tensor, torch.Tensor, torch.Tensor], indices: torch.Tensor,
    targets: Dict[str, torch.Tensor],
) -> torch.Tensor:
    error_logits, advantage, round_delta = output
    error_target = targets["direct_error"][indices]
    positives = torch.clamp(error_target.sum(), min=1)
    negatives = torch.clamp(len(error_target) - error_target.sum(), min=1)
    error_loss = nn.functional.binary_cross_entropy_with_logits(
        error_logits, error_target, pos_weight=(negatives / positives).detach()
    )
    complete = targets["complete"][indices]
    if not bool(complete.any()):
        return error_loss
    advantage_loss = nn.functional.smooth_l1_loss(
        advantage[complete], targets["advantage"][indices][complete], reduction="none"
    )
    round_loss = nn.functional.smooth_l1_loss(
        round_delta[complete], targets["round_delta"][indices][complete], reduction="none"
    )
    advantage_loss = (advantage_loss * targets["advantage_weight"][indices][complete]).mean()
    round_loss = (round_loss * targets["round_weight"][indices][complete]).mean()
    return error_loss + 2 * advantage_loss + 0.5 * round_loss


def _fit(
    features: np.ndarray, targets: Dict[str, np.ndarray], train_idx: np.ndarray,
    val_idx: np.ndarray, hidden_sizes: Sequence[int], dropout: float, epochs: int,
    batch_size: int, learning_rate: float, weight_decay: float, seed: int,
    device_name: str,
) -> tuple[SelectiveAdvantageMLP, StandardScaler, Dict[str, List[float]], int]:
    scaler = StandardScaler().fit(features[train_idx])
    scaled = scaler.transform(features).astype("float32")
    device = torch.device(device_name if device_name.startswith("cuda") and torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    model = SelectiveAdvantageMLP(scaled.shape[1], hidden_sizes, dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    x = torch.from_numpy(scaled).to(device)
    tensors = _target_tensors(targets, device)
    rng = np.random.default_rng(seed)
    history = {"train_loss": [], "val_loss": []}
    best = (float("inf"), 1, None)
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(train_idx)
        losses = []
        for start in range(0, len(order), batch_size):
            batch = torch.as_tensor(order[start:start + batch_size], device=device)
            loss = _loss(model(x[batch]), batch, tensors)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            validation = torch.as_tensor(val_idx, device=device)
            val_loss = float(_loss(model(x[validation]), validation, tensors).cpu())
        train_loss = float(np.mean(losses))
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(
            f"[advantage] epoch={epoch:03d}/{epochs} train_loss={train_loss:.5f} "
            f"val_loss={val_loss:.5f}", flush=True,
        )
        if val_loss < best[0]:
            best = (val_loss, epoch, copy.deepcopy(model.state_dict()))
    model.load_state_dict(best[2])
    return model.cpu().eval(), scaler, history, int(best[1])


def _heads(
    model: SelectiveAdvantageMLP, scaler: StandardScaler, features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scaled = scaler.transform(features).astype("float32")
    collected: List[List[np.ndarray]] = [[], [], []]
    with torch.no_grad():
        for start in range(0, len(scaled), 512):
            output = model(torch.from_numpy(scaled[start:start + 512]))
            collected[0].append(torch.sigmoid(output[0]).numpy())
            collected[1].append(output[1].numpy())
            collected[2].append(output[2].numpy())
    return tuple(np.concatenate(values) for values in collected)  # type: ignore[return-value]


def _fbeta(target: np.ndarray, prediction: np.ndarray, beta: float) -> tuple[float, float, float]:
    tp = float(np.sum((target == 1) & (prediction == 1)))
    fp = float(np.sum((target == 0) & (prediction == 1)))
    fn = float(np.sum((target == 1) & (prediction == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    b2 = beta * beta
    score = (1 + b2) * precision * recall / (b2 * precision + recall) if precision + recall else 0.0
    return score, precision, recall


def _gate_threshold(
    score: np.ndarray,
    target: np.ndarray,
    beta: float,
    max_reflection_rate: float = 1.0,
) -> tuple[float, Dict[str, float]]:
    best_key = None
    best = None
    for threshold in np.unique(np.concatenate([np.linspace(0, 1, 501), score])):
        prediction = (score >= threshold).astype("int64")
        reflection_rate = float(prediction.mean())
        if reflection_rate > max_reflection_rate + 1e-12:
            continue
        fbeta, precision, recall = _fbeta(target, prediction, beta)
        key = (fbeta, precision, -reflection_rate)
        if best_key is None or key > best_key:
            best_key = key
            best = (float(threshold), {
                "fbeta": fbeta, "precision": precision, "recall": recall,
                "accuracy": float(np.mean(prediction == target)),
                "reflection_rate": reflection_rate,
                "max_reflection_rate": float(max_reflection_rate),
            })
    assert best is not None
    return best


def _round_threshold(predicted: np.ndarray, target: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    values = predicted[mask]
    labels = target[mask] > 0
    weights = _weights(target, mask)[mask]
    best_key = None
    best = (0.0, 0.0)
    for threshold in np.unique(np.concatenate([[0.0], values])):
        accuracy = float(np.average((values > threshold) == labels, weights=weights))
        key = (accuracy, -abs(float(threshold)))
        if best_key is None or key > best_key:
            best_key = key
            best = (float(threshold), accuracy)
    return best


def _plot(histories: Sequence[Dict[str, List[float]]], output: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    train = np.mean([history["train_loss"] for history in histories], axis=0)
    val = np.mean([history["val_loss"] for history in histories], axis=0)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(range(1, len(train) + 1), train, label="OOF mean train")
    axis.plot(range(1, len(val) + 1), val, label="OOF mean validation")
    axis.set(xlabel="epoch", ylabel="multi-task loss", title="Selective advantage policy")
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def train_policy(
    rows: List[Dict[str, Any]], output: str, encoder_name: str, feature_cache: str,
    curve_output: str, history_output: str, lambda_cost: float = 0.0001,
    layers: Sequence[int] = (-4, -3, -2, -1), pooling: str = "mean",
    max_length: int = 1024, max_text_chars: int = 8000,
    encoder_batch_size: int = 8, hidden_sizes: Sequence[int] = (128, 32),
    dropout: float = 0.2, epochs: int = 40, train_batch_size: int = 128,
    learning_rate: float = 5e-4, weight_decay: float = 1e-3, folds: int = 5,
    random_state: int = 42, device: str = "cuda", gate_beta: float = 1.5,
    gate_budget_multiplier: float = 2.5, gate_budget_cap: float = 0.2,
) -> Dict[str, Any]:
    targets = build_targets(rows, lambda_cost)
    features, encoder_info = encode_features(
        rows, encoder_name, layers, pooling, max_length, max_text_chars,
        encoder_batch_size, device, feature_cache,
    )
    splitter = StratifiedKFold(folds, shuffle=True, random_state=random_state)
    oof = [np.zeros(len(rows), dtype="float32") for _ in range(3)]
    histories = []
    best_epochs = []
    for fold, (train_idx, val_idx) in enumerate(splitter.split(features, targets["gate_target"]), 1):
        print(f"[advantage] fold={fold}/{folds} train={len(train_idx)} val={len(val_idx)}", flush=True)
        model, scaler, history, best_epoch = _fit(
            features, targets, train_idx, val_idx, hidden_sizes, dropout, epochs,
            train_batch_size, learning_rate, weight_decay, random_state + fold, device,
        )
        predicted = _heads(model, scaler, features[val_idx])
        for destination, values in zip(oof, predicted):
            destination[val_idx] = values
        histories.append(history)
        best_epochs.append(best_epoch)
    scale = max(float(np.median(np.abs(targets["advantage"][targets["complete"]]))), 1e-3)
    gate_score = oof[0] / (1 + np.exp(-np.clip(oof[1] / scale, -30, 30)))
    positive_rate = float(targets["gate_target"].mean())
    max_reflection_rate = min(
        float(gate_budget_cap),
        max(positive_rate, float(gate_budget_multiplier) * positive_rate),
    )
    gate_threshold, gate_metrics = _gate_threshold(
        gate_score, targets["gate_target"], gate_beta, max_reflection_rate
    )
    gate_metrics.update(training_positive_rate=positive_rate, budget_multiplier=float(gate_budget_multiplier))
    round_mask = targets["complete"] & (targets["gate_target"] == 1)
    round_threshold, round_accuracy = _round_threshold(oof[2], targets["round_delta"], round_mask)
    final_epochs = max(1, int(round(float(np.median(best_epochs)))))
    print(
        f"[advantage] OOF threshold={gate_threshold:.6f} metrics={gate_metrics} "
        f"round_threshold={round_threshold:.6f} final_epochs={final_epochs}", flush=True,
    )
    all_idx = np.arange(len(rows))
    model, scaler, _, _ = _fit(
        features, targets, all_idx, all_idx, hidden_sizes, dropout, final_epochs,
        train_batch_size, learning_rate, weight_decay, random_state, device,
    )
    _plot(histories, curve_output)
    history = {
        "fold_histories": histories, "fold_best_epochs": best_epochs,
        "final_epochs": final_epochs, "oof_gate_metrics": gate_metrics,
        "oof_round_weighted_accuracy": round_accuracy,
    }
    save_json(history, history_output)
    bundle = {
        "policy_type": "selective_advantage_mlp", "model": model, "scaler": scaler,
        "encoder_name": encoder_name, "encoder_info": encoder_info,
        "layers": list(layers), "pooling": pooling, "pure_text": True,
        "max_length": max_length, "max_text_chars": max_text_chars,
        "encoder_batch_size": encoder_batch_size, "device": device,
        "hidden_sizes": list(hidden_sizes), "lambda_cost": lambda_cost,
        "gate_threshold": gate_threshold, "round_threshold": round_threshold,
        "gate_budget_multiplier": gate_budget_multiplier,
        "gate_budget_cap": gate_budget_cap,
        "advantage_scale": scale, "oof_gate_metrics": gate_metrics,
        "oof_round_weighted_accuracy": round_accuracy, "final_epochs": final_epochs,
        "curve_output": curve_output, "history_output": history_output,
        "n_rows": len(rows), "n_complete_counterfactual": int(targets["complete"].sum()),
        "n_direct_only": int((~targets["complete"]).sum()),
        "n_positive_advantage": int(targets["gate_target"].sum()),
        "training_assumption": (
            "Direct-correct rows without reflections supervise the error head and conservative "
            "direct gate target only; their advantage and round losses are masked."
        ),
    }
    save_pickle(bundle, output)
    return bundle


def predict_from_features(
    bundle: Dict[str, Any], features: np.ndarray,
) -> tuple[List[str], Dict[str, np.ndarray]]:
    error, advantage, round_delta = _heads(bundle["model"], bundle["scaler"], features)
    scale = max(float(bundle["advantage_scale"]), 1e-6)
    score = error / (1 + np.exp(-np.clip(advantage / scale, -30, 30)))
    reflect = score >= float(bundle["gate_threshold"])
    use_r2 = round_delta > float(bundle["round_threshold"])
    actions = np.where(reflect, np.where(use_r2, "reflect_2", "reflect_1"), "direct")
    return actions.tolist(), {
        "gate_score": score, "direct_error_probability": error,
        "predicted_advantage": advantage, "predicted_round_delta": round_delta,
    }


def predict_actions(
    bundle: Dict[str, Any], rows: Sequence[Dict[str, Any]], cache_path: str | None = None,
) -> tuple[List[str], Dict[str, np.ndarray]]:
    features, _ = encode_features(
        rows, bundle["encoder_name"], bundle["layers"], bundle["pooling"],
        bundle["max_length"], bundle["max_text_chars"], bundle["encoder_batch_size"],
        bundle.get("device", "auto"), cache_path,
    )
    return predict_from_features(bundle, features)


def predict_action(bundle: Dict[str, Any], row: Dict[str, Any]) -> str:
    return predict_actions(bundle, [row])[0][0]


def _ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an OOF selective advantage policy.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--encoder_name", required=True)
    parser.add_argument("--feature_cache", required=True)
    parser.add_argument("--curve_output", required=True)
    parser.add_argument("--history_output", required=True)
    parser.add_argument("--lambda_cost", type=float, default=0.0001)
    parser.add_argument("--layers", default="-4,-3,-2,-1")
    parser.add_argument("--pooling", default="mean", choices=["mean", "last", "mean_last"])
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_text_chars", type=int, default=8000)
    parser.add_argument("--encoder_batch_size", type=int, default=8)
    parser.add_argument("--hidden_sizes", default="128,32")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--train_batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gate_beta", type=float, default=1.5)
    parser.add_argument("--gate_budget_multiplier", type=float, default=2.5)
    parser.add_argument("--gate_budget_cap", type=float, default=0.2)
    args = parser.parse_args()
    bundle = train_policy(
        read_jsonl(args.dataset), args.output, args.encoder_name, args.feature_cache,
        args.curve_output, args.history_output, args.lambda_cost, _ints(args.layers),
        args.pooling, args.max_length, args.max_text_chars, args.encoder_batch_size,
        _ints(args.hidden_sizes), args.dropout, args.epochs, args.train_batch_size,
        args.learning_rate, args.weight_decay, args.folds, args.random_state,
        args.device, args.gate_beta, args.gate_budget_multiplier, args.gate_budget_cap,
    )
    print(f"Saved selective advantage model to {args.output}")
    print(f"OOF gate: {bundle['oof_gate_metrics']}")
    print(f"OOF round weighted accuracy: {bundle['oof_round_weighted_accuracy']:.4f}")
    print(f"Curve: {bundle['curve_output']}")


if __name__ == "__main__":
    main()
