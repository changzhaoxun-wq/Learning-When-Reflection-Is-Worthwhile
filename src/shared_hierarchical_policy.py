from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn

from src.neural_policy import _encode_hf_hidden, build_policy_text
from src.reward import compute_reward
from src.utils import read_jsonl, save_pickle


class SharedHierarchicalMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes: Sequence[int] = (64, 16)) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        previous = int(input_dim)
        for size in hidden_sizes:
            layers.extend([nn.Linear(previous, int(size)), nn.ReLU()])
            previous = int(size)
        self.trunk = nn.Sequential(*layers)
        self.gate_head = nn.Linear(previous, 2)
        self.round_head = nn.Linear(previous, 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.trunk(x)
        return self.gate_head(hidden), self.round_head(hidden)


def _candidate(row: Dict[str, Any], action: str) -> Dict[str, Any]:
    return next(item for item in row.get("candidate_results", []) if str(item.get("action")) == action)


def _best_reflection(row: Dict[str, Any]) -> str:
    candidates = [_candidate(row, "reflect_1"), _candidate(row, "reflect_2")]
    return max(candidates, key=lambda item: (int(float(item.get("correct", 0))), -int(float(item.get("token_cost", 0)))))["action"]


def _targets(rows: List[Dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    gate = []
    round_targets = []
    for row in rows:
        direct = _candidate(row, "direct")
        reflection = _candidate(row, _best_reflection(row))
        direct_key = (int(float(direct.get("correct", 0))), -int(float(direct.get("token_cost", 0))))
        reflection_key = (int(float(reflection.get("correct", 0))), -int(float(reflection.get("token_cost", 0))))
        gate.append(0 if direct_key >= reflection_key else 1)
        r1 = _candidate(row, "reflect_1")
        r2 = _candidate(row, "reflect_2")
        r1_key = (int(float(r1.get("correct", 0))), -int(float(r1.get("token_cost", 0))))
        r2_key = (int(float(r2.get("correct", 0))), -int(float(r2.get("token_cost", 0))))
        round_targets.append(0 if r1_key >= r2_key else 1)
    return np.asarray(gate, dtype="int64"), np.asarray(round_targets, dtype="int64")


def _chosen_action(row: Dict[str, Any], gate_pred: int, round_pred: int) -> str:
    return "direct" if int(gate_pred) == 0 else ("reflect_1" if int(round_pred) == 0 else "reflect_2")


def _answer_accuracy(rows: List[Dict[str, Any]], gate_pred: np.ndarray, round_pred: np.ndarray) -> float:
    if not rows:
        return 0.0
    return float(np.mean([float(_candidate(row, _chosen_action(row, g, r)).get("correct", 0)) for row, g, r in zip(rows, gate_pred, round_pred)]))


def _reward(rows: List[Dict[str, Any]], gate_pred: np.ndarray, round_pred: np.ndarray, lambda_cost: float) -> float:
    if not rows:
        return 0.0
    values = []
    for row, g, r in zip(rows, gate_pred, round_pred):
        item = _candidate(row, _chosen_action(row, g, r))
        values.append(compute_reward(int(float(item.get("correct", 0))), int(float(item.get("token_cost", 0))), lambda_cost))
    return float(np.mean(values))


def _calibrate_gate(
    rows: List[Dict[str, Any]],
    gate_prob: np.ndarray,
    round_pred: np.ndarray,
    lambda_cost: float,
) -> tuple[float, np.ndarray, float, float]:
    thresholds = np.unique(np.concatenate([np.linspace(0.0, 1.0, 201), gate_prob]))
    best = None
    best_key = None
    for threshold in thresholds:
        gate_pred = np.where(gate_prob >= threshold, 0, 1).astype("int64")
        accuracy = _answer_accuracy(rows, gate_pred, round_pred)
        reward = _reward(rows, gate_pred, round_pred, lambda_cost)
        key = (accuracy, reward, float(threshold))
        if best_key is None or key > best_key:
            best_key = key
            best = (float(threshold), gate_pred, accuracy, reward)
    assert best is not None
    return best


def train(
    rows: List[Dict[str, Any]],
    encoder_name: str,
    output: str,
    layers: Sequence[int],
    pooling: str,
    max_length: int,
    encoder_batch_size: int,
    train_batch_size: int,
    hidden_sizes: Sequence[int],
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    test_size: float,
    random_state: int,
    device: str,
    lambda_cost: float,
    curve_output: str,
    history_output: str,
) -> Dict[str, Any]:
    texts = [build_policy_text(row, include_state=False) for row in rows]
    features, encoder_info = _encode_hf_hidden(
        texts,
        encoder_name=encoder_name,
        max_length=max_length,
        batch_size=encoder_batch_size,
        device=device,
        pooling=pooling,
        layer_indices=list(layers),
    )
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    gate_targets, round_targets = _targets(rows)
    indices = np.arange(len(rows))
    train_idx, val_idx = train_test_split(indices, test_size=test_size, random_state=random_state, stratify=gate_targets)
    scaler = StandardScaler().fit(features[train_idx])
    features = scaler.transform(features).astype("float32")

    resolved = device if device not in {"", "auto", None} else ("cuda" if torch.cuda.is_available() else "cpu")
    train_device = torch.device(resolved if resolved.startswith("cuda") and torch.cuda.is_available() else "cpu")
    torch.manual_seed(random_state)
    model = SharedHierarchicalMLP(features.shape[1], hidden_sizes).to(train_device)
    gate_counts = np.bincount(gate_targets[train_idx], minlength=2).astype("float32")
    round_counts = np.bincount(round_targets[train_idx], minlength=2).astype("float32")
    gate_weights = len(train_idx) / np.maximum(2 * gate_counts, 1.0)
    round_weights = len(train_idx) / np.maximum(2 * round_counts, 1.0)
    gate_loss = nn.CrossEntropyLoss(weight=torch.tensor(gate_weights, device=train_device))
    round_loss = nn.CrossEntropyLoss(weight=torch.tensor(round_weights, device=train_device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    x = torch.from_numpy(features)
    gate_y = torch.from_numpy(gate_targets)
    round_y = torch.from_numpy(round_targets)
    rng = np.random.default_rng(random_state)
    history = {"train_loss": [], "val_loss": [], "train_answer_accuracy": [], "val_answer_accuracy": [], "train_reward": [], "val_reward": [], "gate_accuracy": [], "round_accuracy": []}
    best_state = None
    best_epoch = 0
    best_threshold = 0.5
    best_score = float("-inf")

    for epoch in range(1, int(epochs) + 1):
        model.train()
        order = rng.permutation(train_idx)
        for start in range(0, len(order), max(1, int(train_batch_size))):
            batch_idx = order[start : start + max(1, int(train_batch_size))]
            gate_logits, round_logits = model(x[batch_idx].to(train_device))
            loss = gate_loss(gate_logits, gate_y[batch_idx].to(train_device)) + round_loss(round_logits, round_y[batch_idx].to(train_device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            train_gate_logits, train_round_logits = model(x[train_idx].to(train_device))
            val_gate_logits, val_round_logits = model(x[val_idx].to(train_device))
            train_loss_value = float((gate_loss(train_gate_logits, gate_y[train_idx].to(train_device)) + round_loss(train_round_logits, round_y[train_idx].to(train_device))).item())
            val_loss_value = float((gate_loss(val_gate_logits, gate_y[val_idx].to(train_device)) + round_loss(val_round_logits, round_y[val_idx].to(train_device))).item())
            train_gate_prob = torch.softmax(train_gate_logits, dim=1)[:, 0].cpu().numpy()
            val_gate_prob = torch.softmax(val_gate_logits, dim=1)[:, 0].cpu().numpy()
            train_round_pred = train_round_logits.argmax(dim=1).cpu().numpy()
            val_round_pred = val_round_logits.argmax(dim=1).cpu().numpy()
            train_gate_pred = np.where(train_gate_prob >= 0.5, 0, 1).astype("int64")
        train_rows = [rows[i] for i in train_idx]
        val_rows = [rows[i] for i in val_idx]
        threshold, val_gate_pred, val_accuracy, val_reward = _calibrate_gate(val_rows, val_gate_prob, val_round_pred, lambda_cost)
        train_gate_pred_cal = np.where(train_gate_prob >= threshold, 0, 1).astype("int64")
        train_accuracy = _answer_accuracy(train_rows, train_gate_pred_cal, train_round_pred)
        train_reward = _reward(train_rows, train_gate_pred_cal, train_round_pred, lambda_cost)
        history["train_loss"].append(train_loss_value)
        history["val_loss"].append(val_loss_value)
        history["train_answer_accuracy"].append(train_accuracy)
        history["val_answer_accuracy"].append(val_accuracy)
        history["train_reward"].append(train_reward)
        history["val_reward"].append(val_reward)
        history["gate_accuracy"].append(float(np.mean(train_gate_pred == gate_targets[train_idx])))
        history["round_accuracy"].append(float(np.mean(train_round_pred == round_targets[train_idx])))
        print(f"[shared-policy] epoch={epoch:03d}/{epochs} train_loss={train_loss_value:.4f} val_loss={val_loss_value:.4f} train_answer_acc={train_accuracy:.4f} val_answer_acc={val_accuracy:.4f} threshold={threshold:.4f} val_reward={val_reward:.4f}", flush=True)
        if val_accuracy > best_score:
            best_score = val_accuracy
            best_epoch = epoch
            best_threshold = threshold
            best_state = copy.deepcopy(model.state_dict())

    assert best_state is not None
    model.load_state_dict(best_state)
    model = model.to("cpu").eval()
    Path(curve_output).parent.mkdir(parents=True, exist_ok=True)
    Path(history_output).parent.mkdir(parents=True, exist_ok=True)
    Path(history_output).write_text(json.dumps(history, indent=2))
    bundle = {"policy_type": "shared_hierarchical_torch_mlp", "model": model, "encoder_name": encoder_name, "encoder_info": encoder_info, "scaler": scaler, "layers": list(layers), "pooling": pooling, "pure_text": True, "max_length": max_length, "device": device, "gate_threshold": best_threshold, "best_epoch": best_epoch, "best_val_answer_accuracy": best_score, "hidden_sizes": list(hidden_sizes), "history": history, "curve_output": curve_output, "history_output": history_output}
    save_pickle(bundle, output)
    return bundle


def predict(bundle: Dict[str, Any], row: Dict[str, Any]) -> str:
    text = build_policy_text(row, include_state=False)
    features, _ = _encode_hf_hidden([text], encoder_name=bundle["encoder_name"], max_length=int(bundle["max_length"]), batch_size=1, device=bundle.get("device", "auto"), pooling=bundle["pooling"], layer_indices=list(bundle["layers"]))
    features = bundle["scaler"].transform(np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)).astype("float32")
    with torch.no_grad():
        gate_logits, round_logits = bundle["model"](torch.from_numpy(features))
        gate_prob = float(torch.softmax(gate_logits, dim=1)[0, 0].item())
        if gate_prob >= float(bundle.get("gate_threshold", 0.5)):
            return "direct"
        return "reflect_1" if int(round_logits.argmax(dim=1).item()) == 0 else "reflect_2"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a shared-encoder hierarchical policy with two action heads.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--encoder_name", required=True)
    parser.add_argument("--layers", default="-2,-1")
    parser.add_argument("--pooling", choices=["mean", "last", "mean_last"], default="mean_last")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--encoder_batch_size", type=int, default=4)
    parser.add_argument("--train_batch_size", type=int, default=128)
    parser.add_argument("--hidden_sizes", default="64,16")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.001)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lambda_cost", type=float, default=0.0001)
    parser.add_argument("--curve_output", required=True)
    parser.add_argument("--history_output", required=True)
    args = parser.parse_args()
    rows = read_jsonl(args.dataset)
    train(rows, args.encoder_name, args.output, [int(x) for x in args.layers.split(",") if x], args.pooling, args.max_length, args.encoder_batch_size, args.train_batch_size, [int(x) for x in args.hidden_sizes.split(",") if x], args.epochs, args.learning_rate, args.weight_decay, args.test_size, args.random_state, args.device, args.lambda_cost, args.curve_output, args.history_output)
    print(f"Saved shared hierarchical policy to {args.output}")


if __name__ == "__main__":
    main()
