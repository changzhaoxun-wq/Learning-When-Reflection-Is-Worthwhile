from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn

from src.neural_policy import _encode_hf_hidden, build_policy_text
from src.utils import read_jsonl, save_pickle


class ContrastiveRoundMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes: tuple[int, int] = (64, 16)) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_sizes[0]), nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]), nn.ReLU(),
        )
        self.classifier = nn.Linear(hidden_sizes[1], 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.trunk(x)
        return self.classifier(z), z


def _correct(row: Dict[str, Any], action: str) -> int:
    item = next(x for x in row["candidate_results"] if str(x.get("action")) == action)
    return int(float(item.get("correct", 0)))


def _answer_accuracy(rows: List[Dict[str, Any]], preds: np.ndarray) -> float:
    values = []
    for row, pred in zip(rows, preds):
        action = "reflect_1" if int(pred) == 0 else "reflect_2"
        values.append(_correct(row, action))
    return float(np.mean(values)) if values else 0.0


def _contrastive_loss(z: torch.Tensor, labels: torch.Tensor, margin: float = 0.5) -> torch.Tensor:
    if z.shape[0] < 2:
        return z.sum() * 0.0
    normalized = F.normalize(z, dim=1)
    similarity = normalized @ normalized.T
    same = labels[:, None].eq(labels[None, :])
    eye = torch.eye(labels.shape[0], dtype=torch.bool, device=labels.device)
    positive = (1.0 - similarity)[same & ~eye]
    negative = F.relu(similarity[~same] - margin)
    terms = []
    if positive.numel():
        terms.append(positive.mean())
    if negative.numel():
        terms.append(negative.mean())
    return torch.stack(terms).mean() if terms else z.sum() * 0.0


def train(args: argparse.Namespace) -> Dict[str, Any]:
    rows = read_jsonl(args.dataset)
    texts = [build_policy_text(row, include_state=False) for row in rows]
    features, encoder_info = _encode_hf_hidden(
        texts, encoder_name=args.encoder_name, max_length=args.max_length,
        batch_size=args.encoder_batch_size, device=args.device,
        pooling=args.pooling, layer_indices=[int(x) for x in args.layers.split(",") if x],
    )
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    targets = np.asarray([0 if row["best_action"] == "reflect_1" else 1 for row in rows], dtype="int64")
    disagreement = np.asarray([_correct(row, "reflect_1") != _correct(row, "reflect_2") for row in rows], dtype=bool)
    indices = np.arange(len(rows))
    train_idx, val_idx = train_test_split(indices, test_size=args.test_size, random_state=args.random_state, stratify=targets)
    scaler = StandardScaler().fit(features[train_idx])
    features = scaler.transform(features).astype("float32")
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.random_state)
    model = ContrastiveRoundMLP(features.shape[1], tuple(int(x) for x in args.hidden_sizes.split(","))).to(device)
    counts = np.bincount(targets[train_idx], minlength=2).astype("float32")
    weights = len(train_idx) / np.maximum(2 * counts, 1.0)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    x = torch.from_numpy(features)
    y = torch.from_numpy(targets)
    rng = np.random.default_rng(args.random_state)
    history = {"train_loss": [], "val_loss": [], "train_answer_accuracy": [], "val_answer_accuracy": [], "train_action_accuracy": [], "val_action_accuracy": []}
    best_state = None
    best_epoch = 0
    best_score = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(train_idx)
        for start in range(0, len(order), args.train_batch_size):
            batch_idx = order[start:start + args.train_batch_size]
            logits, z = model(x[batch_idx].to(device))
            loss = criterion(logits, y[batch_idx].to(device))
            local_disagreement = disagreement[batch_idx]
            if local_disagreement.sum() >= 2:
                contrastive = _contrastive_loss(z[torch.from_numpy(local_disagreement).to(device)], y[batch_idx][local_disagreement].to(device))
                loss = loss + args.contrastive_weight * contrastive
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            train_logits, _ = model(x[train_idx].to(device))
            val_logits, _ = model(x[val_idx].to(device))
            train_loss = float(criterion(train_logits, y[train_idx].to(device)).item())
            val_loss = float(criterion(val_logits, y[val_idx].to(device)).item())
            train_pred = train_logits.argmax(dim=1).cpu().numpy()
            val_pred = val_logits.argmax(dim=1).cpu().numpy()
        train_rows = [rows[i] for i in train_idx]
        val_rows = [rows[i] for i in val_idx]
        train_answer = _answer_accuracy(train_rows, train_pred)
        val_answer = _answer_accuracy(val_rows, val_pred)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_answer_accuracy"].append(train_answer)
        history["val_answer_accuracy"].append(val_answer)
        history["train_action_accuracy"].append(float(np.mean(train_pred == targets[train_idx])))
        history["val_action_accuracy"].append(float(np.mean(val_pred == targets[val_idx])))
        print(f"[contrastive-round] epoch={epoch:03d}/{args.epochs} train_loss={train_loss:.4f} val_loss={val_loss:.4f} train_answer_acc={train_answer:.4f} val_answer_acc={val_answer:.4f}", flush=True)
        if val_answer > best_score:
            best_score = val_answer
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
    assert best_state is not None
    model.load_state_dict(best_state)
    model = model.to("cpu").eval()
    Path(args.history_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.history_output).write_text(json.dumps(history, indent=2))
    bundle = {"policy_type": "contrastive_round_torch_mlp", "model": model, "encoder_name": args.encoder_name, "encoder_info": encoder_info, "scaler": scaler, "layers": [int(x) for x in args.layers.split(",") if x], "pooling": args.pooling, "pure_text": True, "max_length": args.max_length, "device": args.device, "hidden_sizes": [int(x) for x in args.hidden_sizes.split(",")], "contrastive_weight": args.contrastive_weight, "best_epoch": best_epoch, "best_val_answer_accuracy": best_score, "history_output": args.history_output}
    save_pickle(bundle, args.output)
    print(f"Saved contrastive round policy to {args.output}; best_epoch={best_epoch} val_answer_accuracy={best_score:.4f}")
    return bundle


def predict(bundle: Dict[str, Any], row: Dict[str, Any]) -> str:
    text = build_policy_text(row, include_state=False)
    features, _ = _encode_hf_hidden([text], encoder_name=bundle["encoder_name"], max_length=int(bundle["max_length"]), batch_size=1, device=bundle.get("device", "auto"), pooling=bundle["pooling"], layer_indices=list(bundle["layers"]))
    features = bundle["scaler"].transform(np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)).astype("float32")
    with torch.no_grad():
        logits, _ = bundle["model"](torch.from_numpy(features))
    return "reflect_1" if int(logits.argmax(dim=1).item()) == 0 else "reflect_2"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--encoder_name", required=True)
    parser.add_argument("--layers", default="-2,-1")
    parser.add_argument("--pooling", default="mean_last")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--encoder_batch_size", type=int, default=4)
    parser.add_argument("--train_batch_size", type=int, default=128)
    parser.add_argument("--hidden_sizes", default="64,16")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.001)
    parser.add_argument("--contrastive_weight", type=float, default=0.1)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--history_output", required=True)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
