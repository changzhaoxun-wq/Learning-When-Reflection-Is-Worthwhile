from __future__ import annotations

import argparse
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .neural_policy import build_policy_text, _encode_hf_hidden
from .reward import compute_reward
from .utils import read_jsonl, save_pickle

ACTIONS = ["direct", "reflect_1", "reflect_2"]


class ActorMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes: Sequence[int], n_actions: int = 3) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        previous = input_dim
        for size in hidden_sizes:
            layers.extend([nn.Linear(previous, int(size)), nn.ReLU()])
            previous = int(size)
        layers.append(nn.Linear(previous, n_actions))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def _available_actions(rows: List[Dict[str, Any]]) -> List[str]:
    requested = list(rows[0].get("action_space", [])) if rows else []
    present = {str(item.get("action")) for row in rows for item in row.get("candidate_results", [])}
    actions = requested or [action for action in ACTIONS if action in present]
    if not actions:
        raise ValueError("No candidate actions found in policy dataset")
    return actions


def _reward_matrix(rows: List[Dict[str, Any]], lambda_cost: float, actions: List[str]) -> np.ndarray:
    matrix = []
    for row in rows:
        candidates = {str(item.get("action")): item for item in row.get("candidate_results", [])}
        values = []
        for action in actions:
            item = candidates[action]
            values.append(compute_reward(
                int(float(item.get("correct", 0))),
                int(float(item.get("token_cost", 0))),
                lambda_cost,
            ))
        matrix.append(values)
    return np.asarray(matrix, dtype="float32")


def train_actor_policy(
    rows: List[Dict[str, Any]],
    encoder_name: str,
    output_path: str,
    lambda_cost: float = 0.0001,
    max_length: int = 1024,
    encoder_batch_size: int = 8,
    train_batch_size: int = 128,
    hidden_sizes: Sequence[int] = (128, 32),
    epochs: int = 80,
    learning_rate: float = 1e-3,
    entropy_coef: float = 0.005,
    test_size: float = 0.2,
    random_state: int = 42,
    device: str = "cuda",
    pooling: str = "mean",
    max_text_chars: int = 8000,
) -> Dict[str, Any]:
    actions = _available_actions(rows)
    texts = [build_policy_text(row, max_text_chars=max_text_chars, include_state=False) for row in rows]
    features, encoder_info = _encode_hf_hidden(
        texts,
        encoder_name=encoder_name,
        max_length=max_length,
        batch_size=encoder_batch_size,
        device=device,
        pooling=pooling,
    )
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    train_idx, val_idx = train_test_split(
        np.arange(len(rows)), test_size=test_size, random_state=random_state
    )
    scaler = StandardScaler()
    scaler.fit(features[train_idx])
    features = scaler.transform(features).astype("float32")
    rewards = _reward_matrix(rows, lambda_cost, actions)

    torch.manual_seed(random_state)
    train_device = torch.device(device if device != "auto" and torch.cuda.is_available() else "cpu")
    model = ActorMLP(features.shape[1], hidden_sizes, n_actions=len(actions)).to(train_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    x = torch.from_numpy(features).to(train_device)
    r = torch.from_numpy(rewards).to(train_device)
    rng = np.random.default_rng(random_state)

    model.train()
    for _ in range(int(epochs)):
        order = rng.permutation(train_idx)
        for start in range(0, len(order), int(train_batch_size)):
            batch_idx = torch.as_tensor(order[start : start + int(train_batch_size)], device=train_device)
            logits = model(x[batch_idx])
            probs = torch.softmax(logits, dim=-1)
            log_probs = torch.log_softmax(logits, dim=-1)
            advantages = r[batch_idx] - r[batch_idx].mean(dim=1, keepdim=True)
            expected_advantage = (probs * advantages).sum(dim=1).mean()
            entropy = -(probs * log_probs).sum(dim=1).mean()
            loss = -expected_advantage - float(entropy_coef) * entropy
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def metrics(indices: List[int] | np.ndarray) -> Dict[str, float]:
        if len(indices) == 0:
            return {"action_agreement": 0.0, "avg_reward": 0.0}
        with torch.no_grad():
            predicted = model(x[torch.as_tensor(indices, device=train_device)]).argmax(dim=-1).cpu().numpy()
        target = rewards[indices].argmax(axis=1)
        chosen_reward = rewards[indices, predicted].mean()
        return {
            "action_agreement": float(np.mean(predicted == target)),
            "avg_reward": float(chosen_reward),
        }

    train_metrics = metrics(train_idx)
    val_metrics = metrics(val_idx)
    model = model.to("cpu").eval()
    bundle = {
        "policy_type": "actor_mlp",
        "backend": "hf_hidden_actor_mlp",
        "encoder_info": encoder_info,
        "encoder_name": encoder_name,
        "model": model,
        "scaler": scaler,
        "action_labels": actions,
        "lambda_cost": float(lambda_cost),
        "max_length": int(max_length),
        "encoder_batch_size": int(encoder_batch_size),
        "device": device,
        "pooling": pooling,
        "max_text_chars": int(max_text_chars),
        "pure_text": True,
        "hidden_sizes": list(hidden_sizes),
        "epochs": int(epochs),
        "learning_rate": float(learning_rate),
        "entropy_coef": float(entropy_coef),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
    }
    save_pickle(bundle, output_path)
    return bundle


def predict_actor_action(bundle: Dict[str, Any], row: Dict[str, Any]) -> str:
    text = build_policy_text(row, max_text_chars=int(bundle.get("max_text_chars", 8000)), include_state=False)
    features, _ = _encode_hf_hidden(
        [text],
        encoder_name=bundle["encoder_name"],
        max_length=int(bundle.get("max_length", 1024)),
        batch_size=1,
        device=bundle.get("device", "auto"),
        pooling=bundle.get("pooling", "mean"),
    )
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    features = bundle["scaler"].transform(features).astype("float32")
    with torch.no_grad():
        logits = bundle["model"](torch.from_numpy(features))
    actions = list(bundle.get("action_labels") or ACTIONS)
    return actions[int(logits.argmax(dim=-1).item())]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an offline actor policy directly from action rewards.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--encoder_name", required=True)
    parser.add_argument("--lambda_cost", type=float, default=0.0001)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--encoder_batch_size", type=int, default=8)
    parser.add_argument("--train_batch_size", type=int, default=128)
    parser.add_argument("--hidden_sizes", default="128,32")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--entropy_coef", type=float, default=0.005)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pooling", choices=["mean", "last", "mean_last"], default="mean")
    args = parser.parse_args()
    rows = read_jsonl(args.dataset)
    hidden_sizes = tuple(int(x.strip()) for x in args.hidden_sizes.split(",") if x.strip())
    bundle = train_actor_policy(
        rows=rows,
        encoder_name=args.encoder_name,
        output_path=args.output,
        lambda_cost=args.lambda_cost,
        max_length=args.max_length,
        encoder_batch_size=args.encoder_batch_size,
        train_batch_size=args.train_batch_size,
        hidden_sizes=hidden_sizes,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        entropy_coef=args.entropy_coef,
        device=args.device,
        pooling=args.pooling,
    )
    print(f"Saved actor policy model to {args.output}")
    print(f"Train action agreement: {bundle['train_metrics']['action_agreement']:.4f}")
    print(f"Validation action agreement: {bundle['val_metrics']['action_agreement']:.4f}")
    print(f"Train average reward: {bundle['train_metrics']['avg_reward']:.4f}")
    print(f"Validation average reward: {bundle['val_metrics']['avg_reward']:.4f}")


if __name__ == "__main__":
    main()
