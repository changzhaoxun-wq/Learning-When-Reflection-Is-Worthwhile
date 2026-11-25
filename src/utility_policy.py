from __future__ import annotations

import argparse
from typing import Any, Dict, List, Sequence

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from .neural_policy import build_policy_text, _encode_hf_hidden
from .reward import compute_reward
from .utils import load_pickle, project_root, read_jsonl, save_pickle

ACTIONS = ["direct", "reflect_1", "reflect_2"]


def _split_indices(n: int, test_size: float, random_state: int) -> tuple[List[int], List[int]]:
    indices = np.arange(n)
    if test_size <= 0 or n < 5:
        return indices.tolist(), []
    from sklearn.model_selection import train_test_split
    train, val = train_test_split(indices, test_size=test_size, random_state=random_state)
    return train.tolist(), val.tolist()


def _available_actions(rows: List[Dict[str, Any]]) -> List[str]:
    requested = list(rows[0].get("action_space", [])) if rows else []
    present = {str(item.get("action")) for row in rows for item in row.get("candidate_results", [])}
    actions = requested or [action for action in ACTIONS if action in present]
    if not actions:
        raise ValueError("No candidate actions found in policy dataset")
    return actions


def _reward_vector(row: Dict[str, Any], lambda_cost: float, actions: List[str]) -> List[float]:
    candidates = {str(item.get("action")): item for item in row.get("candidate_results", [])}
    values = []
    for action in actions:
        item = candidates.get(action)
        if item is None:
            raise ValueError(f"Missing candidate {action} in row {row.get('id')}")
        correct = int(float(item.get("correct", 0)))
        token_cost = int(float(item.get("token_cost", 0)))
        values.append(compute_reward(correct, token_cost, lambda_cost))
    return values


def train_utility_policy(
    rows: List[Dict[str, Any]],
    encoder_name: str,
    output_path: str,
    lambda_cost: float = 0.0001,
    max_length: int = 1024,
    batch_size: int = 8,
    hidden_sizes: Sequence[int] = (64, 16),
    max_iter: int = 500,
    alpha: float = 1e-4,
    test_size: float = 0.2,
    random_state: int = 42,
    device: str = "auto",
    pooling: str = "mean",
    max_text_chars: int = 8000,
) -> Dict[str, Any]:
    actions = _available_actions(rows)
    texts = [build_policy_text(row, max_text_chars=max_text_chars, include_state=False) for row in rows]
    features, encoder_info = _encode_hf_hidden(
        texts,
        encoder_name=encoder_name,
        max_length=max_length,
        batch_size=batch_size,
        device=device,
        pooling=pooling,
    )
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    train_idx, val_idx = _split_indices(len(rows), test_size, random_state)
    scaler = StandardScaler()
    scaler.fit(features[train_idx])
    features = scaler.transform(features).astype("float32")
    targets = np.asarray([_reward_vector(row, lambda_cost, actions) for row in rows], dtype="float32")

    model = MLPRegressor(
        hidden_layer_sizes=tuple(int(x) for x in hidden_sizes),
        activation="relu",
        solver="adam",
        alpha=float(alpha),
        learning_rate="adaptive",
        early_stopping=False,
        max_iter=int(max_iter),
        random_state=int(random_state),
    )
    model.fit(features[train_idx], targets[train_idx])
    train_mse = float(np.mean((model.predict(features[train_idx]) - targets[train_idx]) ** 2))
    val_mse = None
    if val_idx:
        val_mse = float(np.mean((model.predict(features[val_idx]) - targets[val_idx]) ** 2))

    bundle = {
        "policy_type": "utility_mlp",
        "backend": "hf_hidden_utility_mlp",
        "encoder_info": encoder_info,
        "encoder_name": encoder_name,
        "model": model,
        "scaler": scaler,
        "action_labels": actions,
        "lambda_cost": float(lambda_cost),
        "max_length": int(max_length),
        "batch_size": int(batch_size),
        "device": device,
        "pooling": pooling,
        "max_text_chars": int(max_text_chars),
        "pure_text": True,
        "hidden_sizes": list(hidden_sizes),
        "train_mse": train_mse,
        "val_mse": val_mse,
        "n_train": len(train_idx),
        "n_val": len(val_idx),
    }
    save_pickle(bundle, output_path)
    return bundle


def predict_utility_action(bundle: Dict[str, Any], row: Dict[str, Any]) -> str:
    text = build_policy_text(
        row,
        max_text_chars=int(bundle.get("max_text_chars", 8000)),
        include_state=False,
    )
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
    utilities = bundle["model"].predict(features)[0]
    actions = list(bundle.get("action_labels") or ACTIONS)
    return actions[int(np.argmax(utilities))]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a pure-text utility policy without best_action labels.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--encoder_name", required=True)
    parser.add_argument("--lambda_cost", type=float, default=0.0001)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--hidden_sizes", default="64,16")
    parser.add_argument("--max_iter", type=int, default=500)
    parser.add_argument("--alpha", type=float, default=1e-4)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--pooling", choices=["mean", "last", "mean_last"], default="mean")
    args = parser.parse_args()
    hidden_sizes = tuple(int(x.strip()) for x in args.hidden_sizes.split(",") if x.strip())
    rows = read_jsonl(args.dataset)
    bundle = train_utility_policy(
        rows=rows,
        encoder_name=args.encoder_name,
        output_path=args.output,
        lambda_cost=args.lambda_cost,
        max_length=args.max_length,
        batch_size=args.batch_size,
        hidden_sizes=hidden_sizes,
        max_iter=args.max_iter,
        alpha=args.alpha,
        test_size=args.test_size,
        device=args.device,
        pooling=args.pooling,
    )
    print(f"Saved utility policy model to {args.output}")
    print(f"Train MSE: {bundle['train_mse']:.6f}")
    if bundle.get("val_mse") is not None:
        print(f"Validation MSE: {bundle['val_mse']:.6f}")
    print("Action targets: reward(" + ", ".join(bundle["action_labels"]) + f"), lambda={args.lambda_cost}")


if __name__ == "__main__":
    main()
