from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn

from .neural_policy import _encode_hf_hidden, build_policy_text
from .reward import compute_reward
from .utils import read_jsonl, save_pickle

DEFAULT_ACTIONS = ["direct", "reflect_1", "reflect_2"]


class TorchPolicyMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes: Sequence[int], n_actions: int) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        previous = int(input_dim)
        for size in hidden_sizes:
            layers.extend([nn.Linear(previous, int(size)), nn.ReLU()])
            previous = int(size)
        layers.append(nn.Linear(previous, int(n_actions)))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def _ordered_actions(rows: List[Dict[str, Any]], labels: Iterable[str]) -> List[str]:
    present = set(labels)
    requested = list(rows[0].get("action_space", [])) if rows else []
    ordered = requested or [action for action in DEFAULT_ACTIONS if action in present]
    ordered.extend(sorted(present - set(ordered)))
    if len(ordered) < 2:
        raise ValueError("Transformer policy requires at least two action labels")
    return ordered


def _parse_layer_indices(value: str) -> List[int]:
    result = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not result:
        raise ValueError("LAYERS must contain at least one layer index, e.g. -4,-3,-2,-1")
    return result


def _reward_for_action(row: Dict[str, Any], action: str, lambda_cost: float) -> float:
    for item in row.get("candidate_results", []):
        if str(item.get("action")) == action:
            if item.get("reward") is not None:
                return float(item["reward"])
            correct = int(float(item.get("correct", 0)))
            token_cost = int(float(item.get("token_cost", 0)))
            return compute_reward(correct, token_cost, lambda_cost)
    return 0.0


def _chosen_rewards(rows: List[Dict[str, Any]], predictions: np.ndarray, actions: List[str], lambda_cost: float) -> np.ndarray:
    return np.asarray([
        _reward_for_action(row, actions[int(index)], lambda_cost)
        for row, index in zip(rows, predictions)
    ], dtype="float32")


def _chosen_answer_accuracy(rows: List[Dict[str, Any]], predictions: np.ndarray, actions: List[str]) -> float:
    if not rows:
        return 0.0
    values = []
    for row, index in zip(rows, predictions):
        action = actions[int(index)]
        item = next((candidate for candidate in row.get("candidate_results", []) if str(candidate.get("action")) == action), None)
        values.append(float((item or {}).get("correct", 0)))
    return float(np.mean(values))


def _calibrate_binary_threshold(
    rows: List[Dict[str, Any]],
    direct_probabilities: np.ndarray,
    actions: List[str],
    lambda_cost: float,
    objective: str,
    target_labels: np.ndarray | None = None,
) -> tuple[float, np.ndarray, float, float]:
    direct_index = actions.index("direct")
    reflect_actions = [action for action in actions if action != "direct"]
    if len(actions) != 2 or len(reflect_actions) != 1:
        predictions = np.argmax(np.stack([direct_probabilities, 1.0 - direct_probabilities], axis=1), axis=1)
        return 0.5, predictions, _chosen_answer_accuracy(rows, predictions, actions), float(np.mean(_chosen_rewards(rows, predictions, actions, lambda_cost)))
    reflect_index = actions.index(reflect_actions[0])
    thresholds = np.unique(np.concatenate([np.linspace(0.0, 1.0, 201), direct_probabilities]))
    best_key = None
    best = None
    for threshold in thresholds:
        predictions = np.where(direct_probabilities >= threshold, direct_index, reflect_index).astype("int64")
        answer_accuracy = _chosen_answer_accuracy(rows, predictions, actions)
        reward = float(np.mean(_chosen_rewards(rows, predictions, actions, lambda_cost)))
        if objective == "action_accuracy":
            if target_labels is None:
                raise ValueError("target_labels are required for action_accuracy calibration")
            action_accuracy = _accuracy(predictions, target_labels)
            direct_rate = float(np.mean(predictions == direct_index))
            key = (action_accuracy, direct_rate)
        elif objective == "answer_accuracy":
            key = (answer_accuracy, threshold)
        else:
            key = (reward, answer_accuracy, -threshold)
        if best_key is None or key > best_key:
            best_key = key
            best = (float(threshold), predictions, answer_accuracy, reward)
    assert best is not None
    return best


def _accuracy(predictions: np.ndarray, targets: np.ndarray) -> float:
    return float(np.mean(predictions == targets)) if len(targets) else 0.0


def _plot_history(history: Dict[str, List[float]], output_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="validation")
    axes[0].set_title("Cross-entropy loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()
    axes[1].plot(epochs, history["train_accuracy"], label="train action")
    axes[1].plot(epochs, history["val_accuracy"], label="validation action")
    if "train_answer_accuracy" in history:
        axes[1].plot(epochs, history["train_answer_accuracy"], "--", label="train answer")
        axes[1].plot(epochs, history["val_answer_accuracy"], "--", label="validation answer")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].legend()
    axes[2].plot(epochs, history["train_reward"], label="train")
    axes[2].plot(epochs, history["val_reward"], label="validation")
    axes[2].set_title("Chosen average reward")
    axes[2].set_xlabel("epoch")
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _save_history(history: Dict[str, List[float]], output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(history, indent=2))


def train_transformer_policy(
    rows: List[Dict[str, Any]],
    encoder_name: str,
    output_path: str,
    layers: Sequence[int] = (-4, -3, -2, -1),
    pooling: str = "mean",
    pure_text: bool = True,
    max_length: int = 1024,
    encoder_batch_size: int = 8,
    train_batch_size: int = 128,
    hidden_sizes: Sequence[int] = (128, 32),
    epochs: int = 80,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    test_size: float = 0.2,
    random_state: int = 42,
    device: str = "cuda",
    lambda_cost: float = 0.0001,
    balance_classes: bool = True,
    curve_output: str = "outputs/training_curves/transformer_policy.png",
    history_output: str = "outputs/training_curves/transformer_policy_history.json",
    epoch_plot_dir: str = "outputs/training_curves/epochs",
    save_epoch_plots: bool = True,
    selection_metric: str = "val_answer_accuracy",
    threshold_metric: str = "answer_accuracy",
    max_text_chars: int = 8000,
) -> Dict[str, Any]:
    if not rows:
        raise ValueError("Cannot train transformer policy on an empty dataset.")
    if selection_metric not in {"val_loss", "val_accuracy", "val_reward", "val_answer_accuracy"}:
        raise ValueError("selection_metric must be val_loss, val_accuracy, val_reward, or val_answer_accuracy")
    if threshold_metric not in {"action_accuracy", "answer_accuracy", "reward"}:
        raise ValueError("threshold_metric must be action_accuracy, answer_accuracy, or reward")
    labels = [str(row["best_action"]) for row in rows]
    actions = _ordered_actions(rows, labels)
    label_to_id = {action: index for index, action in enumerate(actions)}
    targets = np.asarray([label_to_id[label] for label in labels], dtype="int64")
    texts = [
        build_policy_text(row, max_text_chars=max_text_chars, include_state=not pure_text)
        for row in rows
    ]
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
    features = np.clip(features, -1e6, 1e6).astype("float32")

    indices = np.arange(len(rows))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=targets if len(set(targets.tolist())) > 1 else None,
    )
    scaler = StandardScaler()
    scaler.fit(features[train_idx])
    features = scaler.transform(features).astype("float32")

    resolved_device = device if device not in {"", "auto", None} else ("cuda" if torch.cuda.is_available() else "cpu")
    train_device = torch.device(resolved_device if resolved_device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    torch.manual_seed(random_state)
    model = TorchPolicyMLP(features.shape[1], hidden_sizes, len(actions)).to(train_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if balance_classes:
        counts = np.bincount(targets[train_idx], minlength=len(actions)).astype("float32")
        weights = len(train_idx) / np.maximum(len(actions) * counts, 1.0)
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, device=train_device))
    else:
        criterion = nn.CrossEntropyLoss()

    x = torch.from_numpy(features)
    y = torch.from_numpy(targets)
    history: Dict[str, List[float]] = {
        "train_loss": [], "val_loss": [],
        "train_accuracy": [], "val_accuracy": [],
        "train_answer_accuracy": [], "val_answer_accuracy": [],
        "train_reward": [], "val_reward": [],
    }
    rng = np.random.default_rng(random_state)
    best_epoch = 0
    best_score = float("inf") if selection_metric == "val_loss" else float("-inf")
    best_state = None
    best_threshold = 0.5

    for epoch in range(1, int(epochs) + 1):
        model.train()
        order = rng.permutation(train_idx)
        for start in range(0, len(order), max(1, int(train_batch_size))):
            batch_indices = order[start : start + max(1, int(train_batch_size))]
            batch_x = x[batch_indices].to(train_device)
            batch_y = y[batch_indices].to(train_device)
            loss = criterion(model(batch_x), batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            train_logits = model(x[train_idx].to(train_device))
            val_logits = model(x[val_idx].to(train_device)) if len(val_idx) else None
            train_loss = float(criterion(train_logits, y[train_idx].to(train_device)).item())
            val_loss = float(criterion(val_logits, y[val_idx].to(train_device)).item()) if val_logits is not None else train_loss
            train_pred = train_logits.argmax(dim=1).cpu().numpy()
            val_pred = val_logits.argmax(dim=1).cpu().numpy() if val_logits is not None else train_pred
        train_rows = [rows[i] for i in train_idx]
        val_rows = [rows[i] for i in val_idx]
        if len(actions) == 2 and "direct" in actions:
            direct_index = actions.index("direct")
            train_probabilities = torch.softmax(train_logits, dim=1)[:, direct_index].cpu().numpy()
            val_probabilities = torch.softmax(val_logits, dim=1)[:, direct_index].cpu().numpy() if val_logits is not None else train_probabilities
            threshold, threshold_val_pred, _, val_reward = _calibrate_binary_threshold(
                val_rows,
                val_probabilities,
                actions,
                lambda_cost,
                threshold_metric,
                targets[val_idx] if threshold_metric == "action_accuracy" else None,
            )
            threshold_train_pred = np.where(
                train_probabilities >= threshold,
                direct_index,
                1 - direct_index,
            ).astype("int64")
            train_action_pred = threshold_train_pred
            val_action_pred = threshold_val_pred
            train_answer_accuracy = _chosen_answer_accuracy(train_rows, threshold_train_pred, actions)
            val_answer_accuracy = _chosen_answer_accuracy(val_rows, threshold_val_pred, actions) if val_rows else train_answer_accuracy
            train_reward = float(np.mean(_chosen_rewards(train_rows, threshold_train_pred, actions, lambda_cost)))
        else:
            threshold = 0.5
            threshold_train_pred = train_pred
            threshold_val_pred = val_pred
            train_action_pred = train_pred
            val_action_pred = val_pred
            train_answer_accuracy = _chosen_answer_accuracy(train_rows, train_pred, actions)
            val_answer_accuracy = _chosen_answer_accuracy(val_rows, val_pred, actions) if val_rows else train_answer_accuracy
            train_reward = float(np.mean(_chosen_rewards(train_rows, train_pred, actions, lambda_cost)))
            val_reward = float(np.mean(_chosen_rewards(val_rows, val_pred, actions, lambda_cost))) if val_rows else train_reward
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_accuracy"].append(_accuracy(train_action_pred, targets[train_idx]))
        history["val_accuracy"].append(
            _accuracy(val_action_pred, targets[val_idx])
            if len(val_idx)
            else history["train_accuracy"][-1]
        )
        history["train_answer_accuracy"].append(train_answer_accuracy)
        history["val_answer_accuracy"].append(val_answer_accuracy)
        history["train_reward"].append(train_reward)
        history["val_reward"].append(val_reward)
        print(
            f"[transformer-policy] epoch={epoch:03d}/{epochs} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"train_acc={history['train_accuracy'][-1]:.4f} "
            f"val_acc={history['val_accuracy'][-1]:.4f} "
            f"val_answer_acc={val_answer_accuracy:.4f} "
            f"threshold={threshold:.4f} "
            f"train_reward={train_reward:.4f} val_reward={val_reward:.4f}",
            flush=True,
        )
        if save_epoch_plots:
            _plot_history(history, str(Path(epoch_plot_dir) / f"epoch_{epoch:04d}.png"))
        current_score = history[selection_metric][-1]
        is_better = current_score < best_score if selection_metric == "val_loss" else current_score > best_score
        if is_better:
            best_score = current_score
            best_epoch = epoch
            best_threshold = threshold
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
    _plot_history(history, curve_output)
    _save_history(history, history_output)
    model = model.to("cpu").eval()
    bundle = {
        "policy_type": "torch_embedding_mlp",
        "backend": "hf_hidden_torch_mlp",
        "model": model,
        "encoder_info": encoder_info,
        "encoder_name": encoder_name,
        "scaler": scaler,
        "action_labels": actions,
        "label_to_id": label_to_id,
        "layers": list(layers),
        "pooling": pooling,
        "pure_text": pure_text,
        "max_length": int(max_length),
        "encoder_batch_size": int(encoder_batch_size),
        "device": device,
        "max_text_chars": int(max_text_chars),
        "hidden_sizes": list(hidden_sizes),
        "epochs": int(epochs),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "lambda_cost": float(lambda_cost),
        "curve_output": curve_output,
        "history_output": history_output,
        "epoch_plot_dir": epoch_plot_dir,
        "history": history,
        "selection_metric": selection_metric,
        "threshold_metric": threshold_metric,
        "direct_threshold": float(best_threshold),
        "best_epoch": int(best_epoch),
        "best_score": float(best_score),
        "train_accuracy": history["train_accuracy"][best_epoch - 1],
        "val_accuracy": history["val_accuracy"][best_epoch - 1],
        "n_train": len(train_idx),
        "n_val": len(val_idx),
    }
    save_pickle(bundle, output_path)
    return bundle


def predict_transformer_actions(
    bundle: Dict[str, Any],
    rows: Sequence[Dict[str, Any]],
) -> List[str]:
    if not rows:
        return []
    texts = [
        build_policy_text(
            row,
            max_text_chars=int(bundle.get("max_text_chars", 8000)),
            include_state=not bool(bundle.get("pure_text", True)),
        )
        for row in rows
    ]
    features, _ = _encode_hf_hidden(
        texts,
        encoder_name=bundle["encoder_name"],
        max_length=int(bundle.get("max_length", 1024)),
        batch_size=int(bundle.get("encoder_batch_size", 8)),
        device=bundle.get("device", "auto"),
        pooling=bundle.get("pooling", "mean"),
        layer_indices=list(bundle.get("layers", [-1])),
    )
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    features = bundle["scaler"].transform(features).astype("float32")
    with torch.no_grad():
        logits = bundle["model"](torch.from_numpy(features))
    actions = list(bundle["action_labels"])
    if len(actions) == 2 and "direct" in actions and "direct_threshold" in bundle:
        probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
        direct_index = actions.index("direct")
        reflect_index = next(index for index, action in enumerate(actions) if action != "direct")
        threshold = float(bundle["direct_threshold"])
        indices = np.where(
            probabilities[:, direct_index] >= threshold,
            direct_index,
            reflect_index,
        )
    else:
        indices = logits.argmax(dim=-1).cpu().numpy()
    return [actions[int(index)] for index in indices]


def predict_transformer_action(bundle: Dict[str, Any], row: Dict[str, Any]) -> str:
    return predict_transformer_actions(bundle, [row])[0]

def main() -> None:
    parser = argparse.ArgumentParser(description="Train a PyTorch policy head on several frozen Transformer hidden layers.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--encoder_name", required=True)
    parser.add_argument("--layers", default="-4,-3,-2,-1")
    parser.add_argument("--pooling", choices=["mean", "last", "mean_last"], default="mean")
    parser.add_argument("--pure_text", action="store_true")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_text_chars", type=int, default=8000)
    parser.add_argument("--encoder_batch_size", type=int, default=8)
    parser.add_argument("--train_batch_size", type=int, default=128)
    parser.add_argument("--hidden_sizes", default="128,32")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lambda_cost", type=float, default=0.0001)
    parser.add_argument("--no_class_balance", action="store_true")
    parser.add_argument("--curve_output", required=True)
    parser.add_argument("--history_output", required=True)
    parser.add_argument("--epoch_plot_dir", required=True)
    parser.add_argument("--no_epoch_plots", action="store_true")
    parser.add_argument("--selection_metric", choices=["val_loss", "val_accuracy", "val_reward", "val_answer_accuracy"], default="val_answer_accuracy")
    parser.add_argument(
        "--threshold_metric",
        choices=["action_accuracy", "answer_accuracy", "reward"],
        default="answer_accuracy",
    )
    args = parser.parse_args()
    layers = _parse_layer_indices(args.layers)
    rows = read_jsonl(args.dataset)
    bundle = train_transformer_policy(
        rows=rows,
        encoder_name=args.encoder_name,
        output_path=args.output,
        layers=layers,
        pooling=args.pooling,
        pure_text=args.pure_text,
        max_length=args.max_length,
        max_text_chars=args.max_text_chars,
        encoder_batch_size=args.encoder_batch_size,
        train_batch_size=args.train_batch_size,
        hidden_sizes=tuple(int(x.strip()) for x in args.hidden_sizes.split(",") if x.strip()),
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        test_size=args.test_size,
        random_state=args.random_state,
        device=args.device,
        lambda_cost=args.lambda_cost,
        balance_classes=not args.no_class_balance,
        curve_output=args.curve_output,
        history_output=args.history_output,
        epoch_plot_dir=args.epoch_plot_dir,
        save_epoch_plots=not args.no_epoch_plots,
        selection_metric=args.selection_metric,
        threshold_metric=args.threshold_metric,
    )
    print(f"Saved transformer policy model to {args.output}")
    print(f"Layers: {bundle['layers']} embedding_dim={bundle['encoder_info']['embedding_dim']}")
    print(f"Train accuracy: {bundle['train_accuracy']:.4f}")
    print(f"Validation accuracy: {bundle['val_accuracy']:.4f}")
    print(f"Curve: {bundle['curve_output']}")
    print(f"History: {bundle['history_output']}")
    print(f"Best epoch: {bundle['best_epoch']} ({bundle['selection_metric']}={bundle['best_score']:.4f})")
    print(f"Direct threshold: {bundle['direct_threshold']:.4f} ({bundle['threshold_metric']})")


if __name__ == "__main__":
    main()
