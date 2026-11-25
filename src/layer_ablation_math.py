"""Offline last-1 through last-6 hidden-layer ablation for the hierarchical math policy.

The script reuses frozen candidate traces. It never calls the Base model or the
Reflector. Hidden states are extracted once from the small policy encoder and
stored as resumable shards; Gate and Round heads are retrained independently
for each layer set using the same locked train/validation split.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from .neural_policy import _ensure_torchvision_nms_compat, build_policy_text
from .train_fixed_math_policy import ACTIONS, audit_split, best_action, decision_metrics, digest, write_json
from .transformer_policy import TorchPolicyMLP
from .utils import read_jsonl


DEFAULT_ENCODER = "Qwen/Qwen3-0.6B"
ENCODE_LAYERS = [-6, -5, -4, -3, -2, -1]
LAYER_SETS = {
    "last1": [-1],
    "last2": [-2, -1],
    "last3": [-3, -2, -1],
    "last4": [-4, -3, -2, -1],
    "last5": [-5, -4, -3, -2, -1],
    "last6": [-6, -5, -4, -3, -2, -1],
}


def _hash_texts(rows: list[dict[str, Any]]) -> list[str]:
    return [hashlib.sha256(build_policy_text(row, max_text_chars=0, include_state=False).encode()).hexdigest()
            for row in rows]


def _load_rows(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(str(path))


def _encode_sharded(
    rows: list[dict[str, Any]],
    encoder_name: str,
    output: Path,
    max_length: int,
    batch_size: int,
    device: str,
    shard_size: int = 32,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Encode all rows with the last six hidden layers, resuming complete .npy shards."""
    _ensure_torchvision_nms_compat(torch)
    from transformers import AutoModel, AutoTokenizer
    import transformers

    texts = [build_policy_text(row, max_text_chars=0, include_state=False) for row in rows]
    resolved = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if resolved.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {resolved}, but CUDA is unavailable")
    encoder = Path(encoder_name).resolve()
    if not encoder.is_dir():
        raise FileNotFoundError(f"Missing encoder directory: {encoder}")

    tokenizer = AutoTokenizer.from_pretrained(str(encoder), local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    lengths = [len(tokenizer(text, truncation=False)["input_ids"]) for text in texts]
    truncated_ids = [row["id"] for row, length in zip(rows, lengths) if length > max_length]
    if truncated_ids:
        raise RuntimeError(
            f"{len(truncated_ids)} policy inputs exceed max_length={max_length}; "
            "increase MAX_LENGTH instead of silently truncating. First IDs: "
            + ", ".join(truncated_ids[:5])
        )

    signature = {
        "version": 2,
        "encoder": str(encoder),
        "encoder_config_sha256": digest(encoder / "config.json"),
        "tokenizer_sha256": digest(encoder / "tokenizer.json"),
        "layer_indices": ENCODE_LAYERS,
        "pooling": "mean_last",
        "hidden_size": 1024,
        "max_length": max_length,
        "batch_size": batch_size,
        "ids": [row["id"] for row in rows],
        "text_sha256": _hash_texts(rows),
        "transformers": transformers.__version__,
    }
    fingerprint = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    cache = output / "features" / fingerprint[:20]
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "manifest.json").write_text(json.dumps(signature, indent=2, ensure_ascii=False) + "\n")

    model = None
    parts: list[np.ndarray] = []
    started = time.monotonic()
    for start in range(0, len(rows), shard_size):
        end = min(start + shard_size, len(rows))
        shard = cache / f"{start:06d}_{end:06d}.npy"
        expected_shape = (end - start, 2 * len(ENCODE_LAYERS) * 1024)
        values = None
        if shard.exists():
            candidate = np.load(shard, allow_pickle=False)
            if candidate.shape == expected_shape and np.isfinite(candidate).all():
                values = candidate.astype("float32", copy=False)
        if values is None:
            if model is None:
                dtype = torch.bfloat16 if resolved.startswith("cuda") and torch.cuda.is_bf16_supported() else torch.float32
                model = AutoModel.from_pretrained(
                    str(encoder), local_files_only=True, dtype=dtype, trust_remote_code=True
                )
                model.to(resolved).eval().requires_grad_(False)
            encoded: list[np.ndarray] = []
            for batch_start in range(start, end, max(1, batch_size)):
                batch_end = min(batch_start + max(1, batch_size), end)
                batch = tokenizer(
                    texts[batch_start:batch_end], padding=True, truncation=True,
                    max_length=max_length, return_tensors="pt"
                )
                batch = {key: value.to(resolved) for key, value in batch.items()}
                with torch.inference_mode():
                    hidden_states = model(**batch, output_hidden_states=True, use_cache=False).hidden_states
                    mask = batch["attention_mask"]
                    mask_float = mask.unsqueeze(-1).to(hidden_states[-1].dtype)
                    lengths_tensor = mask.sum(1).clamp_min(1) - 1
                    pieces = []
                    for layer in ENCODE_LAYERS:
                        hidden = hidden_states[layer]
                        mean = (hidden * mask_float).sum(1) / mask_float.sum(1).clamp_min(1.0)
                        last = hidden[torch.arange(hidden.shape[0], device=hidden.device), lengths_tensor]
                        pieces.extend([mean, last])
                    encoded.append(torch.cat(pieces, dim=1).float().cpu().numpy())
            values = np.concatenate(encoded).astype("float32")
            if values.shape != expected_shape or not np.isfinite(values).all():
                raise ValueError(f"Invalid encoder output for rows {start}:{end}: {values.shape}")
            temporary = shard.with_suffix(".tmp")
            with temporary.open("wb") as stream:
                np.save(stream, values, allow_pickle=False)
            temporary.replace(shard)
        parts.append(values)
        elapsed = time.monotonic() - started
        print(f"[layers] encoded {end}/{len(rows)} elapsed={elapsed:.0f}s", flush=True)

    features = np.concatenate(parts, axis=0)
    info = {
        "cache": str(cache), "fingerprint": fingerprint, "encoder_name": str(encoder),
        "layer_indices": ENCODE_LAYERS, "pooling": "mean_last", "embedding_dim": 2 * len(ENCODE_LAYERS) * 1024,
        "max_length": max_length, "device": resolved, "encoder_clipped_count": len(truncated_ids),
        "rows": len(rows),
    }
    (cache / "complete.json").write_text(json.dumps(info, indent=2) + "\n")
    return features, info


def _slice_layers(features: np.ndarray, name: str, hidden_size: int = 1024) -> np.ndarray:
    selected = LAYER_SETS[name]
    all_layers = ENCODE_LAYERS
    positions = [all_layers.index(layer) for layer in selected]
    columns = []
    # Each layer contributes [masked_mean, last_valid_token].
    for position in positions:
        columns.extend([slice(position * 2 * hidden_size, (position * 2 + 1) * hidden_size),
                        slice((position * 2 + 1) * hidden_size, (position * 2 + 2) * hidden_size)])
    return np.concatenate([features[:, column] for column in columns], axis=1).astype("float32")


def _mlp(input_dim: int, hidden: tuple[int, ...], dropout: float) -> torch.nn.Module:
    layers: list[torch.nn.Module] = []
    previous = input_dim
    for width in hidden:
        layers.extend([torch.nn.Linear(previous, width), torch.nn.ReLU()])
        if dropout:
            layers.append(torch.nn.Dropout(dropout))
        previous = width
    layers.append(torch.nn.Linear(previous, 2))
    return torch.nn.Sequential(*layers)


def _train_epoch(model, optimizer, criterion, x, y, batch_size, rng):
    model.train()
    order = rng.permutation(len(y))
    for start in range(0, len(order), batch_size):
        index = order[start:start + batch_size]
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x[index]), y[index])
        loss.backward()
        optimizer.step()


def _predict(model, x) -> np.ndarray:
    model.eval()
    with torch.inference_mode():
        return model(x).softmax(1).cpu().numpy()


def _round_actions(probabilities: np.ndarray) -> list[str]:
    return ["reflect_1" if int(index) == 0 else "reflect_2" for index in probabilities.argmax(1)]


def _select_threshold(rows, gate_prob_direct, round_actions, lambda_cost):
    choices = []
    for threshold in [*np.linspace(0.0, 1.0, 21), 1.000001]:
        actions = ["direct" if p >= threshold else action for p, action in zip(gate_prob_direct, round_actions)]
        metrics = decision_metrics(rows, actions, lambda_cost)
        choices.append((float(threshold), metrics, actions))
    return max(choices, key=lambda item: (item[1]["correct"], -item[1]["avg_token_cost"], -item[0]))


def _train_layer(name, train, val, test, train_features, val_features, test_features, output, args):
    input_dim = train_features.shape[1]
    gate_targets = np.asarray([int(str(row["best_action"]) != "direct") for row in train], dtype="int64")
    round_indices = np.asarray([i for i, row in enumerate(train) if set(c["action"] for c in row["candidate_results"]) == set(ACTIONS)], dtype="int64")
    round_targets = np.asarray([int(best_action(train[i], ACTIONS[1:]) == "reflect_2") for i in round_indices], dtype="int64")
    val_gate_targets = np.asarray([int(str(row["best_action"]) != "direct") for row in val], dtype="int64")
    val_round_targets = np.asarray([int(best_action(row, ACTIONS[1:]) == "reflect_2") for row in val], dtype="int64")
    if len(np.unique(gate_targets)) < 2 or len(np.unique(round_targets)) < 2:
        raise ValueError(f"{name}: both classes are required for Gate and Round")

    gate_scaler = StandardScaler().fit(train_features)
    round_scaler = StandardScaler().fit(train_features[round_indices])
    gate_x = torch.from_numpy(gate_scaler.transform(train_features).astype("float32"))
    gate_vx = torch.from_numpy(gate_scaler.transform(val_features).astype("float32"))
    gate_test_x = torch.from_numpy(gate_scaler.transform(test_features).astype("float32"))
    round_x = torch.from_numpy(round_scaler.transform(train_features[round_indices]).astype("float32"))
    round_vx = torch.from_numpy(round_scaler.transform(val_features).astype("float32"))
    round_test_x = torch.from_numpy(round_scaler.transform(test_features).astype("float32"))
    gate_y = torch.from_numpy(gate_targets)
    gate_vy = torch.from_numpy(val_gate_targets)
    round_y = torch.from_numpy(round_targets)
    round_vy = torch.from_numpy(val_round_targets)
    if not all(torch.isfinite(x).all() for x in (gate_x, gate_vx, gate_test_x, round_x, round_vx, round_test_x)):
        raise ValueError(f"{name}: non-finite standardized features")

    gate = _mlp(input_dim, (16,), 0.2)
    round_model = _mlp(input_dim, (64, 16), 0.0)
    gate_optimizer = torch.optim.AdamW(gate.parameters(), lr=3e-4, weight_decay=0.01)
    round_optimizer = torch.optim.AdamW(round_model.parameters(), lr=1e-3, weight_decay=1e-3)
    gate_criterion = torch.nn.CrossEntropyLoss()
    round_counts = np.bincount(round_targets, minlength=2).astype("float32")
    round_weights = torch.tensor(len(round_targets) / (2.0 * round_counts), dtype=torch.float32)
    round_criterion = torch.nn.CrossEntropyLoss(weight=round_weights)
    rng = np.random.default_rng(args.seed)
    history = []
    best = None
    output.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        _train_epoch(gate, gate_optimizer, gate_criterion, gate_x, gate_y, args.batch_size, rng)
        _train_epoch(round_model, round_optimizer, round_criterion, round_x, round_y, args.batch_size, rng)
        gate_val = _predict(gate, gate_vx)
        round_val = _predict(round_model, round_vx)
        threshold, val_metrics, val_actions = _select_threshold(val, gate_val[:, 0], _round_actions(round_val), args.lambda_cost)
        with torch.inference_mode():
            gate_train_loss = float(torch.nn.functional.cross_entropy(gate(gate_x), gate_y))
            gate_val_loss = float(torch.nn.functional.cross_entropy(gate(gate_vx), gate_vy))
            round_train_loss = float(torch.nn.functional.cross_entropy(round_model(round_x), round_y))
            round_val_loss = float(torch.nn.functional.cross_entropy(round_model(round_vx), round_vy))
        entry = {
            "epoch": epoch,
            "gate_train_loss": gate_train_loss, "gate_val_loss": gate_val_loss,
            "round_train_loss": round_train_loss, "round_val_loss": round_val_loss,
            "threshold": threshold, "validation": val_metrics,
        }
        history.append(entry)
        key = (val_metrics["correct"], -val_metrics["avg_token_cost"], -val_metrics["reflection_rate"])
        if best is None or key > best["key"]:
            best = {"key": key, "epoch": epoch, "threshold": threshold,
                    "gate_state": copy.deepcopy(gate.state_dict()),
                    "round_state": copy.deepcopy(round_model.state_dict()),
                    "validation": val_metrics}
        print(f"[layer-ablation] {name} epoch={epoch}/{args.epochs} "
              f"val_acc={val_metrics['accuracy']:.4f} val_tokens={val_metrics['avg_token_cost']:.1f} "
              f"reflect={val_metrics['reflection_rate']:.3f} threshold={threshold:.4f}", flush=True)

    assert best is not None
    gate.load_state_dict(best["gate_state"])
    round_model.load_state_dict(best["round_state"])
    test_gate = _predict(gate, gate_test_x)
    test_round = _predict(round_model, round_test_x)
    test_actions = ["direct" if p >= best["threshold"] else action
                    for p, action in zip(test_gate[:, 0], _round_actions(test_round))]
    test_metrics = decision_metrics(test, test_actions, args.lambda_cost)
    torch.save({"gate": gate.state_dict(), "round": round_model.state_dict(),
                "gate_scaler_mean": gate_scaler.mean_, "gate_scaler_scale": gate_scaler.scale_,
                "round_scaler_mean": round_scaler.mean_, "round_scaler_scale": round_scaler.scale_,
                "layer_set": LAYER_SETS[name], "best_epoch": best["epoch"],
                "threshold": best["threshold"]}, output / "checkpoint.pt")
    (output / "history.json").write_text(json.dumps(history, indent=2, ensure_ascii=False) + "\n")
    return {
        "layer_set": LAYER_SETS[name], "embedding_dim": input_dim,
        "best_epoch": best["epoch"], "threshold": best["threshold"],
        "validation": best["validation"], "test": test_metrics,
        "train_counts": {"gate": dict(Counter(gate_targets.tolist())),
                          "round": dict(Counter(round_targets.tolist()))},
        "checkpoint": str(output / "checkpoint.pt"),
    }


def _baseline(rows, action, lambda_cost):
    return decision_metrics(rows, [action] * len(rows), lambda_cost)


def _write_report(path: Path, results: dict[str, Any], val_baselines: dict[str, Any], test_baselines: dict[str, Any]):
    lines = ["# GSM8K policy encoder layer ablation", "",
             "This is an offline representation ablation. All candidate traces, prompts, answers, "
             "token accounting, and train/validation split are reused; no Base or Reflector calls are made.", "",
             "## Protocol", "",
             "- Gate input: question + initial answer; Round input: the same state.",
             "- Each layer contributes masked-mean and last-valid-token pooling.",
             "- The three variants use the same optimizer, class weighting, epoch selection, and threshold grid.",
             "- Epoch and threshold are selected only on the fixed validation split; the test set is evaluated once after locking them.", "",
             "## Validation baselines", "", "| Method | Accuracy | Total tokens | Reflection rate |", "| --- | ---: | ---: | ---: |"]
    for name, metrics in val_baselines.items():
        lines.append(f"| {name} | {metrics['accuracy']:.4f} | {metrics['avg_token_cost']:.2f} | {metrics['reflection_rate']:.4f} |")
    lines += ["", "## Test baselines", "", "| Method | Accuracy | Total tokens | Reflection rate |", "| --- | ---: | ---: | ---: |"]
    for name, metrics in test_baselines.items():
        lines.append(f"| {name} | {metrics['accuracy']:.4f} | {metrics['avg_token_cost']:.2f} | {metrics['reflection_rate']:.4f} |")
    lines += ["", "## Layer policies", "", "| Layer set | Dim. | Epoch | Val accuracy | Val tokens | Test accuracy | Test tokens | Test reflection | Fix/Harm |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, result in results.items():
        val, test = result["validation"], result["test"]
        lines.append(f"| {name} | {result['embedding_dim']} | {result['best_epoch']} | {val['accuracy']:.4f} | "
                     f"{val['avg_token_cost']:.2f} | {test['accuracy']:.4f} | {test['avg_token_cost']:.2f} | "
                     f"{test['reflection_rate']:.4f} | {test['fixes']}/{test['harms']} |")
    lines += ["", "## Interpretation boundary", "",
              "The layer with the highest test accuracy must not be selected after inspecting this table. "
              "For a final paper claim, select the representation from validation performance and report this test table as a locked ablation.",
              "The experiment changes only the frozen encoder representation; it does not compare different Base/Reflector generations.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/math_full_training_v1"))
    parser.add_argument("--test-trace", type=Path, default=Path("outputs/traces/gsm8k_test_policy_trace_detailed.jsonl"))
    parser.add_argument("--encoder", default=DEFAULT_ENCODER)
    parser.add_argument("--output", type=Path, default=Path("outputs/layer_ablation_gsm8k_v1"))
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--encoder-batch-size", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lambda-cost", type=float, default=0.0001)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.epochs < 1 or args.encoder_batch_size < 1 or args.batch_size < 1:
        parser.error("epochs and batch sizes must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    train, val, audit = audit_split(args.root, allow_partial=True)
    test = _load_rows(args.test_trace)
    if not test or any("candidate_results" not in row for row in test):
        raise ValueError("Test trace is empty or lacks candidate_results")
    all_rows = train + val + test
    features, encoder_info = _encode_sharded(all_rows, args.encoder, args.output, args.max_length,
                                             args.encoder_batch_size, args.device)
    n_train, n_val = len(train), len(val)
    train_features = features[:n_train]
    val_features = features[n_train:n_train + n_val]
    test_features = features[n_train + n_val:]
    val_baselines = {name: _baseline(val, action, args.lambda_cost) for name, action in
                     (("direct", "direct"), ("fixed_reflect_1", "reflect_1"),
                      ("fixed_reflect_2", "reflect_2"), ("independent", "independent"))}
    test_baselines = {name: _baseline(test, action, args.lambda_cost) for name, action in
                      (("direct", "direct"), ("fixed_reflect_1", "reflect_1"),
                       ("fixed_reflect_2", "reflect_2"))}
    results = {}
    for name in LAYER_SETS:
        current_train = _slice_layers(train_features, name)
        current_val = _slice_layers(val_features, name)
        current_test = _slice_layers(test_features, name)
        results[name] = _train_layer(name, train, val, test, current_train, current_val, current_test,
                                     args.output / name, args)
    payload = {"protocol": vars(args) | {"root": str(args.root), "test_trace": str(args.test_trace), "output": str(args.output)},
               "encoder": encoder_info, "coverage": audit, "validation_baselines": val_baselines,
               "test_baselines": test_baselines, "results": results, "test_evaluated": True}
    write_json(args.output / "results.json", payload)
    _write_report(args.output / "REPORT.md", results, val_baselines, test_baselines)
    print(f"[layer-ablation] completed: {args.output}", flush=True)


if __name__ == "__main__":
    main()
