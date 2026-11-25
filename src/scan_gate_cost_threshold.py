from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from src.neural_policy import _encode_hf_hidden, build_policy_text
from src.shared_hierarchical_policy import _targets
from src.transformer_policy import TorchPolicyMLP  # noqa: F401
from src.utils import load_pickle, read_jsonl, save_pickle


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan gate thresholds on the validation split for lower token cost.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--gate_model", required=True)
    parser.add_argument("--round_model", required=True)
    parser.add_argument("--output_gate_model", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--lambda_cost", type=float, default=0.0001)
    parser.add_argument("--objective", choices=["accuracy", "reward"], default="accuracy")
    args = parser.parse_args()

    rows = read_jsonl(args.dataset)
    gate_bundle = load_pickle(args.gate_model)
    round_bundle = load_pickle(args.round_model)
    gate_targets, _ = _targets(rows)
    indices = np.arange(len(rows))
    _, val_idx = train_test_split(indices, test_size=args.test_size, random_state=args.random_state, stratify=gate_targets)
    val_rows = [rows[i] for i in val_idx]
    texts = [build_policy_text(row, include_state=False) for row in val_rows]

    gate_features, _ = _encode_hf_hidden(texts, encoder_name=gate_bundle["encoder_name"], max_length=int(gate_bundle["max_length"]), batch_size=8, device=args.device, pooling=gate_bundle["pooling"], layer_indices=list(gate_bundle["layers"]))
    gate_features = gate_bundle["scaler"].transform(np.nan_to_num(gate_features, nan=0.0, posinf=0.0, neginf=0.0)).astype("float32")
    with torch.no_grad():
        gate_logits = gate_bundle["model"](torch.from_numpy(gate_features))
        direct_prob = torch.softmax(gate_logits, dim=-1)[:, 0].cpu().numpy()

    round_features, _ = _encode_hf_hidden(texts, encoder_name=round_bundle["encoder_name"], max_length=int(round_bundle["max_length"]), batch_size=8, device=args.device, pooling=round_bundle["pooling"], layer_indices=list(round_bundle["layers"]))
    round_features = round_bundle["scaler"].transform(np.nan_to_num(round_features, nan=0.0, posinf=0.0, neginf=0.0)).astype("float32")
    with torch.no_grad():
        round_pred = round_bundle["model"](torch.from_numpy(round_features)).argmax(dim=-1).cpu().numpy()

    records = []
    thresholds = np.unique(np.concatenate([np.linspace(0.0, 1.0, 201), direct_prob]))
    for threshold in thresholds:
        correct = 0
        total_tokens = 0.0
        reflected = 0
        for row, probability, round_index in zip(val_rows, direct_prob, round_pred):
            gate_direct = probability >= threshold
            action = "direct" if gate_direct else ("reflect_1" if int(round_index) == 0 else "reflect_2")
            item = next(candidate for candidate in row["candidate_results"] if candidate["action"] == action)
            correct += int(float(item.get("correct", 0)))
            total_tokens += float(item.get("token_cost", 0))
            reflected += int(not gate_direct)
        n = len(val_rows)
        accuracy = correct / n
        avg_token_cost = total_tokens / n
        records.append({
            "threshold": float(threshold),
            "accuracy": accuracy,
            "avg_token_cost": avg_token_cost,
            "reflection_rate": reflected / n,
            "reward": accuracy - args.lambda_cost * avg_token_cost,
        })

    if args.objective == "reward":
        best = max(records, key=lambda item: (item["reward"], item["accuracy"], -item["avg_token_cost"]))
        candidates = [item for item in records if item["reward"] == best["reward"]]
        print(f"Validation best reward: {best['reward']:.4f} lambda={args.lambda_cost:g}")
    else:
        best_accuracy = max(item["accuracy"] for item in records)
        candidates = [item for item in records if item["accuracy"] >= best_accuracy]
        best = min(candidates, key=lambda item: (item["avg_token_cost"], -item["threshold"]))
        print(f"Validation best accuracy: {best_accuracy:.4f}")
    print(f"Selected threshold: {best['threshold']:.6f} accuracy={best['accuracy']:.4f} avg_token_cost={best['avg_token_cost']:.1f} reflection_rate={best['reflection_rate']:.4f} reward={best['reward']:.4f}")
    print("Best validation candidates:")
    sort_key = (lambda x: (-x["reward"], -x["accuracy"], x["avg_token_cost"])) if args.objective == "reward" else (lambda x: (x["avg_token_cost"], -x["threshold"]))
    for item in sorted(candidates, key=sort_key)[:10]:
        print(item)
    updated = copy.deepcopy(gate_bundle)
    updated["direct_threshold"] = float(best["threshold"])
    updated["threshold_source"] = f"validation_{args.objective}_scan"
    updated["threshold_lambda_cost"] = float(args.lambda_cost)
    save_pickle(updated, args.output_gate_model)
    print(f"Saved calibrated gate model to {args.output_gate_model}")


if __name__ == "__main__":
    main()
