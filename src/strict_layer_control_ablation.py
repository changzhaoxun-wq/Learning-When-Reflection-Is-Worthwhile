"""Strict Last-1--Last-6 control ablation on frozen GSM8K traces.

This runner uses one common training protocol for every representation layer.
Only the layer set, Gate loss (weighted CE versus weighted CE+Rank), and
deployment branch (Gate-Round versus Gate-R1) are varied. No LLM is called.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from .control_ablation_last3 import (
    BUDGETS,
    RANKING_COEFFICIENT,
    _baseline,
    _predict,
    _round_actions,
    _train_gate_epoch,
    _train_round,
    _thresholds,
)
from .layer_ablation_math import ENCODE_LAYERS, LAYER_SETS, _mlp, _slice_layers
from .neural_policy import build_policy_text
from .train_fixed_math_policy import (
    ACTIONS, audit_split, best_action, candidates, decision_metrics, write_json
)
from .utils import read_jsonl


SEED_LAYERS = tuple(LAYER_SETS)
TRIALS = ("weighted_ce", "weighted_ce_plus_rank")
BRANCHES = ("gate_round", "gate_r1")


def _load_six_layer_cache(results_path: Path, rows: list[dict]) -> tuple[np.ndarray, dict]:
    results = json.loads(results_path.read_text())
    cache = Path(results["encoder"]["cache"])
    manifest = json.loads((cache / "manifest.json").read_text())
    expected_ids = [row["id"] for row in rows]
    if manifest["ids"] != expected_ids:
        raise ValueError("Feature cache IDs do not match the locked row order")
    if manifest["layer_indices"] != ENCODE_LAYERS:
        raise ValueError("Expected the six-layer feature cache")
    if manifest["pooling"] != "mean_last" or manifest["hidden_size"] != 1024:
        raise ValueError("Unexpected feature cache pooling or hidden size")
    expected_hashes = [
        hashlib.sha256(
            build_policy_text(row, max_text_chars=0, include_state=False).encode()
        ).hexdigest()
        for row in rows
    ]
    if manifest["text_sha256"] != expected_hashes:
        raise ValueError("Feature cache text fingerprint differs from current policy input")

    parts = []
    next_start = 0
    for shard in sorted(cache.glob("*.npy")):
        start, end = map(int, shard.stem.split("_"))
        values = np.load(shard, allow_pickle=False)
        expected_shape = (end - start, 2 * len(ENCODE_LAYERS) * 1024)
        if start != next_start or values.shape != expected_shape:
            raise ValueError(f"Invalid feature shard: {shard}")
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite feature shard: {shard}")
        parts.append(values.astype("float32", copy=False))
        next_start = end
    if next_start != len(rows):
        raise ValueError("Feature cache does not cover all rows")
    return np.concatenate(parts, axis=0), {
        "cache": str(cache),
        "manifest_sha256": hashlib.sha256((cache / "manifest.json").read_bytes()).hexdigest(),
        "layer_indices": ENCODE_LAYERS,
        "pooling": manifest["pooling"],
        "hidden_size": manifest["hidden_size"],
    }


def _make_orders(count: int, batch_size: int, epochs: int, seed: int, device):
    rng = np.random.default_rng(seed)
    batches = max(1, int(np.ceil(count / batch_size)))
    return [
        [
            torch.from_numpy(order.astype(np.int64)).to(device)
            for order in np.array_split(rng.permutation(count), batches)
        ]
        for _ in range(epochs)
    ]


def _action_counts(actions):
    values, counts = np.unique(actions, return_counts=True)
    return {str(value): int(count) for value, count in zip(values, counts)}


def _metric_arrays(rows, round_actions):
    direct, reflected, base = [], [], []
    for row, round_action in zip(rows, round_actions, strict=True):
        cs = candidates(row)
        base.append(cs["direct"]["correct"])
        direct.append(cs["direct"])
        reflected.append(cs[round_action])
    keys = ("correct", "prompt_tokens", "completion_tokens", "token_cost")
    return (
        np.asarray(base, dtype=np.int64),
        {key: np.asarray([item[key] for item in direct], dtype=np.float64) for key in keys},
        {key: np.asarray([item[key] for item in reflected], dtype=np.float64) for key in keys},
    )


def _vectorized_metrics(base, direct, reflected, use_reflection, round_actions, lambda_cost):
    correct = np.where(use_reflection, reflected["correct"], direct["correct"])
    prompt = np.where(use_reflection, reflected["prompt_tokens"], direct["prompt_tokens"])
    completion = np.where(use_reflection, reflected["completion_tokens"], direct["completion_tokens"])
    total = np.where(use_reflection, reflected["token_cost"], direct["token_cost"])
    base_correct = base.astype(np.int64)
    correct_int = correct.astype(np.int64)
    fixes = int(((base_correct == 0) & (correct_int == 1)).sum())
    harms = int(((base_correct == 1) & (correct_int == 0)).sum())
    round_actions_array = np.asarray(round_actions)
    actions = {"direct": int((~use_reflection).sum())}
    for action in ("reflect_1", "reflect_2"):
        if action in set(round_actions):
            actions[action] = int((use_reflection & (round_actions_array == action)).sum())
    return {
        "n": int(len(correct)), "correct": int(correct_int.sum()),
        "accuracy": float(correct_int.mean()),
        "avg_prompt_tokens": float(prompt.mean()),
        "avg_completion_tokens": float(completion.mean()),
        "avg_token_cost": float(total.mean()),
        "avg_reward": float((correct_int - lambda_cost * total).mean()),
        "reflection_rate": float(use_reflection.mean()), "actions": actions,
        "fixes": fixes, "harms": harms, "net_fix": fixes - harms,
        "fix_rate": fixes / int((base_correct == 0).sum()) if (base_correct == 0).any() else None,
        "harm_rate": harms / int((base_correct == 1).sum()) if (base_correct == 1).any() else None,
    }


def _select_fast(rows, direct_prob, round_actions, budget, strong_cost, lambda_cost):
    base, direct, reflected = _metric_arrays(rows, round_actions)
    points = []
    for threshold in _thresholds(direct_prob):
        use_reflection = np.asarray(direct_prob) < threshold
        metrics = _vectorized_metrics(base, direct, reflected, use_reflection, round_actions, lambda_cost)
        if metrics["reflection_rate"] <= budget + 1e-12 and metrics["avg_token_cost"] < strong_cost:
            points.append({
                "threshold": float(threshold), "metrics": metrics,
                "actions": ["direct" if not use else action for use, action in zip(use_reflection, round_actions, strict=True)],
            })
    if not points:
        return None
    return max(points, key=lambda point: (
        point["metrics"]["correct"], -point["metrics"]["avg_token_cost"],
        -point["metrics"]["reflection_rate"], -point["threshold"]))


def _run_layer(
    layer_name,
    raw_features,
    train,
    validation,
    test,
    args,
    round_orders,
    gate_orders,
    baseline_data,
):
    features = _slice_layers(raw_features, layer_name)
    n_train, n_val = len(train), len(validation)
    train_features = features[:n_train]
    val_features = features[n_train : n_train + n_val]
    test_features = features[n_train + n_val :]

    gate_labels = torch.tensor(
        [int(row["best_action"] != "direct") for row in train], dtype=torch.long
    )
    round_indices = np.asarray(
        [
            i
            for i, row in enumerate(train)
            if {candidate["action"] for candidate in row["candidate_results"]}
            == set(ACTIONS)
        ],
        dtype=np.int64,
    )
    round_labels = torch.tensor(
        [int(best_action(train[i], ACTIONS[1:]) == "reflect_2") for i in round_indices],
        dtype=torch.long,
    )
    if len(torch.unique(gate_labels)) < 2 or len(torch.unique(round_labels)) < 2:
        raise ValueError(f"{layer_name}: both Gate and Round classes are required")

    gate_scaler = StandardScaler().fit(train_features)
    round_scaler = StandardScaler().fit(train_features[round_indices])
    device = torch.device(args.device)
    gate_x = torch.from_numpy(gate_scaler.transform(train_features).astype("float32")).to(device)
    gate_vx = torch.from_numpy(gate_scaler.transform(val_features).astype("float32")).to(device)
    gate_test_x = torch.from_numpy(gate_scaler.transform(test_features).astype("float32")).to(device)
    round_x = torch.from_numpy(
        round_scaler.transform(train_features[round_indices]).astype("float32")
    ).to(device)
    round_vx = torch.from_numpy(round_scaler.transform(val_features).astype("float32")).to(device)
    round_test_x = torch.from_numpy(
        round_scaler.transform(test_features).astype("float32")
    ).to(device)
    gate_weights = torch.tensor(
        [
            1.0,
            min(
                5.0,
                float(np.sqrt((gate_labels == 0).sum() / (gate_labels == 1).sum())),
            ),
        ],
        dtype=torch.float32,
    ).to(device)
    gate_labels = gate_labels.to(device)
    round_labels = round_labels.to(device)
    round_counts = torch.bincount(round_labels, minlength=2).float()
    round_weights = (len(round_labels) / (2.0 * round_counts)).to(device)

    # One common Round protocol is shared by both Gate-loss trials and branches.
    torch.manual_seed(args.seed + 10000)
    round_model = _mlp(features.shape[1], (64, 16), 0.0).to(device)
    round_optimizer = torch.optim.AdamW(
        round_model.parameters(), lr=1e-3, weight_decay=1e-3
    )
    for orders in round_orders:
        _train_round(round_model, round_optimizer, round_x, round_labels, round_weights, orders)
    round_val_actions = _round_actions(_predict(round_model, round_vx))
    round_test_actions = _round_actions(_predict(round_model, round_test_x))

    rows = []
    for trial_name, coefficient in (
        ("weighted_ce", 0.0),
        ("weighted_ce_plus_rank", RANKING_COEFFICIENT),
    ):
        # Identical initialization and minibatch orders isolate the loss term.
        torch.manual_seed(args.seed + 30000)
        template = _mlp(features.shape[1], (16,), 0.2).to(device)
        gate = _mlp(features.shape[1], (16,), 0.2).to(device)
        gate.load_state_dict(copy.deepcopy(template.state_dict()))
        optimizer = torch.optim.AdamW(gate.parameters(), lr=3e-4, weight_decay=1e-2)
        best = {}
        history = []
        for epoch, orders in enumerate(gate_orders, start=1):
            _train_gate_epoch(
                gate, optimizer, gate_x, gate_labels, gate_weights, orders, coefficient
            )
            val_prob = _predict(gate, gate_vx)[:, 0]
            selected = {}
            for branch, branch_actions in (
                ("gate_round", round_val_actions),
                ("gate_r1", ["reflect_1"] * len(validation)),
            ):
                for budget in BUDGETS:
                    point = _select_fast(
                        validation,
                        val_prob,
                        branch_actions,
                        budget,
                        baseline_data["validation"]["independent"]["avg_token_cost"],
                        args.lambda_cost,
                    )
                    key = (branch, round(budget, 2))
                    selected[f"{branch}_{int(budget * 100)}"] = point
                    if point is None:
                        continue
                    order_key = (
                        point["metrics"]["correct"],
                        -point["metrics"]["avg_token_cost"],
                        -point["metrics"]["reflection_rate"],
                        -point["threshold"],
                    )
                    if key not in best or order_key > best[key]["order"]:
                        best[key] = {
                            "trial": trial_name,
                            "branch": branch,
                            "budget": budget,
                            "epoch": epoch,
                            "threshold": point["threshold"],
                            "validation": point["metrics"],
                            "order": order_key,
                            "gate_state": copy.deepcopy(gate.state_dict()),
                        }
            history.append(
                {
                    "epoch": epoch,
                    "selected": {
                        key: (value["metrics"] if value is not None else None)
                        for key, value in selected.items()
                    },
                }
            )

        for _, chosen in best.items():
            branch = chosen["branch"]
            gate.load_state_dict(chosen["gate_state"])
            test_prob = _predict(gate, gate_test_x)[:, 0]
            branch_test_actions = (
                round_test_actions
                if branch == "gate_round"
                else ["reflect_1"] * len(test)
            )
            test_actions = [
                "direct" if p >= chosen["threshold"] else action
                for p, action in zip(test_prob, branch_test_actions)
            ]
            rows.append(
                {
                    "layer": layer_name,
                    "trial": trial_name,
                    "branch": branch,
                    "budget": chosen["budget"],
                    "epoch": chosen["epoch"],
                    "threshold": chosen["threshold"],
                    "validation": chosen["validation"],
                    "test": decision_metrics(test, test_actions, args.lambda_cost),
                    "history": history,
                }
            )
    return rows


def run(args):
    root = Path(args.root)
    train, validation, audit = audit_split(root, allow_partial=True)
    test = read_jsonl(str(args.test_trace))
    all_rows = train + validation + test
    raw_features, cache_info = _load_six_layer_cache(
        Path(args.feature_results), all_rows
    )
    baseline_data = {
        "validation": {
            name: _baseline(validation, action, args.lambda_cost)
            for name, action in (
                ("direct", "direct"),
                ("fixed_reflect_1", "reflect_1"),
                ("fixed_reflect_2", "reflect_2"),
                ("independent", "independent"),
            )
        },
        "test": {
            name: _baseline(test, action, args.lambda_cost)
            for name, action in (
                ("direct", "direct"),
                ("fixed_reflect_1", "reflect_1"),
                ("fixed_reflect_2", "reflect_2"),
            )
        },
    }
    if all(row.get("independent") is not None for row in test):
        baseline_data["test"]["independent"] = _baseline(
            test, "independent", args.lambda_cost
        )

    round_count = sum(
        {
            candidate["action"] for candidate in row["candidate_results"]
        }
        == set(ACTIONS)
        for row in train
    )
    round_orders = _make_orders(
        round_count, args.batch_size, args.epochs, args.seed + 20000, args.device
    )
    gate_orders = _make_orders(
        len(train), args.batch_size, args.epochs, args.seed + 40000, args.device
    )

    results = []
    for layer_name in SEED_LAYERS:
        print(f"[strict] seed={args.seed} layer={layer_name} start", flush=True)
        results.extend(
            _run_layer(
                layer_name,
                raw_features,
                train,
                validation,
                test,
                args,
                round_orders,
                gate_orders,
                baseline_data,
            )
        )
        print(f"[strict] seed={args.seed} layer={layer_name} done", flush=True)

    protocol = {
        "seed": args.seed,
        "layers": list(SEED_LAYERS),
        "layer_sets": LAYER_SETS,
        "trials": list(TRIALS),
        "branches": list(BRANCHES),
        "budgets": list(BUDGETS),
        "embedding_dims": {
            name: 2 * len(layers) * 1024 for name, layers in LAYER_SETS.items()
        },
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gate_loss": "weighted cross entropy; positive weight=min(5,sqrt(n_negative/n_positive))",
        "ranking_loss": "0.5 * mean softplus(score_negative-score_positive) within batch",
        "common_round": True,
        "same_gate_initialization_and_orders_across_losses": True,
        "selection": "validation reflection-rate budget and strong-independent token cap; then correct count, lower total tokens, lower call rate",
        "validation_token_cap": baseline_data["validation"]["independent"][
            "avg_token_cost"
        ],
        "root": str(root),
        "test_trace": str(args.test_trace),
        "feature_results": str(args.feature_results),
        "feature_info": cache_info,
        "coverage": audit,
        "test_evaluated": True,
        "new_llm_calls": 0,
        "warning": "Fixed test trace was previously inspected; results are retrospective strict-protocol ablation evidence.",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "protocol.json", protocol)
    write_json(args.output / "baselines.json", baseline_data)
    write_json(
        args.output / "results.json",
        {"protocol": protocol, "baselines": baseline_data, "results": results},
    )
    print(f"[strict] completed {args.output}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("outputs/math_full_training_v1"))
    parser.add_argument(
        "--test-trace",
        type=Path,
        default=Path(
            "outputs/math_full_test_failure_accounted_v1/policy_test_accounted.jsonl"
        ),
    )
    parser.add_argument("--feature-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lambda-cost", type=float, default=0.0001)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError(f"Output is not empty: {args.output}")
    torch.set_num_threads(4)
    run(args)


if __name__ == "__main__":
    main()
