"""Last-3 controlled ablations for Gate loss and reflection-round selection.

The experiment reuses the six-layer hidden-state cache produced by the layer
ablation. It slices Last-3 features, trains the same small Gate and a shared
Round head, and compares weighted CE with weighted CE plus a pairwise ranking
term. The Gate+Round and Gate+fixed-R1 branches are evaluated from the same
Gate checkpoints, so the round-selection ablation does not retrain the Gate.
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

from .layer_ablation_math import _mlp, _round_actions, _slice_layers
from .neural_policy import build_policy_text
from .train_fixed_math_policy import ACTIONS, audit_split, best_action, decision_metrics, write_json
from .utils import read_jsonl


BUDGETS = (0.05, 0.10, 0.20)
LAYER_NAME = "last3"
RANKING_COEFFICIENT = 0.5


def _load_cache(results_path: Path, rows: list[dict]) -> tuple[np.ndarray, dict]:
    results = json.loads(results_path.read_text())
    encoder = results["encoder"]
    cache = Path(encoder["cache"])
    manifest = json.loads((cache / "manifest.json").read_text())
    expected_ids = [row["id"] for row in rows]
    if manifest["ids"] != expected_ids:
        raise ValueError("Feature cache IDs do not match the locked row order")
    if manifest["layer_indices"] != [-6, -5, -4, -3, -2, -1]:
        raise ValueError("Expected the six-layer cache used by the layer ablation")
    if manifest["pooling"] != "mean_last" or manifest["hidden_size"] != 1024:
        raise ValueError("Unexpected feature cache pooling or hidden size")
    text_hashes = [hashlib.sha256(build_policy_text(row, max_text_chars=0,
                                                    include_state=False).encode()).hexdigest()
                   for row in rows]
    if manifest["text_sha256"] != text_hashes:
        raise ValueError("Feature cache text fingerprint differs from current policy input")
    parts = []
    next_start = 0
    for shard in sorted(cache.glob("*.npy")):
        start, end = map(int, shard.stem.split("_"))
        values = np.load(shard, allow_pickle=False)
        if start != next_start or values.shape != (end - start, 12288):
            raise ValueError(f"Invalid feature shard: {shard}")
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite feature shard: {shard}")
        parts.append(_slice_layers(values, LAYER_NAME))
        next_start = end
    if next_start != len(rows):
        raise ValueError("Feature cache does not cover all rows")
    return np.concatenate(parts).astype("float32"), {
        "cache": str(cache),
        "fingerprint": hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
        "layer_set": [-3, -2, -1],
        "embedding_dim": 6144,
        "manifest_sha256": hashlib.sha256((cache / "manifest.json").read_bytes()).hexdigest(),
    }


def _rank_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    scores = logits[:, 1] - logits[:, 0]
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    if not len(positive) or not len(negative):
        return logits.sum() * 0.0
    return torch.nn.functional.softplus(negative.unsqueeze(0) - positive.unsqueeze(1)).mean()


def _train_gate_epoch(model, optimizer, x, y, weights, orders, coefficient):
    model.train()
    rank_total = 0.0
    for order in orders:
        optimizer.zero_grad(set_to_none=True)
        logits = model(x[order])
        ce = torch.nn.functional.cross_entropy(logits, y[order], weight=weights)
        ranking = _rank_loss(logits, y[order])
        (ce + coefficient * ranking).backward()
        optimizer.step()
        rank_total += float(ranking.detach())
    return rank_total / max(len(orders), 1)


def _train_round(round_model, optimizer, x, y, weights, orders):
    round_model.train()
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    for order in orders:
        optimizer.zero_grad(set_to_none=True)
        criterion(round_model(x[order]), y[order]).backward()
        optimizer.step()


def _predict(model, x):
    model.eval()
    with torch.inference_mode():
        return model(x).softmax(1).cpu().numpy()


def _thresholds(probabilities):
    values = np.unique(np.asarray(probabilities, dtype="float32"))
    result = {0.0, 1.000001}
    result.update(float(x) for x in np.linspace(0.0, 1.0, 201))
    result.update(float(np.nextafter(value, np.float32(np.inf))) for value in values)
    return sorted(result)


def _select(rows, direct_prob, round_actions, budget, strong_cost, lambda_cost):
    points = []
    for threshold in _thresholds(direct_prob):
        actions = ["direct" if p >= threshold else action
                   for p, action in zip(direct_prob, round_actions)]
        metrics = decision_metrics(rows, actions, lambda_cost)
        if metrics["reflection_rate"] <= budget + 1e-12 and metrics["avg_token_cost"] < strong_cost:
            points.append({"threshold": threshold, "metrics": metrics, "actions": actions})
    if not points:
        return None
    return max(points, key=lambda point: (
        point["metrics"]["correct"],
        -point["metrics"]["avg_token_cost"],
        -point["metrics"]["reflection_rate"],
        -point["threshold"],
    ))


def _baseline(rows, action, lambda_cost):
    return decision_metrics(rows, [action] * len(rows), lambda_cost)


def _action_counts(actions):
    values, counts = np.unique(actions, return_counts=True)
    return {str(value): int(count) for value, count in zip(values, counts)}


def run(args):
    root = Path(args.root)
    train, validation, audit = audit_split(root, allow_partial=True)
    test = read_jsonl(str(args.test_trace))
    all_rows = train + validation + test
    feature_results = Path(args.feature_results)
    features, feature_info = _load_cache(feature_results, all_rows)
    n_train, n_val = len(train), len(validation)
    train_features = features[:n_train]
    val_features = features[n_train:n_train + n_val]
    test_features = features[n_train + n_val:]

    gate_labels = torch.tensor([int(row["best_action"] != "direct") for row in train], dtype=torch.long)
    round_indices = np.asarray([
        i for i, row in enumerate(train)
        if {candidate["action"] for candidate in row["candidate_results"]} == set(ACTIONS)
    ], dtype=np.int64)
    round_labels = torch.tensor([
        int(best_action(train[i], ACTIONS[1:]) == "reflect_2") for i in round_indices
    ], dtype=torch.long)
    if len(torch.unique(gate_labels)) < 2 or len(torch.unique(round_labels)) < 2:
        raise ValueError("Both Gate and Round classes are required")

    gate_scaler = StandardScaler().fit(train_features)
    round_scaler = StandardScaler().fit(train_features[round_indices])
    gate_x = torch.from_numpy(gate_scaler.transform(train_features).astype("float32"))
    gate_vx = torch.from_numpy(gate_scaler.transform(val_features).astype("float32"))
    gate_test_x = torch.from_numpy(gate_scaler.transform(test_features).astype("float32"))
    round_x = torch.from_numpy(round_scaler.transform(train_features[round_indices]).astype("float32"))
    round_vx = torch.from_numpy(round_scaler.transform(val_features).astype("float32"))
    round_test_x = torch.from_numpy(round_scaler.transform(test_features).astype("float32"))
    gate_weights = torch.tensor([
        1.0, min(5.0, float(np.sqrt((gate_labels == 0).sum() / (gate_labels == 1).sum())))
    ], dtype=torch.float32)
    round_counts = torch.bincount(round_labels, minlength=2).float()
    round_weights = len(round_labels) / (2.0 * round_counts)

    val_baselines = {name: _baseline(validation, action, args.lambda_cost) for name, action in (
        ("direct", "direct"), ("fixed_reflect_1", "reflect_1"),
        ("fixed_reflect_2", "reflect_2"), ("independent", "independent"))}
    test_baselines = {name: _baseline(test, action, args.lambda_cost) for name, action in (
        ("direct", "direct"), ("fixed_reflect_1", "reflect_1"),
        ("fixed_reflect_2", "reflect_2"))}
    if all(row.get("independent") is not None for row in test):
        test_baselines["independent"] = _baseline(test, "independent", args.lambda_cost)

    # Train one Round head shared by both Gate-loss trials.
    torch.manual_seed(args.seed + 10000)
    round_model = _mlp(train_features.shape[1], (64, 16), 0.0)
    round_optimizer = torch.optim.AdamW(round_model.parameters(), lr=1e-3, weight_decay=1e-3)
    round_orders_rng = np.random.default_rng(args.seed + 20000)
    round_orders = [
        [torch.from_numpy(order.astype(np.int64)) for order in
         np.array_split(round_orders_rng.permutation(len(round_labels)),
                        max(1, int(np.ceil(len(round_labels) / args.batch_size))))]
        for _ in range(args.epochs)
    ]
    for orders in round_orders:
        _train_round(round_model, round_optimizer, round_x, round_labels, round_weights, orders)
    round_val_actions = _round_actions(_predict(round_model, round_vx))
    round_test_actions = _round_actions(_predict(round_model, round_test_x))

    results = []
    histories = {}
    trial_specs = [("weighted_ce", 0.0), ("weighted_ce_plus_rank", RANKING_COEFFICIENT)]
    for trial_name, coefficient in trial_specs:
        torch.manual_seed(args.seed + 30000)
        template = _mlp(train_features.shape[1], (16,), 0.2)
        gate = _mlp(train_features.shape[1], (16,), 0.2)
        gate.load_state_dict(copy.deepcopy(template.state_dict()))
        optimizer = torch.optim.AdamW(gate.parameters(), lr=3e-4, weight_decay=1e-2)
        gate_orders_rng = np.random.default_rng(args.seed + 40000)
        gate_orders = [
            [torch.from_numpy(order.astype(np.int64)) for order in
             np.array_split(gate_orders_rng.permutation(len(gate_labels)),
                            max(1, int(np.ceil(len(gate_labels) / args.batch_size))))]
            for _ in range(args.epochs)
        ]
        best = {}
        history = []
        trial_dir = args.output / trial_name
        trial_dir.mkdir(parents=True, exist_ok=True)
        for epoch, orders in enumerate(gate_orders, start=1):
            batch_rank = _train_gate_epoch(gate, optimizer, gate_x, gate_labels,
                                           gate_weights, orders, coefficient)
            val_prob = _predict(gate, gate_vx)[:, 0]
            entry = {"epoch": epoch, "batch_mean_ranking_loss": batch_rank, "selected": {}}
            for branch, branch_actions in (("gate_round", round_val_actions),
                                           ("gate_r1", ["reflect_1"] * len(validation))):
                points = []
                for budget in BUDGETS:
                    point = _select(validation, val_prob, branch_actions, budget,
                                    val_baselines["independent"]["avg_token_cost"], args.lambda_cost)
                    key = f"{branch}_{int(budget * 100)}"
                    entry["selected"][key] = point
                    if point is None:
                        continue
                    order_key = (point["metrics"]["correct"], -point["metrics"]["avg_token_cost"],
                                 -point["metrics"]["reflection_rate"], -point["threshold"])
                    if key not in best or order_key > best[key]["order"]:
                        best[key] = {
                            "order": order_key, "trial": trial_name, "branch": branch,
                            "budget": budget, "epoch": epoch, "threshold": point["threshold"],
                            "validation": point["metrics"],
                            "gate_state": copy.deepcopy(gate.state_dict()),
                        }
            history.append(entry)
            print(f"[control] seed={args.seed} trial={trial_name} epoch={epoch}/{args.epochs}", flush=True)
        histories[trial_name] = history
        for key, chosen in best.items():
            branch = chosen["branch"]
            gate.load_state_dict(chosen["gate_state"])
            test_prob = _predict(gate, gate_test_x)[:, 0]
            actions = ["direct" if p >= chosen["threshold"] else action
                       for p, action in zip(test_prob, round_test_actions if branch == "gate_round"
                                             else ["reflect_1"] * len(test))]
            test_metrics = decision_metrics(test, actions, args.lambda_cost)
            checkpoint = trial_dir / f"{branch}_{int(chosen['budget'] * 100)}.pt"
            torch.save({"gate": chosen["gate_state"], "scaler": gate_scaler,
                        "layer_set": [-3, -2, -1], "branch": branch,
                        "budget": chosen["budget"], "threshold": chosen["threshold"],
                        "epoch": chosen["epoch"], "seed": args.seed}, checkpoint)
            results.append({"trial": trial_name, "branch": branch, "budget": chosen["budget"],
                           "epoch": chosen["epoch"], "threshold": chosen["threshold"],
                           "validation": chosen["validation"], "test": test_metrics,
                           "checkpoint": str(checkpoint)})
        (trial_dir / "history.json").write_text(json.dumps(history, indent=2, ensure_ascii=False) + "\n")

    protocol = {
        "seed": args.seed, "layer_set": [-3, -2, -1], "embedding_dim": 6144,
        "root": str(root), "test_trace": str(args.test_trace),
        "feature_results": str(feature_results), "feature_info": feature_info,
        "epochs": args.epochs, "batch_size": args.batch_size,
        "gate_hidden": [16], "round_hidden": [64, 16], "gate_dropout": 0.2,
        "gate_loss": "weighted cross entropy; positive weight=min(5,sqrt(n_negative/n_positive))",
        "ranking_loss": "0.5 * mean softplus(score_negative-score_positive) within batch",
        "branches": ["gate_round", "gate_r1"], "budgets": BUDGETS,
        "selection": "validation final correct count, then lower total tokens, then lower call rate",
        "test_evaluated": True, "new_llm_calls": 0,
        "warning": "Fixed test trace was previously inspected; results are retrospective ablation evidence.",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "protocol.json", protocol)
    write_json(args.output / "baselines.json", {"validation": val_baselines, "test": test_baselines})
    write_json(args.output / "results.json", {"protocol": protocol, "baselines": {"validation": val_baselines,
        "test": test_baselines}, "results": results, "round_action_counts": {
            "validation": _action_counts(round_val_actions),
            "test": _action_counts(round_test_actions)},
        "histories": {name: str(args.output / name / "history.json") for name, _ in trial_specs},
        "test_evaluated": True})
    lines = ["# Last-3 control ablation", "", "No new LLM calls; all hidden vectors and candidate traces are frozen.",
             "Epochs and thresholds are selected on validation only. The fixed test trace was previously inspected.", "",
             "| Loss | Branch | Budget | Epoch | Val Acc | Test Acc | Test Tokens | Test Call | Fix/Harm |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in sorted(results, key=lambda r: (r["trial"], r["branch"], r["budget"])):
        val, test_metrics = row["validation"], row["test"]
        lines.append(f"| {row['trial']} | {row['branch']} | {row['budget']:.0%} | {row['epoch']} | "
                     f"{val['accuracy']:.4f} | {test_metrics['accuracy']:.4f} | {test_metrics['avg_token_cost']:.2f} | "
                     f"{test_metrics['reflection_rate']:.4f} | {test_metrics['fixes']}/{test_metrics['harms']} |")
    (args.output / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(f"[control] completed {args.output}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("outputs/math_full_training_v1"))
    p.add_argument("--test-trace", type=Path,
                   default=Path("outputs/math_full_test_failure_accounted_v1/policy_test_accounted.jsonl"))
    p.add_argument("--feature-results", type=Path, required=True,
                   help="results.json from a completed six-layer ablation run")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lambda-cost", type=float, default=0.0001)
    args = p.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError(f"Output is not empty: {args.output}")
    torch.set_num_threads(4)
    run(args)


if __name__ == "__main__":
    main()
