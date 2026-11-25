"""Budget-constrained last-1/2/3 layer ablation using frozen math traces."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from .layer_ablation_math import (
    DEFAULT_ENCODER,
    LAYER_SETS,
    _encode_sharded,
    _load_rows,
    _mlp,
    _predict,
    _round_actions,
    _slice_layers,
    _train_epoch,
)
from .neural_policy import build_policy_text
from .train_fixed_math_policy import ACTIONS, audit_split, best_action, decision_metrics, write_json


BUDGETS = (0.05, 0.10, 0.20)


def _thresholds(probabilities: np.ndarray) -> list[float]:
    values = np.unique(np.asarray(probabilities, dtype="float32"))
    result = {0.0, 1.000001}
    result.update(float(x) for x in np.linspace(0.0, 1.0, 201))
    result.update(float(np.nextafter(value, np.float32(np.inf))) for value in values)
    return sorted(result)


def _select_budget_point(rows, direct_prob, round_actions, budget, strong_cost, lambda_cost):
    points = []
    for threshold in _thresholds(direct_prob):
        actions = ["direct" if p >= threshold else action
                   for p, action in zip(direct_prob, round_actions)]
        metrics = decision_metrics(rows, actions, lambda_cost)
        if metrics["reflection_rate"] <= budget + 1e-12 and metrics["avg_token_cost"] < strong_cost:
            points.append({"threshold": threshold, "metrics": metrics, "actions": actions})
    if not points:
        return None
    return max(points, key=lambda item: (
        item["metrics"]["correct"],
        -item["metrics"]["avg_token_cost"],
        -item["metrics"]["reflection_rate"],
        -item["threshold"],
    ))


def _baseline(rows, action, lambda_cost):
    return decision_metrics(rows, [action] * len(rows), lambda_cost)


def _run_layer(name, train, val, test, train_features, val_features, test_features, output, args, strong_cost):
    dim = train_features.shape[1]
    gate_targets = np.asarray([int(str(row["best_action"]) != "direct") for row in train], dtype="int64")
    round_indices = np.asarray([
        index for index, row in enumerate(train)
        if {candidate["action"] for candidate in row["candidate_results"]} == set(ACTIONS)
    ], dtype="int64")
    round_targets = np.asarray([
        int(best_action(train[index], ACTIONS[1:]) == "reflect_2") for index in round_indices
    ], dtype="int64")
    gate_val_targets = np.asarray([int(best_action(row) != "direct") for row in val], dtype="int64")
    round_val_targets = np.asarray([int(best_action(row, ACTIONS[1:]) == "reflect_2") for row in val], dtype="int64")

    gate_scaler = StandardScaler().fit(train_features)
    round_scaler = StandardScaler().fit(train_features[round_indices])
    gate_x = torch.from_numpy(gate_scaler.transform(train_features).astype("float32"))
    gate_vx = torch.from_numpy(gate_scaler.transform(val_features).astype("float32"))
    gate_test_x = torch.from_numpy(gate_scaler.transform(test_features).astype("float32"))
    round_x = torch.from_numpy(round_scaler.transform(train_features[round_indices]).astype("float32"))
    round_vx = torch.from_numpy(round_scaler.transform(val_features).astype("float32"))
    round_test_x = torch.from_numpy(round_scaler.transform(test_features).astype("float32"))
    gate_y = torch.from_numpy(gate_targets)
    round_y = torch.from_numpy(round_targets)
    round_counts = np.bincount(round_targets, minlength=2).astype("float32")
    round_weights = torch.tensor(len(round_targets) / (2.0 * round_counts), dtype=torch.float32)

    torch.manual_seed(args.seed)
    gate = _mlp(dim, (16,), 0.2)
    round_model = _mlp(dim, (64, 16), 0.0)
    gate_optimizer = torch.optim.AdamW(gate.parameters(), lr=3e-4, weight_decay=0.01)
    round_optimizer = torch.optim.AdamW(round_model.parameters(), lr=1e-3, weight_decay=1e-3)
    gate_criterion = torch.nn.CrossEntropyLoss()
    round_criterion = torch.nn.CrossEntropyLoss(weight=round_weights)
    rng = np.random.default_rng(args.seed)
    best = {budget: None for budget in BUDGETS}
    history = []

    for epoch in range(1, args.epochs + 1):
        _train_epoch(gate, gate_optimizer, gate_criterion, gate_x, gate_y, args.batch_size, rng)
        _train_epoch(round_model, round_optimizer, round_criterion, round_x, round_y, args.batch_size, rng)
        gate_val = _predict(gate, gate_vx)
        round_val = _predict(round_model, round_vx)
        round_actions = _round_actions(round_val)
        selected = {}
        for budget in BUDGETS:
            point = _select_budget_point(val, gate_val[:, 0], round_actions, budget, strong_cost, args.lambda_cost)
            selected[f"{budget:.2f}"] = point
            if point is None:
                continue
            key = (point["metrics"]["correct"], -point["metrics"]["avg_token_cost"],
                   -point["metrics"]["reflection_rate"], -point["threshold"])
            if best[budget] is None or key > best[budget]["key"]:
                best[budget] = {"key": key, "epoch": epoch, "threshold": point["threshold"],
                                "validation": point["metrics"],
                                "gate_state": copy.deepcopy(gate.state_dict()),
                                "round_state": copy.deepcopy(round_model.state_dict())}
        history.append({"epoch": epoch, "selected": {
            key: (value["metrics"] if value is not None else None)
            for key, value in selected.items()
        }, "thresholds": {key: (value["threshold"] if value is not None else None)
                           for key, value in selected.items()}})
        print(f"[budget-ablation] {name} epoch={epoch}/{args.epochs} " + " | ".join(
            f"{int(float(budget)*100)}% acc={point['metrics']['accuracy']:.4f} "
            f"tokens={point['metrics']['avg_token_cost']:.1f} call={point['metrics']['reflection_rate']:.3f}"
            if point is not None else f"{int(float(budget)*100)}% infeasible"
            for budget, point in selected.items()), flush=True)

    results = {}
    output.mkdir(parents=True, exist_ok=True)
    for budget in BUDGETS:
        chosen = best[budget]
        if chosen is None:
            results[f"{int(budget * 100)}%"] = None
            continue
        gate.load_state_dict(chosen["gate_state"])
        round_model.load_state_dict(chosen["round_state"])
        gate_test = _predict(gate, gate_test_x)
        round_test = _predict(round_model, round_test_x)
        actions = ["direct" if p >= chosen["threshold"] else action
                   for p, action in zip(gate_test[:, 0], _round_actions(round_test))]
        test_metrics = decision_metrics(test, actions, args.lambda_cost)
        results[f"{int(budget * 100)}%"] = {
            "layer_set": LAYER_SETS[name], "embedding_dim": dim,
            "budget": budget, "best_epoch": chosen["epoch"], "threshold": chosen["threshold"],
            "validation": chosen["validation"], "test": test_metrics,
            "checkpoint": str(output / f"checkpoint_{int(budget * 100)}.pt"),
        }
        torch.save({"gate": gate.state_dict(), "round": round_model.state_dict(),
                    "layer_set": LAYER_SETS[name], "best_epoch": chosen["epoch"],
                    "threshold": chosen["threshold"], "budget": budget},
                   output / f"checkpoint_{int(budget * 100)}.pt")
    (output / "history.json").write_text(json.dumps(history, indent=2, ensure_ascii=False) + "\n")
    return results


def _report(path, results, val_baselines, test_baselines):
    lines = ["# GSM8K layer ablation under fixed reflection budgets", "",
             "All rows reuse frozen candidate traces. Epochs and thresholds are selected on validation only.", "",
             "## Main threshold and checkpoint selection", "",
             "The main experiment evaluates reflection-rate caps of 5%, 10%, and 20%. The Gate score is the predicted Direct-class probability p_D; a validation row remains Direct when p_D >= threshold and uses the predicted reflection action otherwise.", "",
             "For every validation checkpoint and budget, candidate thresholds are the union of the regular grid {0.000, 0.005, ..., 1.000}, 1.000001 (the no-reflection endpoint), and np.nextafter(p_D, +infinity) for every distinct validation score. A candidate is feasible only when validation reflection rate <= the budget and average total token cost is strictly below the validation independent-candidate baseline (strong_cost). Among feasible candidates, the code maximizes (correct answer count, -average total tokens, -reflection rate, -threshold). The best checkpoint and threshold are selected independently for each budget using this same lexicographic rule, then frozen for test evaluation without test-side recalibration.", "",
             "## Baselines", "", "| Split | Method | Accuracy | Total tokens | Reflection rate |", "| --- | --- | ---: | ---: | ---: |"]
    for split, table in (("validation", val_baselines), ("test", test_baselines)):
        for name, metrics in table.items():
            lines.append(f"| {split} | {name} | {metrics['accuracy']:.4f} | {metrics['avg_token_cost']:.2f} | {metrics['reflection_rate']:.4f} |")
    lines += ["", "## Layer and budget results", "",
              "| Layer set | Budget | Epoch | Val acc. | Val tokens | Test acc. | Test tokens | Test call rate | Fix/Harm |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for layer, budget_results in results.items():
        for budget, result in budget_results.items():
            if result is None:
                lines.append(f"| {layer} | {budget} | -- | -- | -- | -- | -- | -- | -- |")
                continue
            val, test = result["validation"], result["test"]
            lines.append(f"| {layer} | {budget} | {result['best_epoch']} | {val['accuracy']:.4f} | {val['avg_token_cost']:.2f} | "
                         f"{test['accuracy']:.4f} | {test['avg_token_cost']:.2f} | {test['reflection_rate']:.4f} | "
                         f"{test['fixes']}/{test['harms']} |")
    lines += ["", "## Boundary", "",
              "The test set is evaluated after validation locking. This is a representation ablation, not a new model-generation experiment.",
              "The independent baseline is reported only when every test row contains a saved independent candidate.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/math_full_training_v1"))
    parser.add_argument("--test-trace", type=Path, default=Path("outputs/traces/gsm8k_test_policy_trace_detailed.jsonl"))
    parser.add_argument("--encoder", default=DEFAULT_ENCODER)
    parser.add_argument("--output", type=Path, default=Path("outputs/layer_ablation_gsm8k_budget_v1"))
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--encoder-batch-size", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lambda-cost", type=float, default=0.0001)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    train, val, audit = audit_split(args.root, allow_partial=True)
    test = _load_rows(args.test_trace)
    all_rows = train + val + test
    features, encoder_info = _encode_sharded(all_rows, args.encoder, args.output, args.max_length,
                                             args.encoder_batch_size, args.device)
    n_train, n_val = len(train), len(val)
    train_features, val_features, test_features = features[:n_train], features[n_train:n_train+n_val], features[n_train+n_val:]
    val_baselines = {name: _baseline(val, action, args.lambda_cost) for name, action in
                     (("direct", "direct"), ("fixed_reflect_1", "reflect_1"),
                      ("fixed_reflect_2", "reflect_2"), ("independent", "independent"))}
    test_baselines = {name: _baseline(test, action, args.lambda_cost) for name, action in
                      (("direct", "direct"), ("fixed_reflect_1", "reflect_1"), ("fixed_reflect_2", "reflect_2"))}
    # Report the independent baseline only when it is saved for every test row.
    if all(row.get("independent") is not None for row in test):
        test_baselines["independent"] = _baseline(test, "independent", args.lambda_cost)
    results = {}
    for name in LAYER_SETS:
        current_train = _slice_layers(train_features, name)
        current_val = _slice_layers(val_features, name)
        current_test = _slice_layers(test_features, name)
        results[name] = _run_layer(name, train, val, test, current_train, current_val, current_test,
                                   args.output / name, args, val_baselines["independent"]["avg_token_cost"])
    payload = {"protocol": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
               "encoder": encoder_info, "coverage": audit, "budgets": BUDGETS,
               "validation_baselines": val_baselines, "test_baselines": test_baselines,
               "results": results, "test_evaluated": True}
    write_json(args.output / "results.json", payload)
    _report(args.output / "REPORT.md", results, val_baselines, test_baselines)
    print(f"[budget-ablation] completed: {args.output}", flush=True)


if __name__ == "__main__":
    main()
