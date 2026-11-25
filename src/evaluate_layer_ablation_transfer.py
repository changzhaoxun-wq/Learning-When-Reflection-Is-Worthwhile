"""Evaluate the current Last-1--Last-6 GSM8K routers on math transfer data.

This is an offline transfer evaluator. It reuses the current main experiment's
checkpoints, training-fitted standardizers, and locked Base/Reflector
candidate traces. It never retrains on a target dataset, recalibrates a
threshold, or calls a generative model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from .layer_ablation_math import (
    DEFAULT_ENCODER,
    LAYER_SETS,
    _encode_sharded,
    _mlp,
    _predict,
    _round_actions,
    _slice_layers,
)
from .train_fixed_math_policy import ACTIONS, decision_metrics
from .utils import read_jsonl


BUDGETS = (0.05, 0.10, 0.20)
DATASETS = ("gsm1k", "svamp", "multiarith")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _load_feature_cache(cache: Path, manifest: dict[str, Any]) -> np.ndarray:
    shards = sorted(cache.glob("*.npy"))
    if not shards:
        raise FileNotFoundError(f"No feature shards found in {cache}")
    parts = [np.load(path, allow_pickle=False).astype("float32", copy=False)
             for path in shards]
    features = np.concatenate(parts, axis=0)
    expected = len(manifest["ids"])
    if features.shape[0] != expected:
        raise ValueError(f"Feature cache rows {features.shape[0]} != manifest rows {expected}")
    if not np.isfinite(features).all():
        raise ValueError(f"Non-finite values in feature cache {cache}")
    return features


def _find_main_cache(main_output: Path) -> tuple[Path, dict[str, Any], np.ndarray]:
    manifests = sorted((main_output / "features").glob("*/manifest.json"))
    if len(manifests) != 1:
        raise ValueError(f"Expected exactly one main feature manifest, found {manifests}")
    manifest_path = manifests[0]
    manifest = json.loads(manifest_path.read_text())
    return manifest_path.parent, manifest, _load_feature_cache(manifest_path.parent, manifest)


def _load_main_rows(project_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    train = read_jsonl(str(project_root / "outputs/math_full_training_v1/policy_train_partial.jsonl"))
    val = read_jsonl(str(project_root / "outputs/math_full_training_v1/policy_validation_partial.jsonl"))
    test = read_jsonl(str(project_root / "outputs/math_full_test_failure_accounted_v1/policy_test_accounted.jsonl"))
    return train, val, test


def _check_feature_order(manifest: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    ids = [str(row["id"]) for row in rows]
    if manifest["ids"] != ids:
        for index, (expected, actual) in enumerate(zip(manifest["ids"], ids)):
            if expected != actual:
                raise ValueError(f"Feature/row order mismatch at {index}: {expected} != {actual}")
        raise ValueError(f"Feature/row count mismatch: {len(manifest['ids'])} != {len(ids)}")


def _load_transfer_rows(transfer_root: Path) -> dict[str, list[dict[str, Any]]]:
    result = {}
    for dataset in DATASETS:
        path = transfer_root / dataset / "policy_test_accounted.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Missing transfer candidate trace: {path}")
        rows = read_jsonl(str(path))
        if not rows:
            raise ValueError(f"Empty transfer candidate trace: {path}")
        for row in rows:
            actions = {candidate["action"] for candidate in row.get("candidate_results", [])}
            if not set(ACTIONS[:3]).issubset(actions):
                raise ValueError(f"{path}: {row['id']} lacks direct/R1/R2 candidates: {actions}")
        result[dataset] = rows
    return result


def _predict_policy(checkpoint: Path, features: np.ndarray, gate_scaler: StandardScaler,
                    round_scaler: StandardScaler, device: str) -> tuple[np.ndarray, list[str]]:
    bundle = torch.load(checkpoint, map_location="cpu", weights_only=False)
    dim = features.shape[1]
    gate = _mlp(dim, (16,), 0.2)
    round_model = _mlp(dim, (64, 16), 0.0)
    gate.load_state_dict(bundle["gate"])
    round_model.load_state_dict(bundle["round"])
    gate_x = torch.from_numpy(gate_scaler.transform(features).astype("float32"))
    round_x = torch.from_numpy(round_scaler.transform(features).astype("float32"))
    if device != "cpu":
        gate.to(device)
        round_model.to(device)
        gate_x = gate_x.to(device)
        round_x = round_x.to(device)
    gate_prob = _predict(gate, gate_x)
    round_prob = _predict(round_model, round_x)
    return gate_prob[:, 0], _round_actions(round_prob)


def _policy_metrics(rows: list[dict[str, Any]], actions: list[str]) -> dict[str, Any]:
    return decision_metrics(rows, actions, 0.0001)


def _evaluate_dataset(rows: list[dict[str, Any]], features: np.ndarray,
                      layer: str, layer_dir: Path, gate_scaler: StandardScaler,
                      round_scaler: StandardScaler, device: str) -> dict[str, Any]:
    direct = _policy_metrics(rows, ["direct"] * len(rows))
    fixed_r1 = _policy_metrics(rows, ["reflect_1"] * len(rows))
    fixed_r2 = _policy_metrics(rows, ["reflect_2"] * len(rows))
    baselines = {"direct": direct, "fixed_reflect_1": fixed_r1, "fixed_reflect_2": fixed_r2}
    policies: dict[str, Any] = {}
    current_features = _slice_layers(features, layer)
    for budget in BUDGETS:
        checkpoint = layer_dir / f"checkpoint_{int(budget * 100)}.pt"
        bundle = torch.load(checkpoint, map_location="cpu", weights_only=False)
        gate_prob, round_actions = _predict_policy(
            checkpoint, current_features, gate_scaler, round_scaler, device
        )
        actions = ["direct" if p >= float(bundle["threshold"]) else action
                   for p, action in zip(gate_prob, round_actions)]
        metrics = _policy_metrics(rows, actions)
        policies[f"{int(budget * 100)}%"] = {
            "checkpoint": str(checkpoint),
            "layer_set": bundle["layer_set"],
            "embedding_dim": int(current_features.shape[1]),
            "budget": float(budget),
            "gsm8k_epoch": int(bundle["best_epoch"]),
            "gsm8k_threshold": float(bundle["threshold"]),
            "metrics": metrics,
        }
    return {"n": len(rows), "baselines": baselines, "policies": policies}


def _report(output: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Current Last-1--Last-6 transfer evaluation", "",
        "This is an offline transfer evaluation using the current GSM8K main experiment's Last-1--Last-6 checkpoints.",
        "No target-domain retraining, threshold recalibration, or new LLM generation was performed.", "",
        "## Protocol", "",
        f"- Main policy output: `{payload['main_output']}`",
        f"- Transfer candidate root: `{payload['transfer_root']}`",
        f"- Main generation protocol SHA256: `{payload['main_generation_protocol_sha256']}`",
        f"- Transfer generation protocol SHA256: `{payload['transfer_generation_protocol_sha256']}`",
        f"- Encoder: `{payload['encoder']['encoder_name']}`; max length `{payload['encoder']['max_length']}`; pooling `{payload['encoder']['pooling']}`",
        "- Gate score is the predicted Direct-class probability; each threshold is copied from the corresponding GSM8K checkpoint.",
        "- Candidate traces contain Direct, R1, and R2 for every transfer item; fixed baselines use the same traces.", "",
        "## Results", "",
        "| Dataset | Method | Cap | Acc. | Tokens | Reflection rate | Fix/Harm |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for dataset, result in payload["datasets"].items():
        for method, metrics in result["baselines"].items():
            label = {"direct": "Direct", "fixed_reflect_1": "Fixed R1", "fixed_reflect_2": "Fixed R2"}[method]
            lines.append(f"| {dataset} | {label} | -- | {metrics['accuracy']:.4f} | {metrics['avg_token_cost']:.2f} | {metrics['reflection_rate']:.4f} | {metrics['fixes']}/{metrics['harms']} |")
        for layer, layer_result in result["layers"].items():
            for cap, policy in layer_result.items():
                m = policy["metrics"]
                lines.append(f"| {dataset} | {layer} | {cap} | {m['accuracy']:.4f} | {m['avg_token_cost']:.2f} | {m['reflection_rate']:.4f} | {m['fixes']}/{m['harms']} |")
    lines += ["", "## Interpretation boundary", "", "The transfer policies are the frozen GSM8K routers applied without target-domain fitting. Results are reported for the current seed-42 main checkpoints and should not be mixed with the older k=2/k=3 transfer reports.", ""]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--main-output", type=Path, default=Path("outputs/layer_ablation_gsm8k_budget_layers6_v1"))
    parser.add_argument("--transfer-root", type=Path, default=Path("outputs/math_transfer_locked_v1"))
    parser.add_argument("--output", type=Path, default=Path("outputs/math_transfer_layers6_v1"))
    parser.add_argument("--encoder", default=DEFAULT_ENCODER)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--encoder-batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()
    root = args.project_root.resolve()
    main_output = (root / args.main_output).resolve()
    transfer_root = (root / args.transfer_root).resolve()
    output = (root / args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    main_payload = json.loads((main_output / "results.json").read_text())
    train, val, main_test = _load_main_rows(root)
    cache, manifest, main_features = _find_main_cache(main_output)
    _check_feature_order(manifest, train + val + main_test)
    if len(train) != int(main_payload["coverage"]["train"]["observed"]):
        raise ValueError("Main train row count does not match current coverage")
    if len(val) != int(main_payload["coverage"]["validation"]["observed"]):
        raise ValueError("Main validation row count does not match current coverage")

    transfer = _load_transfer_rows(transfer_root)
    transfer_rows = [row for dataset in DATASETS for row in transfer[dataset]]
    transfer_features, encoder_info = _encode_sharded(
        transfer_rows, args.encoder, output, args.max_length, args.encoder_batch_size, args.device
    )
    offsets: dict[str, tuple[int, int]] = {}
    cursor = 0
    for dataset in DATASETS:
        end = cursor + len(transfer[dataset])
        offsets[dataset] = (cursor, end)
        cursor = end

    train_features = main_features[:len(train)]
    dataset_features = {dataset: transfer_features[start:end] for dataset, (start, end) in offsets.items()}
    results: dict[str, Any] = {}
    for layer, layer_set in LAYER_SETS.items():
        train_layer = _slice_layers(train_features, layer)
        round_indices = np.asarray([
            index for index, row in enumerate(train)
            if {candidate["action"] for candidate in row["candidate_results"]} == set(ACTIONS)
        ], dtype="int64")
        gate_scaler = StandardScaler().fit(train_layer)
        round_scaler = StandardScaler().fit(train_layer[round_indices])
        layer_dir = main_output / layer
        results[layer] = {}
        for dataset in DATASETS:
            dataset_result = _evaluate_dataset(
                transfer[dataset], dataset_features[dataset], layer, layer_dir,
                gate_scaler, round_scaler, args.device
            )
            results[layer][dataset] = dataset_result

    main_protocol = root / "outputs/math_full_training_v1/protocol.json"
    transfer_protocol = transfer_root / "generation_protocol.json"
    payload = {
        "main_output": str(main_output),
        "transfer_root": str(transfer_root),
        "main_generation_protocol_sha256": _sha256(main_protocol),
        "transfer_generation_protocol_sha256": _sha256(transfer_protocol),
        "main_feature_cache": str(cache),
        "main_feature_manifest_sha256": _sha256(cache / "manifest.json"),
        "encoder": encoder_info,
        "no_target_retraining": True,
        "no_target_threshold_recalibration": True,
        "no_new_llm_calls": True,
        "seed": int(main_payload["protocol"]["seed"]),
        "datasets": {},
        "layers": list(LAYER_SETS),
    }
    for dataset in DATASETS:
        payload["datasets"][dataset] = {
            "n": len(transfer[dataset]),
            "baselines": results["last1"][dataset]["baselines"],
            "layers": {layer: {cap: value for cap, value in results[layer][dataset]["policies"].items()}
                       for layer in LAYER_SETS},
        }
    (output / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    (output / "protocol.json").write_text(json.dumps({
        "main_output": str(main_output), "transfer_root": str(transfer_root),
        "main_generation_protocol_sha256": payload["main_generation_protocol_sha256"],
        "transfer_generation_protocol_sha256": payload["transfer_generation_protocol_sha256"],
        "main_seed": payload["seed"], "layers": LAYER_SETS, "budgets": BUDGETS,
        "encoder": encoder_info, "no_target_retraining": True,
        "no_target_threshold_recalibration": True, "no_new_llm_calls": True,
        "candidate_counts": {dataset: len(transfer[dataset]) for dataset in DATASETS},
    }, indent=2, ensure_ascii=False) + "\n")
    _report(output, payload)
    print(f"[transfer-layers6] completed: {output}", flush=True)


if __name__ == "__main__":
    main()
