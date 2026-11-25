"""Apply the strict Last-3 policy to frozen transfer traces.

This command does not call a generative model. It rebuilds the small k=3
policy heads from the locked GSM8K train/validation features, encodes the
transfer questions with the frozen 0.6B encoder, and evaluates all policies
without target-domain fitting or threshold recalibration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from .evaluate_accounted_math_test import failure_metrics
from .control_ablation_last3 import (
    BUDGETS,
    RANKING_COEFFICIENT,
    _load_cache,
    _predict,
    _round_actions,
    _select,
    _train_gate_epoch,
    _train_round,
)
from .layer_ablation_math import _mlp
from .strict_layer_control_ablation import _select_fast
from .neural_policy import build_policy_text, _ensure_torchvision_nms_compat
from .train_fixed_math_policy import ACTIONS, audit_split, decision_metrics, digest
from .utils import read_jsonl


PROJECT = Path(__file__).resolve().parents[1]
DATASETS = ("gsm1k", "svamp", "multiarith")
TRIALS = (("weighted_ce", 0.0), ("weighted_ce_plus_rank", RANKING_COEFFICIENT))
BRANCHES = ("gate_round", "gate_r1")
ENCODER = Path(os.environ.get("ENCODER_NAME", "Qwen/Qwen3-0.6B"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _orders(count: int, batch_size: int, epochs: int, seed: int):
    rng = np.random.default_rng(seed)
    batches = max(1, int(np.ceil(count / batch_size)))
    return [
        [torch.from_numpy(order.astype(np.int64))
         for order in np.array_split(rng.permutation(count), batches)]
        for _ in range(epochs)
    ]


def _load_training_features(seed: int, train: list[dict], validation: list[dict], test: list[dict]):
    rows = train + validation + test
    result_path = PROJECT / f"outputs/layer_ablation_gsm8k_budget_layers6_seed{seed}/results.json"
    features, info = _load_cache(result_path, rows)
    return features[:len(train) + len(validation)], info


def _train_k3(seed: int, train: list[dict], validation: list[dict], features: np.ndarray,
              output: Path, epochs: int, batch_size: int, lambda_cost: float):
    output.mkdir(parents=True, exist_ok=True)
    n_train = len(train)
    train_features = features[:n_train]
    val_features = features[n_train:]
    gate_labels = torch.tensor(
        [int(row["best_action"] != "direct") for row in train], dtype=torch.long
    )
    round_indices = np.asarray([
        i for i, row in enumerate(train)
        if {candidate["action"] for candidate in row["candidate_results"]} == set(ACTIONS)
    ], dtype=np.int64)
    round_labels = torch.tensor([
        int(max(
            (candidate for candidate in train[i]["candidate_results"]
             if candidate["action"] in ACTIONS[1:]),
            key=lambda candidate: (candidate["correct"], -candidate["token_cost"]),
        )["action"] == "reflect_2")
        for i in round_indices
    ], dtype=torch.long)
    if len(torch.unique(gate_labels)) < 2 or len(torch.unique(round_labels)) < 2:
        raise ValueError(f"seed={seed}: Gate and Round both need two classes")

    gate_scaler = StandardScaler().fit(train_features)
    round_scaler = StandardScaler().fit(train_features[round_indices])
    gate_x = torch.from_numpy(gate_scaler.transform(train_features).astype("float32"))
    gate_vx = torch.from_numpy(gate_scaler.transform(val_features).astype("float32"))
    round_x = torch.from_numpy(round_scaler.transform(train_features[round_indices]).astype("float32"))
    round_vx = torch.from_numpy(round_scaler.transform(val_features).astype("float32"))
    gate_weights = torch.tensor([
        1.0,
        min(5.0, float(np.sqrt((gate_labels == 0).sum() / (gate_labels == 1).sum()))),
    ])
    round_counts = torch.bincount(round_labels, minlength=2).float()
    round_weights = len(round_labels) / (2.0 * round_counts)
    strong_cost = decision_metrics(
        validation, ["independent"] * len(validation), lambda_cost
    )["avg_token_cost"]

    round_orders = _orders(len(round_indices), batch_size, epochs, seed + 20000)
    gate_orders = _orders(len(train), batch_size, epochs, seed + 40000)
    torch.manual_seed(seed + 10000)
    round_model = _mlp(6144, (64, 16), 0.0)
    round_optimizer = torch.optim.AdamW(round_model.parameters(), lr=1e-3, weight_decay=1e-3)
    for orders in round_orders:
        _train_round(round_model, round_optimizer, round_x, round_labels, round_weights, orders)
    round_val_actions = _round_actions(_predict(round_model, round_vx))

    all_policies = []
    for trial, coefficient in TRIALS:
        torch.manual_seed(seed + 30000)
        template = _mlp(6144, (16,), 0.2)
        gate = _mlp(6144, (16,), 0.2)
        gate.load_state_dict(template.state_dict())
        optimizer = torch.optim.AdamW(gate.parameters(), lr=3e-4, weight_decay=1e-2)
        best = {}
        for epoch, orders in enumerate(gate_orders, start=1):
            _train_gate_epoch(gate, optimizer, gate_x, gate_labels, gate_weights, orders, coefficient)
            val_prob = _predict(gate, gate_vx)[:, 0]
            for branch, branch_actions in (
                ("gate_round", round_val_actions),
                ("gate_r1", ["reflect_1"] * len(validation)),
            ):
                for budget in BUDGETS:
                    point = _select_fast(validation, val_prob, branch_actions, budget, strong_cost, lambda_cost)
                    if point is None:
                        continue
                    key = (point["metrics"]["correct"], -point["metrics"]["avg_token_cost"],
                           -point["metrics"]["reflection_rate"], -point["threshold"])
                    identity = (branch, round(budget, 2))
                    if identity not in best or key > best[identity]["order"]:
                        best[identity] = {
                            "trial": trial, "branch": branch, "budget": budget,
                            "epoch": epoch, "threshold": point["threshold"],
                            "validation": point["metrics"], "order": key,
                            "gate_state": {k: v.detach().cpu().clone() for k, v in gate.state_dict().items()},
                            "round_state": {k: v.detach().cpu().clone() for k, v in round_model.state_dict().items()},
                        }
        for chosen in best.values():
            name = f"{trial}_{chosen['branch']}_seed{seed}_{int(chosen['budget'] * 100)}"
            path = output / f"{name}.pt"
            torch.save({
                "gate": chosen["gate_state"], "round": chosen["round_state"],
                "gate_mean": gate_scaler.mean_.astype("float32"),
                "gate_scale": gate_scaler.scale_.astype("float32"),
                "round_mean": round_scaler.mean_.astype("float32"),
                "round_scale": round_scaler.scale_.astype("float32"),
                "layer_set": [-3, -2, -1], "embedding_dim": 6144,
                "trial": trial, "branch": chosen["branch"], "seed": seed,
                "budget": chosen["budget"], "epoch": chosen["epoch"],
                "threshold": chosen["threshold"], "validation": chosen["validation"],
            }, path)
            all_policies.append({**chosen, "name": name, "checkpoint": str(path)})
    return all_policies


def _encode_transfer(rows: list[dict], output: Path, gpu: str, max_length: int):
    texts = [build_policy_text(row, max_text_chars=0, include_state=False) for row in rows]
    hashes = [hashlib.sha256(text.encode()).hexdigest() for text in texts]
    manifest_path = output / "manifest.json"
    feature_path = output / "features.npy"
    if manifest_path.exists() and feature_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("text_sha256") == hashes and manifest.get("layers") == [-3, -2, -1]:
            return np.load(feature_path, allow_pickle=False), manifest

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for transfer encoder extraction")
    _ensure_torchvision_nms_compat(torch)
    from transformers import AutoModel, AutoTokenizer
    from torch.nn.attention import SDPBackend, sdpa_kernel
    tokenizer = AutoTokenizer.from_pretrained(str(ENCODER), local_files_only=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    lengths = [len(tokenizer(text, truncation=False)["input_ids"]) for text in texts]
    device = torch.device("cuda:0")
    model = AutoModel.from_pretrained(str(ENCODER), local_files_only=True,
                                      dtype=torch.float32, attn_implementation="sdpa").to(device).eval()
    parts = []
    for start in range(0, len(texts), 2):
        batch = tokenizer(texts[start:start + 2], padding=True, truncation=True,
                          max_length=max_length, return_tensors="pt").to(device)
        with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
            hidden_states = model(**batch, output_hidden_states=True, use_cache=False).hidden_states
            mask = batch["attention_mask"]
            last = mask.sum(1) - 1
            pieces = []
            for layer in (-3, -2, -1):
                hidden = hidden_states[layer].float()
                mean = (hidden * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
                pieces.extend([mean, hidden[torch.arange(len(last), device=device), last]])
            parts.append(torch.cat(pieces, 1).cpu().numpy().astype("float32"))
        if (start // 2) % 50 == 0:
            print(f"[k3-transfer] encoded {min(start + 2, len(texts))}/{len(texts)}", flush=True)
    del model
    torch.cuda.empty_cache()
    features = np.concatenate(parts)
    manifest = {
        "encoder": str(ENCODER), "layers": [-3, -2, -1], "pooling": "mean_last",
        "embedding_dim": 6144, "max_length": max_length, "dtype": "float32",
        "ids": [row["id"] for row in rows], "text_sha256": hashes,
        "lengths": lengths, "clipped_ids": [row["id"] for row, n in zip(rows, lengths) if n > max_length],
        "physical_gpu": str(gpu),
    }
    output.mkdir(parents=True, exist_ok=True)
    np.save(feature_path, features, allow_pickle=False)
    write_json(manifest_path, manifest)
    return features, manifest


def _apply(bundle: dict, features: np.ndarray, rows: list[dict]):
    gate = _mlp(6144, (16,), 0.2)
    round_model = _mlp(6144, (64, 16), 0.0)
    gate.load_state_dict(bundle["gate"])
    round_model.load_state_dict(bundle["round"])
    gate_scaler = StandardScaler()
    gate_scaler.mean_, gate_scaler.scale_ = np.asarray(bundle["gate_mean"]), np.asarray(bundle["gate_scale"])
    round_scaler = StandardScaler()
    round_scaler.mean_, round_scaler.scale_ = np.asarray(bundle["round_mean"]), np.asarray(bundle["round_scale"])
    gate_scaler.n_features_in_ = 6144
    round_scaler.n_features_in_ = 6144
    with torch.inference_mode():
        gate_prob = gate(torch.from_numpy(gate_scaler.transform(features).astype("float32"))).softmax(1).numpy()[:, 0]
        round_logits = round_model(torch.from_numpy(round_scaler.transform(features).astype("float32")))
        round_actions = _round_actions(round_logits.softmax(1).numpy())
    actions = []
    for row, probability, round_action in zip(rows, gate_prob, round_actions):
        if row.get("initial_execution_status") != "success":
            actions.append("direct")
        elif probability >= bundle["threshold"]:
            actions.append("direct")
        else:
            actions.append(round_action if bundle["branch"] == "gate_round" else "reflect_1")
    return actions, gate_prob


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=PROJECT / "outputs/math_transfer_locked_v1")
    parser.add_argument("--output", type=Path, default=PROJECT / "outputs/math_transfer_k3_v1")
    parser.add_argument("--gpu", default="2")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lambda-cost", type=float, default=0.0001)
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    torch.set_num_threads(4)
    output.mkdir(parents=True, exist_ok=True)
    generation = json.loads((source / "generation_protocol.json").read_text())
    roles = generation["roles"]
    if roles["base"]["model"] != "Qwen3-8B-Instruct" or roles["qwen14"]["model"] != "Qwen3-14B":
        raise ValueError(f"Transfer traces are not the current Base/Reflector pair: {roles}")
    transfer_rows = {}
    for dataset in DATASETS:
        path = source / dataset / "policy_test_accounted.jsonl"
        rows = read_jsonl(path)
        if not rows:
            raise ValueError(f"Empty transfer trace: {path}")
        transfer_rows[dataset] = rows
    all_transfer = [row for dataset in DATASETS for row in transfer_rows[dataset]]

    train, validation, audit = audit_split(PROJECT / "outputs/math_full_training_v1", allow_partial=True)
    test_rows = read_jsonl(PROJECT / "outputs/math_full_test_failure_accounted_v1/policy_test_accounted.jsonl")
    all_gsm = train + validation + test_rows
    transfer_features, encoder_manifest = _encode_transfer(all_transfer, output / "features", args.gpu, args.max_length)

    policies = []
    for seed in (7, 21, 42):
        gsm_features, feature_info = _load_training_features(seed, train, validation, test_rows)
        seed_policies = _train_k3(seed, train, validation, gsm_features,
                                  output / "policies" / f"seed{seed}", args.epochs,
                                  args.batch_size, args.lambda_cost)
        for policy in seed_policies:
            bundle = torch.load(policy["checkpoint"], map_location="cpu", weights_only=False)
            for dataset in DATASETS:
                rows = transfer_rows[dataset]
                start = sum(len(transfer_rows[name]) for name in DATASETS[:DATASETS.index(dataset)])
                end = start + len(rows)
                actions, probabilities = _apply(bundle, transfer_features[start:end], rows)
                metrics = failure_metrics(rows, actions)
                policies.append({
                    "seed": seed, "trial": policy["trial"], "branch": policy["branch"],
                    "budget": policy["budget"], "epoch": policy["epoch"],
                    "threshold": policy["threshold"], "dataset": dataset,
                    "checkpoint": policy["checkpoint"], "metrics": metrics,
                    "actions": actions, "direct_probability": probabilities.tolist(),
                })
                print(f"[k3-transfer] {dataset} seed={seed} {policy['trial']} {policy['branch']} "
                      f"budget={policy['budget']:.0%} acc={metrics['accuracy']:.4f} "
                      f"tokens={metrics['avg_token_cost']:.1f}", flush=True)

    summary = {"source": str(source), "datasets": {d: len(transfer_rows[d]) for d in DATASETS},
               "generation_roles": roles, "encoder": encoder_manifest, "training_audit": audit,
               "policies": policies, "no_target_retraining": True, "no_target_threshold_recalibration": True,
               "no_new_llm_calls": True}
    write_json(output / "summary.json", summary)
    lines = ["# k=3 transfer evaluation", "",
             "Frozen Qwen3-8B-Instruct/Qwen3-14B traces are reused. The policy is trained on GSM8K only;"
             " transfer data are used only for offline evaluation.", "",
             "| Dataset | Direct Acc. | Recipe | Branch | Cap | Acc. mean +/- SD | Tokens mean +/- SD | Call mean +/- SD |", 
             "|---|---:|---|---|---:|---:|---:|---:|"]
    for dataset in DATASETS:
        rows = transfer_rows[dataset]
        direct = failure_metrics(rows, ["direct"] * len(rows))
        subset = [p for p in policies if p["dataset"] == dataset]
        lines.append(f"| {dataset} | {direct['accuracy']:.4f} | -- | -- | -- | -- | {direct['avg_token_cost']:.2f} | -- |")
        for trial in ("weighted_ce", "weighted_ce_plus_rank"):
            for branch in BRANCHES:
                for budget in BUDGETS:
                    values = [p["metrics"] for p in subset if p["trial"] == trial and p["branch"] == branch and abs(p["budget"] - budget) < 1e-9]
                    acc = np.asarray([v["accuracy"] for v in values])
                    tok = np.asarray([v["avg_token_cost"] for v in values])
                    call = np.asarray([v["reflection_rate"] for v in values])
                    lines.append(f"| {dataset} |  | {trial} | {branch} | {budget:.0%} | "
                                 f"{acc.mean():.4f} +/- {acc.std(ddof=1):.4f} | "
                                 f"{tok.mean():.2f} +/- {tok.std(ddof=1):.2f} | "
                                 f"{call.mean():.4f} +/- {call.std(ddof=1):.4f} |")
    lines += ["", "Execution failures remain in the denominator and their paid tokens are included.",
              "Thresholds were selected on GSM8K validation only; no target-domain retraining or recalibration was performed.",
              "The transfer traces were generated with the active Base/Reflector roles recorded in generation_protocol.json."]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(output / "protocol.json", {
        "source": str(source), "source_generation_protocol_sha256": digest(source / "generation_protocol.json"),
        "encoder": encoder_manifest, "layer_set": [-3, -2, -1], "embedding_dim": 6144,
        "seeds": [7, 21, 42], "budgets": list(BUDGETS), "trials": [x[0] for x in TRIALS],
        "branches": list(BRANCHES), "no_target_retraining": True,
        "no_target_threshold_recalibration": True, "no_new_llm_calls": True,
    })
    print(f"[k3-transfer] completed: {output / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
