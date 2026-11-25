"""Offline controls requested for the GSM8K reflection policy.

This module deliberately uses only saved candidates and saved frozen encoder
features.  It does not load a language model or make any LLM calls.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


ACTIONS = ("direct", "reflect_1", "reflect_2")
SEEDS = (7, 21, 42)


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.open() if line.strip()]


def candidates(row):
    out = {str(item["action"]): item for item in row["candidate_results"]}
    missing = set(ACTIONS) - set(out)
    if missing:
        raise ValueError(f"Missing candidates for {row['id']}: {sorted(missing)}")
    return out


def best_action(row):
    cs = candidates(row)
    return max(ACTIONS, key=lambda action: (int(cs[action]["correct"]), -int(cs[action]["token_cost"])))


def load_features(manifest_path: Path):
    manifest = json.loads(manifest_path.read_text())
    pieces = []
    next_start = 0
    for path in sorted(manifest_path.parent.glob("*.npy")):
        start, end = map(int, path.stem.split("_"))
        if start != next_start:
            raise ValueError(f"Feature shard gap at {path}")
        values = np.load(path, allow_pickle=False)
        if values.shape[0] != end - start or not np.isfinite(values).all():
            raise ValueError(f"Invalid feature shard {path}")
        pieces.append(values.astype("float32"))
        next_start = end
    values = np.concatenate(pieces)
    if len(values) != len(manifest["ids"]):
        raise ValueError("Feature manifest and shards have different lengths")
    return manifest, {ident: values[i] for i, ident in enumerate(manifest["ids"])}


def last_k_features(values, manifest, k=3):
    indices = list(manifest["layer_indices"])
    if len(indices) < k:
        raise ValueError(f"Feature cache has only {len(indices)} layers; need {k}")
    hidden = int(manifest["hidden_size"])
    width = 2 * hidden
    # The extractor stores [mean,last] for each layer in layer_indices order.
    start = (len(indices) - k) * width
    return values[start : start + k * width]


def metric(rows, actions):
    chosen = [candidates(row)[action] for row, action in zip(rows, actions, strict=True)]
    direct = np.asarray([int(candidates(row)["direct"]["correct"]) for row in rows])
    final = np.asarray([int(item["correct"]) for item in chosen])
    total = np.asarray([int(item["token_cost"]) for item in chosen])
    prompt = np.asarray([int(item["prompt_tokens"]) for item in chosen])
    completion = np.asarray([int(item["completion_tokens"]) for item in chosen])
    return {
        "n": len(rows),
        "accuracy": float(final.mean()),
        "avg_prompt_tokens": float(prompt.mean()),
        "avg_completion_tokens": float(completion.mean()),
        "avg_token_cost": float(total.mean()),
        "reflection_rate": float(np.mean(np.asarray(actions) != "direct")),
        "actions": dict(Counter(actions)),
        "fixes": int(((direct == 0) & (final == 1)).sum()),
        "harms": int(((direct == 1) & (final == 0)).sum()),
    }


def choose_threshold(rows, probabilities, cap):
    points = []
    for threshold in np.linspace(0.0, 1.0, 101):
        actions = ["reflect_1" if p >= threshold else "direct" for p in probabilities]
        m = metric(rows, actions)
        if m["reflection_rate"] <= cap + 1e-12:
            points.append((m["accuracy"], -m["avg_token_cost"], -m["reflection_rate"], -threshold, threshold, m))
    if not points:
        raise ValueError(f"No threshold satisfies reflection cap {cap}")
    return max(points, key=lambda point: point[:4])


def train_mlp(x_train, y_train, seed, classes, balance=True):
    counts = Counter(y_train)
    sample_weight = np.ones(len(y_train), dtype="float32")
    if balance:
        total = len(y_train)
        for label, count in counts.items():
            weight = min(5.0, total / (len(classes) * count))
            sample_weight[np.asarray(y_train) == label] = weight
    model = MLPClassifier(
        hidden_layer_sizes=(64, 16), activation="relu", solver="adam",
        alpha=1e-3, batch_size=128, learning_rate_init=1e-3,
        max_iter=300, early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=20, random_state=seed,
    )
    try:
        model.fit(x_train, y_train, sample_weight=sample_weight)
    except TypeError:
        rng = np.random.default_rng(seed)
        probabilities = sample_weight / sample_weight.sum()
        indices = rng.choice(len(y_train), size=len(y_train), replace=True, p=probabilities)
        model.fit(x_train[indices], np.asarray(y_train)[indices])
    return model, {str(label): int(count) for label, count in counts.items()}


def summarize_classifier(model, x, labels):
    predicted = model.predict(x)
    out = {"action_accuracy": float(accuracy_score(labels, predicted)),
           "n_iter": int(model.n_iter_), "labels": dict(Counter(labels))}
    if len(model.classes_) == 2:
        score = model.predict_proba(x)[:, list(model.classes_).index(1)]
        out["roc_auc"] = float(roc_auc_score(labels, score)) if len(set(labels)) == 2 else None
        out["average_precision"] = float(average_precision_score(labels, score)) if len(set(labels)) == 2 else None
    return out


def run(args):
    root = Path(args.root)
    train_rows = {row["id"]: row for row in read_jsonl(root / "policy_all.jsonl")}
    test_rows = read_jsonl(Path(args.test_trace))
    split = json.loads((root / "manifest.json").read_text())
    manifest, feature_map = load_features(Path(args.feature_manifest))
    def available(ids):
        return [train_rows[ident] for ident in ids if ident in train_rows and ident in feature_map and {item["action"] for item in train_rows[ident].get("candidate_results", [])} == set(ACTIONS)]
    def all_available(ids):
        return [train_rows[ident] for ident in ids if ident in train_rows and ident in feature_map]
    train = available(split["train_ids"])
    val = available(split["validation_ids"])
    all_train = all_available(split["train_ids"])
    all_val = all_available(split["validation_ids"])
    test = [row for row in test_rows if row["id"] in feature_map]
    if not train or not val or not all_train or not all_val or len(test) != len(test_rows):
        raise ValueError("Missing rows/features in the locked split")
    x = {row["id"]: last_k_features(feature_map[row["id"]], manifest, 3) for row in all_train + all_val + test}
    scaler = StandardScaler().fit(np.stack([x[row["id"]] for row in all_train]))
    train_x = scaler.transform(np.stack([x[row["id"]] for row in train])).astype("float32")
    val_x = scaler.transform(np.stack([x[row["id"]] for row in val])).astype("float32")
    all_train_x = scaler.transform(np.stack([x[row["id"]] for row in all_train])).astype("float32")
    all_val_x = scaler.transform(np.stack([x[row["id"]] for row in all_val])).astype("float32")
    test_x = scaler.transform(np.stack([x[row["id"]] for row in test])).astype("float32")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    result = {
        "protocol": {
            "scope": "offline controls on saved GSM8K candidates",
            "feature_manifest": str(args.feature_manifest),
            "layer_set": "last3",
            "representation_dim": int(train_x.shape[1]),
            "encoder_layers": manifest["layer_indices"][-3:],
            "train_rows": len(train), "validation_rows": len(val), "all_train_rows": len(all_train), "all_validation_rows": len(all_val), "test_rows": len(test),
            "training_coverage": "complete three-action rows only; missing counterfactuals are not synthesized",
            "seeds": list(SEEDS), "hidden_sizes": [64, 16], "max_iter": 300,
            "no_new_llm_calls": True,
            "label_rule": "accuracy first, then lower token cost",
            "gate_reflection_cap": float(args.reflection_cap),
        },
        "baselines": {
            "direct": metric(test, ["direct"] * len(test)),
            "fixed_reflect_1": metric(test, ["reflect_1"] * len(test)),
            "fixed_reflect_2": metric(test, ["reflect_2"] * len(test)),
            "oracle_accuracy_first": metric(test, [best_action(row) for row in test]),
        },
        "three_class_mlp": [],
        "gate_target_comparison": {"direct_error": [], "r1_repairable": []},
    }

    for seed in SEEDS:
        y_train = np.asarray([ACTIONS.index(best_action(row)) for row in train], dtype="int64")
        val_labels = np.asarray([ACTIONS.index(best_action(row)) for row in val], dtype="int64")
        test_labels = np.asarray([ACTIONS.index(best_action(row)) for row in test], dtype="int64")
        model, counts = train_mlp(train_x, y_train, seed, (0, 1, 2), balance=False)
        val_actions = [ACTIONS[int(value)] for value in model.predict(val_x)]
        test_actions = [ACTIONS[int(value)] for value in model.predict(test_x)]
        result["three_class_mlp"].append({
            "seed": seed, "train_label_counts": {ACTIONS[int(label)]: count for label, count in counts.items()},
            "train": summarize_classifier(model, train_x, y_train.tolist()),
            "validation": {**summarize_classifier(model, val_x, val_labels.tolist()),
                           "answer": metric(val, val_actions)},
            "test": {**summarize_classifier(model, test_x, test_labels.tolist()),
                     "answer": metric(test, test_actions)},
        })

        for name in ("direct_error", "r1_repairable"):
            def direct_is_correct(row):
                return bool(next(item for item in row["candidate_results"] if item["action"] == "direct")["correct"])
            def target(row):
                if name == "direct_error":
                    return int(not direct_is_correct(row))
                cs = candidates(row)
                return int(not bool(cs["direct"]["correct"]) and bool(cs["reflect_1"]["correct"]))
            gate_train, gate_val = (all_train, all_val) if name == "direct_error" else (train, val)
            gate_train_x, gate_val_x = (all_train_x, all_val_x) if name == "direct_error" else (train_x, val_x)
            tr_y = np.asarray([target(row) for row in gate_train], dtype="int64")
            va_y = np.asarray([target(row) for row in gate_val], dtype="int64")
            te_y = np.asarray([target(row) for row in test], dtype="int64")
            model, counts = train_mlp(gate_train_x, tr_y, seed, (0, 1), balance=True)
            va_p = model.predict_proba(gate_val_x)[:, list(model.classes_).index(1)]
            te_p = model.predict_proba(test_x)[:, list(model.classes_).index(1)]
            _, _, _, _, threshold, va_metric = choose_threshold(gate_val, va_p, args.reflection_cap)
            te_actions = ["reflect_1" if p >= threshold else "direct" for p in te_p]
            result["gate_target_comparison"][name].append({
                "seed": seed, "train_label_counts": counts,
                "positive_definition": "direct incorrect" if name == "direct_error" else "direct incorrect and R1 correct",
                "validation_classifier": {**summarize_classifier(model, gate_val_x, va_y.tolist()),
                                           "answer": va_metric, "threshold": float(threshold)},
                "test_classifier": summarize_classifier(model, test_x, te_y.tolist()),
                "test_answer": metric(test, te_actions),
            })

    (output / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    write_report(output / "REPORT.md", result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


def mean_std(values):
    values = np.asarray(values, dtype=float)
    return float(values.mean()), float(values.std(ddof=1)) if len(values) > 1 else 0.0


def fmt(values, scale=1.0):
    mean, std = mean_std([float(v) * scale for v in values])
    return f"{mean:.4f} +/- {std:.4f}"


def write_report(path, result):
    lines = [
        "# Teacher-requested GSM8K policy controls",
        "",
        "All experiments reuse saved candidates and saved Last-3 frozen features; no LLM was called.",
        "The test set was historically observed, so these are retrospective diagnostic controls.",
        "",
        "## Protocol",
        "",
        f"- Representation: Last-3, dimension {result['protocol']['representation_dim']}; layers {result['protocol']['encoder_layers']}",
        f"- Rows: train={result['protocol']['train_rows']}, validation={result['protocol']['validation_rows']}, test={result['protocol']['test_rows']}",
        "- Standardizer is fitted on the training split only.",
        "- Three-class labels use accuracy first, then lower total token cost.",
        f"- Gate comparison uses a validation reflection-rate cap of {result["protocol"]["gate_reflection_cap"]:.0%}, then selects accuracy first and token cost second.",
        "",
        "## Frozen baselines and Oracle",
        "",
        "| Method | Accuracy | Total token | Reflection rate | Fix/Harm |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, m in result["baselines"].items():
        lines.append(f"| {name} | {m['accuracy']:.4f} | {m['avg_token_cost']:.2f} | {m['reflection_rate']:.2%} | {m['fixes']}/{m['harms']} |")
    lines += ["", "## Direct three-class MLP: Direct / R1 / R2", "",
              "This is the direct multiclass control on the same representation. It does not decompose the action into a Gate and a Round head.", "",
              "| Seed | Val action acc. | Val answer acc. | Test action acc. | Test answer acc. | Test token | Test reflection |",
              "|---:|---:|---:|---:|---:|---:|---:|"]
    for row in result["three_class_mlp"]:
        lines.append(f"| {row['seed']} | {row['validation']['action_accuracy']:.4f} | {row['validation']['answer']['accuracy']:.4f} | "
                     f"{row['test']['action_accuracy']:.4f} | {row['test']['answer']['accuracy']:.4f} | "
                     f"{row['test']['answer']['avg_token_cost']:.2f} | {row['test']['answer']['reflection_rate']:.2%} |")
    lines += ["", "## Gate target comparison", "",
              "`direct_error` predicts whether Direct is wrong. `r1_repairable` predicts whether Direct is wrong and R1 is correct; both deploy fixed R1 when positive.", "",
              "| Target | Seed | Val classifier acc. | Val answer acc. | Test classifier acc. | Test answer acc. | Test token | Test reflection |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, rows in result["gate_target_comparison"].items():
        for row in rows:
            lines.append(f"| {name} | {row['seed']} | {row['validation_classifier']['action_accuracy']:.4f} | "
                         f"{row['validation_classifier']['answer']['accuracy']:.4f} | {row['test_classifier']['action_accuracy']:.4f} | "
                         f"{row['test_answer']['accuracy']:.4f} | {row['test_answer']['avg_token_cost']:.2f} | "
                         f"{row['test_answer']['reflection_rate']:.2%} |")
    lines += ["", "## Interpretation", "",
              "- The three-class MLP tests whether one flat action classifier can replace the hierarchical decomposition under identical frozen features.",
              "- The Gate target comparison separates generic error detection from the more deployment-relevant event that a fixed R1 can actually repair the Direct answer.",
              "- These controls are not used to select a new main model; the test set is retained for retrospective diagnosis only.", ""]
    path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("outputs/math_full_training_v1"))
    parser.add_argument("--test-trace", type=Path, default=Path("outputs/math_full_test_failure_accounted_v1/policy_test_accounted.jsonl"))
    parser.add_argument("--feature-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reflection-cap", type=float, default=0.10)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
