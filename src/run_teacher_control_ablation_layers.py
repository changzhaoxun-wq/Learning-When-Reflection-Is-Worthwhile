"""Run the teacher-requested offline controls for Last-1 through Last-6.

All controls reuse the current saved GSM8K traces and the six-layer frozen
feature cache. This script makes no LLM calls.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

from .run_teacher_control_ablation import (
    ACTIONS,
    SEEDS,
    best_action,
    candidates,
    choose_threshold,
    last_k_features,
    load_features,
    metric,
    read_jsonl,
    summarize_classifier,
    train_mlp,
)

CAP = 0.10


def direct_correct(row: dict) -> bool:
    return bool(
        next(item for item in row["candidate_results"] if item["action"] == "direct")["correct"]
    )


def round_action(row: dict) -> str:
    cs = candidates(row)
    return max(
        ("reflect_1", "reflect_2"),
        key=lambda action: (int(cs[action]["correct"]), -int(cs[action]["token_cost"])),
    )


def complete_candidates(row: dict) -> bool:
    return {item["action"] for item in row.get("candidate_results", [])} == set(ACTIONS)


def flat_action(row: dict) -> str:
    """Return a valid flat label without fabricating unknown counterfactuals."""
    if direct_correct(row):
        return "direct"
    if not complete_candidates(row):
        raise ValueError(f"Cannot label incorrect Direct row without R1/R2: {row['id']}")
    return max(ACTIONS, key=lambda action: (int(candidates(row)[action]["correct"]), -int(candidates(row)[action]["token_cost"])))


def repairable_label(row: dict) -> int:
    """Label whether saved R1 repairs Direct, when that fact is observable."""
    if direct_correct(row):
        return 0
    if not complete_candidates(row):
        raise ValueError(f"Cannot label incorrect Direct row without R1: {row['id']}")
    return int(bool(candidates(row)["reflect_1"]["correct"]))


def choose_threshold_with_actions(rows, probabilities, reflected_actions, cap):
    points = []
    for threshold in np.linspace(0.0, 1.0, 101):
        actions = [
            reflected if probability >= threshold else "direct"
            for probability, reflected in zip(probabilities, reflected_actions, strict=True)
        ]
        values = metric(rows, actions)
        if values["reflection_rate"] <= cap + 1e-12:
            points.append(
                (
                    values["accuracy"],
                    -values["avg_token_cost"],
                    -values["reflection_rate"],
                    -threshold,
                    float(threshold),
                    values,
                )
            )
    if not points:
        raise ValueError(f"No threshold satisfies cap={cap}")
    return max(points, key=lambda point: point[:4])


def transform_rows(rows, feature_map, manifest, scaler, k):
    raw = np.stack(
        [last_k_features(feature_map[row["id"]], manifest, k) for row in rows]
    )
    return scaler.transform(raw).astype("float32")


def mean_std(values):
    values = np.asarray(values, dtype=float)
    return float(values.mean()), float(values.std(ddof=1)) if len(values) > 1 else 0.0


def metric_summary(rows):
    accuracy = mean_std([row["accuracy"] for row in rows])
    tokens = mean_std([row["avg_token_cost"] for row in rows])
    rate = mean_std([row["reflection_rate"] for row in rows])
    return {
        "accuracy_mean": accuracy[0],
        "accuracy_sd": accuracy[1],
        "avg_token_mean": tokens[0],
        "avg_token_sd": tokens[1],
        "reflection_rate_mean": rate[0],
        "reflection_rate_sd": rate[1],
    }


def run(args):
    root = Path(args.root)
    train_rows = {row["id"]: row for row in read_jsonl(root / "policy_all.jsonl")}
    split = json.loads((root / "manifest.json").read_text())
    test = read_jsonl(Path(args.test_trace))
    manifest, feature_map = load_features(Path(args.feature_manifest))

    def rows_for(ids):
        return [train_rows[ident] for ident in ids if ident in train_rows and ident in feature_map]

    def complete(rows):
        return [
            row for row in rows
            if {item["action"] for item in row.get("candidate_results", [])} == set(ACTIONS)
        ]

    all_train = rows_for(split["train_ids"])
    all_val = rows_for(split["validation_ids"])
    train = complete(all_train)
    val = complete(all_val)
    test = [row for row in test if row["id"] in feature_map]
    if not all_train or not all_val or not train or not val or len(test) != len(read_jsonl(Path(args.test_trace))):
        raise ValueError("Missing rows/features in the locked split")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    flat_train_labels = [flat_action(row) for row in all_train]
    repair_train_labels = [repairable_label(row) for row in all_train]
    round_train_labels = [round_action(row) for row in train]
    result = {
        "protocol": {
            "scope": "offline teacher controls across Last-1 through Last-6",
            "feature_manifest": str(args.feature_manifest),
            "layer_sets": {str(k): manifest["layer_indices"][-k:] for k in range(1, 7)},
            "representation_dims": {str(k): 2 * k * int(manifest["hidden_size"]) for k in range(1, 7)},
            "seeds": list(SEEDS),
            "hidden_sizes": [64, 16],
            "max_iter": 300,
            "reflection_cap": CAP,
            "all_train_rows": len(all_train),
            "all_validation_rows": len(all_val),
            "complete_train_rows": len(train),
            "complete_validation_rows": len(val),
            "flat_train_rows": len(all_train),
            "repairability_train_rows": len(all_train),
            "test_rows": len(test),
            "standardizer_fit": "all training split rows for Gate, flat MLP, and repairability; complete rows for Round",
            "flat_train_label_counts": {action: flat_train_labels.count(action) for action in ACTIONS},
            "repairability_train_label_counts": {str(label): repair_train_labels.count(label) for label in (0, 1)},
            "round_train_label_counts": {action: round_train_labels.count(action) for action in ("reflect_1", "reflect_2")},
            "missing_counterfactuals": "never synthesized; Direct-correct rows receive logically determined negative/Direct labels",
            "no_new_llm_calls": True,
            "test_historically_observed": True,
        },
        "baselines": {
            "direct": metric(test, ["direct"] * len(test)),
            "fixed_reflect_1": metric(test, ["reflect_1"] * len(test)),
            "fixed_reflect_2": metric(test, ["reflect_2"] * len(test)),
            "oracle_accuracy_first": metric(test, [best_action(row) for row in test]),
        },
        "layers": {},
    }

    all_ids = [row["id"] for row in all_train + all_val + test]
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("Unexpected duplicate IDs across split sections")
    for k in range(1, 7):
        scaler = StandardScaler().fit(
            np.stack([last_k_features(feature_map[row["id"]], manifest, k) for row in all_train])
        )
        all_train_x = transform_rows(all_train, feature_map, manifest, scaler, k)
        all_val_x = transform_rows(all_val, feature_map, manifest, scaler, k)
        train_x = transform_rows(train, feature_map, manifest, scaler, k)
        val_x = transform_rows(val, feature_map, manifest, scaler, k)
        test_x = transform_rows(test, feature_map, manifest, scaler, k)
        layer_result = {
            "protocol": {
                "k": k,
                "layers": manifest["layer_indices"][-k:],
                "representation_dim": int(2 * k * int(manifest["hidden_size"])),
                "all_train_rows": len(all_train),
                "complete_train_rows": len(train),
            },
            "three_class_mlp": [],
            "gate_target_comparison": {"direct_error": [], "r1_repairable": []},
            "same_gate_round": [],
        }

        for seed in SEEDS:
            flat_train_y = np.asarray(
                [ACTIONS.index(flat_action(row)) for row in all_train], dtype="int64"
            )
            flat_val_y = np.asarray(
                [ACTIONS.index(flat_action(row)) for row in all_val], dtype="int64"
            )
            flat_test_y = np.asarray(
                [ACTIONS.index(flat_action(row)) for row in test], dtype="int64"
            )
            flat_model, flat_counts = train_mlp(
                all_train_x, flat_train_y, seed, (0, 1, 2), balance=False
            )
            flat_val_actions = [
                ACTIONS[int(value)] for value in flat_model.predict(all_val_x)
            ]
            flat_test_actions = [
                ACTIONS[int(value)] for value in flat_model.predict(test_x)
            ]
            layer_result["three_class_mlp"].append(
                {
                    "seed": seed,
                    "train_label_counts": {
                        ACTIONS[int(label)]: count
                        for label, count in flat_counts.items()
                    },
                    "validation_classifier": summarize_classifier(
                        flat_model, all_val_x, flat_val_y.tolist()
                    ),
                    "validation_answer": metric(all_val, flat_val_actions),
                    "test_classifier": summarize_classifier(
                        flat_model, test_x, flat_test_y.tolist()
                    ),
                    "test_answer": metric(test, flat_test_actions),
                }
            )

            gate_train_y = np.asarray(
                [int(not direct_correct(row)) for row in all_train], dtype="int64"
            )
            gate_val_y = np.asarray(
                [int(not direct_correct(row)) for row in all_val], dtype="int64"
            )
            gate_test_y = np.asarray(
                [int(not direct_correct(row)) for row in test], dtype="int64"
            )
            gate_model, gate_counts = train_mlp(
                all_train_x, gate_train_y, seed, (0, 1), balance=True
            )
            gate_val_p = gate_model.predict_proba(all_val_x)[
                :, list(gate_model.classes_).index(1)
            ]
            gate_test_p = gate_model.predict_proba(test_x)[
                :, list(gate_model.classes_).index(1)
            ]
            _, _, _, _, gate_threshold, gate_val_answer = choose_threshold(
                all_val, gate_val_p, CAP
            )
            gate_test_actions = [
                "reflect_1" if probability >= gate_threshold else "direct"
                for probability in gate_test_p
            ]
            layer_result["gate_target_comparison"]["direct_error"].append(
                {
                    "seed": seed,
                    "train_label_counts": gate_counts,
                    "validation_auc": summarize_classifier(
                        gate_model, all_val_x, gate_val_y.tolist()
                    ).get("roc_auc"),
                    "validation_answer": gate_val_answer,
                    "threshold": float(gate_threshold),
                    "test_classifier": summarize_classifier(
                        gate_model, test_x, gate_test_y.tolist()
                    ),
                    "test_answer": metric(test, gate_test_actions),
                }
            )

            repair_train_y = np.asarray(
                [repairable_label(row) for row in all_train], dtype="int64"
            )
            repair_val_y = np.asarray(
                [repairable_label(row) for row in all_val], dtype="int64"
            )
            repair_model, repair_counts = train_mlp(
                all_train_x, repair_train_y, seed, (0, 1), balance=True
            )
            repair_val_p = repair_model.predict_proba(all_val_x)[
                :, list(repair_model.classes_).index(1)
            ]
            repair_test_p = repair_model.predict_proba(test_x)[
                :, list(repair_model.classes_).index(1)
            ]
            _, _, _, _, repair_threshold, repair_val_answer = choose_threshold(
                all_val, repair_val_p, CAP
            )
            repair_test_actions = [
                "reflect_1" if probability >= repair_threshold else "direct"
                for probability in repair_test_p
            ]
            layer_result["gate_target_comparison"]["r1_repairable"].append(
                {
                    "seed": seed,
                    "train_label_counts": repair_counts,
                    "validation_auc": summarize_classifier(
                        repair_model, val_x, repair_val_y.tolist()
                    ).get("roc_auc"),
                    "validation_answer": repair_val_answer,
                    "threshold": float(repair_threshold),
                    "test_classifier": summarize_classifier(
                        repair_model, test_x, [
                            repairable_label(row)
                            for row in test
                        ],
                    ),
                    "test_answer": metric(test, repair_test_actions),
                }
            )

            round_train_y = np.asarray([int(round_action(row) == "reflect_2") for row in train])
            round_model, round_counts = train_mlp(
                train_x, round_train_y, seed, (0, 1), balance=True
            )
            round_val_actions = [
                "reflect_2" if int(value) else "reflect_1"
                for value in round_model.predict(val_x)
            ]
            round_test_actions = [
                "reflect_2" if int(value) else "reflect_1"
                for value in round_model.predict(test_x)
            ]
            val_probability_by_id = {
                row["id"]: probability for row, probability in zip(all_val, gate_val_p, strict=True)
            }
            complete_val_p = np.asarray([val_probability_by_id[row["id"]] for row in val])
            _, _, _, _, same_threshold, same_val_round = choose_threshold_with_actions(
                val, complete_val_p, round_val_actions, CAP
            )
            same_val_fixed = [
                "reflect_1" if probability >= same_threshold else "direct"
                for probability in complete_val_p
            ]
            same_test_fixed = [
                "reflect_1" if probability >= same_threshold else "direct"
                for probability in gate_test_p
            ]
            same_test_round = [
                round_action_value if probability >= same_threshold else "direct"
                for probability, round_action_value in zip(
                    gate_test_p, round_test_actions, strict=True
                )
            ]
            layer_result["same_gate_round"].append(
                {
                    "seed": seed,
                    "gate_threshold": float(same_threshold),
                    "round_train_label_counts": round_counts,
                    "validation_fixed_r1": metric(val, same_val_fixed),
                    "validation_round": same_val_round,
                    "test_fixed_r1": metric(test, same_test_fixed),
                    "test_round": metric(test, same_test_round),
                }
            )
        result["layers"][str(k)] = layer_result

    (output / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    )
    write_report(output / "REPORT.md", result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


def write_report(path, result):
    lines = [
        "# Teacher controls across Last-1 to Last-6",
        "",
        "All results reuse the current saved GSM8K candidates and frozen features; no LLM was called.",
        "The test trace was historically observed, so the test rows are retrospective diagnostics.",
        "",
        "## Protocol",
        "",
        f"- Direct-error Gate training rows: {result['protocol']['all_train_rows']}",
        f"- Flat three-way MLP training rows: {result['protocol']['flat_train_rows']}",
        f"- R1-repairability training rows: {result['protocol']['repairability_train_rows']}",
        f"- Round training rows with complete candidates: {result['protocol']['complete_train_rows']}",
        f"- Validation reflection-rate cap: {result['protocol']['reflection_cap']:.0%}",
        "- Standardizers are fitted on the training split only.",
        "",
        "## Formal Gate labels",
        "",
        "For item $i$, let $d_i\\in\\{0,1\\}$ denote the correctness of the saved Direct candidate: $d_i=1$ if Direct is correct and $d_i=0$ otherwise. The Direct-error Gate target is",
        "",
        "$$y_i^{\\mathrm{DE}}=\\mathbf{1}[d_i=0]=1-d_i.$$",
        "",
        "The positive class $y_i^{\\mathrm{DE}}=1$ therefore means that Direct was incorrect and the policy is allowed to trigger reflection; it does **not** mean that a reflection candidate is guaranteed to repair the error. The negative class means that Direct was correct. This target is defined for every available Direct observation, so the Direct-error Gate is trained on all 5976 training rows, including rows without saved reflection counterfactuals.",
        "",
        "For the separate R1-repairability control, the label is defined whenever it is logically or directly observable:",
        "",
        "$$y_i^{\\mathrm{R1\\text{-}repair}}=\\begin{cases}0,&d_i=1,\\\\ \\mathbf{1}[r_{1i}=1],&d_i=0\\text{ and R1 is observed}.\\end{cases}$$",
        "",
        "A Direct-correct row is therefore a known negative even when its reflection counterfactual was not generated. An incorrect Direct row still requires an observed R1 candidate; such rows are excluded rather than relabelled if R1 is missing. In the current training split, all 185 incorrect Direct rows have complete candidates, so the repairability control covers all 5976 training rows.",
        "",
        "The fair flat three-way label follows the same correctness-first, cost-second rule: Direct-correct rows are labelled Direct without inventing R1/R2 outputs; Direct-incorrect rows use the best observed action among Direct, R1, and R2. In the current training split, this yields 5791 Direct, 145 R1, and 40 R2 labels.",
        "",
        "## Reproduction-critical details",
        "",
        "### Exact generation prompts",
        "",
        "The current locked main GSM8K trace uses the following Direct prompt:",
        "",
        "~~~text",
        "You are solving a grade school math problem.",
        "Question:",
        "{question}",
        "Show concise reasoning and end with: Final answer: <number>",
        "~~~",
        "",
        "The current locked main trace uses one reflection template for both rounds:",
        "",
        "~~~text",
        "You are revising an existing solution to a grade school math problem.",
        "Question:",
        "{question}",
        "Current answer:",
        "{current_answer}",
        "Reflection round: {round_id}",
        "Check whether the solution answers the requested quantity and uses the problem's",
        "conditions correctly. Check the arithmetic. Correct any errors you find; if the",
        "solution is correct, retain its answer. Give concise reasoning for the final solution.",
        "End with a final line exactly in this format: Final answer: <number>",
        "After the colon write only the numeric answer, without units, Markdown, or LaTeX.",
        "~~~",
        "",
        "For R1, {round_id}=1 and {current_answer} is the Direct answer. For R2, {round_id}=2 and {current_answer} is the visible answer from R1. These are the prompts used to generate the saved candidates; the newer MATH_REFLECTION_PROMPT_ROUND_1/2 templates in src/prompts.py are not used for this locked trace.",
        "",
        "### Gate/Round sample counts",
        "",
        "| Component | Train | Validation | Test |",
        "|---|---:|---:|---:|",
        f"| Direct-error Gate | {result['protocol']['all_train_rows']} | {result['protocol']['all_validation_rows']} | {result['protocol']['test_rows']} |",
        f"| Flat Direct/R1/R2 MLP control | {result['protocol']['flat_train_rows']} | {result['protocol']['all_validation_rows']} | {result['protocol']['test_rows']} |",
        f"| R1-repairability control | {result['protocol']['repairability_train_rows']} | {result['protocol']['all_validation_rows']} | {result['protocol']['test_rows']} |",
        f"| Round action head | {result['protocol']['complete_train_rows']} complete rows | {result['protocol']['complete_validation_rows']} complete rows | {result['protocol']['test_rows']} |",
        "",
        "The Gate, flat classifier, and repairability control use all rows for which their labels are logically determined. The Round head still requires complete Direct/R1/R2 candidates because its R1-versus-R2 target cannot be inferred from Direct correctness alone.",
        "",
        "### Teacher-control threshold search (10% cap)",
        "",
        "For each seed and layer configuration, the teacher-control Gate threshold is searched on the validation split over the grid {0.00, 0.01, ..., 1.00}. Here p is the predicted positive-class probability for the Direct-error Gate, and a validation example triggers reflection when p >= threshold. A threshold is feasible only when the validation reflection rate is at most the 10% cap. Among feasible thresholds, selection is lexicographic:",
        "",
        "1. maximize validation answer accuracy;",
        "2. among ties, minimize validation average total token cost;",
        "3. among remaining ties, minimize reflection rate;",
        "4. among remaining ties, choose the smaller threshold.",
        "",
        "The selected threshold is then applied to the test scores without test-side threshold recalibration. The same-Gate fixed-R1 and Round comparison uses this one threshold and the same Gate scores within each seed and layer.",
        "",
        "### Main budgeted-policy threshold search",
        "",
        "The separate main budgeted-policy experiment in src/layer_ablation_budget_math.py uses budgets b in {5%, 10%, 20%}. Its Gate score is p_D, the predicted probability of the Direct class; a row is sent to the predicted reflection action only when p_D < threshold, and remains Direct when p_D >= threshold. For each validation checkpoint and budget, candidate thresholds are the union of the grid {0.000, 0.005, ..., 1.000}, the no-reflection endpoint 1.000001, and np.nextafter(p_D, +infinity) for every distinct validation score. The latter event points evaluate every change in the selected question set rather than relying only on the regular grid.",
        "",
        "A candidate is feasible only if its validation reflection rate is at most b and its average total token cost is strictly below the validation independent-candidate baseline (the code variable strong_cost). Among feasible candidates, the code maximizes the tuple (correct answer count, -average total tokens, -reflection rate, -threshold). This rule is applied after every training epoch; the best epoch/checkpoint is retained separately for each budget using the same tuple. The selected threshold and checkpoint are then frozen for test evaluation; no test-side threshold or budget recalibration is performed. This main-policy rule is distinct from the teacher-control 0.01-grid search above.",
        "",
        "### Answer extraction and scoring",
        "",
        "The current math trace calls extract_final_number on the final response. It first checks the text after the last ####; otherwise it searches, in order, for labelled numeric patterns such as Final answer, answer is, answer, therefore, so, or boxed{...} and takes the last matching value. If no labelled value is found, it takes the last numeric token. Signs, commas, and decimal values are accepted; commas are removed before numeric normalization. If no number can be extracted, the prediction is None and is scored incorrect.",
        "",
        "The extracted prediction and gold answer are then normalized by answers_equal. Finite numeric values are considered equal within an absolute tolerance of 1e-6; otherwise normalized strings must match exactly. GSM8K candidate correctness therefore uses numeric equivalence rather than raw string equality.",
        "",
        "## Layer summary",
        "",
        "| k | Flat MLP acc | Flat tokens | Flat reflect | Direct-error Gate acc | Gate tokens | Repairable Gate acc | Same-Gate fixed R1 | Same-Gate Round |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for k in range(1, 7):
        layer = result["layers"][str(k)]
        flat = metric_summary([x["test_answer"] for x in layer["three_class_mlp"]])
        direct = metric_summary([x["test_answer"] for x in layer["gate_target_comparison"]["direct_error"]])
        repair = metric_summary([x["test_answer"] for x in layer["gate_target_comparison"]["r1_repairable"]])
        same_fixed = metric_summary([x["test_fixed_r1"] for x in layer["same_gate_round"]])
        same_round = metric_summary([x["test_round"] for x in layer["same_gate_round"]])
        lines.append(
            f"| {k} | {flat['accuracy_mean']:.4f} +/- {flat['accuracy_sd']:.4f} | "
            f"{flat['avg_token_mean']:.1f} +/- {flat['avg_token_sd']:.1f} | "
            f"{flat['reflection_rate_mean']:.2%} | "
            f"{direct['accuracy_mean']:.4f} +/- {direct['accuracy_sd']:.4f} | "
            f"{direct['avg_token_mean']:.1f} +/- {direct['avg_token_sd']:.1f} | "
            f"{repair['accuracy_mean']:.4f} +/- {repair['accuracy_sd']:.4f} | "
            f"{same_fixed['accuracy_mean']:.4f} / {same_fixed['avg_token_mean']:.1f} | "
            f"{same_round['accuracy_mean']:.4f} / {same_round['avg_token_mean']:.1f} |"
        )
    lines += [
        "",
        "## Same-Gate statistical accounting",
        "",
        "Within each seed and layer, fixed-R1 and Round use the identical Direct-error Gate scores, the identical validation-selected threshold, and therefore the identical per-question Gate decision. The only intervention is the post-trigger action: every triggered item receives fixed R1 in one branch, while the Round head can choose R1 or R2 in the other branch. This makes the comparison paired by question, rather than a comparison of independent samples.",
        "",
        "The relevant question-level correctness contrast is $\\Delta_i=\\mathbf{1}[\\text{Round}_i\\text{ correct}]-\\mathbf{1}[\\text{fixed-R1}_i\\text{ correct}]$; token and reflection contrasts should likewise be computed on the same question IDs. The table reports averages across the three training seeds, and `+/-` is the sample standard deviation across seeds. It is not a question-level confidence interval, and it does not provide a significance test for the branch difference.",
        "",
        "The current teacher-control artifact stores aggregate branch metrics but does not store the paired per-question correctness vectors or calculate a same-Gate McNemar test / paired bootstrap interval. Consequently, small displayed differences, such as the Last-3 Round versus fixed-R1 accuracy difference, are descriptive only and must not be described as statistically significant. A confirmatory analysis would use per-question paired McNemar for binary correctness and a paired question-level bootstrap for accuracy and token differences, with the three seeds treated as repeated training realizations rather than independent test questions.",
        "",
        "## Interpretation",
        "",
        "- The fair flat classifier is evaluated as a direct three-way action predictor on all 5976 logically labelable training rows and is not given a separate reflection-rate threshold.",
        "- Direct-error Gate uses the $y^{\\mathrm{DE}}$ target above; R1-repairability uses all Direct-correct negatives plus observed R1 outcomes for Direct-incorrect rows.",
        "- Same-Gate fixed-R1 versus Round uses the Direct-error Gate trigger scores and one shared validation-selected threshold per seed and layer; only the post-trigger action changes.",
        "- Oracle and matched-rate random reflection are representation-independent fixed-trace controls. Their current evidence is documented in outputs/math_full_test_failure_accounted_v1/evaluation/summary.json.",
        "",
        "## Evidence boundaries",
        "",
        "- These traces and the official test set have been observed during project development.",
        "- Direct-correct rows do not need saved R1/R2 candidates for the flat or repairability labels; only an incorrect Direct row with missing R1 remains unlabelled.",
        "- Token cost includes saved Base/Reflector input and output tokens; local policy forward cost is not converted into generation tokens.",
    ]
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("outputs/math_full_training_v1"))
    parser.add_argument(
        "--test-trace",
        type=Path,
        default=Path("outputs/math_full_test_failure_accounted_v1/policy_test_accounted.jsonl"),
    )
    parser.add_argument("--feature-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
