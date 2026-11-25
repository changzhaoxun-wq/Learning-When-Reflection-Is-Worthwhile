"""Generate held-out QA candidates and apply a development-locked rule once."""

import argparse
import copy
import fcntl
import json
import os
from pathlib import Path

import numpy as np

from .calibrate_qa_hard_development import (
    RULE_VERSION, cached_scores, outcomes, route, summarize,
)
from .evaluate_qa_dual_candidate_offline import _oracle_source, evaluate
from .prepare_qa_hard_development import PROJECT, ROOT, file_hash, locked_write, rows
from .qa_dual_candidate_policy import CANDIDATE_SOURCES, align_dual_rows
from .qa_hard_development import check_lock, preflight, require_service, run_module, validate_trace
from .utils import load_json, read_jsonl, save_json, write_jsonl


BOOTSTRAP_ITERATIONS = 10000
BOOTSTRAP_SEED = 20260910
RAW_HOTPOT = os.environ.get("RAW_HOTPOT_ROOT", "data/hotpotqa/distractor")


def lock_json(path, value):
    locked_write(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def confirmation_config(original, dataset):
    cfg = copy.deepcopy(original)
    for key in ("data_path", "eval_data_path"):
        cfg["data"][key] = str(dataset)
    for key in ("train_data", "eval_data"):
        cfg["experiment_protocol"][key] = str(dataset)
    cfg["experiment_protocol"]["evaluation_role"] = "held_out_confirmation"
    return cfg


def validate_split(manifest, items):
    ids = [item["id"] for item in items]
    if not ids or ids != manifest["confirmation_ids"] or len(set(ids)) != len(ids):
        raise ValueError("Confirmation IDs/order do not match the reserved split")
    if set(ids) & set(manifest["development_ids"]):
        raise ValueError("Development/confirmation overlap")
    return ids


def prepare(development, root):
    manifest = check_lock(development)
    policy_path = development / "source_calibration/policy_lock.json"
    policy = load_json(policy_path)
    if policy["rule_version"] != RULE_VERSION or policy["scope"] != "development_only":
        raise ValueError("Unsupported development policy lock")
    if policy["manifest_sha256"] != file_hash(development / "manifest.json"):
        raise ValueError("Development manifest changed since calibration")
    if policy["reference_policy_sha256"] != manifest["reference_policy_sha256"]:
        raise ValueError("Reference weights do not match the policy lock")
    if policy["calibration_code_sha256"] != file_hash(PROJECT / "src/calibrate_qa_hard_development.py"):
        raise ValueError("Locked routing implementation changed")
    for name, digest in policy["input_sha256"].items():
        if file_hash(development / name) != digest:
            raise ValueError(f"Development calibration input changed: {name}")
    items = list(rows(development / "confirmation.jsonl"))
    ids = validate_split(manifest, items)
    root.mkdir(parents=True, exist_ok=True)
    locked_write(root / "confirmation.jsonl", (development / "confirmation.jsonl").read_text(encoding="utf-8"))
    locked_write(root / "policy_lock.json", policy_path.read_text(encoding="utf-8"))
    configs = {}
    for name in ("direct", "v5", "v7"):
        configs[name] = confirmation_config(load_json(development / f"configs/{name}.json"), root / "confirmation.jsonl")
        lock_json(root / f"configs/{name}.json", configs[name])
    code = (
        "qa_hard_confirmation.py", "qa_hard_development.py", "qa_top_layer_policy.py",
        "qa_dual_candidate_policy.py", "evaluate_qa_dual_candidate_offline.py",
        "evaluate_policy_offline.py", "build_policy_dataset.py", "build_frozen_base_reflector_trace.py",
        "preflight_qa_context.py", "audit_trace_completion.py", "analyze_decoupling_significance.py",
    )
    protocol = {
        "role": "held_out_confirmation", "development_root": str(development),
        "development_manifest_sha256": file_hash(development / "manifest.json"),
        "policy_lock_sha256": file_hash(policy_path), "confirmation_ids": ids,
        "artifacts_sha256": {name: file_hash(root / name) for name in
                             ("confirmation.jsonl", "policy_lock.json", "configs/direct.json", "configs/v5.json", "configs/v7.json")},
        "runner_code_sha256": {name: file_hash(PROJECT / "src" / name) for name in code},
        "frozen_generation_code_sha256": manifest["frozen_code_sha256"],
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS, "bootstrap_seed": BOOTSTRAP_SEED,
        "primary_comparison": "locked source-calibrated policy minus Direct",
        "primary_binary_test": "two-sided exact McNemar on project Acc@0.8",
        "secondary_intervals": "paired percentile bootstrap; descriptive, no multiplicity correction",
        "threshold_search": False, "weight_updates": 0,
        "invocation_cap_note": "10% was a development constraint, not a confirmation hard cap",
    }
    lock_json(root / "manifest.json", protocol)
    return manifest, policy, configs


def generate(root, manifest, configs, workers):
    ids = manifest["confirmation_ids"]
    output = root / "direct.jsonl"
    preflight(root, configs["direct"], manifest, dataset_name="confirmation.jsonl", expected_ids=ids)
    count = validate_trace(output, manifest, ["direct"], complete=False, expected_ids=ids)["rows"] if output.exists() else 0
    if count < len(ids):
        require_service(configs["direct"]["base_llm"])
        run_module("src.build_policy_dataset", "--config", root / "configs/direct.json", "--split", "train",
                   "--actions", "direct", "--output", output, "--resume")
    save_json(validate_trace(output, manifest, ["direct"], expected_ids=ids), root / "audits/direct.json")
    for source in ("v5", "v7"):
        cfg = configs[source]
        preflight(root, cfg, manifest, source, expected_ids=ids)
        output = root / f"{source}_raw.jsonl"
        profile = cfg["qa_reflection_profile"]
        count = validate_trace(output, manifest, ["direct", "reflect_1"], profile, complete=False,
                               expected_ids=ids)["rows"] if output.exists() else 0
        if count < len(ids):
            require_service(cfg["reflector_llm"])
            run_module("src.build_frozen_base_reflector_trace", "--canonical", root / "direct.jsonl",
                       "--config", root / f"configs/{source}.json", "--output", output,
                       "--reflection-actions", "reflect_1", "--max-workers", workers, "--resume")
        save_json(validate_trace(output, manifest, ["direct", "reflect_1"], profile, expected_ids=ids),
                  root / f"audits/{source}.json")
        run_module("src.relabel_qa_policy_dataset", "--input", output, "--output", root / f"{source}.jsonl",
                   "--quarantine_output", root / f"audits/{source}_quarantine.jsonl",
                   "--summary_output", root / f"audits/{source}_relabel.json", "--raw_hotpot_root", RAW_HOTPOT,
                   "--qa_f1_threshold", "0.8", "--lambda_cost", "0.0001")
    old, new = read_jsonl(root / "v5.jsonl"), read_jsonl(root / "v7.jsonl")
    pairs = align_dual_rows(old, new, require_complete=True)
    retained = [pair["id"] for pair in pairs]
    if not retained or not set(retained) <= set(ids):
        raise ValueError("Invalid retained confirmation IDs")
    for pair in pairs:
        if pair["old_row"]["initial_call"] != pair["new_row"]["initial_call"]:
            raise ValueError("Base calls differ between sources")
    save_json({"role": "confirmation", "raw_rows": len(ids), "aligned_rows": len(pairs),
               "excluded_ids": sorted(set(ids) - set(retained))}, root / "generation_complete.json")
    return old, new, pairs


def paired_intervals(differences):
    from .analyze_decoupling_significance import _bootstrap

    output = {}
    for name, values in differences.items():
        values = np.asarray(values, dtype="float64")
        if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
            raise ValueError("Paired differences must be a nonempty finite vector")
        interval = _bootstrap(values, BOOTSTRAP_ITERATIONS, BOOTSTRAP_SEED, chunk_size=250)
        # The uncentered bootstrap tail fraction is not used as a hypothesis-test p-value.
        output[name] = {key: interval[key] for key in ("observed_difference", "ci95_lower", "ci95_upper")}
    return output


def confirm(root, policy, old, new, pairs):
    inference_inputs = {name: file_hash(root / name) for name in ("v5.jsonl", "v7.jsonl", "policy_lock.json", "manifest.json")}
    cache_lock = root / "prediction_lock.json"
    if cache_lock.exists():
        cache = load_json(cache_lock)
        if cache["inputs"] != inference_inputs or cache["decisions_sha256"] != file_hash(root / "reference_decisions.jsonl"):
            raise ValueError("Prediction cache provenance mismatch")
        decisions = read_jsonl(root / "reference_decisions.jsonl")
    else:
        _, _, _, analysis, decisions = evaluate(old, new, policy["reference_policy"], None, None, 0.0001)
        write_jsonl(decisions, root / "reference_decisions.jsonl")
        save_json(analysis, root / "reference_analysis.json")
        lock_json(cache_lock, {"inputs": inference_inputs, "decisions_sha256": file_hash(root / "reference_decisions.jsonl")})
    gate, scores, _ = cached_scores(pairs, decisions)
    thresholds = [policy[key] for key in ("gate_threshold", "v5_margin_threshold", "v7_margin_threshold")]
    # Selection is made before reading candidate outcomes. Never call scan here.
    selected, invoked = route(gate, scores, thresholds)
    data = outcomes(pairs)
    n = len(pairs)
    index = np.arange(n)
    zeros = np.zeros(n, dtype="int64")
    oracle = np.asarray([_oracle_source(pair) for pair in pairs])
    methods = [
        summarize("direct", zeros, zeros.astype(bool), data),
        summarize("fixed_v5", np.ones(n, dtype="int64"), np.ones(n, dtype=bool), data, False),
        summarize("fixed_v7", np.full(n, 2), np.ones(n, dtype=bool), data, False),
        summarize("locked_source_policy", selected, invoked, data),
        summarize("dual_correctness_oracle", oracle, oracle > 0, data),
    ]
    for method in methods:
        method["paired_statistics_role"] = "held-out; primary comparison is locked policy versus Direct; other tests descriptive"
    correct, f1, em, utility, usage, dual = data
    costs = np.where(invoked[:, None], dual, usage[:, 0])
    differences = {
        "accuracy": correct[index, selected] - correct[:, 0],
        "standard_em": em[index, selected] - em[:, 0],
        "standard_f1": f1[index, selected] - f1[:, 0],
        "prompt_tokens": costs[:, 0] - usage[:, 0, 0],
        "completion_tokens": costs[:, 1] - usage[:, 0, 1],
        "total_tokens": costs[:, 2] - usage[:, 0, 2],
        "reward": utility[index, selected] - utility[:, 0] - .0001 * (costs[:, 2] - usage[:, 0, 2]),
    }
    report = {
        "methods": methods, "paired_policy_minus_direct": paired_intervals(differences),
        "generation": load_json(root / "generation_complete.json"), "thresholds": thresholds,
        "threshold_search": False, "weight_updates": 0, "confirmation_evaluated": True,
        "manifest_sha256": file_hash(root / "manifest.json"),
        "prediction_lock_sha256": file_hash(cache_lock),
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS, "bootstrap_seed": BOOTSTRAP_SEED,
        "limitations": ["Filtered official train-hard holdout, not official test",
                        "Acc@0.8 is a custom binary metric, not official EM",
                        "Local selector latency is not represented by LLM token cost",
                        "Secondary intervals are unadjusted; non-significance does not establish equivalence"],
    }
    lock_json(root / "confirmation_report.json", report)
    write_jsonl([{"id": pair["id"], "selected_source": CANDIDATE_SOURCES[int(action)], "invoked": bool(call),
                  "split": "confirmation", "test_label_used_for_action_selection": False}
                 for pair, action, call in zip(pairs, selected, invoked)], root / "confirmation_decisions.jsonl")
    lines = ["# HotpotQA Held-out Confirmation", "", "Locked development thresholds; no training or threshold search.", "",
             "| Method | Acc@0.8 | EM | F1 | Prompt | Completion | Total | Invoke | Fix/Harm |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in methods:
        lines.append(f"| {row['method']} | {row['accuracy']:.4f} | {row['standard_em']:.4f} | {row['standard_f1']:.4f} | "
                     f"{row['avg_prompt_tokens']:.1f} | {row['avg_completion_tokens']:.1f} | {row['avg_total_tokens']:.1f} | "
                     f"{row['invocation_rate']:.4f} | {row['fixes']}/{row['harms']} |")
    lines += ["", f"Raw/retained: {report['generation']['raw_rows']}/{n}.",
              "Dual invocation charges both candidates even when Direct is retained.",
              "Acc@0.8 is not official EM. Oracle is not deployable.",
              "", "Policy minus Direct, paired percentile 95% intervals:"]
    for name, ci in report["paired_policy_minus_direct"].items():
        lines.append(f"- {name}: {ci['observed_difference']:.6f} [{ci['ci95_lower']:.6f}, {ci['ci95_upper']:.6f}]")
    lines += ["", f"Primary exact McNemar: {methods[3]['mcnemar_vs_direct']}."]
    locked_write(root / "REPORT.md", "\n".join(lines) + "\n")
    lock_json(root / "evaluation_complete.json", {
        "confirmation_evaluated": True, "report_sha256": file_hash(root / "confirmation_report.json"),
        "prediction_lock_sha256": file_hash(cache_lock), "manifest_sha256": file_hash(root / "manifest.json"),
    })
    print("\n".join(lines), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-root", type=Path, default=ROOT)
    parser.add_argument("--root", type=Path, default=ROOT.parent / "hotpotqa_hard_confirmation_v1")
    parser.add_argument("--phase", choices=("prepare", "preflight", "all"), default="all")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    root, development = args.root.resolve(), args.development_root.resolve()
    if root == development or args.workers < 1:
        parser.error("Separate output directory and positive workers required")
    root.mkdir(parents=True, exist_ok=True)
    with (root / "pipeline.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest, policy, configs = prepare(development, root)
        if args.phase == "prepare":
            print("[qa-confirm] locked confirmation protocol; no generation", flush=True)
            return
        if args.phase == "preflight":
            preflight(root, configs["direct"], manifest, dataset_name="confirmation.jsonl", expected_ids=manifest["confirmation_ids"])
            return
        if (root / "evaluation_complete.json").exists():
            completion = load_json(root / "evaluation_complete.json")
            for field, name in (("report_sha256", "confirmation_report.json"), ("prediction_lock_sha256", "prediction_lock.json"), ("manifest_sha256", "manifest.json")):
                if completion[field] != file_hash(root / name):
                    raise ValueError("Completed confirmation artifact changed")
            print("[qa-confirm] already evaluated; no regeneration or retuning", flush=True)
            return
        save_json({"stage": "generation", "confirmation_evaluated": False}, root / "status.json")
        try:
            old, new, pairs = generate(root, manifest, configs, args.workers)
            save_json({"stage": "locked_policy_inference", "confirmation_evaluated": False}, root / "status.json")
            confirm(root, policy, old, new, pairs)
            save_json({"stage": "complete", "confirmation_evaluated": True}, root / "status.json")
        except Exception as exc:
            save_json({"stage": "failed", "error": str(exc), "resume_supported": True}, root / "status.json")
            raise


if __name__ == "__main__":
    main()
