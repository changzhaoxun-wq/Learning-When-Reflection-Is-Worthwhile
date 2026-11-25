"""Lock source-specific QA thresholds on development, never on confirmation."""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .evaluate_policy_offline import _candidate_usage, _candidate_utility
from .evaluate_qa_dual_candidate_offline import _mcnemar, _oracle_source
from .prepare_qa_hard_development import ROOT, file_hash, locked_write
from .qa_dual_candidate_policy import CANDIDATE_SOURCES, align_dual_rows, dual_usage, pair_candidates
from .qa_hard_development import check_lock
from .utils import load_json, read_jsonl, save_json, write_jsonl


GRID_POINTS = 21
MAX_INVOCATION_RATE = 0.10
DISABLED = 2.0
RULE_VERSION = "qa_source_margin_development_v1"


def validate_probabilities(gate, candidates):
    gate = np.asarray(gate, dtype="float64")
    candidates = np.asarray(candidates, dtype="float64")
    if gate.ndim != 1 or not len(gate) or candidates.shape != (len(gate), 3):
        raise ValueError("Expected nonempty gate[n] and candidate[n,3]")
    for values in (gate, candidates):
        if not np.all(np.isfinite(values)) or np.any((values < 0) | (values > 1)):
            raise ValueError("Probabilities must be finite and in [0,1]")
    if not np.allclose(candidates[:, 0], 1 - gate, atol=1e-6):
        raise ValueError("Direct score must equal one minus gate error probability")
    return gate, candidates


def route(gate, candidates, thresholds):
    gate, candidates = validate_probabilities(gate, candidates)
    values = np.asarray(thresholds, dtype="float64")
    if values.shape != (3,) or not np.all(np.isfinite(values)):
        raise ValueError("Expected finite [gate, v5 margin, v7 margin] thresholds")
    invoked = gate >= values[0]
    margin = candidates[:, 1:] - candidates[:, 0:1]
    eligible = margin >= values[1:]
    scores = np.where(eligible, candidates[:, 1:], -np.inf)
    best = np.argmax(scores, axis=1) + 1
    selected = np.where(invoked & eligible.any(axis=1), best, 0)
    return selected, invoked


def scan(gate, candidates, correct, f1, direct_tokens, dual_tokens):
    gate, candidates = validate_probabilities(gate, candidates)
    correct, f1 = np.asarray(correct), np.asarray(f1, dtype="float64")
    n = len(gate)
    if correct.shape != (n, 3) or f1.shape != (n, 3):
        raise ValueError("Candidate outcome shapes must match the scores")
    if not np.all(np.isin(correct, [0, 1])) or not np.all(np.isfinite(f1)):
        raise ValueError("Invalid development outcomes")
    if np.any((f1 < 0) | (f1 > 1)):
        raise ValueError("F1 outside [0,1]")
    direct_tokens, dual_tokens = np.asarray(direct_tokens), np.asarray(dual_tokens)
    if direct_tokens.shape != (n,) or dual_tokens.shape != (n,):
        raise ValueError("Token shapes do not match scores")
    if not np.all(np.isfinite(direct_tokens)) or not np.all(np.isfinite(dual_tokens)):
        raise ValueError("Non-finite cost")
    if np.any(direct_tokens < 0) or np.any(dual_tokens < direct_tokens):
        raise ValueError("Dual invocation cannot cost less than direct")

    index = np.arange(n)
    margins = candidates[:, 1:] - candidates[:, 0:1]
    quantiles = np.linspace(0, 1, GRID_POINTS)
    # Strict successors handle quantile ties while keeping a <=10% integer budget.
    gates = np.unique(np.r_[np.nextafter(np.quantile(gate, quantiles), np.inf), DISABLED])
    grids = [np.unique(np.r_[np.quantile(margins[:, j], quantiles), DISABLED]) for j in (0, 1)]
    best_thresholds = [DISABLED] * 3
    direct_correct = int(correct[:, 0].sum())
    best_key = (direct_correct, float(f1[:, 0].sum()), -float(direct_tokens.sum()))
    feasible = 0
    budget = math.floor(MAX_INVOCATION_RATE * n + 1e-9)
    for gate_threshold in gates:
        invoked = gate >= gate_threshold
        if int(invoked.sum()) > budget:
            continue
        cost = float(np.where(invoked, dual_tokens, direct_tokens).sum())
        for v5_threshold in grids[0]:
            for v7_threshold in grids[1]:
                selected, _ = route(gate, candidates, [gate_threshold, v5_threshold, v7_threshold])
                key = (int(correct[index, selected].sum()), float(f1[index, selected].sum()), -cost)
                feasible += 1
                if key > best_key:
                    best_key = key
                    best_thresholds = [float(gate_threshold), float(v5_threshold), float(v7_threshold)]
    fallback = best_key[0] <= direct_correct
    if fallback:
        best_thresholds = [DISABLED] * 3
    selected, invoked = route(gate, candidates, best_thresholds)
    return {
        "thresholds": best_thresholds,
        "fallback_to_direct": bool(fallback),
        "feasible_combinations": feasible,
        "invocation_budget_count": budget,
        "invocation_count": int(invoked.sum()),
        "net_fixes": int(correct[index, selected].sum()) - direct_correct,
        "selected_source": selected,
        "invoked": invoked,
    }


def outcomes(pairs):
    candidates = [pair_candidates(pair) for pair in pairs]
    correct = np.asarray([[int(c["correct"]) for c in row] for row in candidates], dtype="int64")
    f1 = np.asarray([[float(c["answer_f1"]) for c in row] for row in candidates])
    em = np.asarray([[float(c["exact_match"]) for c in row] for row in candidates])
    utility = np.asarray([[_candidate_utility(c) for c in row] for row in candidates])
    usage = np.asarray([[_candidate_usage(c) for c in row] for row in candidates])
    dual = np.asarray([dual_usage(pair) for pair in pairs])
    if not np.allclose(usage[:, :, 0] + usage[:, :, 1], usage[:, :, 2]):
        raise ValueError("Candidate prompt/completion/total mismatch")
    if not np.allclose(dual[:, 0] + dual[:, 1], dual[:, 2]):
        raise ValueError("Dual prompt/completion/total mismatch")
    return correct, f1, em, utility, usage, dual


def summarize(name, selected, invoked, data, dual_charge=True):
    correct, f1, em, utility, usage, dual = data
    n = len(selected)
    index = np.arange(n)
    chosen = correct[index, selected]
    costs = usage[index, selected].copy()
    if dual_charge:
        costs[invoked] = dual[invoked]
    direct = correct[:, 0]
    fixes = int(((direct == 0) & (chosen == 1)).sum())
    harms = int(((direct == 1) & (chosen == 0)).sum())
    return {
        "method": name, "n": n, "accuracy": float(chosen.mean()),
        "standard_em": float(em[index, selected].mean()),
        "standard_f1": float(f1[index, selected].mean()),
        "avg_prompt_tokens": float(costs[:, 0].mean()),
        "avg_completion_tokens": float(costs[:, 1].mean()),
        "avg_total_tokens": float(costs[:, 2].mean()),
        "avg_reward": float((utility[index, selected] - 0.0001 * costs[:, 2]).mean()),
        "invocation_rate": float(invoked.mean()), "adoption_rate": float((selected > 0).mean()),
        "fixes": fixes, "harms": harms, "net_fixes": fixes - harms,
        "fix_rate": fixes / int((direct == 0).sum()) if np.any(direct == 0) else None,
        "harm_rate": harms / int((direct == 1).sum()) if np.any(direct == 1) else None,
        "mcnemar_vs_direct": _mcnemar(direct, chosen),
        "paired_statistics_role": "development diagnostic, not confirmatory significance",
    }


def cached_scores(pairs, decisions):
    by_id = {}
    for row in decisions:
        if row["id"] in by_id:
            raise ValueError("Duplicate probability ID")
        by_id[row["id"]] = row
    ids = [pair["id"] for pair in pairs]
    if len(set(ids)) != len(ids) or set(ids) != set(by_id):
        raise ValueError("Probability IDs do not match the complete development pairs")
    gate = [by_id[key]["gate_probability"] for key in ids]
    candidates = [[by_id[key]["candidate_probabilities"][source] for source in CANDIDATE_SOURCES] for key in ids]
    return (*validate_probabilities(gate, candidates), [by_id[key] for key in ids])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = check_lock(root)
    completion = load_json(root / "generation_complete.json")
    if completion.get("confirmation_evaluated") is not False:
        raise ValueError("Confirmation must remain reserved")
    old, new = read_jsonl(root / "v5.jsonl"), read_jsonl(root / "v7.jsonl")
    pairs = align_dual_rows(old, new, require_complete=True)
    ids = [pair["id"] for pair in pairs]
    expected = set(manifest["development_ids"])
    if len(pairs) != completion["aligned_rows"] or not set(ids) <= expected:
        raise ValueError("Not the frozen development subset")
    if set(ids) & set(manifest["confirmation_ids"]):
        raise ValueError("Confirmation leakage")
    for pair in pairs:
        if pair["old_row"]["initial_call"] != pair["new_row"]["initial_call"]:
            raise ValueError("Base call differs between sources")
    analysis = load_json(root / "reference_analysis.json")
    if (analysis.get("gate_target_name"), analysis.get("ranker_target_name"),
        analysis.get("selection_score_mode")) != ("direct_error", "candidate_correctness", "correctness_margin"):
        raise ValueError("Unexpected reference policy objective")
    gate, scores, decisions = cached_scores(pairs, read_jsonl(root / "reference_decisions.jsonl"))
    data = outcomes(pairs)
    result = scan(gate, scores, data[0], data[1], data[4][:, 0, 2], data[5][:, 2])
    n = len(pairs)
    direct = np.zeros(n, dtype="int64")
    no_calls = np.zeros(n, dtype=bool)
    all_calls = np.ones(n, dtype=bool)
    oracle = np.asarray([_oracle_source(pair) for pair in pairs])
    reference = np.asarray([CANDIDATE_SOURCES.index(r["selected_source"]) for r in decisions])
    reference_calls = np.asarray([r["invoked"] for r in decisions], dtype=bool)
    methods = [
        summarize("direct", direct, no_calls, data),
        summarize("fixed_v5", np.ones(n, dtype="int64"), all_calls, data, False),
        summarize("fixed_v7", np.full(n, 2), all_calls, data, False),
        summarize("frozen_reference", reference, reference_calls, data),
        summarize("source_calibrated", result["selected_source"], result["invoked"], data),
        summarize("dual_correctness_oracle", oracle, oracle > 0, data),
    ]
    provenance = {
        "manifest_sha256": file_hash(root / "manifest.json"),
        "reference_policy": manifest["reference_policy"],
        "reference_policy_sha256": manifest["reference_policy_sha256"],
        "input_sha256": {name: file_hash(root / name) for name in
                         ("v5.jsonl", "v7.jsonl", "reference_decisions.jsonl", "reference_analysis.json")},
        "calibration_code_sha256": file_hash(Path(__file__)),
    }
    lock = {
        "rule_version": RULE_VERSION, "scope": "development_only",
        "gate_threshold": result["thresholds"][0],
        "v5_margin_threshold": result["thresholds"][1],
        "v7_margin_threshold": result["thresholds"][2],
        "fallback_to_direct": result["fallback_to_direct"],
        "score_rule": "candidate_probability - (1 - gate_error_probability)",
        "routing_rule": "highest probability among source-threshold-eligible reflections; ties v5",
        "grid_points": GRID_POINTS, "max_invocation_rate": MAX_INVOCATION_RATE,
        "grid_rule": "21 full-range quantiles; gate strict successors; finite 2.0 disables a route",
        "objective": ["binary_correct_count", "standard_f1_sum", "negative_actual_total_tokens"],
        "fallback_rule": "direct unless positive net binary fixes on development",
        "development_rows": n, "raw_rows": completion["raw_rows"],
        "excluded_rows": completion["raw_rows"] - n, "weight_updates": 0, "llm_calls": 0,
        "confirmation_evaluated": False, **provenance,
    }
    output = root / "source_calibration"
    output.mkdir(exist_ok=True)
    locked_write(output / "policy_lock.json", json.dumps(lock, ensure_ascii=False, indent=2) + "\n")
    report = {
        "methods": methods, "selection": {k: v for k, v in result.items() if k not in ("selected_source", "invoked")},
        "protocol": lock, "warning": "Development-selected results are not independent test evidence.",
    }
    save_json(report, output / "development_report.json")
    write_jsonl([
        {"id": key, "selected_source": CANDIDATE_SOURCES[int(selected)], "invoked": bool(invoked),
         "split": "development", "thresholds_selected_on_this_split": True}
        for key, selected, invoked in zip(ids, result["selected_source"], result["invoked"])
    ], output / "development_decisions.jsonl")
    lines = [
        "# HotpotQA Hard Development Calibration", "",
        f"Raw: {completion['raw_rows']}; aligned: {n}; quarantined: {completion['raw_rows'] - n}.",
        "Frozen encoder weights; no LLM calls. Confirmation has not been evaluated.", "",
        "| Method | Acc@0.8 | EM | F1 | Prompt | Completion | Total | Invoke | Adopt | Fix/Harm |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in methods:
        lines.append(
            f"| {row['method']} | {row['accuracy']:.4f} | {row['standard_em']:.4f} | {row['standard_f1']:.4f} | "
            f"{row['avg_prompt_tokens']:.1f} | {row['avg_completion_tokens']:.1f} | {row['avg_total_tokens']:.1f} | "
            f"{row['invocation_rate']:.4f} | {row['adoption_rate']:.4f} | {row['fixes']}/{row['harms']} |"
        )
    lines += ["", f"Fallback to direct: {result['fallback_to_direct']}.",
              f"Thresholds (gate, v5, v7): {result['thresholds']}.",
              "Invocation charges both candidates even if direct is retained.",
              "Acc@0.8 is the project's binary F1-threshold metric, not official HotpotQA accuracy.",
              "Metrics and McNemar values are development diagnostics, not confirmatory evidence.",
              "Next: independently generate confirmation candidates with the same protocol, then apply the locked rule once."]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
