from __future__ import annotations

import argparse
import copy
from collections import Counter, defaultdict
from typing import Any, Dict, List

import numpy as np

from .evaluate_policy_offline import _add, _candidate_utility, _rows_from_stats
from .qa_dual_candidate_policy import (
    CANDIDATE_SOURCES,
    _correct,
    align_dual_rows,
    dual_usage,
    pair_candidates,
    predict_dual_candidate,
    reflection_decision_score,
)
from .utils import load_pickle, print_table, read_jsonl, save_json, write_jsonl


METHODS = [
    "direct",
    "fixed_v5_reflect_1",
    "fixed_v7_reflect_1",
    "dual_candidate_ranker",
    "dual_candidate_oracle",
]


def _charged_candidate(
    selected: Dict[str, Any],
    pair: Dict[str, Any],
    invoked: bool,
) -> Dict[str, Any]:
    result = copy.deepcopy(selected)
    if not invoked:
        return result
    prompt, completion, total = dual_usage(pair)
    result["prompt_tokens"] = prompt
    result["completion_tokens"] = completion
    result["token_cost"] = total
    result["reflected_rounds"] = 2
    metadata = copy.deepcopy(result.get("metadata") or {})
    metadata["dual_candidate_token_accounting"] = {
        "base_calls": 1,
        "reflection_calls": 2,
        "formula": "base + v5_reflection_delta + v7_reflection_delta",
    }
    result["metadata"] = metadata
    return result


def _oracle_source(pair: Dict[str, Any]) -> int:
    candidates = pair_candidates(pair)
    if _correct(candidates[0]):
        return 0
    repaired = [index for index in (1, 2) if _correct(candidates[index])]
    if not repaired:
        return 0
    return max(repaired, key=lambda index: _candidate_utility(candidates[index]))


def _transition_counts(
    pairs: List[Dict[str, Any]], selected_source: np.ndarray
) -> Dict[str, int]:
    counts = {
        "fixes": 0,
        "harms": 0,
        "unchanged_correct": 0,
        "unchanged_wrong": 0,
    }
    for pair, source_index in zip(pairs, selected_source):
        candidates = pair_candidates(pair)
        direct_correct = _correct(candidates[0])
        selected_correct = _correct(candidates[int(source_index)])
        if not direct_correct and selected_correct:
            counts["fixes"] += 1
        elif direct_correct and not selected_correct:
            counts["harms"] += 1
        elif direct_correct:
            counts["unchanged_correct"] += 1
        else:
            counts["unchanged_wrong"] += 1
    counts["net_fixes"] = counts["fixes"] - counts["harms"]
    return counts


def _mcnemar(direct: np.ndarray, selected: np.ndarray) -> Dict[str, Any]:
    direct_only = int(np.sum((direct == 1) & (selected == 0)))
    selected_only = int(np.sum((direct == 0) & (selected == 1)))
    discordant = direct_only + selected_only
    if not discordant:
        p_value = 1.0
    else:
        try:
            from scipy.stats import binomtest

            p_value = float(
                binomtest(
                    min(direct_only, selected_only),
                    discordant,
                    p=0.5,
                    alternative="two-sided",
                ).pvalue
            )
        except ImportError:
            p_value = float("nan")
    return {
        "direct_only_correct": direct_only,
        "method_only_correct": selected_only,
        "discordant": discordant,
        "exact_p_value": p_value,
    }


def _add_runtime_columns(
    rows: List[Dict[str, Any]],
    n: int,
    invocation_counts: Dict[str, int],
    adoption_counts: Dict[str, int],
    reflection_calls: Dict[str, int],
) -> None:
    for row in rows:
        method = str(row["method"])
        row["invocation_rate"] = invocation_counts.get(method, 0) / max(n, 1)
        row["candidate_adoption_rate"] = adoption_counts.get(method, 0) / max(
            n, 1
        )
        row["avg_reflection_calls"] = reflection_calls.get(method, 0) / max(n, 1)


def _diagnostics_from_saved_probabilities(
    bundle: Dict[str, Any],
    old_rows: List[Dict[str, Any]],
    new_rows: List[Dict[str, Any]],
    decision_rows: List[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], Dict[str, np.ndarray]]:
    pairs = align_dual_rows(old_rows, new_rows, require_complete=True)
    decisions_by_id: Dict[str, Dict[str, Any]] = {}
    for row in decision_rows:
        row_id = str(row.get("id") or "")
        if not row_id:
            raise ValueError("Saved probability row has no id")
        if row_id in decisions_by_id:
            raise ValueError(f"Duplicate saved probability id: {row_id}")
        decisions_by_id[row_id] = row

    gate_probability = np.zeros(len(pairs), dtype="float32")
    candidate_probability = np.zeros(
        (len(pairs), len(CANDIDATE_SOURCES)), dtype="float32"
    )
    for index, pair in enumerate(pairs):
        row_id = str(pair["id"])
        if row_id not in decisions_by_id:
            raise ValueError(f"Missing saved probabilities for id={row_id}")
        decision = decisions_by_id[row_id]
        probabilities = decision.get("candidate_probabilities") or {}
        missing = [
            source for source in CANDIDATE_SOURCES if source not in probabilities
        ]
        if missing:
            raise ValueError(
                f"Saved probabilities for id={row_id} are missing {missing}"
            )
        gate_probability[index] = float(decision["gate_probability"])
        for source_index, source in enumerate(CANDIDATE_SOURCES):
            candidate_probability[index, source_index] = float(
                probabilities[source]
            )

    extra_ids = set(decisions_by_id) - {str(pair["id"]) for pair in pairs}
    if extra_ids:
        raise ValueError(
            f"Saved probabilities contain {len(extra_ids)} unaligned ids"
        )
    best_reflection, margin = reflection_decision_score(
        candidate_probability,
        bundle.get("selection_score_mode", "correctness_margin"),
    )
    invoked = gate_probability >= float(bundle["gate_threshold"])
    adopted = invoked & (margin >= float(bundle["margin_threshold"]))
    selected_source = np.where(adopted, best_reflection, 0)
    return pairs, {
        "gate_probability": gate_probability,
        "candidate_probability": candidate_probability,
        "best_reflection": best_reflection,
        "margin": margin,
        "invoked": invoked,
        "adopted": adopted,
        "selected_source": selected_source,
    }


def evaluate(
    old_rows: List[Dict[str, Any]],
    new_rows: List[Dict[str, Any]],
    policy_model: str,
    gate_cache: str | None,
    candidate_cache: str | None,
    lambda_cost: float,
    probability_decisions: List[Dict[str, Any]] | None = None,
) -> tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    Dict[str, Any],
    List[Dict[str, Any]],
]:
    bundle = load_pickle(policy_model)
    if probability_decisions is None:
        pairs, diagnostics = predict_dual_candidate(
            bundle,
            old_rows,
            new_rows,
            gate_cache=gate_cache,
            candidate_cache=candidate_cache,
        )
        probability_source = "model_inference"
    else:
        pairs, diagnostics = _diagnostics_from_saved_probabilities(
            bundle,
            old_rows,
            new_rows,
            probability_decisions,
        )
        probability_source = "saved_decisions"
    selected_source = diagnostics["selected_source"].astype("int64")
    invoked = diagnostics["invoked"].astype(bool)
    adopted = diagnostics["adopted"].astype(bool)
    oracle_source = np.asarray([_oracle_source(pair) for pair in pairs], dtype="int64")
    oracle_invoked = oracle_source > 0

    overall: Dict[str, Dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    by_difficulty: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(float))
    )
    by_source: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(float))
    )
    decisions = []
    action_counts: Counter[str] = Counter()
    direct_labels = np.zeros(len(pairs), dtype="int64")
    method_labels = np.zeros(len(pairs), dtype="int64")

    for index, pair in enumerate(pairs):
        candidates = pair_candidates(pair)
        selected_index = int(selected_source[index])
        oracle_index = int(oracle_source[index])
        ranker_candidate = _charged_candidate(
            candidates[selected_index], pair, bool(invoked[index])
        )
        oracle_candidate = _charged_candidate(
            candidates[oracle_index], pair, bool(oracle_invoked[index])
        )
        selected = {
            "direct": candidates[0],
            "fixed_v5_reflect_1": candidates[1],
            "fixed_v7_reflect_1": candidates[2],
            "dual_candidate_ranker": ranker_candidate,
            "dual_candidate_oracle": oracle_candidate,
        }
        row = pair["new_row"]
        difficulty = str(row.get("difficulty") or "unknown")
        source = str(row.get("source") or row.get("dataset") or "unknown")
        for method, candidate in selected.items():
            _add(overall[method], candidate, lambda_cost)
            _add(by_difficulty[difficulty][method], candidate, lambda_cost)
            _add(by_source[source][method], candidate, lambda_cost)

        action = CANDIDATE_SOURCES[selected_index]
        action_counts[action] += 1
        direct_labels[index] = _correct(candidates[0])
        method_labels[index] = _correct(candidates[selected_index])
        prompt, completion, total = dual_usage(pair)
        decisions.append(
            {
                "id": pair["id"],
                "invoked": bool(invoked[index]),
                "adopted": bool(adopted[index]),
                "selected_source": action,
                "oracle_source": CANDIDATE_SOURCES[oracle_index],
                "gate_probability": float(diagnostics["gate_probability"][index]),
                "candidate_probabilities": {
                    source_name: float(
                        diagnostics["candidate_probability"][index, source_id]
                    )
                    for source_id, source_name in enumerate(CANDIDATE_SOURCES)
                },
                "reflection_margin": float(diagnostics["margin"][index]),
                "direct_correct": int(direct_labels[index]),
                "v5_reflect_1_correct": int(_correct(candidates[1])),
                "v7_reflect_1_correct": int(_correct(candidates[2])),
                "selected_correct": int(method_labels[index]),
                "dual_prompt_tokens": prompt,
                "dual_completion_tokens": completion,
                "dual_total_tokens": total,
                "test_label_used_for_action_selection": False,
            }
        )

    overall_rows = _rows_from_stats({"overall": overall}, None, METHODS)
    n = len(pairs)
    invocation_counts = {
        "direct": 0,
        "fixed_v5_reflect_1": n,
        "fixed_v7_reflect_1": n,
        "dual_candidate_ranker": int(np.sum(invoked)),
        "dual_candidate_oracle": int(np.sum(oracle_invoked)),
    }
    adoption_counts = {
        "direct": 0,
        "fixed_v5_reflect_1": n,
        "fixed_v7_reflect_1": n,
        "dual_candidate_ranker": int(np.sum(adopted)),
        "dual_candidate_oracle": int(np.sum(oracle_invoked)),
    }
    reflection_calls = {
        "direct": 0,
        "fixed_v5_reflect_1": n,
        "fixed_v7_reflect_1": n,
        "dual_candidate_ranker": int(2 * np.sum(invoked)),
        "dual_candidate_oracle": int(2 * np.sum(oracle_invoked)),
    }
    _add_runtime_columns(
        overall_rows, n, invocation_counts, adoption_counts, reflection_calls
    )
    analysis = {
        "n": n,
        "policy_type": bundle.get("policy_type"),
        "training_rows": bundle.get("n_rows"),
        "training_complete": bundle.get("n_complete"),
        "training_positive": bundle.get("n_positive"),
        "training_objective": bundle.get("training_objective"),
        "token_accounting": bundle.get("token_accounting"),
        "gate_target_name": bundle.get("gate_target_name", "direct_error"),
        "ranker_target_name": bundle.get(
            "ranker_target_name", "candidate_correctness"
        ),
        "selection_score_mode": bundle.get(
            "selection_score_mode", "correctness_margin"
        ),
        "gate_threshold": bundle.get("gate_threshold"),
        "margin_threshold": bundle.get("margin_threshold"),
        "max_invocation_rate": bundle.get("max_invocation_rate"),
        "recalibration": bundle.get("recalibration"),
        "probability_source": probability_source,
        "oof_metrics": bundle.get("oof_metrics"),
        "union_accuracy_oracle_oof": bundle.get("union_accuracy_oracle"),
        "action_counts": dict(action_counts),
        "invocation_count": int(np.sum(invoked)),
        "adoption_count": int(np.sum(adopted)),
        "ranker_transitions": _transition_counts(pairs, selected_source),
        "oracle_transitions": _transition_counts(pairs, oracle_source),
        "mcnemar_vs_direct": _mcnemar(direct_labels, method_labels),
        "test_labels_used_for_action_selection": False,
        "llm_calls": 0,
    }
    return (
        overall_rows,
        _rows_from_stats(by_difficulty, "difficulty", METHODS),
        _rows_from_stats(by_source, "source", METHODS),
        analysis,
        decisions,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the saved v5/v7 dual-candidate QA policy offline."
    )
    parser.add_argument("--old_trace", required=True)
    parser.add_argument("--new_trace", required=True)
    parser.add_argument("--policy_model", required=True)
    parser.add_argument("--gate_cache")
    parser.add_argument("--candidate_cache")
    parser.add_argument(
        "--probability_decisions",
        help=(
            "Reuse gate/candidate probabilities from a prior decisions JSONL; "
            "this skips encoder inference and only reapplies saved-model thresholds."
        ),
    )
    parser.add_argument("--lambda_cost", type=float, default=0.0001)
    parser.add_argument("--output", required=True)
    parser.add_argument("--difficulty_output", required=True)
    parser.add_argument("--source_output", required=True)
    parser.add_argument("--analysis_output", required=True)
    parser.add_argument("--decisions_output", required=True)
    args = parser.parse_args()
    overall, difficulty, source, analysis, decisions = evaluate(
        read_jsonl(args.old_trace),
        read_jsonl(args.new_trace),
        args.policy_model,
        args.gate_cache,
        args.candidate_cache,
        args.lambda_cost,
        (
            read_jsonl(args.probability_decisions)
            if args.probability_decisions
            else None
        ),
    )
    save_json(overall, args.output)
    save_json(difficulty, args.difficulty_output)
    save_json(source, args.source_output)
    save_json(analysis, args.analysis_output)
    write_jsonl(decisions, args.decisions_output)
    print(
        print_table(
            overall,
            [
                "method",
                "n",
                "accuracy",
                "avg_answer_f1",
                "avg_prompt_tokens",
                "avg_completion_tokens",
                "avg_token_cost",
                "invocation_rate",
                "candidate_adoption_rate",
                "avg_reward",
            ],
        )
    )
    print(f"ranker={analysis['ranker_transitions']}")
    print(f"oracle={analysis['oracle_transitions']}")
    print(f"mcnemar={analysis['mcnemar_vs_direct']}")
    print(f"Read {analysis['n']} saved aligned traces; no LLM calls were made.")


if __name__ == "__main__":
    main()
