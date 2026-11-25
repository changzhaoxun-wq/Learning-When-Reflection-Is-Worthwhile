"""Offline significance tests for full Base/Reflector decoupling experiments.

The test-set text is encoded once per experiment case and then reused by every
policy seed. No LLM endpoint is called.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch

from src.evaluate_policy_offline import _candidate_usage, _policy_row
from src.neural_policy import _encode_hf_hidden, build_policy_text
from src.utils import load_pickle, read_jsonl


METHODS = ("direct", "fixed_reflect_1", "fixed_reflect_2", "learned_policy", "oracle")
BASELINES = ("direct", "fixed_reflect_1", "fixed_reflect_2", "oracle")


def _candidate(candidate: Mapping[str, Any], lambda_cost: float) -> Dict[str, float]:
    prompt_tokens, completion_tokens, total_tokens = _candidate_usage(dict(candidate))
    correct = float(candidate.get("correct", 0))
    return {
        "correct": correct,
        "prompt_tokens": float(prompt_tokens),
        "completion_tokens": float(completion_tokens),
        "token_cost": float(total_tokens),
        "reflection": float(str(candidate.get("action")) != "direct"),
        "reward": correct - lambda_cost * float(total_tokens),
    }


def _validate_shared_encoder(bundles: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    if not bundles:
        raise ValueError("No policy bundles were supplied")
    reference = bundles[0]
    keys = ("encoder_name", "layers", "pooling", "pure_text", "max_length", "max_text_chars")
    expected = {key: reference.get(key) for key in keys}
    for bundle in bundles[1:]:
        actual = {key: bundle.get(key) for key in keys}
        if actual != expected:
            raise ValueError(f"Policy bundles do not share one encoder configuration: {expected} != {actual}")
    return reference


def _encode_rows(
    rows: Sequence[Dict[str, Any]],
    reference: Mapping[str, Any],
    device: str,
    batch_size: int,
) -> np.ndarray:
    texts = [
        build_policy_text(
            _policy_row(row),
            max_text_chars=int(reference.get("max_text_chars", 8000)),
            include_state=not bool(reference.get("pure_text", True)),
        )
        for row in rows
    ]
    features, _ = _encode_hf_hidden(
        texts,
        encoder_name=str(reference["encoder_name"]),
        max_length=int(reference.get("max_length", 1024)),
        batch_size=batch_size,
        device=device,
        pooling=str(reference.get("pooling", "mean")),
        layer_indices=list(reference.get("layers", [-1])),
    )
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(features, -1e6, 1e6).astype("float32")


def _predict(bundle: Mapping[str, Any], raw_features: np.ndarray, batch_size: int = 512) -> np.ndarray:
    features = bundle["scaler"].transform(raw_features).astype("float32")
    model = bundle["model"].to("cpu").eval()
    logits: List[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            batch = torch.from_numpy(features[start : start + batch_size])
            logits.append(model(batch).cpu())
    all_logits = torch.cat(logits, dim=0)
    actions = list(bundle["action_labels"])
    if len(actions) == 2 and "direct" in actions and "direct_threshold" in bundle:
        probabilities = torch.softmax(all_logits, dim=-1).numpy()
        direct_index = actions.index("direct")
        other_index = 1 - direct_index
        indices = np.where(
            probabilities[:, direct_index] >= float(bundle["direct_threshold"]),
            direct_index,
            other_index,
        )
    else:
        indices = all_logits.argmax(dim=-1).numpy()
    return np.asarray([actions[int(index)] for index in indices], dtype=object)


def _oracle_action(candidates: Mapping[str, Mapping[str, Any]], lambda_cost: float) -> str:
    return max(
        ("direct", "reflect_1", "reflect_2"),
        key=lambda action: (
            float(candidates[action].get("correct", 0))
            - lambda_cost * _candidate_usage(dict(candidates[action]))[2],
            -_candidate_usage(dict(candidates[action]))[2],
        ),
    )


def _seed_records(
    rows: Sequence[Dict[str, Any]],
    raw_features: np.ndarray,
    gate: Mapping[str, Any],
    round_policy: Mapping[str, Any],
    lambda_cost: float,
) -> Dict[str, Dict[str, np.ndarray]]:
    gate_actions = _predict(gate, raw_features)
    round_actions = _predict(round_policy, raw_features)
    records: Dict[str, Dict[str, List[float]]] = {
        method: {
            "correct": [],
            "prompt_tokens": [],
            "completion_tokens": [],
            "token_cost": [],
            "reflection": [],
            "reward": [],
        }
        for method in METHODS
    }
    for index, trace in enumerate(rows):
        candidates = {
            str(item.get("action")): item
            for item in trace.get("candidate_results", [])
        }
        missing = {"direct", "reflect_1", "reflect_2"} - set(candidates)
        if missing:
            raise ValueError(f"Trace {trace.get('id')} is missing candidates: {sorted(missing)}")
        learned_action = str(gate_actions[index])
        if learned_action != "direct":
            learned_action = str(round_actions[index])
            if learned_action not in {"reflect_1", "reflect_2"}:
                learned_action = "reflect_1"
        actions = {
            "direct": "direct",
            "fixed_reflect_1": "reflect_1",
            "fixed_reflect_2": "reflect_2",
            "learned_policy": learned_action,
            "oracle": _oracle_action(candidates, lambda_cost),
        }
        for method, action in actions.items():
            values = _candidate(candidates[action], lambda_cost)
            for metric, value in values.items():
                records[method][metric].append(value)
    return {
        method: {
            metric: np.asarray(values, dtype=np.float64)
            for metric, values in metrics.items()
        }
        for method, metrics in records.items()
    }


def _aggregate(records: Mapping[str, Mapping[str, np.ndarray]]) -> List[Dict[str, Any]]:
    output = []
    for method in METHODS:
        metrics = records[method]
        output.append(
            {
                "method": method,
                "n": int(len(metrics["correct"])),
                "accuracy": float(metrics["correct"].mean()),
                "avg_prompt_tokens": float(metrics["prompt_tokens"].mean()),
                "avg_completion_tokens": float(metrics["completion_tokens"].mean()),
                "avg_token_cost": float(metrics["token_cost"].mean()),
                "reflection_rate": float(metrics["reflection"].mean()),
                "avg_reward": float(metrics["reward"].mean()),
            }
        )
    return output


def _bootstrap(
    differences: np.ndarray,
    iterations: int,
    seed: int,
    chunk_size: int = 1000,
) -> Dict[str, float]:
    values = np.asarray(differences, dtype=np.float64)
    rng = np.random.default_rng(seed)
    sampled_means: List[np.ndarray] = []
    for start in range(0, iterations, chunk_size):
        size = min(chunk_size, iterations - start)
        indices = rng.integers(0, len(values), size=(size, len(values)))
        sampled_means.append(values[indices].mean(axis=1))
    samples = np.concatenate(sampled_means)
    lower, upper = np.quantile(samples, [0.025, 0.975])
    return {
        "observed_difference": float(values.mean()),
        "ci95_lower": float(lower),
        "ci95_upper": float(upper),
        "bootstrap_two_sided_p": float(
            min(1.0, 2.0 * min(np.mean(samples <= 0.0), np.mean(samples >= 0.0)))
        ),
    }


def _mcnemar(learned: np.ndarray, baseline: np.ndarray) -> Dict[str, Any]:
    learned_correct = np.asarray(learned, dtype=np.int64)
    baseline_correct = np.asarray(baseline, dtype=np.int64)
    b = int(np.sum((learned_correct == 1) & (baseline_correct == 0)))
    c = int(np.sum((learned_correct == 0) & (baseline_correct == 1)))
    discordant = b + c
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(discordant, index) for index in range(min(b, c) + 1))
        p_value = min(1.0, 2.0 * tail / (2 ** discordant))
    return {
        "learned_correct_baseline_wrong": b,
        "learned_wrong_baseline_correct": c,
        "discordant_pairs": discordant,
        "exact_two_sided_p": float(p_value),
    }


def _mean_std(values: Sequence[float]) -> Tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=1)) if len(array) > 1 else 0.0


def _analyze_case(
    name: str,
    case_dir: Path,
    seeds: Sequence[int],
    lambda_cost: float,
    device: str,
    encoder_batch_size: int,
    bootstrap_iterations: int,
    bootstrap_seed: int,
) -> Tuple[Dict[str, Any], Dict[int, Dict[str, Dict[str, np.ndarray]]]]:
    rows = read_jsonl(case_dir / "policy_test.jsonl")
    ids = [str(row.get("id")) for row in rows]
    bundles: Dict[int, Tuple[Mapping[str, Any], Mapping[str, Any]]] = {}
    for seed in seeds:
        policy_dir = case_dir / "policy"
        bundles[seed] = (
            load_pickle(policy_dir / f"gate_calibrated_seed{seed}.pkl"),
            load_pickle(policy_dir / f"round_seed{seed}.pkl"),
        )
    reference = _validate_shared_encoder(
        [bundle for pair in bundles.values() for bundle in pair]
    )
    print(f"[significance] {name}: encoding {len(rows)} test rows once", flush=True)
    raw_features = _encode_rows(rows, reference, device, encoder_batch_size)

    records_by_seed: Dict[int, Dict[str, Dict[str, np.ndarray]]] = {}
    per_seed_summary: List[Dict[str, Any]] = []
    mcnemar_rows: List[Dict[str, Any]] = []
    for seed in seeds:
        print(f"[significance] {name}: evaluating seed={seed}", flush=True)
        records = _seed_records(
            rows,
            raw_features,
            bundles[seed][0],
            bundles[seed][1],
            lambda_cost,
        )
        records_by_seed[seed] = records
        for row in _aggregate(records):
            row["seed"] = seed
            per_seed_summary.append(row)
        for baseline in BASELINES:
            test = _mcnemar(
                records["learned_policy"]["correct"],
                records[baseline]["correct"],
            )
            test.update({"seed": seed, "baseline": baseline})
            mcnemar_rows.append(test)

    summary: List[Dict[str, Any]] = []
    for method in METHODS:
        selected = [row for row in per_seed_summary if row["method"] == method]
        item: Dict[str, Any] = {"method": method, "n_seeds": len(selected), "n": len(rows)}
        for metric in (
            "accuracy",
            "avg_prompt_tokens",
            "avg_completion_tokens",
            "avg_token_cost",
            "reflection_rate",
            "avg_reward",
        ):
            mean, std = _mean_std([float(row[metric]) for row in selected])
            item[f"mean_{metric}"] = mean
            item[f"std_{metric}"] = std
        summary.append(item)

    paired_bootstrap: List[Dict[str, Any]] = []
    for baseline_index, baseline in enumerate(BASELINES):
        for metric_index, metric in enumerate(("correct", "token_cost", "reward")):
            # The question is the resampling unit. Each question's difference
            # is first averaged over policy seeds.
            differences = np.stack(
                [
                    records_by_seed[seed]["learned_policy"][metric]
                    - records_by_seed[seed][baseline][metric]
                    for seed in seeds
                ],
                axis=0,
            ).mean(axis=0)
            result = _bootstrap(
                differences,
                bootstrap_iterations,
                bootstrap_seed + 100 * baseline_index + metric_index,
            )
            result.update(
                {
                    "baseline": baseline,
                    "metric": metric,
                    "n_questions": len(differences),
                    "n_policy_seeds": len(seeds),
                    "resampling_unit": "question; seed differences averaged within question",
                }
            )
            paired_bootstrap.append(result)

    report = {
        "case": name,
        "case_dir": str(case_dir),
        "trace": str(case_dir / "policy_test.jsonl"),
        "ids": ids,
        "seeds": list(seeds),
        "lambda_cost": lambda_cost,
        "summary": summary,
        "per_seed_summary": per_seed_summary,
        "paired_bootstrap": paired_bootstrap,
        "mcnemar": mcnemar_rows,
    }
    return report, records_by_seed


def _cross_case_tests(
    case_reports: Mapping[str, Dict[str, Any]],
    case_records: Mapping[str, Dict[int, Dict[str, Dict[str, np.ndarray]]]],
    seeds: Sequence[int],
    iterations: int,
    bootstrap_seed: int,
    shared_base_trace: bool,
) -> List[Dict[str, Any]]:
    names = list(case_reports)
    output: List[Dict[str, Any]] = []
    for left_index in range(len(names)):
        for right_index in range(left_index + 1, len(names)):
            left = names[left_index]
            right = names[right_index]
            if case_reports[left]["ids"] != case_reports[right]["ids"]:
                continue
            item: Dict[str, Any] = {
                "left": left,
                "right": right,
                "interpretation": (
                    "controlled reflector comparison: Base prompt, initial answer, state, and direct candidate are frozen"
                    if shared_base_trace
                    else "exploratory: independently generated traces may include decoding noise"
                ),
                "paired_bootstrap": [],
                "mcnemar": [],
            }
            for metric_index, metric in enumerate(("correct", "token_cost", "reward")):
                differences = np.stack(
                    [
                        case_records[left][seed]["learned_policy"][metric]
                        - case_records[right][seed]["learned_policy"][metric]
                        for seed in seeds
                    ],
                    axis=0,
                ).mean(axis=0)
                result = _bootstrap(
                    differences,
                    iterations,
                    bootstrap_seed + 1000 + metric_index,
                )
                result.update({"metric": metric})
                item["paired_bootstrap"].append(result)
            for seed in seeds:
                result = _mcnemar(
                    case_records[left][seed]["learned_policy"]["correct"],
                    case_records[right][seed]["learned_policy"]["correct"],
                )
                result["seed"] = seed
                item["mcnemar"].append(result)
            output.append(item)
    return output


def _fmt_ci(row: Mapping[str, Any], scale: float = 1.0) -> str:
    return (
        f"{scale * float(row['observed_difference']):.4f} "
        f"[{scale * float(row['ci95_lower']):.4f}, {scale * float(row['ci95_upper']):.4f}]"
    )


def _write_markdown(report: Mapping[str, Any], output: Path) -> None:
    lines = [
        "# Full Decoupling Significance Report",
        "",
        "All tests are offline and reuse saved candidate traces. The question is the paired-bootstrap",
        "resampling unit; per-question policy differences are averaged over seeds before resampling.",
        "McNemar is reported separately for each seed because it requires binary paired outcomes.",
        "",
        "## Aggregate Results",
        "",
        "| Case | Method | Accuracy | Avg Tokens | Reflection Rate | Reward |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for case in report["cases"]:
        for row in case["summary"]:
            lines.append(
                f"| {case['case']} | {row['method']} | "
                f"{row['mean_accuracy']:.4f} +/- {row['std_accuracy']:.4f} | "
                f"{row['mean_avg_token_cost']:.1f} +/- {row['std_avg_token_cost']:.1f} | "
                f"{row['mean_reflection_rate']:.4f} | "
                f"{row['mean_avg_reward']:.4f} +/- {row['std_avg_reward']:.4f} |"
            )

    for case in report["cases"]:
        lines += [
            "",
            f"## {case['case']}",
            "",
            "Differences are learned policy minus baseline. Negative token difference means token saving.",
            "",
            "| Baseline | Accuracy Diff [95% CI] | Token Diff [95% CI] | Reward Diff [95% CI] | McNemar p (seed 7/21/42) |",
            "|---|---:|---:|---:|---:|",
        ]
        by_key = {
            (row["baseline"], row["metric"]): row
            for row in case["paired_bootstrap"]
        }
        for baseline in BASELINES:
            p_values = [
                row["exact_two_sided_p"]
                for row in case["mcnemar"]
                if row["baseline"] == baseline
            ]
            lines.append(
                f"| {baseline} | {_fmt_ci(by_key[(baseline, 'correct')])} | "
                f"{_fmt_ci(by_key[(baseline, 'token_cost')])} | "
                f"{_fmt_ci(by_key[(baseline, 'reward')])} | "
                f"{' / '.join(f'{value:.4g}' for value in p_values)} |"
            )

    if report["cross_case"]:
        if report.get("shared_base_trace"):
            comparison_note = [
                "The cases share an identical frozen Base trace: question, Base prompt, initial answer,",
                "state features, verifier result, and direct candidate are equal for every example.",
            ]
        else:
            comparison_note = [
                "This direct comparison is exploratory because each experiment case was generated in a",
                "separate decoding run; within-case learned-versus-baseline tests are the primary evidence.",
            ]
        lines += [
            "",
            "## Cross-Case Comparison",
            "",
            *comparison_note,
            "",
            "| Left minus Right | Accuracy Diff [95% CI] | Token Diff [95% CI] | Reward Diff [95% CI] |",
            "|---|---:|---:|---:|",
        ]
        for comparison in report["cross_case"]:
            by_metric = {
                row["metric"]: row
                for row in comparison["paired_bootstrap"]
            }
            lines.append(
                f"| {comparison['left']} - {comparison['right']} | "
                f"{_fmt_ci(by_metric['correct'])} | "
                f"{_fmt_ci(by_metric['token_cost'])} | "
                f"{_fmt_ci(by_metric['reward'])} |"
            )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze(
    cases: Sequence[Tuple[str, str]],
    seeds: Sequence[int],
    output_dir: str,
    lambda_cost: float,
    device: str,
    encoder_batch_size: int,
    bootstrap_iterations: int,
    bootstrap_seed: int,
    shared_base_trace: bool = False,
) -> Dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    case_reports: Dict[str, Dict[str, Any]] = {}
    case_records: Dict[str, Dict[int, Dict[str, Dict[str, np.ndarray]]]] = {}
    for name, path in cases:
        case_report, records = _analyze_case(
            name,
            Path(path),
            seeds,
            lambda_cost,
            device,
            encoder_batch_size,
            bootstrap_iterations,
            bootstrap_seed,
        )
        case_reports[name] = case_report
        case_records[name] = records

    cross_case = _cross_case_tests(
        case_reports,
        case_records,
        seeds,
        bootstrap_iterations,
        bootstrap_seed,
        shared_base_trace,
    )
    serializable_cases = []
    for case_report in case_reports.values():
        case_report = dict(case_report)
        case_report.pop("ids", None)
        serializable_cases.append(case_report)
    report = {
        "seeds": list(seeds),
        "lambda_cost": lambda_cost,
        "bootstrap_iterations": bootstrap_iterations,
        "bootstrap_seed": bootstrap_seed,
        "shared_base_trace": shared_base_trace,
        "token_accounting": "prompt_tokens + completion_tokens for the selected action; pure-text local policy adds zero LLM decision tokens",
        "cases": serializable_cases,
        "cross_case": cross_case,
    }
    json_path = output / "SIGNIFICANCE_REPORT.json"
    markdown_path = output / "SIGNIFICANCE_REPORT.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_markdown(report, markdown_path)
    print(f"Wrote {json_path}")
    print(f"Wrote {markdown_path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline significance tests for decoupling experiments.")
    parser.add_argument("--case", action="append", nargs=2, metavar=("NAME", "DIR"), required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 21, 42])
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--lambda_cost", type=float, default=0.0001)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--encoder_batch_size", type=int, default=8)
    parser.add_argument("--bootstrap_iterations", type=int, default=10000)
    parser.add_argument("--bootstrap_seed", type=int, default=20260730)
    parser.add_argument(
        "--shared_base_trace",
        action="store_true",
        help="Mark cross-case tests as controlled comparisons sharing frozen Base/state/direct traces.",
    )
    args = parser.parse_args()
    analyze(
        cases=[(name, path) for name, path in args.case],
        seeds=args.seeds,
        output_dir=args.output_dir,
        lambda_cost=args.lambda_cost,
        device=args.device,
        encoder_batch_size=args.encoder_batch_size,
        bootstrap_iterations=args.bootstrap_iterations,
        bootstrap_seed=args.bootstrap_seed,
        shared_base_trace=args.shared_base_trace,
    )


if __name__ == "__main__":
    main()
