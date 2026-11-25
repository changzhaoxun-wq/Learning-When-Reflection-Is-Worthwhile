"""Offline evaluation for a completed model-scale GSM8K trace."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from .math_gate_cpu_ablation import score_model
from .train_fixed_math_policy import best_action, decision_metrics, extract_features, write_json
from .utils import load_pickle, read_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True,
                        help="Directory containing gate.pkl and round.pkl")
    parser.add_argument("--encoder", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--cache", type=Path, default=None)
    args = parser.parse_args()
    root, policy = args.root.resolve(), args.policy.resolve()
    rows = read_jsonl(root / "policy_test.jsonl")
    if not rows or any(r.get("policy_split") != "test" or r.get("candidate_coverage") != "complete" for r in rows):
        raise ValueError("Complete policy_test.jsonl is required; no partial test evaluation is allowed")
    cache = args.cache or root / "test_feature_cache"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    features, info = extract_features(rows, args.encoder, cache, 2048, 2)
    gate = load_pickle(policy / "gate.pkl")
    round_model = load_pickle(policy / "round.pkl")
    direct_score = score_model(gate, features)[:, 0]
    threshold = float(gate["direct_threshold"])
    round_actions = np.array(round_model["action_labels"])[score_model(round_model, features).argmax(1)]
    actions = np.where(direct_score >= threshold, "direct", round_actions)
    n = len(rows)
    baselines = {name: decision_metrics(rows, [action] * n) for name, action in
                 (("direct", "direct"), ("fixed_reflect_1", "reflect_1"),
                  ("fixed_reflect_2", "reflect_2"), ("independent", "independent"))}
    baselines["oracle_accuracy_first"] = decision_metrics(rows, [best_action(r) for r in rows])
    learned = decision_metrics(rows, actions.tolist())
    output = root / "evaluation" / policy.name
    output.mkdir(parents=True, exist_ok=True)
    result = {"n": n, "policy": str(policy), "threshold": threshold,
              "baselines": baselines, "learned_policy": learned,
              "encoder_info": info, "no_test_calibration": True,
              "token_accounting": "input + output for every selected candidate path"}
    write_json(output / "summary.json", result)
    lines = ["# Model-scale GSM8K supplemental evaluation", "",
             f"Base/Reflector trace root: `{root}`", f"Policy: `{policy}`", "",
             "| Method | Accuracy | Input tokens | Output tokens | Total tokens | Reflection rate | Fixes | Harms |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, metrics in [*baselines.items(), ("learned_policy", learned)]:
        lines.append(f"| {name} | {metrics['accuracy']:.4f} | {metrics['avg_prompt_tokens']:.2f} | "
                     f"{metrics['avg_completion_tokens']:.2f} | {metrics['avg_token_cost']:.2f} | "
                     f"{metrics['reflection_rate']:.4f} | {metrics['fixes']} | {metrics['harms']} |")
    lines += ["", "The policy is selected on the fixed internal validation split; test labels are not used for calibration.",
              "Oracle is a non-deployable upper bound over saved candidates."]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
