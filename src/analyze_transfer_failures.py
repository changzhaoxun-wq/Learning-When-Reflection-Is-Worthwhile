"""Offline error-transfer analysis for the pre-locked math policy."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


DATASETS = ("gsm1k", "svamp", "multiarith")


def read_jsonl(path):
    with path.open() as f:
        return [json.loads(x) for x in f if x.strip()]


def cs(row):
    value = row["candidate_results"]
    return value if isinstance(value, dict) else {x["action"]: x for x in value}


def text_len(value):
    return len(re.findall(r"\S+", value or ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("outputs/math_transfer_locked_v1"))
    ap.add_argument("--policy", default="seed42_weighted_ce_control_original_round_0.10")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=False, exist_ok=False)
    report = {"policy": args.policy, "datasets": {}, "new_llm_calls": 0}
    md = ["# 预锁定策略迁移失败分析", "", f"策略：`{args.policy}`。全部结果离线重放，无新的 LLM 调用。", ""]
    md += ["| 数据集 | n | Direct Acc. | Direct 错误 | R1 修复/引错 | R2 修复/引错 | 策略修复/引错 | Gate 漏修复 |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for dataset in DATASETS:
        rows = read_jsonl(args.root / dataset / "policy_test_accounted.jsonl")
        decisions = json.loads((args.root / "evaluation" / dataset / "decisions.json").read_text())
        actions = decisions["policies"][args.policy]["actions"]
        probs = np.array([float(x) for x in decisions["policies"][args.policy]["direct_probability"] if x is not None and np.isfinite(x)])
        direct = np.array([cs(r)["direct"]["correct"] for r in rows], int)
        r1 = np.array([cs(r)["reflect_1"]["correct"] for r in rows], int)
        r2 = np.array([cs(r)["reflect_2"]["correct"] for r in rows], int)
        chosen = np.array([cs(rows[i])[actions[i]]["correct"] for i in range(len(rows))], int)
        direct_wrong = direct == 0
        direct_right = direct == 1
        gate_reflect = np.array([a != "direct" for a in actions])
        best_reflect = np.maximum(r1, r2)
        summary = {
            "n": len(rows),
            "direct_accuracy": float(direct.mean()),
            "policy_accuracy": float(chosen.mean()),
            "direct_wrong": int(direct_wrong.sum()),
            "r1_fixes": int((direct_wrong & (r1 == 1)).sum()),
            "r1_harms": int((direct_right & (r1 == 0)).sum()),
            "r2_fixes": int((direct_wrong & (r2 == 1)).sum()),
            "r2_harms": int((direct_right & (r2 == 0)).sum()),
            "policy_fixes": int((direct_wrong & (chosen == 1)).sum()),
            "policy_harms": int((direct_right & (chosen == 0)).sum()),
            "gate_reflections": int(gate_reflect.sum()),
            "gate_missed_repairable": int((direct_wrong & (best_reflect == 1) & ~gate_reflect).sum()),
            "reflection_unrepairable": int((direct_wrong & (best_reflect == 0)).sum()),
            "selected_wrong_round": int((direct_wrong & gate_reflect & (chosen == 0) & (best_reflect == 1)).sum()),
            "question_words_mean": float(np.mean([text_len(r.get("question", "")) for r in rows])),
            "initial_words_mean": float(np.mean([text_len(r.get("initial_answer", "")) for r in rows])),
            "gate_score_observed_n": int(len(probs)),
            "gate_score_mean": float(np.mean(probs)) if len(probs) else None,
            "gate_score_p25": float(np.quantile(probs, .25)) if len(probs) else None,
            "gate_score_p50": float(np.quantile(probs, .50)) if len(probs) else None,
            "gate_score_p75": float(np.quantile(probs, .75)) if len(probs) else None,
            "reflection_rate": float(gate_reflect.mean()),
        }
        report["datasets"][dataset] = summary
        md.append(f"| {dataset} | {summary['n']} | {summary['direct_accuracy']:.4f} | {summary['direct_wrong']} | "
                  f"{summary['r1_fixes']}/{summary['r1_harms']} | {summary['r2_fixes']}/{summary['r2_harms']} | "
                  f"{summary['policy_fixes']}/{summary['policy_harms']} | {summary['gate_missed_repairable']} |")
    md += ["", "## 解释", "",
           "- `修复/引错` 均相对 Direct 定义；修复是 Direct 错而候选正确，引错是 Direct 正而候选错误。",
           "- `Gate 漏修复` 是 Direct 错、至少一个反思候选正确，但策略仍选择 Direct 的题。",
           "- `反思不可修复` 是 Direct 错且 R1/R2 都错误；这类问题不是 Gate 单独能够解决的。",
           "- `selected_wrong_round` 是已触发反思但选择了错误轮次、另一轮可修复的样本。",
           "- Gate 分数是策略输出分数，不是已校准的错误概率；长度统计是描述性诊断，不是人工难度标签。", ""]
    report["interpretation"] = "diagnostic only; no test-set tuning"
    (args.output / "summary.json").write_text(json.dumps(report, indent=2))
    (args.output / "REPORT.md").write_text("\n".join(md) + "\n")
    print(f"Wrote transfer failure analysis to {args.output}")


if __name__ == "__main__":
    main()
