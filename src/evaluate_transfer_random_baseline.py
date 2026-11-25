"""Offline matched-call random Gate controls for locked math transfer traces."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ACTIONS = ("direct", "reflect_1", "reflect_2")
DATASETS = ("gsm1k", "svamp", "multiarith")


def read_jsonl(path):
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def candidates(row):
    value = row["candidate_results"]
    if isinstance(value, dict):
        return value
    return {item["action"]: item for item in value}


def metrics(rows, actions):
    correct = np.array([candidates(rows[i])[a]["correct"] for i, a in enumerate(actions)], float)
    direct = np.array([candidates(r)["direct"]["correct"] for r in rows], float)
    prompt = np.array([candidates(rows[i])[a]["prompt_tokens"] for i, a in enumerate(actions)], float)
    completion = np.array([candidates(rows[i])[a]["completion_tokens"] for i, a in enumerate(actions)], float)
    cost = np.array([candidates(rows[i])[a]["token_cost"] for i, a in enumerate(actions)], float)
    reflected = np.array([a != "direct" for a in actions], float)
    return {
        "accuracy": float(correct.mean()),
        "avg_prompt_tokens": float(prompt.mean()),
        "avg_completion_tokens": float(completion.mean()),
        "avg_token_cost": float(cost.mean()),
        "reflection_rate": float(reflected.mean()),
        "fixes": int(((direct == 0) & (correct == 1)).sum()),
        "harms": int(((direct == 1) & (correct == 0)).sum()),
    }


def bootstrap_diff(policy_values, reference_values, seed=2026092201, iterations=10000):
    diff = np.asarray(policy_values) - np.asarray(reference_values)
    rng = np.random.default_rng(seed)
    out = np.empty(iterations)
    for start in range(0, iterations, 500):
        size = min(500, iterations - start)
        idx = rng.integers(0, len(diff), size=(size, len(diff)))
        out[start:start + size] = diff[idx].mean(axis=1)
    return {
        "difference": float(diff.mean()),
        "ci95_lower": float(np.quantile(out, .025)),
        "ci95_upper": float(np.quantile(out, .975)),
    }


def control(rows, selected):
    """Return analytic per-question expectation for two matched-call controls."""
    n = len(rows)
    selected = np.asarray(selected)
    rounds = np.where(selected == "reflect_2", 2, 1)
    k = int(np.count_nonzero(selected != "direct"))
    k1 = int(np.count_nonzero(selected == "reflect_1"))
    k2 = int(np.count_nonzero(selected == "reflect_2"))
    out = {}
    for name, probs in (
        ("uniform_gate", np.full(n, k / n)),
        ("round_stratified_gate", np.where(rounds == 1, k1 / max(1, int((rounds == 1).sum())),
                                             k2 / max(1, int((rounds == 2).sum())))),
    ):
        expected = []
        for i, row in enumerate(rows):
            cs = candidates(row)
            direct = cs["direct"]
            r1 = cs["reflect_1"]
            r2 = cs["reflect_2"]
            reflected = r1 if rounds[i] == 1 else r2
            p = float(probs[i])
            expected.append({
                "correct": (1-p)*direct["correct"] + p*reflected["correct"],
                "prompt_tokens": (1-p)*direct["prompt_tokens"] + p*reflected["prompt_tokens"],
                "completion_tokens": (1-p)*direct["completion_tokens"] + p*reflected["completion_tokens"],
                "token_cost": (1-p)*direct["token_cost"] + p*reflected["token_cost"],
                "reflection": p,
                "fix": p*(1-direct["correct"])*reflected["correct"],
                "harm": p*direct["correct"]*(1-reflected["correct"]),
            })
        out[name] = {
            "accuracy": float(np.mean([x["correct"] for x in expected])),
            "avg_prompt_tokens": float(np.mean([x["prompt_tokens"] for x in expected])),
            "avg_completion_tokens": float(np.mean([x["completion_tokens"] for x in expected])),
            "avg_token_cost": float(np.mean([x["token_cost"] for x in expected])),
            "reflection_rate": float(np.mean([x["reflection"] for x in expected])),
            "expected_fixes": float(sum(x["fix"] for x in expected)),
            "expected_harms": float(sum(x["harm"] for x in expected)),
            "accuracy_by_question": np.array([x["correct"] for x in expected]),
            "cost_by_question": np.array([x["token_cost"] for x in expected]),
        }
    return out, {"reflection_calls": k, "reflect_1_calls": k1, "reflect_2_calls": k2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=Path("outputs/math_transfer_locked_v1"))
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=False, exist_ok=False)
    report = {"datasets": {}, "test_evaluated": False, "new_llm_calls": 0,
              "control": "matched number of reflection calls; stratified also matches R1/R2 counts"}
    md = ["# 数学迁移：匹配调用数随机 Gate 对照", "",
          "本报告完全复用已保存候选轨迹和冻结策略决策，不产生新的 LLM 调用。随机 Gate 不使用标准答案或 token 成本挑选题。",
          "相同调用数不等于相同 token：随机选择的题目长度不同，因此同时报告总 token。", ""]
    for dataset in DATASETS:
        rows = read_jsonl(args.source / dataset / "policy_test_accounted.jsonl")
        decisions = json.loads((args.source / "evaluation" / dataset / "decisions.json").read_text())
        policies = decisions["policies"]
        direct_actions = ["direct"] * len(rows)
        base = metrics(rows, direct_actions)
        results = {"baselines": {"direct": base}, "policies": {}}
        for name, obj in policies.items():
            actions = obj["actions"]
            if len(actions) != len(rows):
                raise ValueError(f"{dataset}/{name}: action count mismatch")
            learned = metrics(rows, actions)
            controls, counts = control(rows, actions)
            policy_acc = np.array([candidates(rows[i])[a]["correct"] for i, a in enumerate(actions)], float)
            policy_cost = np.array([candidates(rows[i])[a]["token_cost"] for i, a in enumerate(actions)], float)
            item = {"learned": learned, "matched_calls": counts}
            for mode, c in controls.items():
                item[mode] = {k: v for k, v in c.items() if not k.endswith("_by_question")}
                item[mode]["accuracy_ci_vs_policy"] = bootstrap_diff(policy_acc, c["accuracy_by_question"])
                item[mode]["cost_ci_policy_minus_random"] = bootstrap_diff(policy_cost, c["cost_by_question"])
            results["policies"][name] = item
        report["datasets"][dataset] = results
        (args.output / f"{dataset}.json").write_text(json.dumps(results, indent=2))
        md += [f"## {dataset}", "", "| 策略 | Acc. | Token | 调用率 | 随机 Gate Acc. | 同轮数随机 Acc. |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
        for name, item in results["policies"].items():
            l, u, s = item["learned"], item["uniform_gate"], item["round_stratified_gate"]
            md.append(f"| {name} | {l['accuracy']:.4f} | {l['avg_token_cost']:.1f} | {l['reflection_rate']:.2%} | {u['accuracy']:.4f} | {s['accuracy']:.4f} |")
        md += ["", "说明：随机列是固定调用概率下的解析期望；区间保存在 JSON 中，表示逐题配对差异的 bootstrap 95% 区间。", ""]
    (args.output / "summary.json").write_text(json.dumps(report, indent=2))
    (args.output / "REPORT.md").write_text("\n".join(md))
    (args.output / "status.json").write_text(json.dumps({"stage": "completed", "test_evaluated": False, "new_llm_calls": 0}, indent=2))
    print(f"Wrote matched-call random baseline report to {args.output / 'REPORT.md'}")


if __name__ == "__main__":
    main()
