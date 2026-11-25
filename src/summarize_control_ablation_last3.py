"""Aggregate the three-seed Last-3 control ablation."""
from __future__ import annotations

import argparse
import json
import statistics
from itertools import product
from pathlib import Path


SEEDS = (7, 21, 42)
TRIALS = ("weighted_ce", "weighted_ce_plus_rank")
BRANCHES = ("gate_round", "gate_r1")
BUDGETS = (0.05, 0.10, 0.20)
METRICS = ("accuracy", "correct", "avg_prompt_tokens", "avg_completion_tokens",
           "avg_token_cost", "reflection_rate", "avg_reward", "fixes", "harms")


def mean_std(values):
    values = [float(value) for value in values]
    return {"mean": statistics.mean(values), "std": statistics.stdev(values)}


def load_runs(paths):
    runs = []
    for seed, path in zip(SEEDS, paths):
        data = json.loads((path / "results.json").read_text())
        if data["protocol"]["seed"] != seed:
            raise ValueError(f"Expected seed {seed}, got {data['protocol']['seed']}")
        if data["protocol"]["layer_set"] != [-3, -2, -1]:
            raise ValueError("All runs must use Last-3")
        rows = data["results"]
        keys = {(r["trial"], r["branch"], round(r["budget"], 2)) for r in rows}
        expected = {(t, b, round(budget, 2)) for t, b, budget in product(TRIALS, BRANCHES, BUDGETS)}
        if keys != expected:
            raise ValueError(f"Incomplete result cells for seed {seed}")
        runs.append(data)
    return runs


def aggregate(runs):
    indexes = [{(r["trial"], r["branch"], round(r["budget"], 2)): r for r in data["results"]}
               for data in runs]
    groups = []
    for trial, branch, budget in product(TRIALS, BRANCHES, BUDGETS):
        points = [index[(trial, branch, round(budget, 2))] for index in indexes]
        stats = {}
        for split in ("validation", "test"):
            stats[split] = {metric: mean_std([point[split][metric] for point in points])
                            for metric in METRICS}
        groups.append({"trial": trial, "branch": branch, "budget": budget,
                       "metrics": stats,
                       "seeds": [{"seed": seed, **point} for seed, point in zip(SEEDS, points)]})

    paired_loss = []
    for branch, budget in product(BRANCHES, BUDGETS):
        rows = []
        for index in indexes:
            ce = index[(TRIALS[0], branch, round(budget, 2))]
            rank = index[(TRIALS[1], branch, round(budget, 2))]
            rows.append({split: {metric: rank[split][metric] - ce[split][metric]
                                 for metric in METRICS} for split in ("validation", "test")})
        paired_loss.append({"branch": branch, "budget": budget,
                            "delta_rank_minus_ce": {
                                split: {metric: mean_std([row[split][metric] for row in rows])
                                        for metric in METRICS}
                                for split in ("validation", "test")},
                            "per_seed": [{"seed": seed, **row} for seed, row in zip(SEEDS, rows)]})

    paired_round = []
    for trial, budget in product(TRIALS, BUDGETS):
        rows = []
        for index in indexes:
            full = index[(trial, BRANCHES[0], round(budget, 2))]
            r1 = index[(trial, BRANCHES[1], round(budget, 2))]
            rows.append({split: {metric: full[split][metric] - r1[split][metric]
                                 for metric in METRICS} for split in ("validation", "test")})
        paired_round.append({"trial": trial, "budget": budget,
                             "delta_gate_round_minus_gate_r1": {
                                 split: {metric: mean_std([row[split][metric] for row in rows])
                                         for metric in METRICS}
                                 for split in ("validation", "test")},
                             "per_seed": [{"seed": seed, **row} for seed, row in zip(SEEDS, rows)]})
    return groups, paired_loss, paired_round


def fmt(stat, scale=1.0, digits=4):
    return f"{stat['mean'] * scale:.{digits}f} +/- {stat['std'] * scale:.{digits}f}"


def write_report(path, runs, groups, paired_loss, paired_round):
    baseline = runs[0]["baselines"]
    lines = [
        "# Last-3 控制变量消融：CE/CE+Rank 与 Gate-Round/Gate-R1", "",
        "固定 Last-3（6144 维）隐藏向量、候选轨迹、训练/验证划分、预算和 3 个随机种子。",
        "所有 epoch/threshold 只在验证集选择；测试结果使用固定且已被历史查看过的轨迹，因此是回顾性消融证据，不是新的无偏确认实验。",
        "误差为 3 个策略训练 seed 的样本标准差（ddof=1）。", "",
        "## 固定基线", "", "| 方法 | Val Acc | Test Acc | Test Token |", "|---|---:|---:|---:|"]
    for name in ("direct", "fixed_reflect_1", "fixed_reflect_2", "independent"):
        val = baseline["validation"][name]
        test = baseline["test"].get(name, val)
        lines.append(f"| {name} | {val['accuracy']:.4f} | {test['accuracy']:.4f} | {test['avg_token_cost']:.2f} |")

    lines += ["", "## 主结果（均值 +/- 样本标准差）", "",
              "| Loss | Branch | Budget | Val Acc | Test Acc | Test Token | Test Call | Fix/Harm |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for group in groups:
        val, test = group["metrics"]["validation"], group["metrics"]["test"]
        lines.append(f"| {group['trial']} | {group['branch']} | {group['budget']:.0%} | "
                     f"{fmt(val['accuracy'], 100, 2)}% | {fmt(test['accuracy'], 100, 2)}% | "
                     f"{fmt(test['avg_token_cost'], 1, 2)} | {fmt(test['reflection_rate'], 100, 2)}% | "
                     f"{test['fixes']['mean']:.2f}/{test['harms']['mean']:.2f} |")

    lines += ["", "## 损失函数配对差值：CE+Rank 减 CE", "",
              "| Branch | Budget | Val Acc pp | Test Acc pp | Test Token | Test Fix/Harm |",
              "|---|---:|---:|---:|---:|---:|"]
    for row in paired_loss:
        val = row["delta_rank_minus_ce"]["validation"]
        test = row["delta_rank_minus_ce"]["test"]
        lines.append(f"| {row['branch']} | {row['budget']:.0%} | {fmt(val['accuracy'], 100, 3)} | "
                     f"{fmt(test['accuracy'], 100, 3)} | {fmt(test['avg_token_cost'], 1, 2)} | "
                     f"{test['fixes']['mean']:+.2f}/{test['harms']['mean']:+.2f} |")

    lines += ["", "## 轮次结构配对差值：Gate-Round 减 Gate-R1", "",
              "| Loss | Budget | Val Acc pp | Test Acc pp | Test Token | Test Fix/Harm |",
              "|---|---:|---:|---:|---:|---:|"]
    for row in paired_round:
        val = row["delta_gate_round_minus_gate_r1"]["validation"]
        test = row["delta_gate_round_minus_gate_r1"]["test"]
        lines.append(f"| {row['trial']} | {row['budget']:.0%} | {fmt(val['accuracy'], 100, 3)} | "
                     f"{fmt(test['accuracy'], 100, 3)} | {fmt(test['avg_token_cost'], 1, 2)} | "
                     f"{test['fixes']['mean']:+.2f}/{test['harms']['mean']:+.2f} |")

    lines += ["", "## 解释边界", "",
              "- CE+Rank 的排序项系数固定为 0.5；没有根据测试结果搜索系数。",
              "- Gate-Round 与 Gate-R1 使用同一 Gate checkpoint；Gate-R1 只把反思动作固定为 reflect_1，因而用于隔离第二轮选择的价值。",
              "- 结果使用已有候选池，不能证明重新生成候选时排序损失或第二轮反思必然有效。",
              "- 三个 seed 共享相同题目和轨迹，标准差只描述策略训练随机性，不是独立数据样本的置信区间。",
              "- 测试集已被历史查看，不能将本报告写成新的显著性或泛化确认。", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs=3, type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError(f"Output is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=False)
    runs = load_runs(args.runs)
    groups, paired_loss, paired_round = aggregate(runs)
    payload = {"protocol": {"seeds": SEEDS, "layer_set": [-3, -2, -1],
                             "embedding_dim": 6144, "test_evaluated": True,
                             "new_llm_calls": 0,
                             "warning": "Fixed test trace was previously inspected."},
               "baselines": runs[0]["baselines"], "groups": groups,
               "paired_loss": paired_loss, "paired_round": paired_round}
    (args.output / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    write_report(args.output / "REPORT.md", runs, groups, paired_loss, paired_round)
    (args.output / "status.json").write_text(json.dumps({"stage": "completed", "test_evaluated": True,
                                                            "new_llm_calls": 0}, indent=2) + "\n")
    print(f"[control-summary] completed {args.output / 'REPORT.md'}")


if __name__ == "__main__":
    main()
