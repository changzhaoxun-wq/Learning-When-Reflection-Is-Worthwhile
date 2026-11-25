"""Aggregate the strict six-layer control ablation."""
from __future__ import annotations

import argparse
import json
import statistics
from itertools import product
from pathlib import Path

SEEDS = (7, 21, 42)
LAYERS = ("last1", "last2", "last3", "last4", "last5", "last6")
TRIALS = ("weighted_ce", "weighted_ce_plus_rank")
BRANCHES = ("gate_round", "gate_r1")
BUDGETS = (0.05, 0.10, 0.20)
METRICS = ("accuracy", "avg_token_cost", "reflection_rate", "fixes", "harms")


def stats(values):
    values = [float(value) for value in values]
    return {"mean": statistics.mean(values), "std": statistics.stdev(values)}


def load_runs(paths):
    expected = {
        (layer, trial, branch, round(budget, 2))
        for layer, trial, branch, budget in product(LAYERS, TRIALS, BRANCHES, BUDGETS)
    }
    runs = []
    for seed, path in zip(SEEDS, paths):
        data = json.loads((path / "results.json").read_text())
        if data["protocol"]["seed"] != seed:
            raise ValueError(f"Expected seed {seed}, got {data['protocol']['seed']}")
        rows = data["results"]
        keys = {
            (row["layer"], row["trial"], row["branch"], round(row["budget"], 2))
            for row in rows
        }
        if keys != expected:
            raise ValueError(f"Incomplete strict result cells for seed {seed}")
        runs.append(data)
    return runs


def aggregate(runs):
    indexes = [
        {
            (row["layer"], row["trial"], row["branch"], round(row["budget"], 2)): row
            for row in data["results"]
        }
        for data in runs
    ]
    groups = []
    for layer, trial, branch, budget in product(LAYERS, TRIALS, BRANCHES, BUDGETS):
        points = [index[(layer, trial, branch, round(budget, 2))] for index in indexes]
        metrics = {}
        for split in ("validation", "test"):
            metrics[split] = {
                metric: stats([point[split][metric] for point in points])
                for metric in METRICS
            }
        groups.append({
            "layer": layer,
            "trial": trial,
            "branch": branch,
            "budget": budget,
            "metrics": metrics,
            "per_seed": [
                {"seed": seed, "epoch": point["epoch"], "threshold": point["threshold"],
                 "validation": {metric: point["validation"][metric] for metric in METRICS},
                 "test": {metric: point["test"][metric] for metric in METRICS}}
                for seed, point in zip(SEEDS, points)
            ],
        })

    loss_deltas = []
    branch_deltas = []
    for layer, branch, budget in product(LAYERS, BRANCHES, BUDGETS):
        rows = []
        for index in indexes:
            ce = index[(layer, TRIALS[0], branch, round(budget, 2))]
            rank = index[(layer, TRIALS[1], branch, round(budget, 2))]
            rows.append({
                split: {metric: rank[split][metric] - ce[split][metric] for metric in METRICS}
                for split in ("validation", "test")
            })
        loss_deltas.append({
            "layer": layer, "branch": branch, "budget": budget,
            "delta_rank_minus_ce": {
                split: {metric: stats([row[split][metric] for row in rows]) for metric in METRICS}
                for split in ("validation", "test")
            },
        })
    for layer, trial, budget in product(LAYERS, TRIALS, BUDGETS):
        rows = []
        for index in indexes:
            full = index[(layer, trial, BRANCHES[0], round(budget, 2))]
            r1 = index[(layer, trial, BRANCHES[1], round(budget, 2))]
            rows.append({
                split: {metric: full[split][metric] - r1[split][metric] for metric in METRICS}
                for split in ("validation", "test")
            })
        branch_deltas.append({
            "layer": layer, "trial": trial, "budget": budget,
            "delta_gate_round_minus_gate_r1": {
                split: {metric: stats([row[split][metric] for row in rows]) for metric in METRICS}
                for split in ("validation", "test")
            },
        })
    return groups, loss_deltas, branch_deltas


def fmt(value, scale=1.0, digits=2):
    return f"{value['mean'] * scale:.{digits}f} +/- {value['std'] * scale:.{digits}f}"


def write_report(path, runs, groups, loss_deltas, branch_deltas):
    baseline = runs[0]["baselines"]
    lines = [
        "# 严格控制协议：层数、CE/CE+Rank 与 Gate-Round/Gate-R1",
        "",
        "所有 Last-1 至 Last-6 均使用同一训练协议、同一候选轨迹、同一训练/验证划分、同一 batch 顺序和 3 个随机种子。",
        "只改变表示层集合、Gate 损失和部署分支；Round 头在同一层配置内由两个损失实验共享。",
        "所有 epoch/threshold 仅在验证集选择；测试轨迹固定且已被历史查看，因此本报告是回顾性严格消融证据。",
        "",
        "## 固定协议",
        "",
        "- Gate: weighted CE 或 weighted CE + 0.5 * pairwise ranking loss。",
        "- Gate-Round 与 Gate-R1 使用同一 Gate checkpoint；Gate-R1 将反思动作固定为 reflect_1。",
        "- 验证筛选同时要求反思率不超过预算，且平均总 token 低于验证集 independent baseline。",
        f"- 当前验证 token cap: {baseline['validation']['independent']['avg_token_cost']:.4f} tokens/question。",
        "- 三个 seed 使用同一层切片顺序和同一批次索引；未生成新的 LLM 轨迹。",
        "",
        "## 固定基线",
        "",
        "| 方法 | Val Acc | Test Acc | Test Token |",
        "|---|---:|---:|---:|",
    ]
    for name in ("direct", "fixed_reflect_1", "fixed_reflect_2", "independent"):
        val = baseline["validation"][name]
        test = baseline["test"].get(name, val)
        lines.append(f"| {name} | {val['accuracy']:.4f} | {test['accuracy']:.4f} | {test['avg_token_cost']:.2f} |")

    lines += ["", "## 严格主结果（均值 +/- 样本标准差）", "",
              "| Layer | Loss | Branch | Budget | Val Acc | Test Acc | Test Token | Test Call | Fix/Harm |",
              "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for group in groups:
        val, test = group["metrics"]["validation"], group["metrics"]["test"]
        lines.append(
            f"| {group['layer']} | {group['trial']} | {group['branch']} | {group['budget']:.0%} | "
            f"{fmt(val['accuracy'], 100)}% | {fmt(test['accuracy'], 100)}% | "
            f"{fmt(test['avg_token_cost'])} | {fmt(test['reflection_rate'], 100)}% | "
            f"{test['fixes']['mean']:.2f}/{test['harms']['mean']:.2f} |"
        )

    lines += ["", "## 严格配对：CE+Rank - CE", "",
              "| Layer | Branch | Budget | Val Acc pp | Test Acc pp | Test Token | Test Fix/Harm |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for row in loss_deltas:
        val = row["delta_rank_minus_ce"]["validation"]
        test = row["delta_rank_minus_ce"]["test"]
        lines.append(
            f"| {row['layer']} | {row['branch']} | {row['budget']:.0%} | "
            f"{fmt(val['accuracy'], 100, 3)} | {fmt(test['accuracy'], 100, 3)} | "
            f"{fmt(test['avg_token_cost'])} | {test['fixes']['mean']:+.2f}/{test['harms']['mean']:+.2f} |"
        )

    lines += ["", "## 严格配对：Gate-Round - Gate-R1", "",
              "| Layer | Loss | Budget | Val Acc pp | Test Acc pp | Test Token | Test Fix/Harm |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for row in branch_deltas:
        val = row["delta_gate_round_minus_gate_r1"]["validation"]
        test = row["delta_gate_round_minus_gate_r1"]["test"]
        lines.append(
            f"| {row['layer']} | {row['trial']} | {row['budget']:.0%} | "
            f"{fmt(val['accuracy'], 100, 3)} | {fmt(test['accuracy'], 100, 3)} | "
            f"{fmt(test['avg_token_cost'])} | {test['fixes']['mean']:+.2f}/{test['harms']['mean']:+.2f} |"
        )

    lines += ["", "## 结论边界", "",
              "- 该报告可以严格比较 Last-1 至 Last-6、CE 与 CE+Rank、Gate-Round 与 Gate-R1；不能与旧的非统一层数消融表直接逐项混合。",
              "- 测试集已经被历史查看，均值和 seed 标准差不构成新的无偏泛化或显著性结论。",
              "- token 统计包含实际执行路径的输入和输出 token；本地策略前向成本未折算为生成 token。", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs=3, type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError(f"Output is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=False)
    runs = load_runs(args.runs)
    groups, loss_deltas, branch_deltas = aggregate(runs)
    payload = {
        "protocol": runs[0]["protocol"],
        "baselines": runs[0]["baselines"],
        "groups": groups,
        "paired_loss": loss_deltas,
        "paired_branch": branch_deltas,
        "strictness": {
            "same_candidate_trace": True,
            "same_batch_orders_across_layers": True,
            "same_gate_initialization_across_losses": True,
            "shared_round_within_layer": True,
            "new_llm_calls": 0,
        },
    }
    (args.output / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    write_report(args.output / "REPORT.md", runs, groups, loss_deltas, branch_deltas)
    (args.output / "status.json").write_text(json.dumps({"stage": "completed", "test_evaluated": True, "new_llm_calls": 0}, indent=2) + "\n")
    print(f"[strict-summary] completed {args.output / 'REPORT.md'}")


if __name__ == "__main__":
    main()
