"""Create paper-ready tables and a quality-cost plot from locked transfer traces."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


DATASETS = ("gsm1k", "svamp", "multiarith")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("outputs/math_transfer_locked_v1"))
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=False, exist_ok=False)
    random_root = args.root / "evaluation" / "random_baseline_v1"
    rows = []
    markdown = ["# Locked transfer summary", "", "策略按测试集结果排序仅用于描述，不用于重新选择论文主模型。", ""]
    markdown += ["| Dataset | Method | Accuracy | Avg total token | Reflection rate | Fix/Harm | Random matched-call accuracy | Acc. diff CI95 |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for dataset in DATASETS:
        transfer = json.loads((args.root / "evaluation" / dataset / "full.json").read_text())
        random = json.loads((random_root / f"{dataset}.json").read_text())
        baselines = transfer["baselines"]
        for method in ("direct", "fixed_reflect_1", "fixed_reflect_2", "independent"):
            b = baselines[method]
            rows.append({"dataset": dataset, "method": method, "accuracy": b["accuracy"], "token": b["avg_token_cost"], "reflection_rate": b["reflection_rate"]})
            markdown.append(f"| {dataset} | {method} | {b['accuracy']:.4f} | {b['avg_token_cost']:.1f} | {b['reflection_rate']:.2%} | {b.get('fixes', 0)}/{b.get('harms', 0)} | -- | -- |")
        ranked = sorted(random["policies"].items(), key=lambda x: (-x[1]["learned"]["accuracy"], x[1]["learned"]["avg_token_cost"]))
        for method, item in ranked[:3]:
            learned = item["learned"]
            ref = item["uniform_gate"]
            ci = ref["accuracy_ci_vs_policy"]
            rows.append({"dataset": dataset, "method": method, "accuracy": learned["accuracy"], "token": learned["avg_token_cost"], "reflection_rate": learned["reflection_rate"]})
            markdown.append(f"| {dataset} | {method} | {learned['accuracy']:.4f} | {learned['avg_token_cost']:.1f} | {learned['reflection_rate']:.2%} | {learned['fixes']}/{learned['harms']} | {ref['accuracy']:.4f} | [{ci['ci95_lower']:+.4f}, {ci['ci95_upper']:+.4f}] |")
        (args.output / f"{dataset}_top3.json").write_text(json.dumps({"top3": ranked[:3], "selection": "descriptive only"}, indent=2))
    (args.output / "summary_table.md").write_text("\n".join(markdown) + "\n")
    (args.output / "summary_table.json").write_text(json.dumps(rows, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"gsm1k": "#0072B2", "svamp": "#D55E00", "multiarith": "#009E73"}
    markers = {"direct": "s", "fixed_reflect_1": "^", "fixed_reflect_2": "v", "independent": "*", "learned": "o"}
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=False, constrained_layout=True)
    for ax, dataset in zip(axes, DATASETS):
        transfer = json.loads((args.root / "evaluation" / dataset / "full.json").read_text())
        random = json.loads((random_root / f"{dataset}.json").read_text())
        b = transfer["baselines"]
        for method in ("direct", "fixed_reflect_1", "fixed_reflect_2", "independent"):
            x, y = b[method]["avg_token_cost"], b[method]["accuracy"]
            ax.scatter(x, y, s=48 if method != "independent" else 90, marker=markers[method], color="#444444", label=method if dataset == "gsm1k" else None, zorder=4)
        for method, item in random["policies"].items():
            l = item["learned"]
            ax.scatter(l["avg_token_cost"], l["accuracy"], s=17, alpha=.35, color=colors[dataset], marker="o", label="learned policies" if dataset == "gsm1k" and method == next(iter(random["policies"])) else None)
        ax.set_title(dataset.upper())
        ax.set_xlabel("Average total tokens")
        ax.grid(True, alpha=.22)
        ax.set_ylim(max(0, min(.84, b["direct"]["accuracy"] - .10)), 1.01)
    axes[0].set_ylabel("Accuracy")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(.5, 1.08))
    fig.savefig(args.output / "transfer_quality_cost.pdf")
    fig.savefig(args.output / "transfer_quality_cost.png", dpi=300)
    (args.output / "figure_caption.md").write_text("迁移数据集上的质量—成本散点图。灰色点为 Direct、固定一轮/两轮反思和独立强模型；彩色半透明点为冻结策略在不同反思预算下的结果。横轴包含输入与输出 token，点的颜色只区分数据集，不表示统计显著性。\n")
    print(f"Wrote summary and figure to {args.output}")


if __name__ == "__main__":
    main()
