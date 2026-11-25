"""Report one pre-locked GSM8K policy on held-out transfer traces."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


DATASETS = ("gsm1k", "svamp", "multiarith")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("outputs/math_transfer_locked_v1"))
    ap.add_argument("--policy", default="seed42_weighted_ce_control_original_round_0.10")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=False, exist_ok=False)
    result = {"policy": args.policy, "selection_source": "GSM8K validation protocol", "datasets": {}}
    md = ["# Pre-locked policy transfer report", "",
          f"主策略：`{args.policy}`。该策略由 GSM8K 验证协议预先锁定；本报告不根据迁移集结果选点。", "",
          "| Dataset | Method | Accuracy | Total tokens | Reflection rate | Fix/Harm | Random matched accuracy | Policy - random CI95 |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for dataset in DATASETS:
        full = json.loads((args.root / "evaluation" / dataset / "full.json").read_text())
        data = json.loads((args.root / "evaluation" / "random_baseline_v1" / f"{dataset}.json").read_text())
        item = data["policies"][args.policy]
        result["datasets"][dataset] = item
        for method in ("direct", "fixed_reflect_1", "fixed_reflect_2"):
            b = full["baselines"][method]
            md.append(f"| {dataset} | {method} | {b['accuracy']:.4f} | {b['avg_token_cost']:.1f} | {b['reflection_rate']:.2%} | {b.get('fixes', 0)}/{b.get('harms', 0)} | -- | -- |")
        l, r = item["learned"], item["round_stratified_gate"]
        ci = r["accuracy_ci_vs_policy"]
        md.append(f"| {dataset} | Locked policy | {l['accuracy']:.4f} | {l['avg_token_cost']:.1f} | {l['reflection_rate']:.2%} | {l['fixes']}/{l['harms']} | {r['accuracy']:.4f} | [{ci['ci95_lower']:+.4f}, {ci['ci95_upper']:+.4f}] |")
    result["interpretation"] = {
        "primary_comparison": "locked policy versus Direct and round-stratified matched-call random Gate",
        "test_tuning": False,
        "new_llm_calls": 0,
        "scope": "exploratory because transfer datasets were previously observed"
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2))
    (args.output / "REPORT.md").write_text("\n".join(md) + "\n")
    print(f"Wrote pre-locked transfer report to {args.output}")


if __name__ == "__main__":
    main()
