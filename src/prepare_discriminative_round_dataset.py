from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from src.utils import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Keep only examples where reflect_1 and reflect_2 differ in correctness.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    selected = []
    for row in read_jsonl(args.input):
        candidates = {str(item.get("action")): item for item in row.get("candidate_results", [])}
        r1 = int(float(candidates["reflect_1"].get("correct", 0)))
        r2 = int(float(candidates["reflect_2"].get("correct", 0)))
        if r1 == r2:
            continue
        new_row = dict(row)
        new_row["candidate_results"] = [dict(candidates["reflect_1"]), dict(candidates["reflect_2"])]
        new_row["action_space"] = ["reflect_1", "reflect_2"]
        new_row["best_action"] = "reflect_1" if r1 > r2 else "reflect_2"
        new_row["policy_stage"] = "reflection_round_discriminative"
        selected.append(new_row)

    output = Path(args.output)
    write_jsonl(selected, output)
    print(f"Wrote {len(selected)} discriminative round examples to {output}")
    print(f"Labels: {dict(Counter(row['best_action'] for row in selected))}")


if __name__ == "__main__":
    main()
