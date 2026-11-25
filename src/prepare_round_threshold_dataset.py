from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from src.utils import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Alias reflect_1 as direct to enable binary threshold calibration for round selection.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output_rows = []
    for row in read_jsonl(args.input):
        candidates = {str(item.get("action")): dict(item) for item in row.get("candidate_results", [])}
        first = dict(candidates["reflect_1"])
        first["source_action"] = "reflect_1"
        first["action"] = "direct"
        second = dict(candidates["reflect_2"])
        best = "direct" if int(float(candidates["reflect_1"].get("correct", 0))) >= int(float(candidates["reflect_2"].get("correct", 0))) else "reflect_2"
        new_row = dict(row)
        new_row["candidate_results"] = [first, second]
        new_row["action_space"] = ["direct", "reflect_2"]
        new_row["best_action"] = best
        new_row["policy_stage"] = "reflection_round_threshold"
        output_rows.append(new_row)

    output = Path(args.output)
    write_jsonl(output_rows, output)
    print(f"Wrote {len(output_rows)} threshold-calibrated round examples to {output}")
    print(f"Labels: {dict(Counter(row['best_action'] for row in output_rows))}")


if __name__ == "__main__":
    main()
