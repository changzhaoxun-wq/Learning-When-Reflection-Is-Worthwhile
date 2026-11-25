import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from src.utils import ensure_dir, read_jsonl, write_jsonl, save_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--dedupe', action='store_true')
    parser.add_argument('--summary_output', default=None)
    args = parser.parse_args()

    rows: List[Dict[str, Any]] = []
    seen = set()
    input_counts = {}
    for input_path in args.inputs:
        path = Path(input_path)
        input_rows = read_jsonl(path)
        input_counts[str(path)] = len(input_rows)
        for row in input_rows:
            key = str(row.get('id') or row.get('question'))
            if args.dedupe and key in seen:
                continue
            seen.add(key)
            rows.append(row)

    output = Path(args.output)
    ensure_dir(output.parent)
    write_jsonl(rows, output)

    summary = {
        'output': str(output),
        'n': len(rows),
        'inputs': input_counts,
        'label_distribution': dict(Counter(row.get('best_action') for row in rows)),
        'dataset_distribution': dict(Counter(row.get('dataset', 'unknown') for row in rows)),
        'source_distribution': dict(Counter(row.get('source', 'unknown') for row in rows)),
        'difficulty_distribution': dict(Counter(row.get('difficulty', 'unknown') for row in rows)),
    }
    summary_path = Path(args.summary_output) if args.summary_output else output.with_suffix('.summary.json')
    save_json(summary, summary_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
