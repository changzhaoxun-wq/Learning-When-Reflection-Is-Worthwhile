import argparse
import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from src.utils import ensure_dir, read_jsonl, save_json, write_jsonl


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def recompute_candidate_rewards(row: Dict[str, Any], lambda_cost: float) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    new_row = copy.deepcopy(row)
    candidates = new_row.get('candidate_results') or []
    if not candidates:
        raise ValueError(f"row has no candidate_results: {new_row.get('id')}")

    for candidate in candidates:
        correct = _to_float(candidate.get('correct'), 0.0)
        token_cost = _to_float(candidate.get('token_cost'), 0.0)
        candidate['reward'] = correct - lambda_cost * token_cost

    best = max(
        candidates,
        key=lambda candidate: (
            _to_float(candidate.get('reward'), -1e9),
            -_to_float(candidate.get('token_cost'), 1e9),
        ),
    )
    new_row['best_action'] = best.get('action')
    new_row['lambda_cost'] = lambda_cost
    return new_row, best


def iter_relabelled_rows(inputs: Iterable[str], lambda_cost: float, dedupe: bool) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    rows: List[Dict[str, Any]] = []
    input_counts: Dict[str, int] = {}
    seen = set()

    for input_path in inputs:
        path = Path(input_path)
        input_rows = read_jsonl(path)
        input_counts[str(path)] = len(input_rows)
        for row in input_rows:
            key = str(row.get('id') or row.get('question'))
            if dedupe and key in seen:
                continue
            seen.add(key)
            new_row, _ = recompute_candidate_rewards(row, lambda_cost)
            rows.append(new_row)
    return rows, input_counts


def summarize(rows: List[Dict[str, Any]], input_counts: Dict[str, int], output: Path, lambda_cost: float) -> Dict[str, Any]:
    return {
        'output': str(output),
        'n': len(rows),
        'lambda_cost': lambda_cost,
        'inputs': input_counts,
        'label_distribution': dict(Counter(row.get('best_action') for row in rows)),
        'dataset_distribution': dict(Counter(row.get('dataset', 'unknown') for row in rows)),
        'source_distribution': dict(Counter(row.get('source', 'unknown') for row in rows)),
        'difficulty_distribution': dict(Counter(row.get('difficulty', 'unknown') for row in rows)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Recompute candidate rewards and best_action for existing policy datasets.')
    parser.add_argument('--inputs', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--lambda_cost', type=float, required=True)
    parser.add_argument('--dedupe', action='store_true')
    parser.add_argument('--summary_output', default=None)
    args = parser.parse_args()

    output = Path(args.output)
    ensure_dir(output.parent)
    rows, input_counts = iter_relabelled_rows(args.inputs, args.lambda_cost, args.dedupe)
    write_jsonl(rows, output)

    summary = summarize(rows, input_counts, output, args.lambda_cost)
    summary_path = Path(args.summary_output) if args.summary_output else output.with_suffix('.summary.json')
    save_json(summary, summary_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
