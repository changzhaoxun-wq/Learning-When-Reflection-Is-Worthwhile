"""Freeze transfer inputs and sensitivity subsets before new generation."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path

from .gsm8k_utils import normalize_example_item
from .math_full_test_trace import check_generation_protocol
from .math_reflector_pilot import PROJECT, locked_json
from .prepare_math_reflector_pilot import question_key
from .prepare_qa_hard_development import file_hash, locked_write
from .utils import read_jsonl

DATA = Path(os.environ.get('TRANSFER_DATA_ROOT', PROJECT / 'data' / 'transfer'))
SOURCE = PROJECT / 'outputs/math_full_test_v1'
TRAIN = PROJECT / 'outputs/math_full_training_v1'
DATASETS = ('gsm1k', 'svamp', 'multiarith')


def audit_rows(rows, references):
    ids = [r['id'] for r in rows]
    if not rows or len(ids) != len(set(ids)):
        raise ValueError('Empty dataset or duplicate IDs')
    groups = defaultdict(list)
    overlap = {}
    for row in rows:
        if not row['question'].strip() or row.get('gold_answer') in (None, ''):
            raise ValueError('Missing question or gold answer')
        key = question_key(row['question'])
        groups[key].append(row)
        matches = [ref['id'] for ref in references if question_key(ref['question']) == key]
        if matches:
            overlap[row['id']] = matches
    duplicates = [[r['id'] for r in g] for g in groups.values() if len(g) > 1]
    conflicts = [[r['id'] for r in g] for g in groups.values()
                 if len({str(r['gold_answer']) for r in g}) > 1]
    # All rows remain primary. Sensitivity selection depends on text/order, not outcomes.
    unique = [g[0]['id'] for g in groups.values()]
    return {'n': len(rows), 'unique_questions': len(groups),
            'duplicate_question_groups': duplicates, 'conflicting_gold_groups': conflicts,
            'training_validation_overlap': overlap,
            'subsets': {'full': ids, 'unique_question': unique,
                        'nonoverlap_unique': [key for key in unique if key not in overlap]},
            'normalization': 'NFKC + casefold + whitespace collapse; not semantic deduplication'}


def prepare(root):
    protocol = json.loads((SOURCE / 'protocol.json').read_text())
    check_generation_protocol(protocol['training_protocol'])
    lock = json.loads((SOURCE / 'policy_lock.json').read_text())
    if file_hash(SOURCE / 'policy_lock.json') != protocol['policy_lock_sha256']:
        raise ValueError('Source policy lock changed')
    if len(lock['policies']) != 36 or len({p['name'] for p in lock['policies']}) != 36:
        raise ValueError('Require all 36 locked policies')
    for point in lock['policies']:
        if file_hash(Path(point['model'])) != point['sha256']:
            raise ValueError('Locked Gate changed')
    if file_hash(Path(lock['round_model'])) != lock['round_sha256']:
        raise ValueError('Locked Round changed')
    for name, checksum in protocol['source_code_sha256'].items():
        if file_hash(PROJECT / 'src' / name) != checksum:
            raise ValueError(f'Frozen source changed: {name}')
    references = read_jsonl(TRAIN / 'train_items.jsonl') + read_jsonl(TRAIN / 'validation_items.jsonl')
    datasets, all_rows = {}, {}
    for name in DATASETS:
        source = DATA / name / 'test.jsonl'
        rows = [normalize_example_item(r, i, 'test') for i, r in enumerate(read_jsonl(source))]
        all_rows[name] = rows
        audit = audit_rows(rows, references)
        if audit['conflicting_gold_groups']:
            raise ValueError(f'Conflicting duplicate golds require review: {name}')
        datasets[name] = {'source': str(source), 'source_sha256': file_hash(source), **audit}
    cross = []
    for i, left in enumerate(DATASETS):
        for right in DATASETS[i + 1:]:
            lookup = defaultdict(list)
            for row in all_rows[right]:
                lookup[question_key(row['question'])].append(row['id'])
            cross += [{'left_dataset': left, 'left_id': row['id'], 'right_dataset': right,
                       'right_ids': lookup[question_key(row['question'])]}
                      for row in all_rows[left] if question_key(row['question']) in lookup]
    root.mkdir(parents=True, exist_ok=True)
    for name, rows in all_rows.items():
        path = root / name / 'test_items.jsonl'
        locked_write(path, ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
        datasets[name]['items_sha256'] = file_hash(path)
    locked_json(root / 'policy_lock.json', lock)
    locked_json(root / 'generation_protocol.json', protocol['training_protocol'])
    manifest = {
        'stage': 'prepared_not_generated', 'datasets': datasets, 'cross_dataset_overlap': cross,
        'source_policy_lock': str(SOURCE / 'policy_lock.json'),
        'source_policy_lock_sha256': file_hash(SOURCE / 'policy_lock.json'),
        'policy_lock_sha256': file_hash(root / 'policy_lock.json'),
        'generation_protocol_sha256': file_hash(root / 'generation_protocol.json'),
        'reference_files_sha256': {str(TRAIN / f): file_hash(TRAIN / f)
                                  for f in ('train_items.jsonl', 'validation_items.jsonl')},
        'implementation_sha256': file_hash(Path(__file__)),
        'all_36_policies_required': True, 'target_training_or_threshold_tuning': False,
        'old_generation_cache_reuse': False, 'main_model_selected': False,
        'required_candidates': ['direct', 'reflect_1', 'reflect_2', 'independent'],
        'primary_reporting': 'Each complete dataset separately; no silent exclusion or pooled independence claim',
        'sensitivity_reporting': 'Also report unique_question and nonoverlap_unique; first source occurrence retained',
        'uncertainty': 'Bootstrap normalized-question clusters to avoid treating duplicates as independent',
        'failure_rule': 'Retain denominator and paid retries; exhausted or blocked paths get null prediction and zero system credit, not semantic-error labels',
        'generation_requirements': 'All actions for every item including correct Direct; independent still runs after reflection failure; reserve maximum output window',
        'runtime_pending': 'Generation runner, failure-path tests, GPU placement and encoder preflight still required before launch',
        'historical_limit': 'Target datasets were inspected in earlier protocol experiments; not pristine confirmation sets',
    }
    locked_json(root / 'manifest.json', manifest)
    lines = ['# Frozen Math Transfer Preparation', '',
             'Status: prepared only; no new model calls, policy fitting, or transfer results.', '',
             '| Dataset | Full | Unique | Train/validation overlap rows | Nonoverlap unique |',
             '| --- | ---: | ---: | ---: | ---: |']
    for name, d in datasets.items():
        lines.append(f"| {name} | {d['n']} | {d['unique_questions']} | {len(d['training_validation_overlap'])} | {len(d['subsets']['nonoverlap_unique'])} |")
    lines += ['', 'All 36 source policy checkpoints and thresholds remain locked.',
              'Primary results must retain all input rows; sensitivity subsets are frozen before generation.',
              'Text overlap screening does not rule out semantic overlap or pretraining contamination.',
              'Historical transfer scripts/caches are not compatible with this frozen protocol.',
              'Next: implement and test failure-inclusive generation and frozen-policy evaluation, then launch.', '']
    locked_write(root / 'PREPARATION.md', '\n'.join(lines))
    print('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=PROJECT / 'outputs/math_transfer_locked_v1')
    args = parser.parse_args()
    prepare(args.root.resolve())


if __name__ == '__main__':
    main()
