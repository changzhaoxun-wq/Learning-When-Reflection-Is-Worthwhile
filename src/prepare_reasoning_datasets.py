import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from src.answer_extractor import extract_final_answer
from src.gsm8k_utils import extract_gold_answer
from src.utils import ensure_dir, write_jsonl


DEFAULT_RAW_ROOT = Path(os.environ.get('RAW_DATA_ROOT', 'data/raw'))
DEFAULT_OUTPUT_ROOT = Path(os.environ.get('PREPARED_DATA_ROOT', 'data/prepared'))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_json(path: Path) -> Any:
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def read_parquet(path: Path) -> List[Dict[str, Any]]:
    import pandas as pd

    return pd.read_parquet(path).to_dict('records')


def dump_split(output_root: Path, dataset: str, split: str, rows: List[Dict[str, Any]]) -> None:
    out = output_root / dataset / f'{split}.jsonl'
    ensure_dir(out.parent)
    write_jsonl(rows, out)
    print(f'[prepare] {dataset}/{split}: {len(rows)} -> {out}')


def maybe_limit(rows: List[Dict[str, Any]], limit: int | None) -> List[Dict[str, Any]]:
    return rows[:limit] if limit is not None else rows


def math_row(
    item_id: str,
    dataset: str,
    question: str,
    answer_text: str,
    gold_answer: str | None = None,
    source: str | None = None,
    difficulty: str | None = None,
    metadata: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    gold = gold_answer or extract_gold_answer(answer_text, dataset=dataset, task_type='math')
    return {
        'id': item_id,
        'dataset': dataset,
        'source': source or dataset,
        'task_type': 'math',
        'question': str(question).strip(),
        'answer': str(answer_text).strip(),
        'gold_answer': gold,
        'difficulty': difficulty,
        'metadata': metadata or {},
    }


def qa_row(
    item_id: str,
    dataset: str,
    question: str,
    answer: str,
    context: str,
    source: str | None = None,
    difficulty: str | None = None,
    metadata: Dict[str, Any] | None = None,
    gold_aliases: List[str] | None = None,
) -> Dict[str, Any]:
    return {
        'id': item_id,
        'dataset': dataset,
        'source': source or dataset,
        'task_type': 'qa',
        'question': str(question).strip(),
        'context': str(context).strip(),
        'answer': str(answer).strip(),
        'gold_answer': str(answer).strip(),
        'gold_aliases': gold_aliases or [],
        'difficulty': difficulty,
        'metadata': metadata or {},
    }


def prepare_gsm_hard(raw_root: Path, output_root: Path, limit: int | None) -> None:
    path = raw_root / 'gsm-hard' / 'gsmhardv2.jsonl'
    rows = []
    for idx, item in enumerate(read_jsonl(path)):
        answer = str(item.get('target', '')).strip()
        rows.append(math_row(
            item_id=f'gsm-hard-test-{idx}',
            dataset='gsm-hard',
            question=item.get('input', ''),
            answer_text=f'#### {answer}',
            gold_answer=answer,
            source='gsm-hard',
            difficulty='hard',
            metadata={'code': item.get('code')},
        ))
    dump_split(output_root, 'gsm-hard', 'test', maybe_limit(rows, limit))


def prepare_gsm_plus(raw_root: Path, output_root: Path, limit: int | None) -> None:
    root = raw_root / 'gsm-plus' / 'data'
    for split_name, filename in [('test', 'test-00000-of-00001.jsonl'), ('testmini', 'testmini-00000-of-00001.jsonl')]:
        path = root / filename
        if not path.exists():
            continue
        rows = []
        for idx, item in enumerate(read_jsonl(path)):
            rows.append(math_row(
                item_id=f'gsm-plus-{split_name}-{idx}',
                dataset='gsm-plus',
                question=item.get('question', ''),
                answer_text=item.get('solution') or f"#### {item.get('answer', '')}",
                gold_answer=str(item.get('answer', '')).strip(),
                source=str(item.get('perturbation_type', 'gsm-plus')),
                metadata={
                    'perturbation_type': item.get('perturbation_type'),
                    'seed_answer': item.get('seed_answer'),
                    'seed_question': item.get('seed_question'),
                },
            ))
        dump_split(output_root, 'gsm-plus', split_name, maybe_limit(rows, limit))


def prepare_gsm_symbolic(raw_root: Path, output_root: Path, limit: int | None) -> None:
    root = raw_root / 'gsm-symbolic'
    all_rows: List[Dict[str, Any]] = []
    for subset in ['main', 'p1', 'p2']:
        path = root / subset / 'test.jsonl'
        if not path.exists():
            continue
        rows = []
        for idx, item in enumerate(read_jsonl(path)):
            rows.append(math_row(
                item_id=f'gsm-symbolic-{subset}-{item.get("id", idx)}',
                dataset='gsm-symbolic',
                question=item.get('question', ''),
                answer_text=item.get('answer', ''),
                source=subset,
                metadata={'original_id': item.get('original_id'), 'instance': item.get('instance')},
            ))
        rows = maybe_limit(rows, limit)
        all_rows.extend(rows)
        dump_split(output_root, 'gsm-symbolic', f'{subset}_test', rows)
    dump_split(output_root, 'gsm-symbolic', 'test', all_rows)


def prepare_math_level_1_3(raw_root: Path, output_root: Path, limit: int | None) -> None:
    root = raw_root / 'hendrycks_math'
    by_split = {'train': [], 'test': []}
    for parquet in sorted(root.glob('*/*.parquet')):
        split = 'train' if parquet.name.startswith('train') else 'test'
        subject = parquet.parent.name
        for idx, item in enumerate(read_parquet(parquet)):
            level = str(item.get('level', ''))
            level_num = next((int(ch) for ch in level if ch.isdigit()), None)
            if level_num is None or level_num > 3:
                continue
            solution = str(item.get('solution', ''))
            rowsplit = by_split[split]
            rowsplit.append(math_row(
                item_id=f'math-l1-3-{split}-{subject}-{idx}',
                dataset='math-level-1-3',
                question=item.get('problem', ''),
                answer_text=solution,
                gold_answer=extract_final_answer(solution),
                source=str(item.get('type') or subject),
                difficulty=level,
                metadata={'level': level, 'type': item.get('type'), 'subject': subject},
            ))
    for split, rows in by_split.items():
        dump_split(output_root, 'math-level-1-3', split, maybe_limit(rows, limit))


def prepare_hotpotqa(raw_root: Path, output_root: Path, limit: int | None, max_context_words: int) -> None:
    # This project evaluates a closed-context reader, not a full-Wikipedia
    # retriever. The distractor split always includes the annotated supporting
    # paragraphs; fullwiki validation only does so for a minority of examples.
    root = raw_root / 'hotpot_qa' / 'distractor'
    files = {'train': sorted(root.glob('train-*.parquet')), 'validation': sorted(root.glob('validation-*.parquet'))}
    for split, paths in files.items():
        rows = []
        for path in paths:
            for item in read_parquet(path):
                answer = item.get('answer')
                if answer is None or str(answer).strip() == '':
                    continue
                rows.append(qa_row(
                    item_id=f"hotpotqa-{split}-{item.get('id', len(rows))}",
                    dataset='hotpotqa',
                    question=item.get('question', ''),
                    answer=answer,
                    context=_format_hotpot_context(item.get('context'), item.get('supporting_facts'), max_context_words),
                    source=str(item.get('type', 'hotpotqa')),
                    difficulty=str(item.get('level', 'medium')),
                    metadata={
                        'type': item.get('type'),
                        'level': item.get('level'),
                        'hotpot_subset': 'distractor',
                    },
                ))
        out_split = 'test' if split == 'validation' else split
        if rows:
            dump_split(output_root, 'hotpotqa', out_split, maybe_limit(rows, limit))
        else:
            print(f'[prepare] hotpotqa/{out_split}: skipped empty or unlabeled split')


def prepare_2wiki(raw_root: Path, output_root: Path, limit: int | None, max_context_words: int) -> None:
    root = raw_root / '2wikimultihopqa'
    mapping = {'train': 'train.json', 'validation': 'dev.json', 'test': 'test.json'}
    for split, filename in mapping.items():
        path = root / filename
        if not path.exists():
            continue
        rows = []
        for idx, item in enumerate(read_json(path)):
            answer = item.get('answer')
            if answer is None or str(answer).strip() == '':
                continue
            rows.append(qa_row(
                item_id=f"2wiki-{split}-{item.get('_id', idx)}",
                dataset='2wikimultihopqa',
                question=item.get('question', ''),
                answer=answer,
                context=_format_2wiki_context(item.get('context'), item.get('supporting_facts'), max_context_words),
                source=str(item.get('type', '2wiki')),
                difficulty='hard',
                metadata={'type': item.get('type'), 'evidences': item.get('evidences')},
            ))
        if rows:
            dump_split(output_root, '2wikimultihopqa', split, maybe_limit(rows, limit))
        else:
            print(f'[prepare] 2wikimultihopqa/{split}: skipped empty or unlabeled split')


def prepare_musique(raw_root: Path, output_root: Path, limit: int | None, max_context_words: int) -> None:
    root = raw_root / 'musique'
    mapping = {'train': 'musique_ans_v1.0_train.jsonl', 'validation': 'musique_ans_v1.0_dev.jsonl', 'test': 'musique_ans_v1.0_dev.jsonl'}
    for split, filename in mapping.items():
        path = root / filename
        if not path.exists():
            continue
        rows = []
        for idx, item in enumerate(read_jsonl(path)):
            if item.get('answerable') is False:
                continue
            answer = item.get('answer')
            if answer is None or str(answer).strip() == '':
                continue
            hops = _musique_hops(item.get('question_decomposition'))
            rows.append(qa_row(
                item_id=f"musique-{split}-{item.get('id', idx)}",
                dataset='musique',
                question=item.get('question', ''),
                answer=answer,
                context=_format_musique_context(item.get('paragraphs'), max_context_words),
                source=f'{hops}hop' if hops else 'musique',
                difficulty='hard' if hops and hops >= 3 else 'medium',
                metadata={'answer_aliases': item.get('answer_aliases'), 'hops': hops},
                gold_aliases=item.get('answer_aliases') or [],
            ))
        dump_split(output_root, 'musique', split, maybe_limit(rows, limit))


def _format_hotpot_context(context: Any, supporting_facts: Any, max_words: int) -> str:
    titles, sentences = _hotpot_context_arrays(context)
    support_titles = set(str(t) for t in _safe_get(supporting_facts, 'title', []))
    paragraphs = []
    for title, sents in zip(titles, sentences):
        text = ' '.join(str(s) for s in list(sents))
        paragraphs.append((0 if str(title) in support_titles else 1, f'{title}: {text}'))
    return _truncate_words('\n'.join(text for _, text in sorted(paragraphs, key=lambda x: x[0])), max_words)


def _hotpot_context_arrays(context: Any) -> tuple[list[Any], list[Any]]:
    if isinstance(context, dict):
        return list(context.get('title', [])), list(context.get('sentences', []))
    return [], []


def _format_2wiki_context(context: Any, supporting_facts: Any, max_words: int) -> str:
    support_titles = {str(sf[0]) for sf in supporting_facts or [] if isinstance(sf, (list, tuple)) and sf}
    paragraphs = []
    for pair in context or []:
        if not isinstance(pair, (list, tuple)) or len(pair) < 2:
            continue
        title, sents = pair[0], pair[1]
        text = ' '.join(str(s) for s in list(sents))
        paragraphs.append((0 if str(title) in support_titles else 1, f'{title}: {text}'))
    return _truncate_words('\n'.join(text for _, text in sorted(paragraphs, key=lambda x: x[0])), max_words)


def _format_musique_context(paragraphs: Any, max_words: int) -> str:
    items = []
    for para in paragraphs or []:
        title = para.get('title', '')
        text = para.get('paragraph_text', '')
        priority = 0 if para.get('is_supporting') else 1
        items.append((priority, f'{title}: {text}'))
    return _truncate_words('\n'.join(text for _, text in sorted(items, key=lambda x: x[0])), max_words)


def _truncate_words(text: str, max_words: int) -> str:
    words = str(text).split()
    if len(words) <= max_words:
        return str(text).strip()
    return ' '.join(words[:max_words]).strip()


def _safe_get(obj: Any, key: str, default: Any) -> Any:
    if isinstance(obj, dict):
        value = obj.get(key, default)
        try:
            return list(value)
        except TypeError:
            return value
    return default


def _musique_hops(question_decomposition: Any) -> int | None:
    if isinstance(question_decomposition, list):
        return len(question_decomposition)
    return None


PREPARE_FNS = {
    'gsm-hard': prepare_gsm_hard,
    'gsm-plus': prepare_gsm_plus,
    'gsm-symbolic': prepare_gsm_symbolic,
    'math-level-1-3': prepare_math_level_1_3,
    'hotpotqa': prepare_hotpotqa,
    '2wikimultihopqa': prepare_2wiki,
    'musique': prepare_musique,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_root', default=str(DEFAULT_RAW_ROOT))
    parser.add_argument('--output_root', default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument('--datasets', nargs='+', default=['all'], choices=['all', *PREPARE_FNS.keys()])
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--max_context_words', type=int, default=900)
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    output_root = Path(args.output_root)
    datasets = list(PREPARE_FNS) if 'all' in args.datasets else args.datasets
    for dataset in datasets:
        fn = PREPARE_FNS[dataset]
        if dataset in {'hotpotqa', '2wikimultihopqa', 'musique'}:
            fn(raw_root, output_root, args.limit, args.max_context_words)
        else:
            fn(raw_root, output_root, args.limit)


if __name__ == '__main__':
    main()
