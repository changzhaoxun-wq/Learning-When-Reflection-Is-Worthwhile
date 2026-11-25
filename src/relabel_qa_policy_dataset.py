from __future__ import annotations

import argparse
import copy
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from .answer_extractor import normalize_text_answer
from .reward import compute_match_metrics, compute_utility_reward
from .utils import read_jsonl, save_json, write_jsonl


def _original_hotpot_id(row_id: Any) -> str:
    value = str(row_id or "")
    for prefix in ("hotpotqa-train-", "hotpotqa-test-", "hotpotqa-validation-"):
        if value.startswith(prefix):
            return value[len(prefix):]
    return value


def _entity_like(value: Any) -> bool:
    text = str(value or "").strip()
    if re.search(r"\d", text):
        return False
    words = re.findall(r"[A-Za-z][A-Za-z.'-]*|[A-Z]{2,}", text)
    if not words or len(words) > 8 or re.search(r"[!?;]", text):
        return False
    if normalize_text_answer(text) in {"yes", "no"}:
        return False
    capitalized = sum(word[0].isupper() or word.isupper() for word in words)
    return capitalized >= max(1, len(words) // 2)


def _dedupe_aliases(values: Iterable[Any], gold: Any) -> list[str]:
    gold_norm = normalize_text_answer(gold)
    output: list[str] = []
    seen = {gold_norm} if gold_norm else set()
    for value in values:
        text = str(value or "").strip()
        key = normalize_text_answer(text)
        if text and key and key not in seen:
            seen.add(key)
            output.append(text)
    return output


def _context_aliases(item: dict[str, Any], gold: str) -> list[str]:
    if not _entity_like(gold):
        return []
    context = item.get("context") or {}
    if not isinstance(context, dict):
        return []
    titles = list(context.get("title") or [])
    gold_norm = normalize_text_answer(gold) or ""
    aliases: list[str] = []
    for title in titles:
        title_text = str(title or "").strip()
        if not _entity_like(title_text):
            continue
        title_norm = normalize_text_answer(title_text) or ""
        title_base = re.sub(r"\s*\([^()]*\)\s*$", "", title_text).strip()
        gold_base = re.sub(r"\s*\([^()]*\)\s*$", "", gold).strip()
        title_base_norm = normalize_text_answer(title_base) or ""
        gold_base_norm = normalize_text_answer(gold_base) or ""
        has_disambiguator = title_base != title_text or gold_base != gold
        if has_disambiguator and title_base_norm and title_base_norm == gold_base_norm:
            aliases.append(title_text)
    return _dedupe_aliases(aliases, gold)


def load_hotpot_context_aliases(
    raw_root: Path,
    rows: list[dict[str, Any]],
) -> dict[str, list[str]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required to derive aliases from raw HotpotQA parquet") from exc

    targets = {
        _original_hotpot_id(row.get("id")): (str(row.get("id")), str(row.get("gold_answer") or ""))
        for row in rows
    }
    result: dict[str, list[str]] = {}
    found: set[str] = set()
    paths = sorted(raw_root.glob("train-*.parquet")) + sorted(raw_root.glob("validation-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No HotpotQA parquet files found under {raw_root}")
    for path in paths:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=256, columns=["id", "context"]):
            for item in batch.to_pylist():
                original_id = str(item.get("id") or "")
                target = targets.get(original_id)
                if target is None:
                    continue
                row_id, gold = target
                result[row_id] = _context_aliases(item, gold)
                found.add(original_id)
        if len(found) == len(targets):
            break
    missing = sorted(set(targets) - found)
    if missing:
        raise RuntimeError(f"Raw HotpotQA lookup missed {len(missing)} trace IDs; first={missing[:3]}")
    return result


def suspect_gold_reasons(row: dict[str, Any]) -> list[str]:
    gold = str(row.get("gold_answer") or "").strip()
    word_count = len(gold.split())
    reasons: list[str] = []
    if word_count > 15:
        reasons.append("gold_exceeds_15_words")
    if str(row.get("source") or "").lower() == "comparison" and word_count > 8:
        reasons.append("comparison_gold_exceeds_8_words")
    return reasons


def _score_candidate(
    candidate: dict[str, Any],
    gold: str,
    provided_aliases: list[str],
    derived_aliases: list[str],
    qa_f1_threshold: float,
    lambda_cost: float,
) -> dict[str, Any]:
    output = copy.deepcopy(candidate)
    pred = output.get("pred")
    standard = compute_match_metrics(
        pred,
        gold,
        task_type="qa",
        gold_aliases=provided_aliases,
        qa_f1_threshold=qa_f1_threshold,
    )
    alias_aware = compute_match_metrics(
        pred,
        gold,
        task_type="qa",
        gold_aliases=[*provided_aliases, *derived_aliases],
        qa_f1_threshold=qa_f1_threshold,
    )
    utility = float(alias_aware["enhanced_answer_f1"])
    token_cost = int(float(output.get("token_cost", 0)))
    output.update(
        correct=int(standard["correct"]),
        exact_match=float(standard["exact_match"]),
        answer_f1=float(standard["answer_f1"]),
        matched_gold=standard.get("matched_gold"),
        alias_correct=int(alias_aware["correct"]),
        alias_exact_match=float(alias_aware["exact_match"]),
        alias_answer_f1=float(alias_aware["answer_f1"]),
        alias_matched_gold=alias_aware.get("matched_gold"),
        enhanced_correct=int(alias_aware["enhanced_correct"]),
        enhanced_exact_match=float(alias_aware["enhanced_exact_match"]),
        enhanced_answer_f1=utility,
        enhanced_matched_gold=alias_aware.get("enhanced_matched_gold"),
        qa_utility_score=utility,
        reward=compute_utility_reward(utility, token_cost, lambda_cost),
        reward_metric="alias_aware_enhanced_token_f1",
    )
    return output


def relabel_rows(
    rows: list[dict[str, Any]],
    aliases_by_id: dict[str, list[str]],
    qa_f1_threshold: float,
    lambda_cost: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    clean: list[dict[str, Any]] = []
    quarantine: list[dict[str, Any]] = []
    labels: Counter[str] = Counter()
    alias_rows = 0
    action_stats: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    policy_stats: dict[str, dict[str, float]] = {
        "direct": defaultdict(float),
        "offline_reward_oracle": defaultdict(float),
    }

    for row in rows:
        reasons = suspect_gold_reasons(row)
        if reasons:
            excluded = copy.deepcopy(row)
            excluded["excluded_from_policy_training"] = True
            excluded["qa_gold_audit"] = {
                "status": "quarantined",
                "reasons": reasons,
                "gold_word_count": len(str(row.get("gold_answer") or "").split()),
            }
            quarantine.append(excluded)
            continue

        output = copy.deepcopy(row)
        provided_aliases = [str(x) for x in (row.get("gold_aliases") or []) if str(x).strip()]
        derived_aliases = aliases_by_id.get(str(row.get("id")), [])
        if derived_aliases:
            alias_rows += 1
        candidates = [
            _score_candidate(
                candidate,
                str(row.get("gold_answer") or ""),
                provided_aliases,
                derived_aliases,
                qa_f1_threshold,
                lambda_cost,
            )
            for candidate in row.get("candidate_results") or []
        ]
        if not candidates:
            raise ValueError(f"Row {row.get('id')} has no candidate_results")
        best = max(
            candidates,
            key=lambda item: (
                float(item.get("reward", -1e9)),
                -float(item.get("token_cost", 1e9)),
            ),
        )
        labels[str(best.get("action"))] += 1
        direct_candidate = next(item for item in candidates if item.get("action") == "direct")
        for name, candidate in (("direct", direct_candidate), ("offline_reward_oracle", best)):
            stats = policy_stats[name]
            stats["standard_correct"] += float(candidate.get("correct", 0))
            stats["standard_em"] += float(candidate.get("exact_match", 0))
            stats["standard_f1"] += float(candidate.get("answer_f1", 0))
            stats["enhanced_f1"] += float(candidate.get("enhanced_answer_f1", 0))
            stats["tokens"] += float(candidate.get("token_cost", 0))
            stats["reward"] += float(candidate.get("reward", 0))
            stats["reflection_rate"] += candidate.get("action") != "direct"
        for candidate in candidates:
            stats = action_stats[str(candidate.get("action"))]
            stats["n"] += 1
            stats["standard_em"] += float(candidate.get("exact_match", 0))
            stats["standard_f1"] += float(candidate.get("answer_f1", 0))
            stats["alias_f1"] += float(candidate.get("alias_answer_f1", 0))
            stats["enhanced_f1"] += float(candidate.get("enhanced_answer_f1", 0))
            stats["tokens"] += float(candidate.get("token_cost", 0))
            stats["reward"] += float(candidate.get("reward", 0))

        output["candidate_results"] = candidates
        output["best_action"] = best.get("action")
        output["policy_objective"] = "alias_aware_enhanced_f1_minus_token_cost"
        output["lambda_cost"] = lambda_cost
        output["derived_gold_aliases"] = derived_aliases
        output["qa_relabel"] = {
            "version": "qa_offline_relabel_v1",
            "standard_reporting_metrics": ["exact_match", "answer_f1"],
            "policy_utility_metric": "alias_aware_enhanced_token_f1",
            "qa_f1_threshold_for_binary_reporting": qa_f1_threshold,
            "lambda_cost": lambda_cost,
            "llm_calls": 0,
        }
        clean.append(output)

    rendered_action_stats: dict[str, Any] = {}
    for action, stats in action_stats.items():
        n = int(stats["n"])
        rendered_action_stats[action] = {
            "n": n,
            **{
                f"avg_{key}": value / n
                for key, value in stats.items()
                if key != "n"
            },
        }
    policy_comparison = {
        name: {f"avg_{key}": value / len(clean) for key, value in stats.items()}
        for name, stats in policy_stats.items()
    }
    summary = {
        "input_rows": len(rows),
        "clean_rows": len(clean),
        "quarantined_rows": len(quarantine),
        "rows_with_derived_aliases": alias_rows,
        "label_distribution": dict(labels),
        "action_metrics": rendered_action_stats,
        "policy_comparison": policy_comparison,
        "lambda_cost": lambda_cost,
        "reward_metric": "alias_aware_enhanced_token_f1_minus_token_cost",
        "llm_calls": 0,
    }
    return clean, quarantine, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Relabel saved QA policy traces with alias-aware continuous F1 utility."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--quarantine_output", required=True)
    parser.add_argument("--summary_output", required=True)
    parser.add_argument(
        "--raw_hotpot_root",
        default=os.environ.get("RAW_HOTPOT_ROOT", "data/hotpotqa/fullwiki"),
    )
    parser.add_argument("--qa_f1_threshold", type=float, default=0.8)
    parser.add_argument("--lambda_cost", type=float, default=0.0001)
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    aliases_by_id = load_hotpot_context_aliases(Path(args.raw_hotpot_root), rows)
    clean, quarantine, summary = relabel_rows(
        rows,
        aliases_by_id,
        qa_f1_threshold=args.qa_f1_threshold,
        lambda_cost=args.lambda_cost,
    )
    output = Path(args.output)
    quarantine_output = Path(args.quarantine_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    quarantine_output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(clean, output)
    write_jsonl(quarantine, quarantine_output)
    summary.update(
        input=str(Path(args.input).resolve()),
        output=str(output.resolve()),
        quarantine_output=str(quarantine_output.resolve()),
        raw_hotpot_root=str(Path(args.raw_hotpot_root).resolve()),
    )
    save_json(summary, Path(args.summary_output))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
