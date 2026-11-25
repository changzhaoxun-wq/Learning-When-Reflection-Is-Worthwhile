import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List

from .answer_extractor import extract_answer
from .gsm8k_utils import load_gsm8k
from .llm_client import build_llm_client
from .prompts import direct_prompt, prompt_version
from .reward import compute_match_metrics, compute_reward
from .state_extractor import extract_state_features
from .strategies import ACTIONS, execute_strategy
from .verifier import build_verifier
from .utils import load_json, project_root, read_jsonl, set_seed




def _progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _eta_message(started_at: float, completed: int, total: int) -> str:
    if completed <= 0 or total <= 0:
        return "avg_per_item=? eta=?"
    elapsed = time.time() - started_at
    avg = elapsed / completed
    remaining = max(0, total - completed)
    eta_seconds = avg * remaining
    finish_at = time.strftime("%H:%M:%S", time.localtime(time.time() + eta_seconds))
    return (
        f"avg_per_item={_format_duration(avg)} "
        f"eta={_format_duration(eta_seconds)} "
        f"finish_at={finish_at}"
    )


def _llm_config(config: Dict[str, Any], key: str) -> Dict[str, Any]:
    return config.get(key) or config.get("llm", {"mode": "fake"})

DEFAULT_CONFIG: Dict[str, Any] = {
    "seed": 42,
    "lambda_cost": 0.001,
    "verifier_threshold": 0.75,
    "llm": {"mode": "fake"},
    "data": {"train_split": "train", "train_limit": None, "data_path": None},
    "outputs": {"policy_dataset": "outputs/policy_dataset.jsonl"},
    "actions": ["direct", "reflect_1", "reflect_2"],
}


def build_policy_rows(
    config: Dict[str, Any],
    split: str,
    limit: int | None,
    existing_rows: List[Dict[str, Any]] | None = None,
    on_row: Callable[[Dict[str, Any]], None] | None = None,
) -> List[Dict[str, Any]]:
    set_seed(int(config.get("seed", 42)))
    base_llm = build_llm_client(_llm_config(config, "base_llm"))
    reflector_llm = build_llm_client(_llm_config(config, "reflector_llm"))
    verifier = build_verifier(config.get("verifier", {"mode": "rule_based"}), default_llm=base_llm)
    examples = load_gsm8k(
        split=split,
        limit=limit,
        data_path=config.get("data", {}).get("data_path"),
    )
    rows = list(existing_rows or [])
    completed_ids = {str(row.get("id")) for row in rows}
    lambda_cost = float(config.get("lambda_cost", 0.001))
    verifier_threshold = float(config.get("verifier_threshold", 0.75))
    qa_f1_threshold = float(config.get("qa_f1_threshold", 0.8))
    actions = list(config.get("actions") or ACTIONS)
    invalid_actions = [action for action in actions if action not in ACTIONS]
    if not actions or invalid_actions:
        raise ValueError(f"Invalid actions={actions}; choose from {ACTIONS}")

    total = len(examples)
    started_at = time.time()
    pending_total = sum(str(ex.get("id", index)) not in completed_ids for index, ex in enumerate(examples))
    session_completed = 0
    _progress(f"[build] loaded {total} examples split={split} limit={limit} resumed={len(rows)} pending={pending_total}")

    for idx, ex in enumerate(examples, start=1):
        item_started_at = time.time()
        example_id = ex.get("id", str(idx - 1))
        if str(example_id) in completed_ids:
            continue
        _progress(f"[build] {idx}/{total} id={example_id} initial_answer start")
        task_type = ex.get("task_type", "math")
        gold_aliases = ex.get("gold_aliases", [])
        initial_prompt = direct_prompt(ex["question"], task_type=task_type)
        initial_response = base_llm.generate(initial_prompt, gold_answer=ex["gold_answer"])
        initial_answer = initial_response.text
        initial_pred = extract_answer(initial_answer, task_type=task_type)
        _progress(
            f"[build] {idx}/{total} id={example_id} initial_answer done "
            f"pred={initial_pred} prompt_tokens={initial_response.prompt_tokens} "
            f"completion_tokens={initial_response.completion_tokens} "
            f"total_tokens={initial_response.total_tokens}"
        )
        state = extract_state_features(
            question=ex["question"],
            initial_answer=initial_answer,
            round_id=0,
            token_cost=initial_response.total_tokens,
            verifier=verifier,
            task_type=task_type,
        )

        candidate_results = []
        for action_idx, action in enumerate(actions, start=1):
            _progress(f"[build] {idx}/{total} id={example_id} action {action_idx}/{len(actions)} {action} start")
            result = execute_strategy(
                action=action,
                llm=reflector_llm,
                question=ex["question"],
                initial_answer=initial_answer,
                initial_token_cost=initial_response.total_tokens,
                initial_prompt_tokens=initial_response.prompt_tokens,
                initial_completion_tokens=initial_response.completion_tokens,
                initial_prompt=initial_prompt,
                verifier_threshold=verifier_threshold,
                gold_answer=ex["gold_answer"],
                verifier=verifier,
                task_type=task_type,
                initial_response_metadata=initial_response.metadata,
            )
            match_metrics = compute_match_metrics(
                result.pred,
                ex["gold_answer"],
                task_type=task_type,
                gold_aliases=gold_aliases,
                qa_f1_threshold=qa_f1_threshold,
            )
            correct = int(match_metrics["correct"])
            reward = compute_reward(correct, result.token_cost, lambda_cost)
            result_dict = result.to_dict()
            result_dict.setdefault("metadata", {})["initial_usage"] = {
                "prompt_tokens": initial_response.prompt_tokens,
                "completion_tokens": initial_response.completion_tokens,
                "total_tokens": initial_response.total_tokens,
            }
            result_dict.update({"correct": correct, "reward": reward})
            if str(task_type).lower() == "qa":
                result_dict.update({
                    "exact_match": match_metrics.get("exact_match"),
                    "answer_f1": match_metrics.get("answer_f1"),
                    "matched_gold": match_metrics.get("matched_gold"),
                })
            candidate_results.append(result_dict)
            _progress(
                f"[build] {idx}/{total} id={example_id} action {action} done "
                f"pred={result.pred} correct={correct} "
                f"prompt_tokens={result.prompt_tokens} "
                f"completion_tokens={result.completion_tokens} "
                f"total_tokens={result.token_cost} reward={reward:.4f}"
            )

        best = max(
            candidate_results,
            key=lambda item: (item["reward"], -item["token_cost"]),
        )
        rows.append(
            {
                "id": example_id,
                "question": ex["question"],
                "raw_question": ex.get("raw_question", ex["question"]),
                "context": ex.get("context", ""),
                "metadata": ex.get("metadata", {}),
                "gold_answer": ex["gold_answer"],
                "gold_aliases": gold_aliases,
                "task_type": task_type,
                "prompt_version": prompt_version(task_type),
                "dataset": ex.get("dataset"),
                "source": ex.get("source"),
                "difficulty": ex.get("difficulty", state.get("difficulty")),
                "difficulty_id": ex.get("difficulty_id", state.get("difficulty_id")),
                "difficulty_score": ex.get("difficulty_score", state.get("difficulty_score")),
                "difficulty_features": ex.get("difficulty_features", {}),
                "initial_call": {
                    "prompt": initial_prompt,
                    "output": initial_answer,
                    "prompt_tokens": initial_response.prompt_tokens,
                    "completion_tokens": initial_response.completion_tokens,
                    "total_tokens": initial_response.total_tokens,
                    "response_metadata": initial_response.metadata,
                    "finish_reason": initial_response.metadata.get("finish_reason"),
                },
                "initial_answer": initial_answer,
                "initial_pred": initial_pred,
                "state_features": state,
                "candidate_results": candidate_results,
                "action_space": actions,
                "best_action": best["action"],
            }
        )
        if on_row is not None:
            on_row(rows[-1])
        session_completed += 1
        elapsed = time.time() - item_started_at
        total_elapsed = time.time() - started_at
        _progress(
            f"[build] {idx}/{total} id={example_id} done "
            f"best_action={best['action']} elapsed={_format_duration(elapsed)} "
            f"total_elapsed={_format_duration(total_elapsed)} {_eta_message(started_at, session_completed, pending_total)}"
        )
    return rows


def load_config(path: str | None) -> Dict[str, Any]:
    config = dict(DEFAULT_CONFIG)
    if path:
        config.update(load_json(path))
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(project_root() / "configs" / "default_config.json"))
    parser.add_argument("--split", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--actions", default=None, help="Comma-separated actions, e.g. direct,reflect_1")
    parser.add_argument("--resume", action="store_true", help="Append completed rows and skip existing ids")
    args = parser.parse_args()

    config = load_config(args.config)
    split = args.split or config.get("data", {}).get("train_split", "train")
    limit = args.limit
    if limit is None:
        limit = config.get("data", {}).get("train_limit")
    if args.actions:
        config["actions"] = [item.strip() for item in args.actions.split(",") if item.strip()]
    output = args.output or config.get("outputs", {}).get("policy_dataset", "outputs/policy_dataset.jsonl")
    output_path = Path(output)
    if not output_path.is_absolute():
        output_path = project_root() / output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing_rows: List[Dict[str, Any]] = []
    if args.resume and output_path.exists():
        existing_rows = read_jsonl(output_path)
        _progress(f"[build] resume loaded {len(existing_rows)} completed rows from {output_path}")
    else:
        output_path.write_text("", encoding="utf-8")

    def append_row(row: Dict[str, Any]) -> None:
        with output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    rows = build_policy_rows(
        config=config,
        split=split,
        limit=limit,
        existing_rows=existing_rows,
        on_row=append_row,
    )
    print(f"Wrote {len(rows)} policy examples to {output_path}")


if __name__ == "__main__":
    main()

