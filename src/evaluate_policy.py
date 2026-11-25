import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .answer_extractor import extract_answer
from .gsm8k_utils import load_gsm8k
from .llm_client import build_llm_client
from .prompts import direct_prompt, prompt_version
from .reward import compute_match_metrics, compute_reward
from .state_extractor import extract_state_features
from .strategies import execute_strategy
from .train_policy import predict_action
from .verifier import build_verifier
from .utils import load_json, load_pickle, print_table, project_root, save_json, write_jsonl




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

METHOD_TO_ACTION = {
    "direct": "direct",
    "fixed_reflect_1": "reflect_1",
    "fixed_reflect_2": "reflect_2",
}


def evaluate(
    config: Dict[str, Any],
    policy_path: str,
    split: str,
    limit: int | None,
    trace_output: str | None = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    base_llm = build_llm_client(_llm_config(config, "base_llm"))
    reflector_llm = build_llm_client(_llm_config(config, "reflector_llm"))
    verifier = build_verifier(config.get("verifier", {"mode": "rule_based"}), default_llm=base_llm)
    policy_bundle = load_pickle(policy_path)
    examples = load_gsm8k(
        split=split,
        limit=limit,
        data_path=config.get("data", {}).get("eval_data_path") or config.get("data", {}).get("data_path"),
    )
    lambda_cost = float(config.get("lambda_cost", 0.001))
    verifier_threshold = float(config.get("verifier_threshold", 0.75))
    qa_f1_threshold = float(config.get("qa_f1_threshold", 0.8))
    stats: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    counts: Dict[str, int] = defaultdict(int)
    difficulty_stats: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    difficulty_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    source_stats: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    source_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    traces: List[Dict[str, Any]] = []

    has_qa = any(str(ex.get("task_type", "math")).lower() == "qa" for ex in examples)
    total = len(examples)
    started_at = time.time()
    _progress(f"[eval] loaded {total} examples split={split} limit={limit}")

    for idx, ex in enumerate(examples, start=1):
        item_started_at = time.time()
        example_id = ex.get("id", str(idx - 1))
        _progress(f"[eval] {idx}/{total} id={example_id} initial_answer start")
        task_type = ex.get("task_type", "math")
        gold_aliases = ex.get("gold_aliases", [])
        initial_prompt = direct_prompt(ex["question"], task_type=task_type)
        initial_response = base_llm.generate(initial_prompt, gold_answer=ex["gold_answer"])
        initial_answer = initial_response.text
        initial_pred = extract_answer(initial_answer, task_type=task_type)
        _progress(
            f"[eval] {idx}/{total} id={example_id} initial_answer done "
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
        difficulty = ex.get("difficulty") or state.get("difficulty", "unknown")
        source = ex.get("source") or ex.get("dataset") or "unknown"
        policy_row = {
            "question": ex["question"],
            "initial_answer": initial_answer,
            "initial_pred": initial_pred,
            "state_features": state,
            "task_type": task_type,
        }

        method_actions = dict(METHOD_TO_ACTION)
        learned_action = predict_action(policy_bundle, policy_row)
        method_actions["learned_policy"] = learned_action
        _progress(f"[eval] {idx}/{total} id={example_id} learned_policy action={learned_action}")
        trace_candidates: Dict[str, Dict[str, Any]] = {}
        method_traces: List[Dict[str, Any]] = []

        for method_idx, (method, action) in enumerate(method_actions.items(), start=1):
            _progress(
                f"[eval] {idx}/{total} id={example_id} method "
                f"{method_idx}/{len(method_actions)} {method} action={action} start"
            )
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
            answer_f1 = float(match_metrics["answer_f1"])
            reward = compute_reward(correct, result.token_cost, lambda_cost)
            stats[method]["correct"] += correct
            stats[method]["prompt_tokens"] += result.prompt_tokens
            stats[method]["completion_tokens"] += result.completion_tokens
            stats[method]["token_cost"] += result.token_cost
            stats[method]["reflection"] += float(result.reflected_rounds > 0)
            stats[method]["reward"] += reward
            stats[method]["answer_f1"] += answer_f1
            counts[method] += 1
            difficulty_stats[difficulty][method]["correct"] += correct
            difficulty_stats[difficulty][method]["prompt_tokens"] += result.prompt_tokens
            difficulty_stats[difficulty][method]["completion_tokens"] += result.completion_tokens
            difficulty_stats[difficulty][method]["token_cost"] += result.token_cost
            difficulty_stats[difficulty][method]["reflection"] += float(result.reflected_rounds > 0)
            difficulty_stats[difficulty][method]["reward"] += reward
            difficulty_stats[difficulty][method]["answer_f1"] += answer_f1
            difficulty_counts[difficulty][method] += 1
            source_stats[source][method]["correct"] += correct
            source_stats[source][method]["prompt_tokens"] += result.prompt_tokens
            source_stats[source][method]["completion_tokens"] += result.completion_tokens
            source_stats[source][method]["token_cost"] += result.token_cost
            source_stats[source][method]["reflection"] += float(result.reflected_rounds > 0)
            source_stats[source][method]["reward"] += reward
            source_stats[source][method]["answer_f1"] += answer_f1
            source_counts[source][method] += 1
            if method != "learned_policy":
                candidate_trace = result.to_dict()
                candidate_trace.update({
                    "correct": correct,
                    "reward": reward,
                    "exact_match": match_metrics.get("exact_match"),
                    "answer_f1": answer_f1,
                    "matched_gold": match_metrics.get("matched_gold"),
                })
                trace_candidates[action] = candidate_trace
            method_trace = result.to_dict()
            method_trace.update({
                "method": method,
                "selected_action": action,
                "correct": correct,
                "reward": reward,
                "exact_match": match_metrics.get("exact_match"),
                "answer_f1": answer_f1,
                "matched_gold": match_metrics.get("matched_gold"),
            })
            if method == "learned_policy":
                method_trace["policy_input"] = policy_row
            method_traces.append(method_trace)
            _progress(
                f"[eval] {idx}/{total} id={example_id} method {method} done "
                f"pred={result.pred} correct={correct} "
                f"prompt_tokens={result.prompt_tokens} completion_tokens={result.completion_tokens} "
                f"total_tokens={result.token_cost} reward={reward:.4f}"
            )
        if trace_output:
            oracle = max(
                trace_candidates.values(),
                key=lambda item: (float(item.get("reward", -1e9)), -float(item.get("token_cost", 1e9))),
            )
            traces.append({
                "id": example_id,
                "question": ex["question"],
                "gold_answer": ex["gold_answer"],
                "gold_aliases": gold_aliases,
                "task_type": task_type,
                "prompt_version": prompt_version(task_type),
                "dataset": ex.get("dataset"),
                "source": source,
                "difficulty": difficulty,
                "difficulty_id": ex.get("difficulty_id", state.get("difficulty_id")),
                "difficulty_score": ex.get("difficulty_score", state.get("difficulty_score")),
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
                "candidate_results": list(trace_candidates.values()),
                "method_results": method_traces,
                "oracle_action": oracle.get("action"),
            })
        elapsed = time.time() - item_started_at
        total_elapsed = time.time() - started_at
        _progress(
            f"[eval] {idx}/{total} id={example_id} done "
            f"elapsed={_format_duration(elapsed)} "
            f"total_elapsed={_format_duration(total_elapsed)} {_eta_message(started_at, idx, total)}"
        )

    methods = ["direct", "fixed_reflect_1", "fixed_reflect_2", "learned_policy"]
    rows = []
    for method in methods:
        n = counts[method]
        row = {
            "method": method,
            "accuracy": stats[method]["correct"] / n if n else 0.0,
            "avg_prompt_tokens": stats[method]["prompt_tokens"] / n if n else 0.0,
            "avg_completion_tokens": stats[method]["completion_tokens"] / n if n else 0.0,
            "avg_token_cost": stats[method]["token_cost"] / n if n else 0.0,
            "reflection_rate": stats[method]["reflection"] / n if n else 0.0,
            "avg_reward": stats[method]["reward"] / n if n else 0.0,
        }
        if has_qa:
            row["avg_answer_f1"] = stats[method]["answer_f1"] / n if n else 0.0
        rows.append(row)

    difficulty_rows = []
    difficulty_order = {"easy": 0, "medium": 1, "hard": 2, "unknown": 3}
    for difficulty in sorted(difficulty_counts, key=lambda d: difficulty_order.get(d, 99)):
        for method in methods:
            n = difficulty_counts[difficulty][method]
            if not n:
                continue
            ds = difficulty_stats[difficulty][method]
            row = {
                "difficulty": difficulty,
                "method": method,
                "n": n,
                "accuracy": ds["correct"] / n,
                "avg_prompt_tokens": ds["prompt_tokens"] / n,
                "avg_completion_tokens": ds["completion_tokens"] / n,
                "avg_token_cost": ds["token_cost"] / n,
                "reflection_rate": ds["reflection"] / n,
                "avg_reward": ds["reward"] / n,
            }
            if has_qa:
                row["avg_answer_f1"] = ds["answer_f1"] / n
            difficulty_rows.append(row)
    source_rows = []
    for source in sorted(source_counts):
        for method in methods:
            n = source_counts[source][method]
            if not n:
                continue
            ss = source_stats[source][method]
            row = {
                "source": source,
                "method": method,
                "n": n,
                "accuracy": ss["correct"] / n,
                "avg_prompt_tokens": ss["prompt_tokens"] / n,
                "avg_completion_tokens": ss["completion_tokens"] / n,
                "avg_token_cost": ss["token_cost"] / n,
                "reflection_rate": ss["reflection"] / n,
                "avg_reward": ss["reward"] / n,
            }
            if has_qa:
                row["avg_answer_f1"] = ss["answer_f1"] / n
            source_rows.append(row)
    if trace_output:
        write_jsonl(traces, trace_output)
        _progress(f"[eval] wrote {len(traces)} traces to {trace_output}")
    return rows, difficulty_rows, source_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(project_root() / "configs" / "default_config.json"))
    parser.add_argument("--policy_model", default=str(project_root() / "outputs" / "policy_model.pkl"))
    parser.add_argument("--split", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default=str(project_root() / "outputs" / "eval_results.json"))
    parser.add_argument("--difficulty_output", default=str(project_root() / "outputs" / "eval_results_by_difficulty.json"))
    parser.add_argument("--source_output", default=str(project_root() / "outputs" / "eval_results_by_source.json"))
    parser.add_argument("--trace_output", default=None)
    args = parser.parse_args()

    config = load_json(args.config)
    split = args.split or config.get("data", {}).get("eval_split", "test")
    limit = args.limit
    if limit is None:
        limit = config.get("data", {}).get("eval_limit")
    rows, difficulty_rows, source_rows = evaluate(
        config=config,
        policy_path=args.policy_model,
        split=split,
        limit=limit,
        trace_output=args.trace_output,
    )
    save_json(rows, args.output)
    save_json(difficulty_rows, args.difficulty_output)
    save_json(source_rows, args.source_output)
    metric_columns = [
        "method", "accuracy", "avg_prompt_tokens",
        "avg_completion_tokens", "avg_token_cost",
        "reflection_rate", "avg_reward",
    ]
    if any("avg_answer_f1" in row for row in rows):
        metric_columns.append("avg_answer_f1")
    table = print_table(rows, metric_columns)
    print(table)
    if difficulty_rows:
        print()
        print("By difficulty:")
        difficulty_columns = [
            "difficulty", "method", "n", "accuracy",
            "avg_prompt_tokens", "avg_completion_tokens",
            "avg_token_cost", "reflection_rate", "avg_reward",
        ]
        if any("avg_answer_f1" in row for row in difficulty_rows):
            difficulty_columns.append("avg_answer_f1")
        print(print_table(difficulty_rows, difficulty_columns))
    if source_rows:
        print()
        print("By source:")
        source_columns = [
            "source", "method", "n", "accuracy",
            "avg_prompt_tokens", "avg_completion_tokens",
            "avg_token_cost", "reflection_rate", "avg_reward",
        ]
        if any("avg_answer_f1" in row for row in source_rows):
            source_columns.append("avg_answer_f1")
        print(print_table(source_rows, source_columns))
    print(f"Wrote evaluation results to {args.output}")
    print(f"Wrote difficulty results to {args.difficulty_output}")
    print(f"Wrote source results to {args.source_output}")


if __name__ == "__main__":
    main()

