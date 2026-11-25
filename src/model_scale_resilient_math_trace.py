"""Resume model-scale math traces while quarantining exhausted samples."""
from __future__ import annotations

import argparse
import fcntl
import time
from pathlib import Path

from . import model_scale_math_trace as scale


def failure_path(root: Path) -> Path:
    return root / "generation_failures.jsonl"


def read_failures(root: Path):
    path = failure_path(root)
    return scale.read_jsonl(path) if path.exists() else []


def failure_key(item, stage: str) -> str:
    scope = item.get("transfer_dataset") or item.get("dataset") or "gsm8k"
    return f"{scope}:{stage}:{item['id']}"


def is_unresolved_truncation(exc: Exception) -> bool:
    return isinstance(exc, RuntimeError) and str(exc).startswith("Unresolved truncation for ")


def record_failure(root: Path, item, stage: str, action: str, role: str,
                   call_item_id: str, exc: Exception) -> None:
    key = failure_key(item, stage)
    failures = read_failures(root)
    if key in {row.get("failure_key") for row in failures}:
        return
    call_log = root / "calls" / role / call_item_id / f"{action}.jsonl"
    attempts = []
    if call_log.exists():
        for record in scale.read_jsonl(call_log):
            metadata = record.get("metadata", {})
            attempts.append({
                "max_tokens": record.get("max_tokens"),
                "finish_reason": metadata.get("finish_reason"),
                "prompt_tokens": record.get("prompt_tokens"),
                "completion_tokens": record.get("completion_tokens"),
                "total_tokens": record.get("total_tokens"),
            })
    scale.append_row(failure_path(root), {
        "failure_key": key,
        "id": item["id"],
        "dataset": item.get("dataset"),
        "transfer_dataset": item.get("transfer_dataset"),
        "policy_split": item.get("policy_split"),
        "stage": stage,
        "action": action,
        "role": role,
        "failure_type": "unresolved_truncation",
        "error": str(exc),
        "call_log": str(call_log),
        "attempts": attempts,
        "attempt_total_tokens": sum(row.get("total_tokens") or 0 for row in attempts),
        "excluded_from_policy_rows": True,
    })


def failure_keys(root: Path):
    return {row.get("failure_key") for row in read_failures(root)}


def generate_direct(root, sources, tokenizers, model_folder, args):
    all_items = [item for split in ("train", "validation", "test")
                 if split in sources for item in sources[split]]
    rows_by_split = {
        split: scale.read_jsonl(scale.direct_path(root, split))
        if scale.direct_path(root, split).exists() else []
        for split in sources
    }
    done = {row["id"] for rows in rows_by_split.values() for row in rows}
    failed = failure_keys(root)
    pending = [item for item in all_items
               if item["id"] not in done and failure_key(item, "direct") not in failed]
    if not pending:
        return {row["id"]: row for rows in rows_by_split.values() for row in rows}

    with scale.model_server(root, "base", model_folder, args.gpu_memory_utilization) as client:
        started = time.monotonic()
        for index, item in enumerate(pending, 1):
            prompt = scale.pilot.MATH_DIRECT_PROMPT.format(question=item["question"])
            try:
                first = scale.call_with_resume(root, "base", item["id"], "direct", prompt,
                                               tokenizers["base"], client)
            except RuntimeError as exc:
                if not is_unresolved_truncation(exc):
                    raise
                record_failure(root, item, "direct", "direct", "base", item["id"], exc)
                print(f"[model-scale] quarantined direct id={item['id']} reason=unresolved_truncation", flush=True)
                scale.save_json({"stage": "direct", "processed": index, "total": len(pending),
                                 "quarantined": len(read_failures(root)),
                                 "elapsed_seconds": time.monotonic() - started}, root / "status.json")
                continue
            direct = scale.pilot.candidate("direct", [first], item["gold_answer"])
            row = {**item, "initial_answer": first["answer_text"], "initial_call": first,
                   "initial_pred": direct["pred"], "candidate_results": [direct]}
            scale.append_row(scale.direct_path(root, item["policy_split"]), row)
            print(f"[model-scale] direct {index}/{len(pending)} id={item['id']} correct={direct['correct']} total_tokens={direct['token_cost']}", flush=True)
            scale.save_json({"stage": "direct", "processed": index, "total": len(pending),
                             "quarantined": len(read_failures(root)),
                             "elapsed_seconds": time.monotonic() - started}, root / "status.json")
    return {row["id"]: row for split in sources
            for row in scale.read_jsonl(scale.direct_path(root, split))}


def generate_reflections(root, sources, base_rows, tokenizers, model_folder, args):
    all_items = [item for split in ("train", "validation", "test")
                 if split in sources for item in sources[split]]
    existing = {
        split: scale.read_jsonl(scale.output_path(root, split))
        if scale.output_path(root, split).exists() else []
        for split in sources
    }
    done = {row["id"] for rows in existing.values() for row in rows}
    failed = failure_keys(root)
    pending = [item for item in all_items
               if item["id"] in base_rows and item["id"] not in done
               and failure_key(item, "reflector") not in failed]
    if not pending:
        return

    with scale.model_server(root, "qwen14", model_folder, args.gpu_memory_utilization) as client:
        started = time.monotonic()
        for index, item in enumerate(pending, 1):
            base = base_rows[item["id"]]
            direct = base["candidate_results"][0]
            calls, candidates = [base["initial_call"]], [direct]
            action = "reflect_1"
            try:
                if scale.should_generate_reflection(item, direct):
                    for round_id in (1, 2):
                        action = f"reflect_{round_id}"
                        prompt = scale.pilot.REFLECTION_TEMPLATE.format(
                            question=item["question"], current_answer=calls[-1]["answer_text"],
                            round_id=round_id)
                        call = scale.call_with_resume(root, "qwen14", item["id"], action,
                                                      prompt, tokenizers["qwen14"], client)
                        calls.append(call)
                        candidates.append(scale.pilot.candidate(action, calls, item["gold_answer"]))
                independent = None
                if item["policy_split"] in ("validation", "test"):
                    action = "independent"
                    prompt = scale.pilot.MATH_DIRECT_PROMPT.format(question=item["question"])
                    call = scale.call_with_resume(root, "qwen14", item["id"], action, prompt,
                                                  tokenizers["qwen14"], client)
                    independent = scale.pilot.candidate("independent", [call], item["gold_answer"])
            except RuntimeError as exc:
                if not is_unresolved_truncation(exc):
                    raise
                record_failure(root, item, "reflector", action, "qwen14", item["id"], exc)
                print(f"[model-scale] quarantined reflector id={item['id']} action={action}", flush=True)
                scale.save_json({"stage": "reflector", "processed": index, "total": len(pending),
                                 "quarantined": len(read_failures(root)),
                                 "elapsed_seconds": time.monotonic() - started}, root / "status.json")
                continue
            row = scale.make_policy_row(base, candidates, independent)
            scale.append_row(scale.output_path(root, item["policy_split"]), row)
            actions = ",".join(candidate["action"] for candidate in candidates)
            print(f"[model-scale] reflector {index}/{len(pending)} id={item['id']} actions={actions} best={row['best_action']}", flush=True)
            scale.save_json({"stage": "reflector", "processed": index, "total": len(pending),
                             "quarantined": len(read_failures(root)),
                             "elapsed_seconds": time.monotonic() - started}, root / "status.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=scale.DEFAULT_ROOT)
    parser.add_argument("--base-model", default="Qwen3-4B-Instruct")
    parser.add_argument("--reflector-model", default="Qwen3-8B-Instruct")
    parser.add_argument("--base-gpus", default="1")
    parser.add_argument("--reflector-gpus", default="2")
    parser.add_argument("--reflector-thinking", type=lambda x: x.lower() in {"1", "true", "yes"}, default=False)
    parser.add_argument("--gpu-memory-utilization", type=float, default=.70)
    parser.add_argument("--split", choices=("train", "test", "all"), default="all")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if not 0.1 <= args.gpu_memory_utilization <= .95:
        parser.error("--gpu-memory-utilization must be between 0.1 and .95")
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / "pipeline.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        paths = scale.configure_roles(args)
        sources = scale.load_items(root)
        scale.record_protocol(root, args, paths)
        wanted = {"train", "validation"} if args.split == "train" else {"test"} if args.split == "test" else {"train", "validation", "test"}
        active_sources = {key: value for key, value in sources.items() if key in wanted}
        from transformers import AutoTokenizer
        tokenizers = {role: AutoTokenizer.from_pretrained(str(folder), local_files_only=True,
                     trust_remote_code=False) for role, folder in paths.items()}
        windows = {}
        for role, tokenizer in tokenizers.items():
            values = []
            for items in active_sources.values():
                values.extend(len(tokenizer.encode(scale.pilot.wire_prompt(
                    tokenizer, scale.pilot.MATH_DIRECT_PROMPT.format(question=item["question"]), role),
                    add_special_tokens=False)) for item in items)
            windows[role] = max(values) if values else 0
        scale.save_json({"split": args.split, "rows": {key: len(value) for key, value in active_sources.items()},
                         "direct_required_windows": windows, "context": scale.CONTEXT,
                         "reflection_checked_with_actual_answer": True,
                         "failure_policy": "quarantine_exhausted_truncation_and_continue"}, root / "preflight.json")
        print(f"[model-scale] preflight split={args.split} rows={ {k: len(v) for k, v in active_sources.items()} } windows={windows}", flush=True)
        if args.preflight_only:
            return
        try:
            base_rows = generate_direct(root, active_sources, tokenizers, paths["base"], args)
            generate_reflections(root, active_sources, base_rows, tokenizers, paths["qwen14"], args)
            failures = read_failures(root)
            scale.save_json({
                "stage": "complete_with_failures" if failures else "complete",
                "split": args.split,
                "policy_files": {key: str(scale.output_path(root, key)) for key in active_sources},
                "failure_file": str(failure_path(root)),
                "failure_count": len(failures),
                "partial_results_not_final": bool(failures),
            }, root / "status.json")
        except BaseException as exc:
            scale.save_json({"stage": "failed_or_interrupted", "error": repr(exc),
                             "resume_supported": True, "partial_results_not_final": True}, root / "status.json")
            raise


if __name__ == "__main__":
    main()
