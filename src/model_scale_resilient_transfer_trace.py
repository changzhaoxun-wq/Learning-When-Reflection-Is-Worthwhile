"""Resume transfer traces while quarantining exhausted samples."""
from __future__ import annotations

import argparse
import fcntl
import time
from pathlib import Path

from . import model_scale_math_trace as scale
from . import model_scale_resilient_math_trace as resilient
from . import model_scale_transfer_trace as transfer


def generate(root, items, tokenizers, base_folder, reflector_folder, args):
    flat = [(dataset, row) for dataset in transfer.DATASETS for row in items[dataset]]
    failed = resilient.failure_keys(root)
    base_rows = {}
    direct_pending = []
    for dataset, item in flat:
        path = root / dataset / "direct.jsonl"
        existing = scale.read_jsonl(path) if path.exists() else []
        by_id = {row["id"]: row for row in existing}
        if item["id"] in by_id:
            base_rows[(dataset, item["id"])] = by_id[item["id"]]
        elif resilient.failure_key(item, "direct") not in failed:
            direct_pending.append((dataset, item))

    if direct_pending:
        with scale.model_server(root, "base", base_folder, args.gpu_memory_utilization) as client:
            started = time.monotonic()
            for index, (dataset, item) in enumerate(direct_pending, 1):
                prompt = scale.pilot.MATH_DIRECT_PROMPT.format(question=item["question"])
                try:
                    call = scale.call_with_resume(root, "base", transfer.call_id(dataset, item),
                                                  "direct", prompt, tokenizers["base"], client)
                except RuntimeError as exc:
                    if not resilient.is_unresolved_truncation(exc):
                        raise
                    resilient.record_failure(root, item, "direct", "direct", "base",
                                             transfer.call_id(dataset, item), exc)
                    print(f"[model-scale-transfer] quarantined direct {dataset}/{item['id']}", flush=True)
                    scale.save_json({"stage": "direct", "processed": index,
                                     "total": len(direct_pending),
                                     "quarantined": len(resilient.read_failures(root)),
                                     "elapsed_seconds": time.monotonic() - started}, root / "status.json")
                    continue
                direct = scale.pilot.candidate("direct", [call], item["gold_answer"])
                row = {**item, "initial_call": call, "initial_answer": call["answer_text"],
                       "initial_pred": direct["pred"], "candidate_results": [direct]}
                transfer.locked_append(root / dataset / "direct.jsonl", row)
                base_rows[(dataset, item["id"])] = row
                print(f"[model-scale-transfer] direct {index}/{len(direct_pending)} {dataset}/{item['id']}", flush=True)
                scale.save_json({"stage": "direct", "processed": index,
                                 "total": len(direct_pending),
                                 "quarantined": len(resilient.read_failures(root)),
                                 "elapsed_seconds": time.monotonic() - started}, root / "status.json")

    for dataset, item in flat:
        if (dataset, item["id"]) in base_rows:
            continue
        path = root / dataset / "direct.jsonl"
        rows = scale.read_jsonl(path) if path.exists() else []
        match = [row for row in rows if row["id"] == item["id"]]
        if len(match) == 1:
            base_rows[(dataset, item["id"])] = match[0]

    reflector_pending = []
    for dataset, item in flat:
        path = root / dataset / "policy_test.jsonl"
        existing = scale.read_jsonl(path) if path.exists() else []
        if ((dataset, item["id"]) in base_rows
                and item["id"] not in {row["id"] for row in existing}
                and resilient.failure_key(item, "reflector") not in resilient.failure_keys(root)):
            reflector_pending.append((dataset, item))
    if not reflector_pending:
        return

    with scale.model_server(root, "qwen14", reflector_folder, args.gpu_memory_utilization) as client:
        started = time.monotonic()
        for index, (dataset, item) in enumerate(reflector_pending, 1):
            base = base_rows[(dataset, item["id"])]
            calls, candidates = [base["initial_call"]], [base["candidate_results"][0]]
            action = "reflect_1"
            try:
                for round_id in (1, 2):
                    action = f"reflect_{round_id}"
                    prompt = scale.pilot.REFLECTION_TEMPLATE.format(
                        question=item["question"], current_answer=calls[-1]["answer_text"],
                        round_id=round_id)
                    call = scale.call_with_resume(root, "qwen14", transfer.call_id(dataset, item),
                                                  action, prompt, tokenizers["qwen14"], client)
                    calls.append(call)
                    candidates.append(scale.pilot.candidate(action, calls, item["gold_answer"]))
                action = "independent"
                prompt = scale.pilot.MATH_DIRECT_PROMPT.format(question=item["question"])
                independent_call = scale.call_with_resume(
                    root, "qwen14", transfer.call_id(dataset, item), action, prompt,
                    tokenizers["qwen14"], client)
                independent = scale.pilot.candidate("independent", [independent_call], item["gold_answer"])
            except RuntimeError as exc:
                if not resilient.is_unresolved_truncation(exc):
                    raise
                resilient.record_failure(root, item, "reflector", action, "qwen14",
                                         transfer.call_id(dataset, item), exc)
                print(f"[model-scale-transfer] quarantined reflector {dataset}/{item['id']} action={action}", flush=True)
                scale.save_json({"stage": "reflector", "processed": index,
                                 "total": len(reflector_pending),
                                 "quarantined": len(resilient.read_failures(root)),
                                 "elapsed_seconds": time.monotonic() - started}, root / "status.json")
                continue
            row = scale.make_policy_row(base, candidates, independent)
            transfer.locked_append(root / dataset / "policy_test.jsonl", row)
            print(f"[model-scale-transfer] reflector {index}/{len(reflector_pending)} {dataset}/{item['id']} best={row['best_action']}", flush=True)
            scale.save_json({"stage": "reflector", "processed": index,
                             "total": len(reflector_pending),
                             "quarantined": len(resilient.read_failures(root)),
                             "elapsed_seconds": time.monotonic() - started}, root / "status.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--base-model", default="Qwen3-4B-Instruct")
    parser.add_argument("--reflector-model", default="Qwen3-8B-Instruct")
    parser.add_argument("--base-gpus", default="1")
    parser.add_argument("--reflector-gpus", default="2")
    parser.add_argument("--reflector-thinking", type=lambda x: x.lower() in {"1", "true", "yes"}, default=False)
    parser.add_argument("--gpu-memory-utilization", type=float, default=.70)
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / "pipeline.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        paths = scale.configure_roles(args)
        items = transfer.load_transfer_items(root)
        from transformers import AutoTokenizer
        tokenizers = {role: AutoTokenizer.from_pretrained(str(folder), local_files_only=True,
                     trust_remote_code=False) for role, folder in paths.items()}
        scale.record_protocol(root, args, paths)
        scale.save_json({"datasets": {name: len(rows) for name, rows in items.items()},
                         "context": scale.CONTEXT,
                         "reflection_checked_with_actual_answer": True,
                         "failure_policy": "quarantine_exhausted_truncation_and_continue"},
                        root / "preflight.json")
        generate(root, items, tokenizers, paths["base"], paths["qwen14"], args)
        failures = resilient.read_failures(root)
        scale.save_json({
            "stage": "complete_with_failures" if failures else "complete",
            "datasets": {name: len(rows) for name, rows in items.items()},
            "required_candidates": ["direct", "reflect_1", "reflect_2", "independent"],
            "failure_file": str(resilient.failure_path(root)),
            "failure_count": len(failures),
            "partial_results_not_final": bool(failures),
        }, root / "status.json")


if __name__ == "__main__":
    main()
