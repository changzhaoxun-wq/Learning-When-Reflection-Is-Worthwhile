"""Generate complete transfer traces for the model-scale experiment."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from . import model_scale_math_trace as scale
from .gsm8k_utils import normalize_example_item
from .math_full_training_trace import make_policy_row
from .prepare_qa_hard_development import file_hash, locked_write
from .utils import read_jsonl, save_json


PROJECT = scale.PROJECT
DATA = Path(os.environ.get("TRANSFER_DATA_ROOT", PROJECT / "data" / "transfer"))
DATASETS = ("gsm1k", "svamp", "multiarith")


def load_transfer_items(root: Path):
    result = {}
    manifest = {"datasets": {}, "no_policy_training": True, "required_candidates":
                ["direct", "reflect_1", "reflect_2", "independent"]}
    for name in DATASETS:
        source = DATA / name / "test.jsonl"
        rows = [{**normalize_example_item(row, i, "test"), "policy_split": "test",
                 "transfer_dataset": name}
                for i, row in enumerate(read_jsonl(source))]
        if len({row["id"] for row in rows}) != len(rows):
            raise ValueError(f"Duplicate IDs in {name}")
        result[name] = rows
        target = root / name / "test_items.jsonl"
        locked_write(target, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
        manifest["datasets"][name] = {"source": str(source), "source_sha256": file_hash(source),
                                      "n": len(rows), "items_sha256": file_hash(target)}
    path = root / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("Existing transfer manifest differs; use a new output root")
    locked_write(path, json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return result


def call_id(dataset, item):
    value = f"{dataset}--{item['id']}"
    if "/" in value or "\\" in value:
        raise ValueError(f"Unsafe transfer ID: {value}")
    return value


def generate(root, items, tokenizers, base_folder, reflector_folder, args):
    flat = [(dataset, row) for dataset in DATASETS for row in items[dataset]]
    base_rows = {}
    direct_pending = []
    for dataset, item in flat:
        path = root / dataset / "direct.jsonl"
        existing = read_jsonl(path) if path.exists() else []
        by_id = {row["id"]: row for row in existing}
        if item["id"] in by_id:
            base_rows[(dataset, item["id"])] = by_id[item["id"]]
        else:
            direct_pending.append((dataset, item))
    if direct_pending:
        with scale.model_server(root, "base", base_folder, args.gpu_memory_utilization) as client:
            start = time.monotonic()
            for index, (dataset, item) in enumerate(direct_pending, 1):
                prompt = scale.pilot.MATH_DIRECT_PROMPT.format(question=item["question"])
                call = scale.call_with_resume(root, "base", call_id(dataset, item), "direct", prompt,
                                               tokenizers["base"], client)
                direct = scale.pilot.candidate("direct", [call], item["gold_answer"])
                row = {**item, "initial_call": call, "initial_answer": call["answer_text"],
                       "initial_pred": direct["pred"], "candidate_results": [direct]}
                locked_append(root / dataset / "direct.jsonl", row)
                base_rows[(dataset, item["id"])] = row
                print(f"[model-scale-transfer] direct {index}/{len(direct_pending)} {dataset}/{item['id']}", flush=True)
                save_json({"stage": "direct", "processed": index, "total": len(direct_pending),
                           "elapsed_seconds": time.monotonic() - start}, root / "status.json")
    else:
        for dataset, item in flat:
            rows = read_jsonl(root / dataset / "direct.jsonl")
            match = [row for row in rows if row["id"] == item["id"]]
            if len(match) != 1:
                raise ValueError(f"Missing cached Direct row: {dataset}/{item['id']}")
            base_rows[(dataset, item["id"])] = match[0]

    reflector_pending = []
    for dataset, item in flat:
        path = root / dataset / "policy_test.jsonl"
        existing = read_jsonl(path) if path.exists() else []
        if item["id"] not in {row["id"] for row in existing}:
            reflector_pending.append((dataset, item))
    if not reflector_pending:
        return
    with scale.model_server(root, "qwen14", reflector_folder, args.gpu_memory_utilization) as client:
        start = time.monotonic()
        for index, (dataset, item) in enumerate(reflector_pending, 1):
            base = base_rows[(dataset, item["id"])]
            calls, candidates = [base["initial_call"]], [base["candidate_results"][0]]
            for round_id in (1, 2):
                prompt = scale.pilot.REFLECTION_TEMPLATE.format(
                    question=item["question"], current_answer=calls[-1]["answer_text"], round_id=round_id)
                call = scale.call_with_resume(root, "qwen14", call_id(dataset, item),
                                              f"reflect_{round_id}", prompt, tokenizers["qwen14"], client)
                calls.append(call)
                candidates.append(scale.pilot.candidate(f"reflect_{round_id}", calls, item["gold_answer"]))
            prompt = scale.pilot.MATH_DIRECT_PROMPT.format(question=item["question"])
            independent_call = scale.call_with_resume(root, "qwen14", call_id(dataset, item),
                                                       "independent", prompt, tokenizers["qwen14"], client)
            independent = scale.pilot.candidate("independent", [independent_call], item["gold_answer"])
            row = make_policy_row(base, candidates, independent)
            locked_append(root / dataset / "policy_test.jsonl", row)
            print(f"[model-scale-transfer] reflector {index}/{len(reflector_pending)} {dataset}/{item['id']} best={row['best_action']}", flush=True)
            save_json({"stage": "reflector", "processed": index, "total": len(reflector_pending),
                       "elapsed_seconds": time.monotonic() - start}, root / "status.json")


def locked_append(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


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
    paths = scale.configure_roles(args)
    items = load_transfer_items(root)
    from transformers import AutoTokenizer
    tokenizers = {role: AutoTokenizer.from_pretrained(str(folder), local_files_only=True,
                 trust_remote_code=False) for role, folder in paths.items()}
    scale.record_protocol(root, args, paths)
    save_json({"datasets": {name: len(rows) for name, rows in items.items()},
               "context": scale.CONTEXT, "reflection_checked_with_actual_answer": True},
              root / "preflight.json")
    generate(root, items, tokenizers, paths["base"], paths["qwen14"], args)
    save_json({"stage": "complete", "datasets": {name: len(rows) for name, rows in items.items()},
               "required_candidates": ["direct", "reflect_1", "reflect_2", "independent"]},
              root / "status.json")


if __name__ == "__main__":
    main()
