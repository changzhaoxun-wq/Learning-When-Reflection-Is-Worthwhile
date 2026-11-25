"""Generate fresh GSM8K traces for a smaller Base and a stronger Reflector.

This is deliberately separate from the locked Qwen3-8B/Qwen3-14B protocol.
The prompt, answer extractor, output budgets, split and token accounting stay
the same; only the two generation models and their placement are configurable.
Models are served one at a time, so the experiment does not require both large
models to fit on one GPU simultaneously.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime
import fcntl
import json
import os
import signal
import socket
import subprocess
import time
import urllib.request
from pathlib import Path

from . import math_reflector_pilot as pilot
from .gsm8k_utils import normalize_example_item
from .math_full_training_trace import CachedOnly, make_policy_row
from .prepare_math_reflector_pilot import DATA
from .prepare_qa_hard_development import file_hash, locked_write
from .utils import read_jsonl, save_json


PROJECT = pilot.PROJECT
MODELS = PROJECT.parent / "models"
SPLIT_ROOT = PROJECT / "outputs/math_full_training_v1"
DEFAULT_ROOT = PROJECT / "outputs/model_scale_gsm8k_qwen3_4b_qwen3_8b"
PORT = 8015
CONTEXT = pilot.CONTEXT
OUTPUT_STEPS = pilot.OUTPUT_STEPS
SAFETY = pilot.SAFETY


def model_path(name: str) -> Path:
    path = Path(name)
    if not path.is_absolute():
        path = MODELS / path
    path = path.resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Local model directory does not exist: {path}")
    return path


def model_id(name: str) -> str:
    return model_path(name).name


def configure_roles(args):
    base_name, reflector_name = model_id(args.base_model), model_id(args.reflector_model)
    base_path, reflector_path = model_path(args.base_model), model_path(args.reflector_model)
    base_gpus = str(args.base_gpus)
    reflector_gpus = str(args.reflector_gpus)
    for value in (base_gpus, reflector_gpus):
        if not value or any(not part.isdigit() for part in value.split(",")):
            raise ValueError("GPU lists must contain comma-separated numeric IDs")
    pilot.PORT = PORT
    pilot.CONTEXT = CONTEXT
    pilot.OUTPUT_STEPS = OUTPUT_STEPS
    pilot.SAFETY = SAFETY
    pilot.ROLES["base"] = {"model": base_name, "gpus": base_gpus,
                            "tp": len(base_gpus.split(",")), "thinking": False}
    # Keep the internal role name qwen14 so the existing candidate schema and
    # Gate/Round code remain compatible. The protocol records the actual model.
    pilot.ROLES["qwen14"] = {"model": reflector_name, "gpus": reflector_gpus,
                             "tp": len(reflector_gpus.split(",")),
                             "thinking": args.reflector_thinking}
    return {"base": base_path, "qwen14": reflector_path}


def load_items(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    sources = {}
    for split in ("train", "validation"):
        path = SPLIT_ROOT / f"{split}_items.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Locked split is missing: {path}")
        items = read_jsonl(path)
        sources[split] = items
        locked_write(root / f"{split}_items.jsonl", "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in items))
    test = [normalize_example_item(r, i, "test") for i, r in enumerate(read_jsonl(DATA / "test.jsonl"))]
    sources["test"] = [{**r, "policy_split": "test"} for r in test]
    locked_write(root / "test_items.jsonl", "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in sources["test"]))
    manifest = {
        "name": "model_scale_gsm8k_fixed_split_v1",
        "train_ids": [r["id"] for r in sources["train"]],
        "validation_ids": [r["id"] for r in sources["validation"]],
        "test_ids": [r["id"] for r in sources["test"]],
        "input_sha256": {"train.jsonl": file_hash(DATA / "train.jsonl"),
                         "test.jsonl": file_hash(DATA / "test.jsonl")},
        "split_source": str(SPLIT_ROOT / "manifest.json"),
        "split_source_sha256": file_hash(SPLIT_ROOT / "manifest.json"),
        "no_test_tuning": True,
    }
    manifest_path = root / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Existing model-scale manifest differs; use a new output root")
    locked_write(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return sources


def record_protocol(root: Path, args, paths):
    metadata, stats = {}, {}
    for role, folder in paths.items():
        metadata[role] = {}
        for name in ("config.json", "tokenizer_config.json", "tokenizer.json", "model.safetensors.index.json"):
            path = folder / name
            if path.exists():
                metadata[role][str(path)] = file_hash(path)
        index = folder / "model.safetensors.index.json"
        if index.exists():
            weight_map = json.loads(index.read_text()).get("weight_map", {})
            for shard_name in sorted(set(weight_map.values())):
                shard = folder / shard_name
                stats[str(shard)] = {"size": shard.stat().st_size, "mtime_ns": shard.stat().st_mtime_ns}
    protocol = {
        "name": "model_scale_gsm8k_answer_conditioned_v1",
        "base_model": str(paths["base"]), "reflector_model": str(paths["qwen14"]),
        "roles": pilot.ROLES, "direct_prompt": pilot.MATH_DIRECT_PROMPT,
        "reflection_template": pilot.REFLECTION_TEMPLATE,
        "context": CONTEXT, "output_steps": list(OUTPUT_STEPS), "safety": SAFETY,
        "temperature": 0.0, "seed": 42, "max_request_retries": 0,
        "server": {"port": PORT, "max_num_seqs": 1, "max_num_batched_tokens": 1024,
                   "enforce_eager": True, "gpu_memory_utilization": args.gpu_memory_utilization},
        "model_metadata_sha256": metadata, "model_shard_stats": stats,
        "policy_encoder": os.environ.get("ENCODER_NAME", "Qwen/Qwen3-0.6B"),
        "candidate_policy": "accuracy_first_then_min_total_tokens",
        "token_accounting": "sum every API input/output attempt, including recovered length calls",
        "test_tuning": False,
    }
    path = root / "generation_protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError("Existing generation protocol differs; use a new output root")
    locked_write(path, json.dumps(protocol, ensure_ascii=False, indent=2) + "\n")
    return protocol


def free_port():
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", PORT))


@contextlib.contextmanager
def model_server(root: Path, role: str, model_folder: Path, gpu_memory_utilization: float):
    free_port()
    spec = pilot.ROLES[role]
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log = root / f"vllm_{role}_{stamp}.log"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = spec["gpus"]
    command = [pilot.VLLM_PYTHON, "-m", "vllm.entrypoints.openai.api_server",
               "--model", str(model_folder), "--served-model-name", spec["model"],
               "--host", "127.0.0.1", "--port", str(PORT), "--max-model-len", str(CONTEXT),
               "--gpu-memory-utilization", str(gpu_memory_utilization), "--enforce-eager",
               "--max-num-seqs", "1", "--max-num-batched-tokens", "1024",
               "--tensor-parallel-size", str(spec["tp"])]
    with log.open("x") as handle:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=handle,
                                   stderr=subprocess.STDOUT, env=env, start_new_session=True)
    save_json({"pid": process.pid, "role": role, "model": str(model_folder),
               "command": command, "log": str(log)}, root / "active_server.json")
    print(f"[model-scale] loading {role}={spec['model']} GPU={spec['gpus']} log={log}", flush=True)
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"vLLM exited {process.returncode}; inspect {log}")
            try:
                with opener.open(f"http://127.0.0.1:{PORT}/v1/models", timeout=5) as response:
                    served = json.load(response).get("data", [])
                if any(entry.get("id") == spec["model"] for entry in served):
                    break
            except (OSError, ValueError, KeyError):
                pass
            time.sleep(5)
        else:
            raise RuntimeError(f"Service readiness timed out; inspect {log}")
        yield pilot.OpenAICompatibleClient(f"http://127.0.0.1:{PORT}/v1", "EMPTY", spec["model"],
                                           temperature=0.0, seed=42, timeout=1800,
                                           max_retries=0, api_mode="completions")
    finally:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=45)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=15)
        save_json({"pid": process.pid, "role": role, "stopped": True,
                   "exit_code": process.returncode}, root / "active_server.json")


def call_with_resume(root, role, item_id, action, prompt, tokenizer, client):
    path = root / "calls" / role / item_id / f"{action}.jsonl"
    if path.exists():
        records = read_jsonl(path)
        if records and records[-1].get("metadata", {}).get("finish_reason") == "stop":
            return pilot.complete_call(CachedOnly(), tokenizer, root, role, item_id, action, prompt)
    return pilot.complete_call(client, tokenizer, root, role, item_id, action, prompt)


def output_path(root, split):
    return root / {"train": "policy_train_partial.jsonl",
                   "validation": "policy_validation_partial.jsonl",
                   "test": "policy_test.jsonl"}[split]


def direct_path(root, split):
    return root / f"direct_{split}.jsonl"


def append_row(path, row):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def generate_direct(root, sources, tokenizers, model_folder, args):
    all_items = [r for split in ("train", "validation", "test") if split in sources for r in sources[split]]
    rows_by_split = {split: read_jsonl(direct_path(root, split)) if direct_path(root, split).exists() else []
                     for split in sources}
    done = {r["id"] for rows in rows_by_split.values() for r in rows}
    pending = [r for r in all_items if r["id"] not in done]
    if not pending:
        return {r["id"]: r for rows in rows_by_split.values() for r in rows}
    with model_server(root, "base", model_folder, args.gpu_memory_utilization) as client:
        started = time.monotonic()
        for index, item in enumerate(pending, 1):
            prompt = pilot.MATH_DIRECT_PROMPT.format(question=item["question"])
            first = call_with_resume(root, "base", item["id"], "direct", prompt,
                                     tokenizers["base"], client)
            direct = pilot.candidate("direct", [first], item["gold_answer"])
            row = {**item, "initial_answer": first["answer_text"], "initial_call": first,
                   "initial_pred": direct["pred"], "candidate_results": [direct]}
            append_row(direct_path(root, item["policy_split"]), row)
            print(f"[model-scale] direct {index}/{len(pending)} id={item['id']} correct={direct['correct']} total_tokens={direct['token_cost']}", flush=True)
            save_json({"stage": "direct", "processed": index, "total": len(pending),
                       "elapsed_seconds": time.monotonic() - started}, root / "status.json")
    return {r["id"]: r for split in sources for r in read_jsonl(direct_path(root, split))}


def should_generate_reflection(item, direct):
    return item["policy_split"] in ("validation", "test") or not direct["correct"]


def generate_reflections(root, sources, base_rows, tokenizers, model_folder, args):
    all_items = [r for split in ("train", "validation", "test") if split in sources for r in sources[split]]
    existing = {split: read_jsonl(output_path(root, split)) if output_path(root, split).exists() else []
                for split in sources}
    done = {r["id"] for rows in existing.values() for r in rows}
    pending = [r for r in all_items if r["id"] not in done]
    if not pending:
        return
    with model_server(root, "qwen14", model_folder, args.gpu_memory_utilization) as client:
        started = time.monotonic()
        for index, item in enumerate(pending, 1):
            base = base_rows[item["id"]]
            direct = base["candidate_results"][0]
            calls, candidates = [base["initial_call"]], [direct]
            if should_generate_reflection(item, direct):
                for round_id in (1, 2):
                    prompt = pilot.REFLECTION_TEMPLATE.format(question=item["question"],
                                                               current_answer=calls[-1]["answer_text"],
                                                               round_id=round_id)
                    call = call_with_resume(root, "qwen14", item["id"], f"reflect_{round_id}",
                                            prompt, tokenizers["qwen14"], client)
                    calls.append(call)
                    candidates.append(pilot.candidate(f"reflect_{round_id}", calls, item["gold_answer"]))
            independent = None
            if item["policy_split"] in ("validation", "test"):
                prompt = pilot.MATH_DIRECT_PROMPT.format(question=item["question"])
                call = call_with_resume(root, "qwen14", item["id"], "independent", prompt,
                                        tokenizers["qwen14"], client)
                independent = pilot.candidate("independent", [call], item["gold_answer"])
            row = make_policy_row(base, candidates, independent)
            append_row(output_path(root, item["policy_split"]), row)
            actions = ",".join(c["action"] for c in candidates)
            print(f"[model-scale] reflector {index}/{len(pending)} id={item['id']} actions={actions} best={row['best_action']}", flush=True)
            save_json({"stage": "reflector", "processed": index, "total": len(pending),
                       "elapsed_seconds": time.monotonic() - started}, root / "status.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
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
        parser.error("--gpu-memory-utilization must be between 0.1 and 0.95")
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / "pipeline.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        paths = configure_roles(args)
        sources = load_items(root)
        record_protocol(root, args, paths)
        wanted = {"train", "validation"} if args.split == "train" else {"test"} if args.split == "test" else {"train", "validation", "test"}
        active_sources = {k: v for k, v in sources.items() if k in wanted}
        from transformers import AutoTokenizer
        tokenizers = {role: AutoTokenizer.from_pretrained(str(folder), local_files_only=True,
                     trust_remote_code=False) for role, folder in paths.items()}
        windows = {}
        for role, tokenizer in tokenizers.items():
            values = []
            for items in active_sources.values():
                values.extend(len(tokenizer.encode(pilot.wire_prompt(tokenizer,
                    pilot.MATH_DIRECT_PROMPT.format(question=r["question"]), role),
                    add_special_tokens=False)) for r in items)
            windows[role] = max(values) if values else 0
        save_json({"split": args.split, "rows": {k: len(v) for k, v in active_sources.items()},
                   "direct_required_windows": windows, "context": CONTEXT,
                   "reflection_checked_with_actual_answer": True}, root / "preflight.json")
        print(f"[model-scale] preflight split={args.split} rows={ {k: len(v) for k, v in active_sources.items()} } windows={windows}", flush=True)
        if args.preflight_only:
            return
        try:
            base_rows = generate_direct(root, active_sources, tokenizers, paths["base"], args)
            generate_reflections(root, active_sources, base_rows, tokenizers, paths["qwen14"], args)
            save_json({"stage": "complete", "split": args.split,
                       "policy_files": {k: str(output_path(root, k)) for k in active_sources}}, root / "status.json")
        except BaseException as exc:
            save_json({"stage": "failed_or_interrupted", "error": repr(exc),
                       "resume_supported": True, "partial_results_not_final": True}, root / "status.json")
            raise


if __name__ == "__main__":
    main()
