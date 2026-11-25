from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
import signal
import socket
import subprocess
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

from . import math_reflector_pilot as pilot
from .math_full_training_trace import CachedOnly, make_policy_row
from .prepare_qa_hard_development import file_hash, locked_write
from .utils import read_jsonl, save_json


PROJECT = pilot.PROJECT
MODELS = PROJECT.parent / "models"
TRAIN_ROOT = PROJECT / "outputs/math_full_training_v1"
TEST_ROOT = PROJECT / "outputs/math_full_test_v1"
TRANSFER_ROOT = PROJECT / "outputs/math_transfer_locked_v1"
DEFAULT_ROOT = PROJECT / "outputs/qwen14_sampling_temp06_all_v1"
TRANSFER_DATASETS = ("gsm1k", "svamp", "multiarith")

PORT = 8014
CONTEXT = 40960
OUTPUT_STEPS = (4096, 8192, 16384, 32768)
SAFETY = 512
SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0}


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _by_id(rows):
    result = {}
    for row in rows:
        if row["id"] in result:
            raise ValueError(f"Duplicate row: {row['id']}")
        result[row["id"]] = row
    return result


def _item_map(path: Path):
    return _by_id(read_jsonl(path))


def _gsm_records():
    records = []
    direct_train = _by_id(read_jsonl(TRAIN_ROOT / "direct_all.jsonl"))
    for split, item_file in (("train", "train_items.jsonl"), ("validation", "validation_items.jsonl")):
        items = _item_map(TRAIN_ROOT / item_file)
        for item_id, item in items.items():
            base = direct_train.get(item_id)
            if base is None:
                continue
            direct = base["candidate_results"][0]
            records.append({
                "group": f"gsm8k_{split}",
                "dataset": "gsm8k",
                "split": split,
                "item": item,
                "base": base,
                "source_root": TRAIN_ROOT,
                "source_key": item_id,
                "need_reflection": split == "validation" or not direct["correct"],
                "need_independent": split == "validation",
            })

    direct_test = _by_id(read_jsonl(TEST_ROOT / "direct_test.jsonl"))
    for item_id, item in _item_map(TEST_ROOT / "test_items.jsonl").items():
        base = direct_test.get(item_id)
        if base is None:
            continue
        records.append({
            "group": "gsm8k_test",
            "dataset": "gsm8k",
            "split": "test",
            "item": item,
            "base": base,
            "source_root": TEST_ROOT,
            "source_key": item_id,
            "need_reflection": True,
            "need_independent": True,
        })
    return records


def _transfer_records():
    records = []
    for dataset in TRANSFER_DATASETS:
        source = TRANSFER_ROOT / dataset
        items = _item_map(source / "test_items.jsonl")
        for item_id, item in items.items():
            base_path = source / "rows" / "base" / f"{item_id}.json"
            if not base_path.exists():
                continue
            records.append({
                "group": f"transfer_{dataset}",
                "dataset": dataset,
                "split": "test",
                "item": item,
                "base": _load_json(base_path),
                "source_root": source,
                "source_key": item_id,
                "need_reflection": True,
                "need_independent": True,
            })
    return records


def _output_path(root: Path, record):
    group = record["group"]
    if group == "gsm8k_train":
        return root / "gsm8k" / "policy_train.jsonl"
    if group == "gsm8k_validation":
        return root / "gsm8k" / "policy_validation.jsonl"
    if group == "gsm8k_test":
        return root / "gsm8k" / "policy_test.jsonl"
    return root / "transfer" / record["dataset"] / "policy_test.jsonl"


def _call_key(record):
    if record["group"].startswith("transfer_"):
        return f"{record['dataset']}--{record['source_key']}"
    return record["source_key"]


def _append(path: Path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _record_failure(root: Path, record, action, exc):
    _append(root / "failures.jsonl", {
        "group": record["group"],
        "dataset": record["dataset"],
        "id": record["source_key"],
        "call_key": _call_key(record),
        "action": action,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "timestamp": time.time(),
        "retryable": True,
    })


def _verify_reused_direct(record, tokenizer):
    item = record["item"]
    base = record["base"]
    prompt = pilot.MATH_DIRECT_PROMPT.format(question=item["question"])
    call_root = Path(base["initial_call"]["call_log"]).parents[3]
    cached = pilot.complete_call(
        CachedOnly(), tokenizer, call_root, "base",
        record["source_key"], "direct", prompt,
    )
    if cached != base["initial_call"]:
        raise ValueError(f"Reused Direct call mismatch: {record['group']}/{record['source_key']}")


def _base_for_policy(record):
    base = dict(record["base"])
    item = record["item"]
    direct = pilot.candidate("direct", [base["initial_call"]], item["gold_answer"])
    base.update({
        "policy_split": "test" if record["split"] == "test" else record["split"],
        "initial_answer": base["initial_call"]["answer_text"],
        "initial_pred": direct["pred"],
        "candidate_results": [direct],
    })
    return base


def _append_protocol(root, records, missing_direct, reflector_gpus):
    protocol = {
        "name": "qwen14_official_thinking_sampling_reflection_rerun_v1",
        "purpose": "Reuse Qwen3-8B Direct traces and regenerate covered Qwen3-14B reflection and independent calls.",
        "official_sampling": {
            "enable_thinking": True,
            **SAMPLING,
            "source": "Qwen/Qwen3-14B model card",
        },
        "reflector_model": str(MODELS / "Qwen3-14B"),
        "reflector_role": {
            "model": "Qwen3-14B",
            "gpus": reflector_gpus,
            "tp": 2,
            "thinking": True,
        },
        "reused_direct_sources": {
            "gsm8k_train_validation": str(TRAIN_ROOT / "direct_all.jsonl"),
            "gsm8k_test": str(TEST_ROOT / "direct_test.jsonl"),
            "transfer": str(TRANSFER_ROOT),
        },
        "prompt_source": str(PROJECT / "src/prompts.py"),
        "reflection_template": pilot.REFLECTION_TEMPLATE,
        "context": CONTEXT,
        "output_steps": list(OUTPUT_STEPS),
        "safety": SAFETY,
        "seed": 42,
        "coverage": {
            "gsm8k_train": "R1/R2 only when reused Direct is incorrect",
            "gsm8k_validation": "R1/R2 and independent for every reusable Direct row",
            "gsm8k_test": "R1/R2 and independent for every reusable Direct row",
            "transfer": "R1/R2 and independent for every reusable Direct row",
        },
        "missing_direct": missing_direct,
        "token_accounting": "New Qwen3-14B attempts include every input/output retry; reused Direct costs remain in source call records.",
        "source_sha256": {
            "prompts.py": file_hash(PROJECT / "src/prompts.py"),
            "answer_extractor.py": file_hash(PROJECT / "src/answer_extractor.py"),
        },
    }
    locked_write(root / "protocol.json", json.dumps(protocol, ensure_ascii=False, indent=2) + "\n")


def _prepare_manifest(root, records):
    groups = Counter(record["group"] for record in records)
    ids = defaultdict(list)
    for record in records:
        ids[record["group"]].append(record["source_key"])
    save_json({"groups": dict(groups), "ids": dict(ids)}, root / "manifest.json")


def _call(root, record, action, prompt, tokenizer, client):
    key = _call_key(record)
    path = root / "calls" / "qwen14" / key / f"{action}.jsonl"
    if path.exists():
        saved = read_jsonl(path)
        if saved and saved[-1].get("metadata", {}).get("finish_reason") == "stop":
            return pilot.complete_call(CachedOnly(), tokenizer, root, "qwen14", key, action, prompt)
    return pilot.complete_call(client, tokenizer, root, "qwen14", key, action, prompt)


def _process_record(root, record, tokenizer, client):
    item = record["item"]
    base = _base_for_policy(record)
    calls = [base["initial_call"]]
    candidates = [base["candidate_results"][0]]
    if not record["need_reflection"]:
        return make_policy_row(base, candidates)

    for round_id in (1, 2):
        prompt = pilot.REFLECTION_TEMPLATE.format(
            question=item["question"],
            current_answer=calls[-1]["answer_text"],
            round_id=round_id,
        )
        call = _call(root, record, f"reflect_{round_id}", prompt, tokenizer, client)
        calls.append(call)
        candidates.append(pilot.candidate(f"reflect_{round_id}", calls, item["gold_answer"]))

    independent = None
    if record["need_independent"]:
        prompt = pilot.MATH_DIRECT_PROMPT.format(question=item["question"])
        call = _call(root, record, "independent", prompt, tokenizer, client)
        independent = pilot.candidate("independent", [call], item["gold_answer"])
    return make_policy_row(base, candidates, independent)


@contextlib.contextmanager
def _server(root, reflector_gpus):
    pilot.free_port()
    model = MODELS / "Qwen3-14B"
    command = [
        pilot.VLLM_PYTHON,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        str(model),
        "--served-model-name",
        "Qwen3-14B",
        "--host",
        "127.0.0.1",
        "--port",
        str(PORT),
        "--max-model-len",
        str(CONTEXT),
        "--gpu-memory-utilization",
        "0.46",
        "--enforce-eager",
        "--max-num-seqs",
        "1",
        "--max-num-batched-tokens",
        "1024",
        "--tensor-parallel-size",
        "2",
    ]
    stamp = time.strftime("%Y%m%d_%H%M%S")
    log = root / f"vllm_qwen14_sampling_{stamp}.log"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = reflector_gpus
    with log.open("x", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
    save_json({
        "pid": process.pid,
        "role": "qwen14",
        "model": str(model),
        "gpus": reflector_gpus,
        "command": command,
        "log": str(log),
    }, root / "active_server.json")
    print(f"[qwen14-sampling] loading Qwen3-14B on GPU {reflector_gpus}; log={log}", flush=True)
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"vLLM exited with code {process.returncode}; inspect {log}")
            try:
                with opener.open(f"http://127.0.0.1:{PORT}/v1/models", timeout=5) as response:
                    served = json.load(response).get("data", [])
                if any(entry.get("id") == "Qwen3-14B" for entry in served):
                    break
            except (OSError, ValueError, KeyError):
                pass
            time.sleep(5)
        else:
            raise RuntimeError(f"vLLM readiness timeout; inspect {log}")
        from .llm_client import OpenAICompatibleClient
        client = OpenAICompatibleClient(
            f"http://127.0.0.1:{PORT}/v1",
            "EMPTY",
            "Qwen3-14B",
            temperature=SAMPLING["temperature"],
            top_p=SAMPLING["top_p"],
            top_k=SAMPLING["top_k"],
            min_p=SAMPLING["min_p"],
            seed=42,
            timeout=1800,
            max_retries=0,
            api_mode="completions",
        )
        yield client
    finally:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=15)
        save_json({
            "pid": process.pid,
            "role": "qwen14",
            "stopped": True,
            "exit_code": process.returncode,
        }, root / "active_server.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--reflector-gpus", default="0,2")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / "pipeline.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        from transformers import AutoTokenizer
        base_tokenizer = AutoTokenizer.from_pretrained(
            str(MODELS / "Qwen3-8B-Instruct"),
            local_files_only=True,
            trust_remote_code=False,
        )
        records = _gsm_records() + _transfer_records()
        missing_direct = []
        verified = []
        for record in records:
            try:
                _verify_reused_direct(record, base_tokenizer)
            except Exception as exc:
                missing_direct.append({
                    "group": record["group"],
                    "id": record["source_key"],
                    "reason": str(exc),
                })
            else:
                verified.append(record)

        _prepare_manifest(root, verified)
        _append_protocol(root, verified, missing_direct, args.reflector_gpus)
        save_json({
            "stage": "preflight",
            "total_records": len(records),
            "verified_direct_records": len(verified),
            "missing_or_invalid_direct": len(missing_direct),
            "groups": dict(Counter(record["group"] for record in verified)),
            "official_sampling": SAMPLING,
            "context": CONTEXT,
            "output_steps": list(OUTPUT_STEPS),
            "reflector_gpus": args.reflector_gpus,
        }, root / "preflight.json")
        print(json.dumps(_load_json(root / "preflight.json"), ensure_ascii=False, indent=2), flush=True)
        if args.preflight_only:
            return

        pilot.CONTEXT = CONTEXT
        pilot.OUTPUT_STEPS = OUTPUT_STEPS
        pilot.SAFETY = SAFETY
        pilot.ROLES["qwen14"] = {
            "model": "Qwen3-14B",
            "gpus": args.reflector_gpus,
            "tp": 2,
            "thinking": True,
        }
        reflector_tokenizer = AutoTokenizer.from_pretrained(
            str(MODELS / "Qwen3-14B"),
            local_files_only=True,
            trust_remote_code=False,
        )
        done = defaultdict(set)
        for record in verified:
            path = _output_path(root, record)
            if path.exists():
                done[record["group"]].update(row["id"] for row in read_jsonl(path))

        pending = [record for record in verified if record["source_key"] not in done[record["group"]]]
        with _server(root, args.reflector_gpus) as client:
            started = time.monotonic()
            for index, record in enumerate(pending, 1):
                try:
                    row = _process_record(root, record, reflector_tokenizer, client)
                except Exception as exc:
                    _record_failure(root, record, "reflection_or_independent", exc)
                    print(
                        f"[qwen14-sampling] failed {index}/{len(pending)} "
                        f"{record['group']}/{record['source_key']}: {exc}",
                        flush=True,
                    )
                    continue
                _append(_output_path(root, record), row)
                done[record["group"]].add(record["source_key"])
                elapsed = time.monotonic() - started
                eta = elapsed / index * (len(pending) - index)
                print(
                    f"[qwen14-sampling] {index}/{len(pending)} "
                    f"{record['group']}/{record['source_key']} "
                    f"best={row['best_action']} eta={eta / 3600:.1f}h",
                    flush=True,
                )
                save_json({
                    "stage": "generating",
                    "processed": index,
                    "total": len(pending),
                    "completed_by_group": {key: len(value) for key, value in done.items()},
                    "failure_count": len(read_jsonl(root / "failures.jsonl"))
                    if (root / "failures.jsonl").exists() else 0,
                    "elapsed_seconds": elapsed,
                    "eta_seconds": eta,
                }, root / "status.json")
        save_json({
            "stage": "complete_or_resumable",
            "completed_by_group": {key: len(value) for key, value in done.items()},
            "failure_count": len(read_jsonl(root / "failures.jsonl"))
            if (root / "failures.jsonl").exists() else 0,
            "resume_supported": True,
        }, root / "status.json")


if __name__ == "__main__":
    main()
