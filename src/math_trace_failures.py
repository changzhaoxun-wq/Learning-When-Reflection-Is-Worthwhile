"""Auditable quarantine of exhausted output budgets, never ordinary wrong answers."""

import json
import subprocess
import time
from pathlib import Path

from . import math_reflector_pilot as pilot
from .prepare_qa_hard_development import file_hash
from .utils import read_jsonl, save_json


class OutputCapFailure(RuntimeError):
    def __init__(self, diagnostic):
        super().__init__(f"Verified exhausted output budget: {diagnostic['call_log']}")
        self.diagnostic = diagnostic


def exhausted_call(root, role, key, action, prompt, tokenizer):
    path = root / "calls" / role / key / f"{action}.jsonl"
    records = read_jsonl(path) if path.exists() else []
    if len(records) > len(pilot.OUTPUT_STEPS):
        raise ValueError("Too many saved attempts")
    if len(records) != len(pilot.OUTPUT_STEPS) or records[-1]["metadata"].get("finish_reason") != "length":
        return None
    wire = pilot.wire_prompt(tokenizer, prompt, role)
    count = len(tokenizer.encode(wire, add_special_tokens=False))
    required = pilot.check_window(count)
    fingerprint = pilot.digest(json.dumps({"role": role, "model": pilot.ROLES[role]["model"], "wire": wire,
                                          "output_steps": pilot.OUTPUT_STEPS, "context": pilot.CONTEXT}, sort_keys=True))
    for i, record in enumerate(records):
        pilot.validate_response(record)
        expected = {"fingerprint": fingerprint, "max_tokens": pilot.OUTPUT_STEPS[i], "prompt": prompt,
                    "wire_prompt": wire, "prompt_tokens_preflight": count, "required_window_with_cap": required}
        if any(record.get(k) != v for k, v in expected.items()):
            raise ValueError("Exhausted call does not match frozen prompt/budget")
        request = record["metadata"].get("request", {})
        if any(request.get(k) != v for k, v in {"temperature": 0., "seed": 42,
                                                "effective_max_tokens": pilot.OUTPUT_STEPS[i]}.items()):
            raise ValueError("Exhausted call request settings changed")
        if record["metadata"]["finish_reason"] != "length":
            raise ValueError("Unexpected retry after normal completion")
    return {"call_log": str(path), "call_log_sha256": file_hash(path), "action": action,
            "reason": "output_budget_exhausted", "attempt_count": len(records),
            "max_tokens_cap": pilot.OUTPUT_STEPS[-1], "finish_reason": "length",
            **{field: sum(r[field] for r in records) for field in
               ("prompt_tokens", "completion_tokens", "total_tokens")}}


def paid_logs(root, role, key):
    logs = []
    for path in sorted((root / "calls" / role / key).glob("*.jsonl")):
        records = read_jsonl(path)
        if not records:
            raise ValueError("Empty call log")
        for record in records:
            pilot.validate_response(record)
        logs.append({"path": str(path), "sha256": file_hash(path), "attempt_count": len(records),
                     "final_finish_reason": records[-1]["metadata"]["finish_reason"],
                     **{f: sum(r[f] for r in records) for f in
                        ("prompt_tokens", "completion_tokens", "total_tokens")}})
    return logs


def failure_record(root, role, item, diagnostic):
    return {"id": item["id"], "policy_split": item["policy_split"],
            "question_sha256": pilot.digest(item["question"]), "role": role,
            "status": "quarantined", "reason": "output_budget_exhausted",
            "failed_call": diagnostic, "paid_call_logs": paid_logs(root, role, item["id"]),
            "scored_as_normal_answer": False, "requires_resolution": True}


def blocked_record(root, item):
    source = root / "failures" / "base" / f"{item['id']}.json"
    record = json.loads(source.read_text())
    if record["reason"] != "output_budget_exhausted" or record["id"] != item["id"]:
        raise ValueError("Invalid upstream failure")
    return {"id": item["id"], "policy_split": item["policy_split"],
            "question_sha256": pilot.digest(item["question"]), "role": "qwen14",
            "status": "blocked", "reason": "base_output_unresolved", "paid_call_logs": [],
            "dependency": str(source), "dependency_sha256": file_hash(source),
            "scored_as_normal_answer": False, "requires_resolution": True}


def write_failure(root, role, record):
    pilot.locked_json(root / "failures" / role / f"{record['id']}.json", record)


def read_failures(root, role):
    records = {}
    for path in sorted((root / "failures" / role).glob("*.json")):
        record = json.loads(path.read_text())
        if record["id"] in records or path.stem != record["id"] or record["role"] != role:
            raise ValueError("Invalid failure ledger ID/role")
        records[record["id"]] = record
    return records


def handled_prefix(items, rows, failures):
    ids = [i["id"] for i in items]
    row_ids = [r["id"] for r in rows]
    row_set = set(row_ids)
    if len(row_ids) != len(row_set) or row_set & set(failures):
        raise ValueError("Duplicate or conflicting successful/failed rows")
    handled = row_set | set(failures)
    if handled != set(ids[:len(handled)]) or row_ids != [key for key in ids if key in row_set]:
        raise ValueError("Successful rows and failures do not form an ordered stage prefix")
    return len(handled)


def wait_for_resources(root, role):
    spec = pilot.ROLES[role]
    gpu_ids = spec["gpus"].split(",")
    while True:
        result = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.total,memory.free",
                                 "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True, timeout=15)
        memory = {parts[0].strip(): (int(parts[1]), int(parts[2])) for line in result.stdout.splitlines()
                  if (parts := line.split(","))}
        insufficient = {g: {"free_mib": memory[g][1], "required_mib": int(memory[g][0] * .46) + 1024}
                        for g in gpu_ids if memory[g][1] < int(memory[g][0] * .46) + 1024}
        if not insufficient:
            return
        path = root / "status.json"
        status = json.loads(path.read_text()) if path.exists() else {}
        save_json({**status, "stage": role, "phase": "waiting_for_gpu_memory", "gpu_memory": insufficient,
                   "heartbeat_unix": time.time(), "poll_seconds": 60}, path)
        print(f"[math-full] waiting for {role} GPU memory: {insufficient}; no unrelated process will be stopped", flush=True)
        time.sleep(60)
