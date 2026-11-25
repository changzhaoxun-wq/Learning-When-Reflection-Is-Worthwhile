"""Frozen math pilot with answer-conditioned reflection and durable length retries."""

import argparse
import contextlib
import datetime
import fcntl
import hashlib
import json
import os
import re
import signal
import socket
import subprocess
import time
import urllib.request
from pathlib import Path

from .answer_extractor import answers_equal, extract_final_number
from .llm_client import OpenAICompatibleClient
from .prepare_qa_hard_development import file_hash, locked_write
from .prompts import MATH_DIRECT_PROMPT
from .utils import read_jsonl, save_json, write_jsonl


PROJECT = Path(__file__).resolve().parents[1]
MODELS = Path(os.environ.get("MODEL_ROOT", PROJECT / "models"))
VLLM_PYTHON = os.environ.get("VLLM_PYTHON", "python")
CONTEXT = 32768
OUTPUT_STEPS = (4096, 8192, 16384)
SAFETY = 512
PORT = 8014
REFLECTION_TEMPLATE = """You are revising an existing solution to a grade school math problem.
Question:
{question}
Current answer:
{current_answer}
Reflection round: {round_id}
Check whether the solution answers the requested quantity and uses the problem's
conditions correctly. Check the arithmetic. Correct any errors you find; if the
solution is correct, retain its answer. Give concise reasoning for the final solution.
End with a final line exactly in this format: Final answer: <number>
After the colon write only the numeric answer, without units, Markdown, or LaTeX.
"""
ROLES = {
    "base": {"model": "Qwen3-8B-Instruct", "gpus": "0", "tp": 1, "thinking": False},
    "deepseek": {"model": "DeepSeek-R1-Distill-Qwen-7B", "gpus": "0", "tp": 1, "thinking": None},
    "qwen14": {"model": "Qwen3-14B", "gpus": "1,2", "tp": 2, "thinking": True},
}


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def locked_json(path, value):
    locked_write(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def visible_answer(text, thinking_open=False):
    if "</think>" in text:
        return text.rsplit("</think>", 1)[1].strip()
    if "<think>" in text or thinking_open:
        raise RuntimeError("Unclosed thinking output; stop rather than score as a semantic error")
    return text.strip()


def check_window(prompt_tokens):
    required = prompt_tokens + OUTPUT_STEPS[-1] + SAFETY
    if required > CONTEXT:
        raise RuntimeError(f"Input plus output reserve needs {required}>{CONTEXT}; no input or output cropping allowed")
    return required


def wire_prompt(tokenizer, prompt, role):
    kwargs = {} if ROLES[role]["thinking"] is None else {"enable_thinking": ROLES[role]["thinking"]}
    return tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                         add_generation_prompt=True, **kwargs)


def validate_response(record):
    p, c, total = (record[k] for k in ("prompt_tokens", "completion_tokens", "total_tokens"))
    if any(not isinstance(v, int) or v < 0 for v in (p, c, total)) or p + c != total:
        raise ValueError("Invalid API token accounting")
    usage = record["metadata"].get("raw_usage", {})
    if any(key not in usage for key in ("prompt_tokens", "completion_tokens", "total_tokens")):
        raise ValueError("API usage is required; estimated token costs are not accepted")
    if [usage[k] for k in ("prompt_tokens", "completion_tokens", "total_tokens")] != [p, c, total]:
        raise ValueError("Reported usage does not match response cost")
    if record["metadata"].get("finish_reason") not in ("stop", "length"):
        raise RuntimeError("Missing or unexpected finish_reason")
    for attempt in record["metadata"].get("generation_attempts", []):
        if attempt.get("context_window_retry"):
            raise RuntimeError("Server reduced the output budget; stop instead of comparing clipped generations")


def complete_call(client, tokenizer, root, role, item_id, action, prompt):
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", item_id):
        raise ValueError("Unsafe item ID")
    wire = wire_prompt(tokenizer, prompt, role)
    count = len(tokenizer.encode(wire, add_special_tokens=False))
    required = check_window(count)
    fingerprint = digest(json.dumps({"role": role, "model": ROLES[role]["model"], "wire": wire,
                                     "output_steps": OUTPUT_STEPS, "context": CONTEXT}, sort_keys=True))
    path = root / "calls" / role / item_id / f"{action}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    records = read_jsonl(path) if path.exists() else []
    for i, record in enumerate(records):
        if i >= len(OUTPUT_STEPS) or record["fingerprint"] != fingerprint or record["max_tokens"] != OUTPUT_STEPS[i]:
            raise ValueError("Saved call does not match locked prompt/budget")
        validate_response(record)
        if i < len(records) - 1 and record["metadata"]["finish_reason"] != "length":
            raise ValueError("Saved call continued after normal completion")
    while not records or records[-1]["metadata"]["finish_reason"] == "length":
        if len(records) >= len(OUTPUT_STEPS):
            raise RuntimeError(f"Unresolved truncation for {role}/{item_id}/{action} at {OUTPUT_STEPS[-1]}; attempts retained, no final report")
        budget = OUTPUT_STEPS[len(records)]
        print(f"[math-pilot] {role}/{item_id}/{action} output_budget={budget} prompt_tokens={count}", flush=True)
        response = client.generate(wire, max_tokens=budget, max_tokens_cap=budget,
                                   retry_on_length=False, fail_on_truncation=False)
        record = {"fingerprint": fingerprint, "max_tokens": budget, "prompt": prompt, "wire_prompt": wire,
                  "prompt_tokens_preflight": count, "required_window_with_cap": required,
                  "output": response.text, "prompt_tokens": response.prompt_tokens,
                  "completion_tokens": response.completion_tokens, "total_tokens": response.total_tokens,
                  "metadata": response.metadata}
        # Persist every paid attempt before validation, including unresolved length endings.
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        validate_response(record)
        records.append(record)
    text = records[-1]["output"]
    thinking_open = wire.rfind("<think>") > wire.rfind("</think>")
    answer = visible_answer(text, thinking_open=thinking_open)
    return {"prompt": prompt, "output": text, "answer_text": answer, "finish_reason": "stop",
            "prompt_tokens": sum(r["prompt_tokens"] for r in records),
            "completion_tokens": sum(r["completion_tokens"] for r in records),
            "total_tokens": sum(r["total_tokens"] for r in records),
            "attempt_count": len(records), "recovered_length_calls": len(records) - 1,
            "call_log": str(path), "call_log_sha256": file_hash(path)}


def candidate(action, calls, gold):
    final = calls[-1]["answer_text"]
    pred = extract_final_number(final)
    p = sum(c["prompt_tokens"] for c in calls)
    c = sum(c["completion_tokens"] for c in calls)
    correct = int(answers_equal(pred, str(gold)))
    return {"action": action, "final_answer": final, "pred": pred, "correct": correct,
            "prompt_tokens": p, "completion_tokens": c, "token_cost": p + c, "total_tokens": p + c,
            "reward": correct - .0001 * (p + c),
            "final_marker_present": "final answer:" in final.lower(), "calls": calls}


def prepare(root):
    source = root.parent
    manifest = json.loads((source / "manifest.json").read_text())
    if file_hash(source / "development.jsonl") != manifest["development_sha256"]:
        raise ValueError("Frozen pilot questions changed")
    items = read_jsonl(source / "development.jsonl")
    if [r["id"] for r in items] != manifest["selected_ids"]:
        raise ValueError("Pilot IDs changed")
    model_files = {}
    shard_stats = {}
    for spec in ROLES.values():
        folder = MODELS / spec["model"]
        for name in ("config.json", "tokenizer_config.json", "tokenizer.json", "model.safetensors.index.json"):
            model_files[str(folder / name)] = file_hash(folder / name)
        index = json.loads((folder / "model.safetensors.index.json").read_text())
        for shard in set(index["weight_map"].values()):
            if not (folder / shard).is_file():
                raise ValueError(f"Missing model shard: {folder / shard}")
            stat = (folder / shard).stat()
            shard_stats[str(folder / shard)] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    protocol = {"name": "answer_conditioned_math_pilot_v1", "sample_sha256": file_hash(source / "development.jsonl"),
                "sampling_manifest_sha256": file_hash(source / "manifest.json"), "roles": ROLES,
                "context": CONTEXT, "output_steps": OUTPUT_STEPS, "safety": SAFETY,
                "direct_prompt": MATH_DIRECT_PROMPT, "reflection_template": REFLECTION_TEMPLATE,
                "temperature": 0., "seed": 42, "max_request_retries": 0,
                "model_metadata_sha256": model_files,
                "model_shard_stats": shard_stats,
                "code_sha256": {name: file_hash(PROJECT / "src" / name) for name in
                                ("math_reflector_pilot.py", "llm_client.py", "answer_extractor.py", "prompts.py")},
                "server": {"port": PORT, "gpu_memory_utilization": .46, "max_num_seqs": 1,
                           "max_num_batched_tokens": 1024, "enforce_eager": True},
                "round_2_input": "visible first revision; raw reasoning is retained in call logs",
                "test_data_used": False, "policy_training": False,
                "limitations": ["Native reasoning differs across model families",
                                "Pilot is exploratory and previously used training domain",
                                "Network errors stop execution; unreported server costs cannot be inferred"]}
    locked_json(root / "protocol.json", protocol)
    return items, protocol


def free_port():
    with socket.socket() as sock:
        try:
            # Match server rebinding semantics: TIME_WAIT is not a live listener.
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("127.0.0.1", PORT))
            sock.listen(1)
        except OSError as exc:
            raise RuntimeError(f"Port {PORT} occupied; will not reuse or stop an unrelated service") from exc


@contextlib.contextmanager
def server(root, role):
    free_port()
    spec = ROLES[role]
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log = root / f"vllm_{role}_{stamp}.log"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = spec["gpus"]
    command = [VLLM_PYTHON, "-m", "vllm.entrypoints.openai.api_server", "--model", str(MODELS / spec["model"]),
               "--served-model-name", spec["model"], "--host", "127.0.0.1", "--port", str(PORT),
               "--max-model-len", str(CONTEXT), "--gpu-memory-utilization", "0.46", "--enforce-eager",
               "--max-num-seqs", "1", "--max-num-batched-tokens", "1024", "--tensor-parallel-size", str(spec["tp"])]
    with log.open("x") as handle:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT,
                                   env=env, start_new_session=True)
    save_json({"pid": process.pid, "pgid": process.pid, "role": role, "command": command, "log": str(log)}, root / "active_server.json")
    print(f"[math-pilot] loading {role} GPU={spec['gpus']} log={log}", flush=True)
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"vLLM exited {process.returncode}; inspect {log}")
            try:
                with opener.open(f"http://127.0.0.1:{PORT}/v1/models", timeout=5) as response:
                    served = json.load(response)["data"]
                if any(m["id"] == spec["model"] and int(m.get("max_model_len", 0)) >= CONTEXT for m in served):
                    break
            except (OSError, ValueError, KeyError):
                pass
            time.sleep(5)
        else:
            raise RuntimeError(f"Service readiness timed out; inspect {log}")
        yield OpenAICompatibleClient(f"http://127.0.0.1:{PORT}/v1", "EMPTY", spec["model"],
                                     temperature=0., seed=42, timeout=1800, max_retries=0, api_mode="completions")
    finally:
        # Only the service session created by this context is signalled.
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=45)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=15)
        save_json({"pid": process.pid, "role": role, "stopped": True, "exit_code": process.returncode}, root / "active_server.json")


def summarize(root, items):
    stages = {role: read_jsonl(root / f"{role}.jsonl") for role in ROLES}
    expected = [r["id"] for r in items]
    for stage in stages.values():
        if [r["id"] for r in stage] != expected:
            raise ValueError("Incomplete stage; no partial-sample final report")
    methods = {"direct": [r["candidate_results"][0] for r in stages["base"]]}
    for role in ("deepseek", "qwen14"):
        for index, action in ((1, "reflect_1"), (2, "reflect_2")):
            methods[f"{role}_{action}"] = [r["candidate_results"][index] for r in stages[role]]
        methods[f"{role}_oracle"] = [max(r["candidate_results"], key=lambda c: (c["correct"], -c["token_cost"])) for r in stages[role]]
    methods["qwen14_independent_standalone"] = [r["independent"] for r in stages["qwen14"]]
    methods["base_then_qwen14_independent"] = [candidate("independent", [stages["base"][i]["initial_call"], r["independent"]["calls"][0]], items[i]["gold_answer"]) for i, r in enumerate(stages["qwen14"])]
    n = len(items)
    direct = methods["direct"]
    metrics = []
    for name, candidates in methods.items():
        result = {"method": name, "n": n, "accuracy": sum(c["correct"] for c in candidates) / n}
        for field in ("prompt_tokens", "completion_tokens", "token_cost", "reward"):
            result[f"avg_{field}"] = sum(c[field] for c in candidates) / n
        result["fixes"] = sum(not d["correct"] and c["correct"] for d, c in zip(direct, candidates))
        result["harms"] = sum(d["correct"] and not c["correct"] for d, c in zip(direct, candidates))
        errors = sum(not d["correct"] for d in direct)
        correct = n - errors
        result["fix_rate"] = result["fixes"] / errors if errors else None
        result["harm_rate"] = result["harms"] / correct if correct else None
        result["missing_final_marker"] = sum(not c["final_marker_present"] for c in candidates)
        metrics.append(result)
    files = list((root / "calls").rglob("*.jsonl"))
    records = [read_jsonl(path) for path in files]
    unresolved = sum(r[-1]["metadata"]["finish_reason"] != "stop" for r in records)
    if unresolved:
        raise ValueError("Unresolved calls remain; no final report")
    report = {"methods": metrics, "unresolved_truncation": unresolved,
              "length_attempts_recovered": sum(len(r) - 1 for r in records),
              "total_generation_tokens": sum(a["total_tokens"] for r in records for a in r),
              "protocol_sha256": file_hash(root / "protocol.json"), "role": "exploratory development pilot"}
    save_json(report, root / "results.json")
    lines = ["# Math Reflector Pilot", "", "Complete development coverage; no policy trained or test evaluated.", "",
             "| Method | Accuracy | Prompt | Completion | Total | Fix/Harm |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in metrics:
        lines.append(f"| {row['method']} | {row['accuracy']:.4f} | {row['avg_prompt_tokens']:.1f} | {row['avg_completion_tokens']:.1f} | {row['avg_token_cost']:.1f} | {row['fixes']}/{row['harms']} |")
    lines += ["", "All charged retries include input and output. Oracle is not deployable.",
              "Native reasoning/tokenizers differ; this is a model-system comparison, not an isolated parameter-count effect.",
              f"Recovered length attempts: {report['length_attempts_recovered']}; unresolved truncation: 0."]
    (root / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


def main():
    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"Received signal {signum}")

    signal.signal(signal.SIGTERM, interrupt)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT / "outputs/math_reflector_pilot_v1/run_v1")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / "pipeline.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        items, _ = prepare(root)
        from transformers import AutoTokenizer
        tokenizers = {role: AutoTokenizer.from_pretrained(str(MODELS / spec["model"]), local_files_only=True, trust_remote_code=False) for role, spec in ROLES.items()}
        windows = {}
        for role, tokenizer in tokenizers.items():
            windows[role] = max(check_window(len(tokenizer.encode(wire_prompt(tokenizer, MATH_DIRECT_PROMPT.format(question=r["question"]), role), add_special_tokens=False))) for r in items)
        save_json({"rows": len(items), "direct_required_windows": windows, "reflection_check": "per request with actual prior answer", "context": CONTEXT}, root / "preflight.json")
        print(f"[math-pilot] preflight: {windows}", flush=True)
        if args.preflight_only:
            return
        try:
            base = {}
            for role in ROLES:
                path = root / f"{role}.jsonl"
                existing = read_jsonl(path) if path.exists() else []
                if [r["id"] for r in existing] != [r["id"] for r in items[:len(existing)]] or len(existing) > len(items):
                    raise ValueError("Stage IDs/order do not match sample")
                for row in existing:
                    for c in row["candidate_results"] + ([row["independent"]] if "independent" in row else []):
                        for call in c["calls"]:
                            if file_hash(Path(call["call_log"])) != call["call_log_sha256"]:
                                raise ValueError("Saved call artifact changed")
                if len(existing) < len(items):
                    save_json({"stage": role, "completed": len(existing), "total": len(items)}, root / "status.json")
                    with server(root, role) as client:
                        start = time.monotonic()
                        initial_completed = len(existing)
                        for i, item in enumerate(items[len(existing):], start=len(existing)):
                            question, key = item["question"], item["id"]
                            if role == "base":
                                first = complete_call(client, tokenizers[role], root, role, key, "direct", MATH_DIRECT_PROMPT.format(question=question))
                                row = {**item, "initial_answer": first["answer_text"], "initial_call": first,
                                       "candidate_results": [candidate("direct", [first], item["gold_answer"])]}
                            else:
                                first = base[key]["initial_call"]
                                calls = [first]
                                candidates = [candidate("direct", calls, item["gold_answer"])]
                                for round_id in (1, 2):
                                    prompt = REFLECTION_TEMPLATE.format(question=question, current_answer=calls[-1]["answer_text"], round_id=round_id)
                                    calls = [*calls, complete_call(client, tokenizers[role], root, role, key, f"reflect_{round_id}", prompt)]
                                    candidates.append(candidate(f"reflect_{round_id}", calls, item["gold_answer"]))
                                row = {**item, "initial_answer": first["answer_text"], "initial_call": first, "candidate_results": candidates}
                                if role == "qwen14":
                                    independent = complete_call(client, tokenizers[role], root, role, key, "independent", MATH_DIRECT_PROMPT.format(question=question))
                                    row["independent"] = candidate("independent", [independent], item["gold_answer"])
                            with path.open("a", encoding="utf-8") as output:
                                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                                output.flush()
                                os.fsync(output.fileno())
                            elapsed = time.monotonic() - start
                            eta = elapsed / (i + 1 - initial_completed) * (len(items) - i - 1)
                            print(f"[math-pilot] {role} {i+1}/{len(items)} elapsed={elapsed:.0f}s stage_eta={eta:.0f}s", flush=True)
                            save_json({"stage": role, "completed": i+1, "total": len(items), "stage_eta_seconds": eta}, root / "status.json")
                if role == "base":
                    base = {r["id"]: r for r in read_jsonl(path)}
            summarize(root, items)
            save_json({"stage": "complete", "unresolved_truncation": 0}, root / "status.json")
        except BaseException as exc:
            save_json({"stage": "failed_or_interrupted", "error": str(exc), "resume_supported": True,
                       "partial_results_not_final": True}, root / "status.json")
            raise


if __name__ == "__main__":
    main()
