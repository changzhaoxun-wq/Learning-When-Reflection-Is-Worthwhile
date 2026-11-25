"""Generate complete, locked development candidates without touching confirmation."""

import argparse
import fcntl
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

from . import prompts, reflector
from .audit_trace_completion import audit
from .gsm8k_utils import normalize_example_item
from .preflight_qa_context import _load_tokenizer, _token_count
from .prepare_qa_hard_development import PROJECT, ROOT, file_hash, rows
from .qa_reflection_profile import profile_hash, profile_prompt
from .utils import save_json


def check_lock(root: Path) -> dict:
    manifest = json.loads((root / "manifest.json").read_text())
    for filename, digest in manifest["artifacts_sha256"].items():
        if file_hash(root / filename) != digest:
            raise ValueError(f"Frozen development artifact changed: {filename}")
    for filename, digest in manifest["frozen_code_sha256"].items():
        if file_hash(Path(filename)) != digest:
            raise ValueError(f"Generation/scoring code changed: {filename}; create a new ROOT")
    if file_hash(Path(manifest["reference_policy"])) != manifest["reference_policy_sha256"]:
        raise ValueError("Frozen reference selector weights changed")
    return manifest


def service_status(config: dict) -> dict:
    url = config["base_url"].rstrip("/") + "/models"
    key = os.environ.get(config.get("api_key_env", "OPENAI_API_KEY"), "EMPTY")
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=5) as response:
            models = json.load(response).get("data", [])
        selected = [m for m in models if m.get("id") == config["model_name"]]
        if not selected:
            return {"ready": False, "url": url, "reason": "Expected model name not served"}
        return {"ready": True, "url": url, "model": selected[0]}
    except (OSError, ValueError) as exc:
        return {"ready": False, "url": url, "reason": str(exc)}


def require_service(config: dict):
    status = service_status(config)
    if not status["ready"]:
        raise RuntimeError(f"Required inference service is unavailable: {status}. Start the specified model and rerun this phase; completed rows are retained.")
    length = status["model"].get("max_model_len")
    if length and int(length) < 8192:
        raise RuntimeError(f"Service context window is {length}; development protocol requires at least 8192")


def run_module(module: str, *args):
    command = [sys.executable, "-m", module, *map(str, args)]
    print("[qa-hard-dev] " + " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT, check=True)


def validate_trace(path: Path, manifest: dict, actions: list[str], profile: dict | None = None,
                   complete: bool = True, expected_ids: list[str] | None = None):
    expected = manifest["development_ids"] if expected_ids is None else expected_ids
    actual = []
    for row in rows(path):
        actual.append(row["id"])
        if [c["action"] for c in row["candidate_results"]] != actions:
            raise ValueError(f"Incomplete candidate coverage for {row['id']}")
        if row["initial_call"]["prompt"] != prompts.direct_prompt(row["question"], "qa"):
            raise ValueError(f"Direct prompt mismatch: {row['id']}")
        if profile and row.get("qa_profile_sha256") != profile_hash(profile):
            raise ValueError(f"Reflection profile mismatch: {row['id']}")
    if actual != expected[:len(actual)] or (complete and actual != expected):
        raise ValueError(f"Trace IDs/order differ from locked development split: {path}, {len(actual)}/{len(expected)} rows")
    result = audit(path)
    for field in ("final_truncated_count", "missing_finish_reason_count", "missing_final_answer_count", "usage_mismatch_count"):
        if result[field]:
            raise ValueError(f"Trace audit failed: {field}={result[field]} in {path}")
    return result


def preflight(root: Path, cfg: dict, manifest: dict, source: str = "direct",
              dataset_name: str = "development.jsonl", expected_ids: list[str] | None = None):
    expected = manifest["development_ids"] if expected_ids is None else expected_ids
    settings = cfg["base_llm"] if source == "direct" else cfg["reflector_llm"]
    tokenizer = _load_tokenizer(settings["local_model_path"])
    maximum, worst_id = 0, None
    if source == "direct":
        for index, raw in enumerate(rows(root / dataset_name)):
            row = normalize_example_item(raw, index, "train")
            length = _token_count(tokenizer, prompts.direct_prompt(row["question"], "qa"), "chat_completions", "", "")
            required = length + int(settings["max_tokens_cap"]) + 256
            if required > maximum:
                maximum, worst_id = required, row["id"]
    else:
        profile = cfg["qa_reflection_profile"]
        validate_trace(root / "direct.jsonl", manifest, ["direct"], expected_ids=expected)
        count = lambda text, suffix: _token_count(tokenizer, text, "completions", settings["completion_prompt_prefix"], suffix)
        for row in rows(root / "direct.jsonl"):
            span = reflector.extract_final_text_answer(row["initial_answer"]) or row["initial_answer"]
            primary = profile_prompt(profile, row["question"], row["initial_answer"], 1)
            repair = profile["format_repair"].format(question=row["question"], current_answer=f"Final answer: {span.strip()}", draft_answer="")
            recovery = prompts.qa_truncation_recovery_prompt(row["question"])
            required = max(
                count(primary, profile["completion_suffix"]) + 256,
                count(repair, reflector.QA_FORMAT_REPAIR_COMPLETION_SUFFIX) + 256 + 256,
                count(recovery, reflector.QA_FORMAT_REPAIR_COMPLETION_SUFFIX) + 256,
            ) + 256
            if required > maximum:
                maximum, worst_id = required, row["id"]
    result = {"phase": source, "rows": len(expected), "max_model_len": 8192,
              "max_required_with_safety": maximum, "worst_id": worst_id, "fits": maximum <= 8192,
              "full_scan": True, "llm_calls": 0}
    save_json(result, root / "audits" / f"preflight_{source}.json")
    print(json.dumps(result), flush=True)
    if not result["fits"]:
        raise RuntimeError("Development prompts exceed the frozen 8192 context budget; stop before making requests")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--phase", choices=("prepare", "check", "preflight", "direct", "v5", "v7", "audit", "all"), default="all")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / "pipeline.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.phase == "prepare" or (args.phase == "all" and not (root / "manifest.json").exists()):
            run_module("src.prepare_qa_hard_development", "--root", root)
        manifest = check_lock(root)
        configs = {name: json.loads((root / f"configs/{name}.json").read_text()) for name in ("direct", "v5", "v7")}
        if args.phase in ("prepare", "check"):
            print(json.dumps({"locked": True, "development_rows": len(manifest["development_ids"]),
                              "confirmation_rows_reserved": len(manifest["confirmation_ids"]),
                              "base": service_status(configs["direct"]["base_llm"]),
                              "reflector": service_status(configs["v7"]["reflector_llm"])}, indent=2))
            return
        if args.phase in ("preflight", "direct", "all"):
            preflight(root, configs["direct"], manifest)
        if args.phase in ("direct", "all"):
            output = root / "direct.jsonl"
            completed = 0
            if output.exists():
                completed = validate_trace(output, manifest, ["direct"], complete=False)["rows"]
            if completed < len(manifest["development_ids"]):
                require_service(configs["direct"]["base_llm"])
                run_module("src.build_policy_dataset", "--config", root / "configs/direct.json", "--split", "train", "--actions", "direct", "--output", output, "--resume")
            save_json(validate_trace(output, manifest, ["direct"]), root / "audits/direct.json")
        for name in ("v5", "v7"):
            if args.phase not in (name, "all"):
                continue
            cfg = configs[name]
            preflight(root, cfg, manifest, name)
            output = root / f"{name}_raw.jsonl"
            completed = 0
            if output.exists():
                completed = validate_trace(output, manifest, ["direct", "reflect_1"], cfg["qa_reflection_profile"], complete=False)["rows"]
            if completed < len(manifest["development_ids"]):
                require_service(cfg["reflector_llm"])
                run_module("src.build_frozen_base_reflector_trace", "--canonical", root / "direct.jsonl", "--config", root / f"configs/{name}.json", "--output", output,
                           "--reflection-actions", "reflect_1", "--max-workers", args.workers, "--resume")
            save_json(validate_trace(output, manifest, ["direct", "reflect_1"], cfg["qa_reflection_profile"]), root / f"audits/{name}.json")
        if args.phase in ("audit", "all"):
            for name in ("v5", "v7"):
                output = root / f"{name}_raw.jsonl"
                save_json(validate_trace(output, manifest, ["direct", "reflect_1"], configs[name]["qa_reflection_profile"]), root / f"audits/{name}.json")
                run_module("src.relabel_qa_policy_dataset", "--input", output, "--output", root / f"{name}.jsonl",
                           "--quarantine_output", root / f"audits/{name}_quarantine.jsonl", "--summary_output", root / f"audits/{name}_relabel.json",
                           "--raw_hotpot_root", os.environ.get("RAW_HOTPOT_ROOT", "data/hotpotqa/distractor"), "--qa_f1_threshold", "0.8", "--lambda_cost", "0.0001")
            old = list(rows(root / "v5.jsonl"))
            new = list(rows(root / "v7.jsonl"))
            from .qa_dual_candidate_policy import align_dual_rows
            pairs = align_dual_rows(old, new, require_complete=True)
            for pair in pairs:
                if pair["old_row"]["initial_call"] != pair["new_row"]["initial_call"]:
                    raise ValueError(f"Frozen Base differs: {pair['id']}")
            save_json({"raw_rows": len(manifest["development_ids"]), "aligned_rows": len(pairs),
                       "confirmation_evaluated": False, "next_phase": "source-specific development calibration"}, root / "generation_complete.json")
        print(f"[qa-hard-dev] {args.phase} completed; confirmation split remains reserved", flush=True)


if __name__ == "__main__":
    main()
