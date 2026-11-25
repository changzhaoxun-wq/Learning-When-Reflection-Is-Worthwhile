"""Locked full-training math candidates, with a shared held-out validation split."""

import argparse
import contextlib
import fcntl
import json
import os
import signal
import time
from collections import Counter
from pathlib import Path

from . import math_reflector_pilot as pilot
from .analyze_math_reflector_pilot import audit
from .gsm8k_utils import normalize_example_item
from .math_trace_failures import (OutputCapFailure, blocked_record, exhausted_call, failure_record,
                                  handled_prefix, read_failures, wait_for_resources, write_failure)
from .prepare_math_reflector_pilot import DATA, question_key
from .prepare_qa_hard_development import file_hash, locked_write
from .utils import read_jsonl, save_json


DEFAULT_ROOT = pilot.PROJECT / "outputs/math_full_training_v1"
PILOT_ROOT = pilot.PROJECT / "outputs/math_reflector_pilot_v1/run_v1"


def split_items(train, test, pilot_ids, seed=20260910, validation_size=1495):
    ids = [r["id"] for r in train]
    keys = [question_key(r["question"]) for r in train]
    if len(ids) != len(set(ids)) or len(keys) != len(set(keys)):
        raise ValueError("Duplicate training IDs/questions; cannot freeze split")
    if set(ids) & {r["id"] for r in test} or set(keys) & {question_key(r["question"]) for r in test}:
        raise ValueError("Training/test overlap; investigate before generation")
    if not set(pilot_ids) <= set(ids):
        raise ValueError("Pilot IDs missing from training pool")
    eligible = [r for r in train if r["id"] not in pilot_ids]
    if not 0 < validation_size < len(eligible):
        raise ValueError("Invalid validation size")
    ordered = sorted(eligible, key=lambda r: pilot.digest(f"{seed}:{r['id']}"))
    validation = {r["id"] for r in ordered[:validation_size]}
    return [{**r, "policy_split": "validation" if r["id"] in validation else "train"} for r in train]


def needs_reflection(item, direct):
    return item["policy_split"] == "validation" or not direct["correct"]


def make_policy_row(base, candidates, independent=None):
    best = max(candidates, key=lambda c: (c["correct"], -c["token_cost"]))
    complete = len(candidates) == 3
    if not complete and (base["policy_split"] != "train" or not candidates[0]["correct"]):
        raise ValueError("Only correct training Direct may omit reflection candidates")
    row = {**base, "candidate_results": candidates, "best_action": best["action"],
           "candidate_coverage": "complete" if complete else "direct_only_known_correct",
           "missing_actions": [] if complete else ["reflect_1", "reflect_2"],
           "label_rule": "correctness_first_then_min_total_tokens",
           "label_evidence": "observed_candidates" if complete else "correct_direct_is_cheapest",
           "state_features": {"question": base["question"], "initial_answer": base["initial_answer"]}}
    if independent is not None:
        row["independent"] = independent
    return row


def lock_protocol(root, protocol, accept_revision=False, accept_placement_revision=False):
    if accept_revision and accept_placement_revision:
        raise ValueError("Apply only one audited revision at a time")
    protocol = json.loads(json.dumps(protocol))
    path = root / "protocol.json"
    if not path.exists() or json.loads(path.read_text()) == protocol:
        pilot.locked_json(path, protocol)
        return
    if not (accept_revision or accept_placement_revision):
        raise ValueError("Protocol changed; an explicit audited revision is required once")
    revision = "reflector_gpu_placement_v1" if accept_placement_revision else "exhausted_output_quarantine_v1"
    archive = root / "infrastructure_revisions" / revision
    before = json.loads((archive / "before.json").read_text())
    old = json.loads(path.read_text())
    if old != json.loads((archive / "protocol.json").read_text()):
        raise ValueError("Current protocol differs from pre-revision archive")
    if file_hash(archive / "math_full_training_trace.py") != old["pipeline_code_sha256"]["math_full_training_trace.py"]:
        raise ValueError("Archived source hash mismatch")
    for artifact, sha in before["preserved_artifacts_sha256"].items():
        if file_hash(Path(artifact)) != sha:
            raise ValueError(f"Original data changed before revision: {artifact}")
    expected = json.loads(json.dumps(old))
    expected["pipeline_code_sha256"]["math_full_training_trace.py"] = protocol["pipeline_code_sha256"]["math_full_training_trace.py"]
    if accept_placement_revision:
        expected["roles"]["qwen14"]["gpus"] = protocol["roles"]["qwen14"]["gpus"]
        changed = ["explicit reflector GPU argument and placement", "pipeline protocol migration support"]
    else:
        expected["pipeline_code_sha256"]["math_trace_failures.py"] = protocol["pipeline_code_sha256"]["math_trace_failures.py"]
        expected["roles"]["base"]["gpus"] = protocol["roles"]["base"]["gpus"]
        for field in ("hardware_change", "exception_policy", "resource_policy"):
            expected[field] = protocol[field]
        changed = ["pipeline orchestration", "failure quarantine and reporting", "GPU resource waiting", "Base GPU placement"]
    if expected != protocol:
        raise ValueError("Revision includes unapproved prompt, scoring, split, model or budget changes")
    pilot.locked_json(archive / "protocol_after.json", protocol)
    pilot.locked_json(archive / "audit.json", {
        "before_protocol_sha256": file_hash(path), "after_protocol_sha256": file_hash(archive / "protocol_after.json"),
        "preserved_artifacts_sha256": before["preserved_artifacts_sha256"],
        "changed": changed,
        "unchanged": ["generation/scoring modules", "prompts", "thinking", "budgets", "model metadata", "data and split"]})
    temporary = root / "protocol.pending.json"
    pilot.locked_json(temporary, protocol)
    os.replace(temporary, path)


def role_placements(previous, base_gpu, reflector_gpus=None, saved=None):
    roles = {r: dict(previous["roles"][r]) for r in ("base", "qwen14")}
    roles["base"]["gpus"] = base_gpu
    if reflector_gpus is None:
        reflector_gpus = (saved or previous)["roles"]["qwen14"]["gpus"]
    roles["qwen14"]["gpus"] = reflector_gpus
    for role, spec in roles.items():
        ids = spec["gpus"].split(",")
        if not all(g.isascii() and g.isdigit() for g in ids) or len(set(map(int, ids))) != spec["tp"] or len(ids) != spec["tp"]:
            raise ValueError(f"{role} requires {spec['tp']} distinct numeric GPU IDs; TP must not change")
    return roles


def prepare(root, base_gpu, accept_revision=False, reflector_gpus=None, accept_placement_revision=False):
    # Validate the entire pilot and its locked model/prompt metadata before reuse.
    _, stages, _, audit_info = audit(PILOT_ROOT)
    _, previous = pilot.prepare(PILOT_ROOT)
    train = [normalize_example_item(r, i, "train") for i, r in enumerate(read_jsonl(DATA / "train.jsonl"))]
    test = [normalize_example_item(r, i, "test") for i, r in enumerate(read_jsonl(DATA / "test.jsonl"))]
    if len(train) != 7473:
        raise ValueError("Expected the frozen GSM8K 7473-question training pool")
    pilot_base = {r["id"]: r for r in stages["base"]}
    for item in train:
        if item["id"] in pilot_base:
            for field in ("question", "gold_answer", "answer"):
                if item[field] != pilot_base[item["id"]][field]:
                    raise ValueError(f"Pilot/training mismatch: {item['id']} {field}")
    items = split_items(train, test, set(pilot_base))
    root.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation"):
        selected = [r for r in items if r["policy_split"] == split]
        locked_write(root / f"{split}_items.jsonl", "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in selected))
    manifest = {"seed": 20260910, "total": len(items), "splits": dict(Counter(r["policy_split"] for r in items)),
                "train_ids": [r["id"] for r in items if r["policy_split"] == "train"],
                "validation_ids": [r["id"] for r in items if r["policy_split"] == "validation"],
                "pilot_ids_forced_to_train": sorted(pilot_base),
                "selection": "sha256(seed:id) on non-pilot IDs; no outcome-based validation selection",
                "input_sha256": {name: file_hash(DATA / name) for name in ("train.jsonl", "test.jsonl")},
                "split_sha256": {s: file_hash(root / f"{s}_items.jsonl") for s in ("train", "validation")},
                "test_use": "ID/question overlap exclusion only; no test generation or tuning",
                "historical_limit": "Training domain was used historically; this split is prospective internal validation, not pristine confirmation"}
    pilot.locked_json(root / "manifest.json", manifest)
    saved = json.loads((root / "protocol.json").read_text()) if (root / "protocol.json").exists() else None
    roles = role_placements(previous, base_gpu, reflector_gpus, saved)
    protocol = {**previous, "name": "full_math_selective_train_complete_validation_v1",
                "parent_protocol_sha256": file_hash(PILOT_ROOT / "protocol.json"),
                "roles": roles, "sample_sha256": manifest["input_sha256"]["train.jsonl"],
                "sampling_manifest_sha256": file_hash(root / "manifest.json"),
                "pipeline_code_sha256": {name: file_hash(pilot.PROJECT / "src" / name) for name in
                                         ("math_full_training_trace.py", "prepare_math_reflector_pilot.py", "gsm8k_utils.py",
                                          "analyze_math_reflector_pilot.py", "utils.py", "prepare_qa_hard_development.py",
                                          "math_trace_failures.py")},
                "launcher_sha256": file_hash(pilot.PROJECT / "scripts/run_math_full_training.sh"),
                "train_candidates": "Direct for all; R1/R2 only if Direct incorrect",
                "validation_candidates": "Direct, R1, R2 and independent Qwen3-14B for every question",
                "pilot_reuse": "audited Base for 200 train rows; audited R1/R2 for incorrect pilot rows only",
                "label_rule": "correctness_first_then_min_total_tokens; reward lambda .0001 is reporting only",
                "selector_input": "question and initial_answer only; no gold, candidate scores or manual features",
                "hardware_change": "Base resumes on selected GPU; previously saved Base responses retained; generation settings unchanged",
                "exception_policy": "quarantine only verified exhausted length attempts; keep costs and IDs; no final exports while unresolved",
                "resource_policy": {"action": "wait_for_free_memory", "poll_seconds": 60, "headroom_mib": 1024}}
    lock_protocol(root, protocol, accept_revision, accept_placement_revision)
    pilot.locked_json(root / "pilot_reuse_audit.json", audit_info)
    return items, stages, protocol


class CachedOnly:
    def generate(self, *args, **kwargs):
        raise RuntimeError("Cached call is incomplete; refusing implicit regeneration during audit")


def check_call(call, role, key, action, prompt, tokenizer):
    path = Path(call["call_log"])
    if path.parts[-4:-1] != ("calls", role, key) or path.name != f"{action}.jsonl":
        raise ValueError("Unexpected cached call path")
    if file_hash(path) != call["call_log_sha256"]:
        raise ValueError("Cached call log changed")
    actual = pilot.complete_call(CachedOnly(), tokenizer, path.parents[3], role, key, action, prompt)
    if actual != call:
        raise ValueError("Cached response differs from its durable call record")


def validate_row(row, item, role, tokenizers, base=None):
    for field, value in item.items():
        if row.get(field) != value:
            raise ValueError(f"Changed sample field: {item['id']}/{field}")
    key, question = item["id"], item["question"]
    first = row["initial_call"]
    check_call(first, "base", key, "direct", pilot.MATH_DIRECT_PROMPT.format(question=question), tokenizers["base"])
    if row["initial_answer"] != first["answer_text"]:
        raise ValueError("Initial response mismatch")
    if row["initial_pred"] != pilot.extract_final_number(first["answer_text"]):
        raise ValueError("Initial prediction mismatch")
    if base is not None and first != base["initial_call"]:
        raise ValueError("Base response changed across paths")
    expected = [pilot.candidate("direct", [first], item["gold_answer"])]
    full = role != "base" and needs_reflection(item, expected[0])
    calls = [first]
    if len(row["candidate_results"]) != (3 if full else 1):
        raise ValueError("Candidate coverage mismatch")
    if full:
        for round_id in (1, 2):
            call = row["candidate_results"][round_id]["calls"][-1]
            prompt = pilot.REFLECTION_TEMPLATE.format(question=question, current_answer=calls[-1]["answer_text"], round_id=round_id)
            check_call(call, "qwen14", key, f"reflect_{round_id}", prompt, tokenizers["qwen14"])
            calls = [*calls, call]
            expected.append(pilot.candidate(f"reflect_{round_id}", calls, item["gold_answer"]))
    if row["candidate_results"] != expected:
        raise ValueError("Candidate scoring, cost or prefix mismatch")
    independent = None
    if role != "base" and item["policy_split"] == "validation":
        call = row["independent"]["calls"][0]
        check_call(call, "qwen14", key, "independent", pilot.MATH_DIRECT_PROMPT.format(question=question), tokenizers["qwen14"])
        independent = pilot.candidate("independent", [call], item["gold_answer"])
        if row["independent"] != independent:
            raise ValueError("Independent control mismatch")
    elif "independent" in row:
        raise ValueError("Unexpected independent candidate")
    if role != "base" and row != make_policy_row(base, expected, independent):
        raise ValueError("Policy label or metadata mismatch")


def append_row(path, row):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def validate_failure(root, role, record, item, tokenizers, base=None):
    if record["reason"] == "base_output_unresolved":
        if role != "qwen14" or base is not None or record != blocked_record(root, item):
            raise ValueError("Invalid blocked reflection record")
        return
    key, question = item["id"], item["question"]
    action = record["failed_call"]["action"]
    if (role == "base" and action != "direct") or (role == "qwen14" and action not in ("reflect_1", "reflect_2", "independent")):
        raise ValueError("Unexpected failed action")
    if role == "qwen14" and base is None:
        raise ValueError("Missing successful Base for reflector failure")
    if action == "independent" and item["policy_split"] != "validation":
        raise ValueError("Independent action outside validation")
    prompt = pilot.MATH_DIRECT_PROMPT.format(question=question)
    if action.startswith("reflect_"):
        if not needs_reflection(item, base["candidate_results"][0]):
            raise ValueError("Unscheduled reflection failure")
        current = base["initial_answer"]
        if action == "reflect_2":
            first_prompt = pilot.REFLECTION_TEMPLATE.format(question=question, current_answer=current, round_id=1)
            prior = pilot.complete_call(CachedOnly(), tokenizers[role], root, role, key, "reflect_1", first_prompt)
            current = prior["answer_text"]
        prompt = pilot.REFLECTION_TEMPLATE.format(question=question, current_answer=current, round_id=int(action[-1]))
    diagnostic = exhausted_call(root, role, key, action, prompt, tokenizers[role])
    if diagnostic is None or record != failure_record(root, role, item, diagnostic):
        raise ValueError("Failure ledger no longer matches exhausted call records")


def run_stage(root, role, items, tokenizers, reused, base=None):
    path = root / ("direct_all.jsonl" if role == "base" else "policy_all.jsonl")
    rows = read_jsonl(path) if path.exists() else []
    failures = read_failures(root, role)
    processed = handled_prefix(items, rows, failures)
    item_map = {r["id"]: r for r in items}
    for row in rows:
        item = item_map[row["id"]]
        validate_row(row, item, role, tokenizers, base[row["id"]] if base else None)
    for key, record in failures.items():
        validate_failure(root, role, record, item_map[key], tokenizers, base.get(key) if base else None)
    print(f"[math-full] {role} verified {len(rows)}/{len(items)} saved rows", flush=True)
    save_json({"stage": role, "completed": len(rows), "total": len(items),
               "processed": processed, "quarantined": len(failures),
               "phase": "cached rows verified; generation pending"}, root / "status.json")
    start, initial = time.monotonic(), processed
    with contextlib.ExitStack() as stack:
        client = None

        def complete(item, action, prompt):
            nonlocal client
            diagnostic = exhausted_call(root, role, item["id"], action, prompt, tokenizers[role])
            if diagnostic:
                raise OutputCapFailure(diagnostic)
            if client is None:
                wait_for_resources(root, role)
                client = stack.enter_context(pilot.server(root, role))
            try:
                return pilot.complete_call(client, tokenizers[role], root, role, item["id"], action, prompt)
            except RuntimeError:
                diagnostic = exhausted_call(root, role, item["id"], action, prompt, tokenizers[role])
                if diagnostic:
                    raise OutputCapFailure(diagnostic)
                raise

        def build_row(item):
            key, question = item["id"], item["question"]
            if role == "base":
                if key in reused:
                    first = reused[key]["initial_call"]
                    check_call(first, "base", key, "direct", pilot.MATH_DIRECT_PROMPT.format(question=question), tokenizers[role])
                else:
                    first = complete(item, "direct", pilot.MATH_DIRECT_PROMPT.format(question=question))
                direct = pilot.candidate("direct", [first], item["gold_answer"])
                row = {**item, "initial_call": first, "initial_answer": first["answer_text"],
                       "initial_pred": direct["pred"], "candidate_results": [direct]}
            else:
                initial_row = base[key]
                candidates = initial_row["candidate_results"][:]
                if needs_reflection(item, candidates[0]):
                    calls = [initial_row["initial_call"]]
                    for round_id in (1, 2):
                        prompt = pilot.REFLECTION_TEMPLATE.format(question=question, current_answer=calls[-1]["answer_text"], round_id=round_id)
                        if key in reused:
                            call = reused[key]["candidate_results"][round_id]["calls"][-1]
                            check_call(call, role, key, f"reflect_{round_id}", prompt, tokenizers[role])
                        else:
                            call = complete(item, f"reflect_{round_id}", prompt)
                        calls = [*calls, call]
                        candidates.append(pilot.candidate(f"reflect_{round_id}", calls, item["gold_answer"]))
                independent = None
                if item["policy_split"] == "validation":
                    call = complete(item, "independent", pilot.MATH_DIRECT_PROMPT.format(question=question))
                    independent = pilot.candidate("independent", [call], item["gold_answer"])
                row = make_policy_row(initial_row, candidates, independent)
            return row

        for processed, item in enumerate(items[initial:], start=initial + 1):
            key = item["id"]
            row = None
            if role == "qwen14" and key not in base:
                failures[key] = blocked_record(root, item)
                write_failure(root, role, failures[key])
            else:
                try:
                    row = build_row(item)
                except OutputCapFailure as exc:
                    failures[key] = failure_record(root, role, item, exc.diagnostic)
                    write_failure(root, role, failures[key])
                    print(f"[math-full] quarantined {role}/{key}: {exc}; original calls/costs retained", flush=True)
                else:
                    append_row(path, row)
                    rows.append(row)
            elapsed = time.monotonic() - start
            eta = elapsed / (processed - initial) * (len(items) - processed)
            costs = " ".join(f"{c['action']}:p={c['prompt_tokens']},c={c['completion_tokens']},t={c['token_cost']}" for c in row["candidate_results"]) if row else failures[key]["reason"]
            print(f"[math-full] {role} processed={processed}/{len(items)} valid={len(rows)} pending={len(failures)} {key} {item['policy_split']} {costs} elapsed={elapsed:.0f}s eta~={eta:.0f}s", flush=True)
            save_json({"stage": role, "phase": "generating", "completed": len(rows), "processed": processed,
                       "quarantined": len(failures), "total": len(items), "eta_seconds_rough": eta,
                       "eta_note": "stage estimate; cache reuse and selective coverage vary"}, root / "status.json")
    return rows


def finalize(root, items, tokenizers, base, rows):
    base_failures = read_failures(root, "base")
    failures = read_failures(root, "qwen14")
    if handled_prefix(items, list(base.values()), base_failures) != len(items) or handled_prefix(items, rows, failures) != len(items):
        raise ValueError("Some questions were neither completed nor explicitly recorded as pending")
    item_map = {r["id"]: r for r in items}
    for role, records in (("base", base_failures), ("qwen14", failures)):
        for key, record in records.items():
            validate_failure(root, role, record, item_map[key], tokenizers, base.get(key))
    for row in rows:
        item = item_map[row["id"]]
        validate_row(row, item, "qwen14", tokenizers, base[item["id"]])
    call_paths = {}
    for row in [*base.values(), *rows]:
        for c in row["candidate_results"] + ([row["independent"]] if "independent" in row else []):
            for call in c["calls"]:
                call_paths[call["call_log"]] = call
    unresolved_attempts = 0
    for record in [*base_failures.values(), *failures.values()]:
        for entry in record["paid_call_logs"]:
            path = entry["path"]
            call_paths[path] = {"total_tokens": entry["total_tokens"],
                                "recovered_length_calls": entry["attempt_count"] - 1 if entry["final_finish_reason"] == "stop" else 0}
            if entry["final_finish_reason"] == "length":
                unresolved_attempts += entry["attempt_count"]
    new_paths = {str(p) for p in (root / "calls").rglob("*.jsonl")}
    if not new_paths <= set(call_paths):
        raise ValueError("Unaccounted local call logs")
    pending = set(base_failures) | set(failures)
    unresolved = sum(r["reason"] == "output_budget_exhausted" for r in [*base_failures.values(), *failures.values()])
    report = {"protocol_sha256": file_hash(root / "protocol.json"), "unresolved_truncation": unresolved,
              "unresolved_length_attempts": unresolved_attempts, "expected_total": len(items),
              "pending_ids": sorted(pending), "final_evaluation_ready": not pending,
              "unique_logical_calls": len(call_paths),
              "new_generation_tokens": sum(c["total_tokens"] for p, c in call_paths.items() if p in new_paths),
              "reused_generation_tokens": sum(c["total_tokens"] for p, c in call_paths.items() if p not in new_paths),
              "recovered_length_attempts": sum(c["recovered_length_calls"] for c in call_paths.values()),
              "policy_trained": False, "test_evaluated": False, "splits": {}}
    for split in ("train", "validation"):
        selected = [r for r in rows if r["policy_split"] == split]
        output = root / (f"policy_{split}_partial.jsonl" if pending else f"policy_{split}.jsonl")
        if pending and (root / f"policy_{split}.jsonl").exists():
            raise ValueError("A final export already exists but coverage is now incomplete")
        locked_write(output, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in selected))
        report["splits"][split] = {"n": len(selected), "labels": dict(Counter(r["best_action"] for r in selected)),
                                   "expected_n": sum(i["policy_split"] == split for i in items),
                                   "pending_n": sum(item_map[k]["policy_split"] == split for k in pending),
                                   "complete_candidates": sum(r["candidate_coverage"] == "complete" for r in selected),
                                   "path": str(output), "sha256": file_hash(output)}
    save_json(report, root / ("coverage_audit.json" if pending else "completion_audit.json"))
    save_json({"stage": "awaiting_failure_resolution" if pending else "complete", "rows": len(rows),
               "processed": len(items), "pending": len(pending), "unresolved_truncation": unresolved,
               "generation_pass_finished": True, "final_evaluation_ready": not pending,
               "policy_training_started": False}, root / "status.json")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--base-gpu", default="0")
    parser.add_argument("--reflector-gpus", help="Comma-separated GPU IDs; defaults to the saved placement, preserving TP=2")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--accept-quarantine-revision", action="store_true",
                        help="One-time audited migration; never permits prompt/model/budget changes")
    parser.add_argument("--accept-placement-revision", action="store_true",
                        help="One-time audited reflector GPU placement migration; preserves generation settings and TP")
    args = parser.parse_args()

    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"Received signal {signum}")

    signal.signal(signal.SIGTERM, interrupt)
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / "pipeline.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        items, reused, protocol = prepare(root, args.base_gpu, args.accept_quarantine_revision,
                                         args.reflector_gpus, args.accept_placement_revision)
        # GPU placement changes only in this process, not in the pilot source.
        for role, spec in protocol["roles"].items():
            pilot.ROLES[role]["gpus"] = spec["gpus"]
        from transformers import AutoTokenizer
        tokenizers = {r: AutoTokenizer.from_pretrained(str(pilot.MODELS / spec["model"]), local_files_only=True,
                                                     trust_remote_code=False) for r, spec in protocol["roles"].items()}
        windows = {r: max(pilot.check_window(len(t.encode(pilot.wire_prompt(t, pilot.MATH_DIRECT_PROMPT.format(question=i["question"]), r),
                                                        add_special_tokens=False))) for i in items) for r, t in tokenizers.items()}
        save_json({"rows": len(items), "required_direct_windows": windows, "context": pilot.CONTEXT,
                   "reflection_preflight": "per call using actual prior visible response"}, root / "preflight.json")
        print(f"[math-full] preflight {windows}; train=5978 validation=1495", flush=True)
        if args.preflight_only:
            return
        try:
            base_rows = run_stage(root, "base", items, tokenizers, {r["id"]: r for r in reused["base"]})
            base = {r["id"]: r for r in base_rows}
            rows = run_stage(root, "qwen14", items, tokenizers, {r["id"]: r for r in reused["qwen14"]}, base)
            finalize(root, items, tokenizers, base, rows)
        except BaseException as exc:
            save_json({"stage": "failed_or_interrupted", "error": str(exc), "resume_supported": True,
                       "partial_results_not_final": True}, root / "status.json")
            raise


if __name__ == "__main__":
    main()
