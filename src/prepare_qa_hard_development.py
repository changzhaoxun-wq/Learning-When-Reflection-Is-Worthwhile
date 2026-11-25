"""Reserve untouched train-hard splits and freeze the historical QA protocols."""

import argparse
import copy
import hashlib
import json
import os
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from . import prompts, reflector
from .qa_reflection_profile import profile_prompt, text_hash, validate_profile


PROJECT = Path(__file__).resolve().parents[1]
TRACE_ROOT = PROJECT / "outputs/qa_direct_first"
DATA_ROOT = Path(os.environ.get("QA_DATA_ROOT", "data/prepared/hotpotqa"))
OLD_TRAIN = TRACE_ROOT / "hotpotqa_counterfactual_train_v1/policy_train_counterfactual_augmented.jsonl"
NEW_TRAIN = TRACE_ROOT / "hotpotqa_target_aware_v7_full/relabel_f1/policy_train_f1_reward.jsonl"
CANONICAL_TRAIN = TRACE_ROOT / "hotpotqa_8000_v7/direct_train.jsonl"
POLICY = TRACE_ROOT / "hotpotqa_reflection_sampling_v1/top_layer_policy_filtered_last3/top_layer_policy.pkl"
ROOT = TRACE_ROOT / "hotpotqa_hard_development_v1"


def rows(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def file_hash(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def original_id(row: dict) -> str:
    value = str(row["id"])
    for prefix in ("hotpotqa-train-", "hotpotqa-validation-", "hotpotqa-test-"):
        if value.startswith(prefix):
            return value[len(prefix):]
    return value


def question_key(row: dict) -> str:
    text = row.get("raw_question") or row["question"]
    text = str(text).rsplit("\nQuestion:\n", 1)[-1]
    return text_hash(" ".join(unicodedata.normalize("NFKC", text).casefold().split()))


def select_splits(train_rows, excluded_ids: set, excluded_questions: set,
                  dev_size: int, confirmation_size: int, seed: int):
    if min(dev_size, confirmation_size) <= 0 or dev_size % 5 or confirmation_size % 5:
        raise ValueError("Split sizes must be positive multiples of five for the fixed 80/20 ratio")
    groups = defaultdict(list)
    counts = Counter()
    seen_ids, seen_questions = set(), set()
    for row in train_rows:
        if not str(row["id"]).startswith("hotpotqa-train-"):
            raise ValueError("Development sampling only accepts official HotpotQA train IDs")
        key, qkey = original_id(row), question_key(row)
        if key in seen_ids:
            raise ValueError(f"Duplicate source ID: {key}")
        seen_ids.add(key)
        counts["raw_rows"] += 1
        if key in excluded_ids or qkey in excluded_questions:
            counts["excluded_used_or_test_overlap"] += 1
            continue
        level = (row.get("metadata") or {}).get("level", row.get("difficulty"))
        source = (row.get("metadata") or {}).get("type", row.get("source"))
        if level != "hard" or source not in ("bridge", "comparison"):
            counts["excluded_non_hard"] += 1
            continue
        if qkey in seen_questions:
            counts["excluded_duplicate_question"] += 1
            continue
        seen_questions.add(qkey)
        groups[source].append(row)
    dev, confirmation = [], []
    for source, numerator in (("bridge", 4), ("comparison", 1)):
        group = sorted(groups[source], key=lambda r: text_hash(f"{seed}:{r['id']}"))
        n_dev, n_confirm = dev_size * numerator // 5, confirmation_size * numerator // 5
        if len(group) < n_dev + n_confirm:
            raise ValueError(f"Not enough unused hard {source}: {len(group)}")
        counts[f"eligible_{source}"] = len(group)
        dev.extend(group[:n_dev])
        confirmation.extend(group[n_dev:n_dev + n_confirm])
    order = lambda r: text_hash(f"order:{seed}:{r['id']}")
    return sorted(dev, key=order), sorted(confirmation, key=order), dict(counts)


def _template_from_call(text: str, values: dict) -> str:
    escaped = lambda s: s.replace("{", "{{").replace("}", "}}")
    result = escaped(text)
    for field, value in sorted(values.items(), key=lambda item: -len(item[1])):
        value = escaped(value)
        if not value or result.count(value) != 1:
            raise ValueError(f"Cannot unambiguously recover template field {field}")
        result = result.replace(value, "{" + field + "}", 1)
    return result


def recover_profiles(old_trace: Path, new_trace: Path, protocol: dict):
    old = {"version": "qa_independent_evidence_repair_v5", "prepend_decision": False}
    for row in rows(old_trace):
        for candidate in row["candidate_results"]:
            for call in candidate.get("metadata", {}).get("calls", [])[1:]:
                round_id = int(call["round_id"])
                if round_id in (1, 2):
                    template = _template_from_call(call["prompt"], {"question": row["question"]})
                    key = f"round_{round_id}"
                    if key in old and old[key] != template:
                        raise ValueError("Mixed v5 templates in reference training trace")
                    old[key] = template
                meta = call.get("response_metadata", {})
                request = meta.get("request", {})
                if request:
                    old["completion_suffix"] = request["completion_prompt_suffix"]
                repair = meta.get("qa_format_repair", {})
                if repair.get("repair_prompt"):
                    old["format_repair"] = _template_from_call(
                        repair["repair_prompt"],
                        {"question": row["question"], "draft_answer": repair["original_output"]},
                    )
        if all(key in old for key in ("round_1", "round_2", "format_repair", "completion_suffix")):
            break
    old["reflection_sha256"] = protocol["qa_reflection_prompt_sha256"]
    old["format_repair_sha256"] = protocol["qa_format_repair_prompt_sha256"]
    validate_profile(old)
    new = {
        "version": prompts.QA_REFLECTION_PROMPT_VERSION,
        "round_1": prompts.QA_REFLECTION_PROMPT_ROUND_1,
        "round_2": prompts.QA_REFLECTION_PROMPT_ROUND_2,
        "format_repair": prompts.QA_FORMAT_REPAIR_PROMPT,
        "completion_suffix": reflector.QA_REFLECTION_COMPLETION_SUFFIX,
        "prepend_decision": True,
    }
    new["reflection_sha256"] = text_hash(new["round_1"] + "\n" + new["round_2"])
    new["format_repair_sha256"] = text_hash(new["format_repair"])
    validate_profile(new)
    checked, repair_checked = 0, 0
    for row in rows(new_trace):
        if row.get("reflection_prompt_version") != new["version"]:
            raise ValueError("Current v7 version differs from the training reference")
        for candidate in row["candidate_results"]:
            if candidate["action"] != "reflect_1":
                continue
            call = candidate["metadata"]["calls"][1]
            expected = profile_prompt(new, row["question"], row["initial_answer"], 1)
            if call["prompt"] != expected:
                raise ValueError("Current v7 prompt differs from stored training calls")
            meta = call["response_metadata"]
            if meta["request"]["completion_prompt_suffix"] != new["completion_suffix"]:
                raise ValueError("v7 completion suffix changed")
            repair = meta.get("qa_format_repair", {})
            if repair.get("repair_prompt"):
                span = reflector.extract_final_text_answer(row["initial_answer"]) or row["initial_answer"]
                expected_repair = new["format_repair"].format(
                    question=row["question"], draft_answer=repair["original_output"],
                    current_answer=f"Final answer: {span.strip()}",
                )
                if repair["repair_prompt"] != expected_repair:
                    raise ValueError("v7 format repair prompt changed")
                repair_checked += 1
            checked += 1
        if checked >= 100 and repair_checked:
            break
    if checked < 1 or repair_checked < 1:
        raise ValueError("Insufficient recorded v7 calls for protocol verification")
    if text_hash(prompts.QA_DIRECT_PROMPT) != protocol["qa_direct_prompt_sha256"]:
        raise ValueError("Base prompt changed relative to the frozen reference")
    if text_hash(prompts.QA_TRUNCATION_RECOVERY_PROMPT) != protocol["qa_truncation_recovery_prompt_sha256"]:
        raise ValueError("Truncation recovery prompt changed")
    return {"v5": old, "v7": new}, {"v7_primary_calls_checked": checked, "v7_repairs_checked": repair_checked}


def locked_write(path: Path, content: str):
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise ValueError(f"Locked artifact differs: {path}. Use a new ROOT.")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(content)


def json_text(value) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def prepare(root: Path, train: Path, test: Path, exclusions: list[Path],
            dev_size=2000, confirmation_size=2000, seed=20260908):
    excluded_ids, excluded_questions = set(), set()
    for path in [*exclusions, test]:
        for row in rows(path):
            excluded_ids.add(original_id(row))
            excluded_questions.add(question_key(row))
    dev, confirmation, counts = select_splits(
        rows(train), excluded_ids, excluded_questions, dev_size, confirmation_size, seed
    )
    old_cfg_path = TRACE_ROOT / "hotpotqa_distractor_8000_v13/train_config.json"
    cfg = json.loads(old_cfg_path.read_text())
    protocol = cfg["experiment_protocol"]
    profiles, profile_audit = recover_profiles(OLD_TRAIN, NEW_TRAIN, protocol)
    artifacts = {}
    for name, selected in (("development", dev), ("confirmation", confirmation)):
        artifacts[f"{name}.jsonl"] = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected)
    for name, profile in profiles.items():
        artifacts[f"profiles/{name}.json"] = json_text(profile)
    manifest = {
        "version": "hotpotqa_hard_development_v1", "seed": seed,
        "train_source": str(train), "counts": counts,
        "development_ids": [r["id"] for r in dev],
        "confirmation_ids": [r["id"] for r in confirmation],
        "sampling": "unused official train-hard; fixed 80% bridge / 20% comparison; no outcome filtering",
        "overlap_guard": "original ID and NFKC/casefold/whitespace-normalized question; excludes all supplied training traces and official test",
        "test_labels_used_for_selection": False,
        "confirmation_usage": "reserved; evaluate once after locking the selector on development",
        "reference_policy": str(POLICY), "reference_policy_sha256": file_hash(POLICY),
        "profile_audit": profile_audit,
        "inputs_sha256": {str(p): file_hash(p) for p in sorted(set([train, test, *exclusions, old_cfg_path]))},
        "frozen_code_sha256": {
            str(PROJECT / "src" / name): file_hash(PROJECT / "src" / name)
            for name in ("prompts.py", "reflector.py", "llm_client.py", "qa_reflection_profile.py",
                         "answer_extractor.py", "reward.py", "relabel_qa_policy_dataset.py")
        },
        "next_calibration": {
            "weights": "frozen last-three-block selector; no optimizer steps",
            "max_invocation_rate": 0.10, "grid_points": 21,
            "thresholds": "one gate threshold and one adoption margin per reflector source",
            "selection_order": ["binary answer accuracy", "standard answer F1", "lower total tokens"],
            "no_positive_development_gain": "retain direct and report the negative result",
            "metrics": ["standard EM", "standard F1", "F1>=0.8 accuracy", "fixes", "harms", "paired McNemar", "paired bootstrap"],
        },
        "token_accounting": "Base once + both reflection deltas when invoked; all input/output/retry/recovery/format-repair tokens. Selector GPU latency is separate from LLM-token cost.",
    }
    for name in ("direct", "v5", "v7"):
        current = copy.deepcopy(cfg)
        selected_profile = profiles.get(name, profiles["v7"])
        current["actions"] = ["direct"]
        current["data"].update(
            data_path=str(root / "development.jsonl"),
            eval_data_path=str(root / "development.jsonl"),
            train_limit=None, eval_limit=None,
        )
        current["reflector_llm"]["completion_prompt_suffix"] = selected_profile["completion_suffix"]
        current["experiment_protocol"] = {
            **protocol, "name": "hotpotqa_hard_development_v1",
            "train_data": str(root / "development.jsonl"),
            "eval_data": str(root / "development.jsonl"),
            "qa_reflection_prompt_version": selected_profile["version"],
            "qa_reflection_prompt_sha256": selected_profile["reflection_sha256"],
            "qa_format_repair_prompt_version": (
                protocol["qa_format_repair_prompt_version"] if name == "v5"
                else prompts.QA_FORMAT_REPAIR_PROMPT_VERSION
            ),
            "qa_format_repair_prompt_sha256": selected_profile["format_repair_sha256"],
            "reflector_completion_prompt_suffix": selected_profile["completion_suffix"],
            "development_manifest": str(root / "manifest.json"),
        }
        if name != "direct":
            current["qa_reflection_profile"] = profiles[name]
        artifacts[f"configs/{name}.json"] = json_text(current)
    manifest["artifacts_sha256"] = {name: text_hash(content) for name, content in artifacts.items()}
    artifacts["manifest.json"] = json_text(manifest)
    for name, content in artifacts.items():
        path = root / name
        if path.exists() and path.read_text(encoding="utf-8") != content:
            raise ValueError(f"Locked artifact differs: {path}. Use a new ROOT.")
    for name, content in artifacts.items():
        locked_write(root / name, content)
    print(json_text({"root": str(root), "development": len(dev), "confirmation_reserved": len(confirmation),
                     "counts": counts, "profile_audit": profile_audit, "llm_calls": 0}))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--train", type=Path, default=DATA_ROOT / "train.jsonl")
    parser.add_argument("--test", type=Path, default=DATA_ROOT / "test.jsonl")
    parser.add_argument("--exclude", action="append", type=Path, default=[])
    parser.add_argument("--dev-size", type=int, default=2000)
    parser.add_argument("--confirmation-size", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260908)
    args = parser.parse_args()
    prepare(args.root.resolve(), args.train.resolve(), args.test.resolve(),
            [OLD_TRAIN, NEW_TRAIN, CANONICAL_TRAIN, *args.exclude],
            args.dev_size, args.confirmation_size, args.seed)


if __name__ == "__main__":
    main()
