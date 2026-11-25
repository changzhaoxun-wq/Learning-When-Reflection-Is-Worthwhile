from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from .prompts import (
    QA_DIRECT_PROMPT,
    QA_FORMAT_REPAIR_PROMPT,
    QA_FORMAT_REPAIR_PROMPT_VERSION,
    QA_PROMPT_VERSION,
    QA_REFLECTION_PROMPT_ROUND_1,
    QA_REFLECTION_PROMPT_ROUND_2,
    QA_REFLECTION_PROMPT_VERSION,
    QA_TRUNCATION_RECOVERY_PROMPT,
    QA_TRUNCATION_RECOVERY_PROMPT_VERSION,
)
from .reflector import (
    QA_ANSWER_CANONICALIZER_VERSION,
    QA_FORMAT_REPAIR_COMPLETION_SUFFIX,
    QA_FORMAT_REPAIR_MAX_TOKENS,
    QA_FORMAT_REPAIR_MAX_TOKENS_CAP,
    QA_FORMAT_REPAIR_STOP,
    QA_REFLECTION_PRIMARY_MAX_TOKENS,
    QA_TRUNCATION_RECOVERY_MAX_TOKENS,
    QA_TRUNCATION_RECOVERY_MAX_TOKENS_CAP,
)
from .utils import project_root


DATASETS = {
    "hotpotqa": ("hotpotqa/train.jsonl", "hotpotqa/test.jsonl", "test"),
    "2wikimultihopqa": (
        "2wikimultihopqa/train.jsonl",
        "2wikimultihopqa/validation.jsonl",
        "validation",
    ),
    "musique": ("musique/train.jsonl", "musique/validation.jsonl", "validation"),
}


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _optional_limit(name: str, default: str) -> int | None:
    raw = os.environ[name] if name in os.environ else default
    return int(raw) if str(raw).strip() else None


def _string_list_env(name: str, default: str) -> list[str]:
    raw = _env(name, default)
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} must be a JSON string or list of strings") from exc
    values = [value] if isinstance(value, str) else value
    if not isinstance(values, list) or not all(isinstance(item, str) for item in values):
        raise ValueError(f"{name} must be a JSON string or list of strings")
    return [item for item in values if item]


def _run(args: list[str]) -> None:
    print("[qa-direct-first]", " ".join(args), flush=True)
    subprocess.run(args, cwd=project_root(), check=True)


def _row_count(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _require_complete(source: Path, trace: Path, limit: int | None) -> None:
    if not trace.exists() or trace.stat().st_size == 0:
        raise FileNotFoundError(f"Missing trace: {trace}")
    expected = _row_count(source)
    if limit is not None:
        expected = min(expected, limit)
    actual = _row_count(trace)
    if actual != expected:
        raise RuntimeError(f"Trace is incomplete: {trace} has {actual}/{expected} rows")


def _write_locked_protocol(path: Path, protocol: dict[str, Any], allow_change: bool) -> None:
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous != protocol and not allow_change:
            raise RuntimeError(
                "Protocol mismatch. Use a new ROOT; do not mix QA prompts, models, "
                "or token settings in one trace."
            )
    path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _make_config(template: Path, output: Path, data_path: Path, settings: dict[str, Any]) -> None:
    cfg = json.loads(template.read_text(encoding="utf-8"))
    base = {
        "mode": "openai_compatible",
        "base_url": settings["base_url"],
        "api_key_env": "OPENAI_API_KEY",
        "model_name": settings["base_model"],
        "local_model_path": settings["base_path"],
        "temperature": 0.0,
        "max_tokens": settings["base_max_tokens"],
        "max_tokens_cap": settings["base_max_tokens_cap"],
        "retry_on_length": True,
        "fail_on_truncation": True,
        "timeout": 180,
        "max_retries": 3,
        "retry_backoff": 2.0,
        "enable_thinking": False,
        "seed": 42,
        "api_mode": "chat_completions",
    }
    reflector = {
        "mode": "openai_compatible",
        "base_url": settings["reflector_url"],
        "api_key_env": "OPENAI_API_KEY",
        "model_name": settings["reflector_model"],
        "local_model_path": settings["reflector_path"],
        "temperature": 0.0,
        "max_tokens": settings["reflector_max_tokens"],
        "max_tokens_cap": settings["reflector_max_tokens_cap"],
        "retry_on_length": True,
        "fail_on_truncation": True,
        "timeout": 180,
        "max_retries": 3,
        "retry_backoff": 2.0,
        "api_mode": settings["reflector_api_mode"],
        "completion_prompt_prefix": settings["reflector_prefix"],
        "completion_prompt_suffix": settings["reflector_suffix"],
        "stop": settings["reflector_stop"],
        "seed": 42,
    }
    cfg.update(
        seed=42,
        lambda_cost=settings["lambda_cost"],
        qa_f1_threshold=settings["qa_f1_threshold"],
        llm=dict(base),
        base_llm=dict(base),
        reflector_llm=reflector,
        verifier={"mode": "rule_based"},
        actions=["direct"],
        experiment_protocol=settings["protocol"],
    )
    cfg.setdefault("data", {}).update(
        data_path=str(data_path),
        eval_data_path=str(data_path),
        train_limit=None,
        eval_limit=None,
    )
    output.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="QA direct-first trace generation with strict truncation protection.")
    parser.add_argument(
        "--phase",
        default=_env("PHASE", "preflight"),
        choices=[
            "preflight", "direct_train", "train_reflect", "prepare",
            "direct_eval", "eval_reflect", "audit_train", "audit_eval", "all",
        ],
    )
    parser.add_argument("--dataset", default=_env("DATASET", "hotpotqa"))
    args = parser.parse_args()

    dataset = "2wikimultihopqa" if args.dataset == "2wiki" else args.dataset
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset={dataset}; choose {sorted(DATASETS)}")
    train_rel, eval_rel, eval_split = DATASETS[dataset]
    data_root = Path(_env(
        "DATA_ROOT",
        os.environ.get("DATA_ROOT", "data/prepared/adaptive_reasoning"),
    ))
    root = Path(_env("ROOT", f"outputs/qa_direct_first/{dataset}"))
    if not root.is_absolute():
        root = project_root() / root
    root.mkdir(parents=True, exist_ok=True)
    audit_dir = root / "audits"
    audit_dir.mkdir(parents=True, exist_ok=True)

    train_data = Path(_env("TRAIN_DATA", str(data_root / train_rel)))
    eval_data = Path(_env("EVAL_DATA", str(data_root / eval_rel)))
    eval_split = _env("EVAL_SPLIT", eval_split)
    template = Path(_env("CONFIG_TEMPLATE", "configs/default_config.json"))
    if not template.is_absolute():
        template = project_root() / template
    train_limit = _optional_limit("TRAIN_LIMIT", "8000")
    eval_limit = _optional_limit("EVAL_LIMIT", "")
    reflect_workers = int(_env("REFLECT_WORKERS", "1"))
    if reflect_workers < 1:
        raise ValueError("REFLECT_WORKERS must be at least 1")

    settings: dict[str, Any] = {
        "base_url": _env("BASE_URL", "http://127.0.0.1:8000/v1"),
        "base_model": _env("BASE_MODEL", "Qwen3-8B-Instruct"),
        "base_path": _env("BASE_PATH", "Qwen/Qwen3-8B-Instruct"),
        "reflector_url": _env("REFLECTOR_URL", "http://127.0.0.1:8001/v1"),
        "reflector_model": _env("REFLECTOR_MODEL", "DeepSeek-R1-Distill-Qwen-7B"),
        "reflector_path": _env("REFLECTOR_PATH", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"),
        "max_model_len": int(_env("MAX_MODEL_LEN", "8192")),
        "base_max_tokens": int(_env("BASE_MAX_TOKENS", "1024")),
        "base_max_tokens_cap": int(_env("BASE_MAX_TOKENS_CAP", "2048")),
        "reflector_max_tokens": int(_env("REFLECTOR_MAX_TOKENS", "256")),
        "reflector_max_tokens_cap": int(_env("REFLECTOR_MAX_TOKENS_CAP", "256")),
        "reflector_api_mode": _env("REFLECTOR_API_MODE", "completions"),
        "reflector_prefix": _env("REFLECTOR_PREFIX", "<｜User｜>"),
        "reflector_suffix": _env("REFLECTOR_SUFFIX", "<｜Assistant｜><think>\n</think>\n"),
        "reflector_stop": _string_list_env(
            "REFLECTOR_STOP",
            '["\\n\\nEvidence chain:", "\\n\\nEvidence audit:"]',
        ),
        "lambda_cost": float(_env("LAMBDA_COST", "0.0001")),
        "qa_f1_threshold": float(_env("QA_F1_THRESHOLD", "0.8")),
        "context_safety_tokens": int(_env("CONTEXT_SAFETY_TOKENS", "256")),
    }
    required = [template, train_data, eval_data, Path(settings["base_path"]), Path(settings["reflector_path"])]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Missing required path: {path}")

    reflection_text = "\n".join([QA_REFLECTION_PROMPT_ROUND_1, QA_REFLECTION_PROMPT_ROUND_2])
    protocol = {
        "name": "qa_direct_first_no_truncation_v14_distractor",
        "dataset": dataset,
        "train_data": str(train_data),
        "eval_data": str(eval_data),
        "qa_direct_prompt_version": QA_PROMPT_VERSION,
        "qa_direct_prompt_sha256": hashlib.sha256(QA_DIRECT_PROMPT.encode()).hexdigest(),
        "qa_reflection_prompt_version": QA_REFLECTION_PROMPT_VERSION,
        "qa_reflection_prompt_sha256": hashlib.sha256(reflection_text.encode()).hexdigest(),
        "base_model": settings["base_model"],
        "base_path": settings["base_path"],
        "base_enable_thinking": False,
        "base_temperature": 0.0,
        "base_max_tokens": settings["base_max_tokens"],
        "base_max_tokens_cap": settings["base_max_tokens_cap"],
        "reflector_model": settings["reflector_model"],
        "reflector_path": settings["reflector_path"],
        "reflector_temperature": 0.0,
        "reflector_max_tokens": settings["reflector_max_tokens"],
        "reflector_max_tokens_cap": settings["reflector_max_tokens_cap"],
        "reflector_api_mode": settings["reflector_api_mode"],
        "reflector_completion_prompt_prefix": settings["reflector_prefix"],
        "reflector_completion_prompt_suffix": settings["reflector_suffix"],
        "reflector_stop": settings["reflector_stop"],
        "qa_reflection_primary_max_tokens": QA_REFLECTION_PRIMARY_MAX_TOKENS,
        "qa_reflection_primary_retry_on_length": False,
        "qa_truncation_recovery_enabled": True,
        "qa_truncation_recovery_prompt_version": QA_TRUNCATION_RECOVERY_PROMPT_VERSION,
        "qa_truncation_recovery_prompt_sha256": hashlib.sha256(
            QA_TRUNCATION_RECOVERY_PROMPT.encode()
        ).hexdigest(),
        "qa_truncation_recovery_max_tokens": QA_TRUNCATION_RECOVERY_MAX_TOKENS,
        "qa_truncation_recovery_max_tokens_cap": QA_TRUNCATION_RECOVERY_MAX_TOKENS_CAP,
        "qa_answer_canonicalizer_version": QA_ANSWER_CANONICALIZER_VERSION,
        "qa_format_repair_enabled": True,
        "qa_format_repair_prompt_version": QA_FORMAT_REPAIR_PROMPT_VERSION,
        "qa_format_repair_prompt_sha256": hashlib.sha256(
            QA_FORMAT_REPAIR_PROMPT.encode()
        ).hexdigest(),
        "qa_format_repair_max_tokens": QA_FORMAT_REPAIR_MAX_TOKENS,
        "qa_format_repair_max_tokens_cap": QA_FORMAT_REPAIR_MAX_TOKENS_CAP,
        "qa_format_repair_stop": QA_FORMAT_REPAIR_STOP,
        "qa_format_repair_completion_suffix": QA_FORMAT_REPAIR_COMPLETION_SUFFIX,
        "qa_format_repair_truncation_policy": "discard_repair_and_extract_complete_draft",
        "max_model_len": settings["max_model_len"],
        "retry_on_length": True,
        "fail_on_truncation": True,
    }
    settings["protocol"] = protocol
    _write_locked_protocol(
        root / "protocol.json",
        protocol,
        allow_change=_env("ALLOW_PROTOCOL_CHANGE", "0") == "1",
    )

    train_config = root / "train_config.json"
    eval_config = root / "eval_config.json"
    _make_config(template, train_config, train_data, settings)
    _make_config(template, eval_config, eval_data, settings)

    direct_train = Path(_env("DIRECT_TRAIN", str(root / "direct_train.jsonl")))
    direct_eval = Path(_env("DIRECT_EVAL", str(root / "direct_eval.jsonl")))
    selective_train = Path(_env("SELECTIVE_TRAIN", str(root / "policy_train_selective.jsonl")))
    full_eval = Path(_env("FULL_EVAL", str(root / "policy_eval_full.jsonl")))
    hierarchical_dir = Path(_env("HIERARCHICAL_DIR", str(root / "hierarchical_accuracy")))
    target_train = _env("TARGET_TRAIN", "")
    target_eval = _env("TARGET_EVAL", "")
    resume = [] if _env("FORCE", "0") == "1" else ["--resume"]
    python = sys.executable

    def preflight(split: str, data: Path, output: Path) -> None:
        command = [
            python, "-m", "src.preflight_qa_context", "--data", str(data),
            "--split", split, "--base_tokenizer", settings["base_path"],
            "--reflector_tokenizer", settings["reflector_path"],
            "--max_model_len", str(settings["max_model_len"]),
            "--base_max_tokens_cap", str(settings["base_max_tokens_cap"]),
            "--reflector_max_tokens_cap", str(settings["reflector_max_tokens_cap"]),
            "--safety_tokens", str(settings["context_safety_tokens"]),
            "--reflector_api_mode", settings["reflector_api_mode"],
            "--reflector_prefix", settings["reflector_prefix"],
            "--reflector_suffix", settings["reflector_suffix"], "--output", str(output),
        ]
        if _env("PREFLIGHT_FULL_SCAN", "0") == "1":
            command.append("--full_scan")
        _run(command)

    def direct(split: str, config: Path, output: Path, limit: int | None) -> None:
        command = [python, "-m", "src.build_policy_dataset", "--config", str(config),
                   "--split", split, "--actions", "direct", "--output", str(output), *resume]
        if limit is not None:
            command.extend(["--limit", str(limit)])
        _run(command)

    def audit(trace: Path, output: Path) -> None:
        _run([python, "-m", "src.audit_trace_completion", "--trace", str(trace),
              "--output", str(output), "--require_finish_reason", "--require_final_answer"])

    def train_reflect() -> None:
        _require_complete(train_data, direct_train, train_limit)
        audit(direct_train, audit_dir / "direct_train.json")
        command = [python, "-m", "src.build_frozen_base_reflector_trace",
                   "--canonical", str(direct_train), "--config", str(train_config),
                   "--output", str(selective_train), "--incorrect-only",
                   "--max-workers", str(reflect_workers), *resume]
        if train_limit is not None:
            command.extend(["--limit", str(train_limit)])
        if target_train:
            command.extend(["--target", target_train])
        _run(command)
        audit(selective_train, audit_dir / "policy_train_selective.json")

    def eval_reflect() -> None:
        _require_complete(eval_data, direct_eval, eval_limit)
        audit(direct_eval, audit_dir / "direct_eval.json")
        command = [python, "-m", "src.build_frozen_base_reflector_trace",
                   "--canonical", str(direct_eval), "--config", str(eval_config),
                   "--output", str(full_eval),
                   "--max-workers", str(reflect_workers), *resume]
        if eval_limit is not None:
            command.extend(["--limit", str(eval_limit)])
        if target_eval:
            command.extend(["--target", target_eval])
        _run(command)
        audit(full_eval, audit_dir / "policy_eval_full.json")

    def prepare() -> None:
        _require_complete(train_data, selective_train, train_limit)
        audit(selective_train, audit_dir / "policy_train_selective.json")
        _run([python, "-m", "src.prepare_hierarchical_policy_dataset",
              "--input", str(selective_train), "--output_dir", str(hierarchical_dir),
              "--objective", "accuracy", "--lambda_cost", str(settings["lambda_cost"])])

    phase = args.phase
    if phase in {"preflight", "all"}:
        preflight("train", train_data, audit_dir / "preflight_train.json")
        preflight(eval_split, eval_data, audit_dir / "preflight_eval.json")
    if phase in {"direct_train", "all"}:
        if phase == "direct_train":
            preflight("train", train_data, audit_dir / "preflight_train.json")
        direct("train", train_config, direct_train, train_limit)
        audit(direct_train, audit_dir / "direct_train.json")
    if phase in {"train_reflect", "all"}:
        train_reflect()
    if phase in {"prepare", "all"}:
        prepare()
    if phase in {"direct_eval", "all"}:
        if phase == "direct_eval":
            preflight(eval_split, eval_data, audit_dir / "preflight_eval.json")
        direct(eval_split, eval_config, direct_eval, eval_limit)
        audit(direct_eval, audit_dir / "direct_eval.json")
    if phase in {"eval_reflect", "all"}:
        eval_reflect()
    if phase == "audit_train":
        audit(selective_train, audit_dir / "policy_train_selective.json")
    if phase == "audit_eval":
        audit(full_eval, audit_dir / "policy_eval_full.json")

    print(f"[qa-direct-first] dataset={dataset} phase={phase} complete; root={root}")


if __name__ == "__main__":
    main()
