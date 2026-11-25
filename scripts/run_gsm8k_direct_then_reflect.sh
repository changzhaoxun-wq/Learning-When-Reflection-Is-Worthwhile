#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"
export PYTHONUNBUFFERED=1

PHASE="${PHASE:-direct_train}"
ROOT="${ROOT:-outputs/gsm8k_no_think_rebuild}"
CONFIG_TEMPLATE="${CONFIG_TEMPLATE:-configs/default_config.json}"
TRAIN_DATA="${TRAIN_DATA:-data/gsm8k_train.jsonl}"
TEST_DATA="${TEST_DATA:-data/gsm8k_test.jsonl}"
TRAIN_LIMIT="${TRAIN_LIMIT-}"
TEST_LIMIT="${TEST_LIMIT-}"
FORCE="${FORCE:-0}"
ALLOW_PROTOCOL_CHANGE="${ALLOW_PROTOCOL_CHANGE:-0}"

BASE_URL="${BASE_URL:-http://127.0.0.1:8000/v1}"
BASE_MODEL="${BASE_MODEL:-Qwen3-8B-Instruct}"
BASE_PATH="${BASE_PATH:-Qwen/Qwen3-8B-Instruct}"
REFLECTOR_URL="${REFLECTOR_URL:-http://127.0.0.1:8001/v1}"
REFLECTOR_MODEL="${REFLECTOR_MODEL:-DeepSeek-R1-Distill-Qwen-7B}"
REFLECTOR_PATH="${REFLECTOR_PATH:-deepseek-ai/DeepSeek-R1-Distill-Qwen-7B}"
BASE_MAX_TOKENS="${BASE_MAX_TOKENS:-512}"
REFLECTOR_MAX_TOKENS="${REFLECTOR_MAX_TOKENS:-3072}"
REFLECTOR_API_MODE="${REFLECTOR_API_MODE:-chat_completions}"
LAMBDA_COST="${LAMBDA_COST:-0.0001}"

TRAIN_CONFIG="${ROOT}/train_config.json"
TEST_CONFIG="${ROOT}/test_config.json"
DIRECT_TRAIN="${DIRECT_TRAIN:-${ROOT}/direct_train.jsonl}"
DIRECT_TEST="${DIRECT_TEST:-${ROOT}/direct_test.jsonl}"
TARGET_TRAIN="${TARGET_TRAIN:-}"
SELECTIVE_TRAIN="${SELECTIVE_TRAIN:-${ROOT}/policy_train_selective.jsonl}"
FULL_TEST="${FULL_TEST:-${ROOT}/policy_test_full.jsonl}"
HIERARCHICAL_DIR="${HIERARCHICAL_DIR:-${ROOT}/hierarchical_accuracy}"

for path in "${CONFIG_TEMPLATE}" "${TRAIN_DATA}" "${TEST_DATA}"; do
  [[ -s "${path}" ]] || { echo "Missing required file: ${path}" >&2; exit 1; }
done
mkdir -p "${ROOT}"

lock_protocol() {
  python - "${ROOT}/protocol.json" "${ALLOW_PROTOCOL_CHANGE}" "${BASE_MODEL}" "${BASE_PATH}"     "${REFLECTOR_MODEL}" "${REFLECTOR_PATH}" "${BASE_MAX_TOKENS}" "${REFLECTOR_MAX_TOKENS}" "${REFLECTOR_API_MODE}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

from src.prompts import (
    MATH_DIRECT_PROMPT,
    MATH_PROMPT_VERSION,
    MATH_REFLECTION_PROMPT_ROUND_1,
    MATH_REFLECTION_PROMPT_ROUND_2,
    MATH_REFLECTION_PROMPT_VERSION,
)

output, allow_change, base_model, base_path, reflector_model, reflector_path, base_max, reflector_max, reflector_api_mode = sys.argv[1:]
prompt_path = Path("src/prompts.py")
reflection_text = "\n".join(
    [
        MATH_REFLECTION_PROMPT_ROUND_1,
        MATH_REFLECTION_PROMPT_ROUND_2,
    ]
)
protocol = {
    "name": "gsm8k_no_think_direct_first_v1",
    "prompt_module": str(prompt_path),
    "prompt_sha256": hashlib.sha256(prompt_path.read_bytes()).hexdigest(),
    "math_direct_prompt_version": MATH_PROMPT_VERSION,
    "math_direct_prompt_sha256": hashlib.sha256(
        MATH_DIRECT_PROMPT.encode("utf-8")
    ).hexdigest(),
    "math_reflection_prompt_version": MATH_REFLECTION_PROMPT_VERSION,
    "math_reflection_prompt_sha256": hashlib.sha256(
        reflection_text.encode("utf-8")
    ).hexdigest(),
    "base_model": base_model,
    "base_path": base_path,
    "base_enable_thinking": False,
    "base_temperature": 0.0,
    "base_max_tokens": int(base_max),
    "base_seed": 42,
    "reflector_model": reflector_model,
    "reflector_path": reflector_path,
    "reflector_temperature": 0.0,
    "reflector_max_tokens": int(reflector_max),
    "reflector_api_mode": reflector_api_mode,
    "reflector_completion_prompt_prefix": "<｜User｜>",
    "reflector_completion_prompt_suffix": "<｜Assistant｜><think>\n</think>\n",
}
path = Path(output)
if path.exists():
    previous = json.loads(path.read_text(encoding="utf-8"))
    if previous != protocol and allow_change != "1":
        raise SystemExit(
            "Protocol mismatch. Refusing to mix prompts/models/settings in one trace. "
            "Use a new ROOT, or set ALLOW_PROTOCOL_CHANGE=1 only when intentionally restarting."
        )
path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
}

make_config() {
  local output="$1" data_path="$2"
  python - "${CONFIG_TEMPLATE}" "${output}" "${data_path}" "${ROOT}/protocol.json"     "${BASE_URL}" "${BASE_MODEL}" "${BASE_PATH}" "${REFLECTOR_URL}"     "${REFLECTOR_MODEL}" "${REFLECTOR_PATH}" "${BASE_MAX_TOKENS}"     "${REFLECTOR_MAX_TOKENS}" "${LAMBDA_COST}" "${REFLECTOR_API_MODE}" <<'PY'
import json
import sys

(
    src, dst, data_path, protocol_path, base_url, base_model, base_path,
    reflector_url, reflector_model, reflector_path, base_max, reflector_max,
    lambda_cost, reflector_api_mode,
) = sys.argv[1:]
cfg = json.load(open(src, encoding="utf-8"))
protocol = json.load(open(protocol_path, encoding="utf-8"))
base = {
    "mode": "openai_compatible",
    "base_url": base_url,
    "api_key_env": "OPENAI_API_KEY",
    "model_name": base_model,
    "local_model_path": base_path,
    "temperature": 0.0,
    "max_tokens": int(base_max),
    "timeout": 120,
    "max_retries": 3,
    "retry_backoff": 2.0,
    "enable_thinking": False,
    "seed": 42,
}
reflector = {
    "mode": "openai_compatible",
    "base_url": reflector_url,
    "api_key_env": "OPENAI_API_KEY",
    "model_name": reflector_model,
    "local_model_path": reflector_path,
    "temperature": 0.0,
    "max_tokens": int(reflector_max),
    "timeout": 120,
    "max_retries": 3,
    "retry_backoff": 2.0,
    "api_mode": reflector_api_mode,
    "completion_prompt_prefix": "<｜User｜>",
    "completion_prompt_suffix": "<｜Assistant｜><think>\n</think>\n",
}
cfg["seed"] = 42
cfg["lambda_cost"] = float(lambda_cost)
cfg["llm"] = dict(base)
cfg["base_llm"] = dict(base)
cfg["reflector_llm"] = reflector
cfg["verifier"] = {"mode": "rule_based"}
cfg["actions"] = ["direct"]
cfg.setdefault("data", {})["data_path"] = data_path
cfg["data"]["eval_data_path"] = data_path
cfg["data"]["train_limit"] = None
cfg["data"]["eval_limit"] = None
cfg["experiment_protocol"] = protocol
json.dump(cfg, open(dst, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
PY
}

limit_args() {
  local value="$1"
  if [[ -n "${value}" ]]; then
    printf '%s\n' --limit "${value}"
  fi
}

resume_args=("--resume")
if [[ "${FORCE}" == "1" ]]; then
  resume_args=()
fi

expected_rows() {
  local source="$1" limit="$2"
  local total
  total="$(wc -l < "${source}")"
  if [[ -n "${limit}" && "${limit}" -lt "${total}" ]]; then
    total="${limit}"
  fi
  printf '%s' "${total}"
}

require_complete() {
  local source="$1" trace="$2" limit="$3"
  [[ -s "${trace}" ]] || { echo "Missing direct trace: ${trace}" >&2; exit 1; }
  local expected actual
  expected="$(expected_rows "${source}" "${limit}")"
  actual="$(wc -l < "${trace}")"
  if [[ "${actual}" -lt "${expected}" ]]; then
    echo "Direct trace is incomplete: ${trace} has ${actual}/${expected} required rows." >&2
    echo "Resume its direct phase before generating reflections." >&2
    exit 1
  fi
}

run_direct() {
  local split="$1" config="$2" output="$3" limit="$4"
  local args=()
  if [[ -n "${limit}" ]]; then
    args+=(--limit "${limit}")
  fi
  python -m src.build_policy_dataset     --config "${config}"     --split "${split}"     "${args[@]}"     "${resume_args[@]}"     --actions direct     --output "${output}"
}

run_selective_train() {
  require_complete "${TRAIN_DATA}" "${DIRECT_TRAIN}" "${TRAIN_LIMIT}"
  local args=()
  if [[ -n "${TRAIN_LIMIT}" ]]; then
    args+=(--limit "${TRAIN_LIMIT}")
  fi
  if [[ -n "${TARGET_TRAIN}" ]]; then
    [[ -s "${TARGET_TRAIN}" ]] || { echo "Missing target trace: ${TARGET_TRAIN}" >&2; exit 1; }
    args+=(--target "${TARGET_TRAIN}")
  fi
  python -m src.build_frozen_base_reflector_trace     --canonical "${DIRECT_TRAIN}"     --config "${TRAIN_CONFIG}"     --output "${SELECTIVE_TRAIN}"     --incorrect-only     "${args[@]}"     "${resume_args[@]}"
}

run_full_test() {
  require_complete "${TEST_DATA}" "${DIRECT_TEST}" "${TEST_LIMIT}"
  local args=()
  if [[ -n "${TEST_LIMIT}" ]]; then
    args+=(--limit "${TEST_LIMIT}")
  fi
  python -m src.build_frozen_base_reflector_trace     --canonical "${DIRECT_TEST}"     --config "${TEST_CONFIG}"     --output "${FULL_TEST}"     "${args[@]}"     "${resume_args[@]}"
}

prepare_policy_data() {
  require_complete "${TRAIN_DATA}" "${SELECTIVE_TRAIN}" "${TRAIN_LIMIT}"
  python -m src.prepare_hierarchical_policy_dataset     --input "${SELECTIVE_TRAIN}"     --output_dir "${HIERARCHICAL_DIR}"     --objective accuracy     --lambda_cost "${LAMBDA_COST}"
}

lock_protocol
make_config "${TRAIN_CONFIG}" "${TRAIN_DATA}"
make_config "${TEST_CONFIG}" "${TEST_DATA}"

case "${PHASE}" in
  direct_train)
    run_direct train "${TRAIN_CONFIG}" "${DIRECT_TRAIN}" "${TRAIN_LIMIT}"
    ;;
  train_reflect)
    run_selective_train
    ;;
  prepare)
    prepare_policy_data
    ;;
  direct_test)
    run_direct test "${TEST_CONFIG}" "${DIRECT_TEST}" "${TEST_LIMIT}"
    ;;
  test_reflect)
    run_full_test
    ;;
  all)
    run_direct train "${TRAIN_CONFIG}" "${DIRECT_TRAIN}" "${TRAIN_LIMIT}"
    run_selective_train
    prepare_policy_data
    run_direct test "${TEST_CONFIG}" "${DIRECT_TEST}" "${TEST_LIMIT}"
    run_full_test
    ;;
  *)
    echo "Unknown PHASE=${PHASE}" >&2
    echo "Choose: direct_train, train_reflect, prepare, direct_test, test_reflect, all" >&2
    exit 2
    ;;
esac

echo "[direct-first] phase=${PHASE} complete; root=${ROOT}"
