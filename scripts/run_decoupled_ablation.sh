#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"
export PYTHONUNBUFFERED=1

CONFIG_TEMPLATE="${CONFIG_TEMPLATE:-configs/default_config.json}"
TRAIN_DATA="${TRAIN_DATA:-data/gsm8k_train.jsonl}"
TEST_DATA="${TEST_DATA:-data/gsm8k_test.jsonl}"
LIMIT="${LIMIT-200}"
EVAL_LIMIT="${EVAL_LIMIT-100}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/decoupling_ablation}"
BASE_QWEN3_URL="${BASE_QWEN3_URL:-http://127.0.0.1:8000/v1}"
BASE_QWEN3_MODEL="${BASE_QWEN3_MODEL:-Qwen3-8B-Instruct}"
BASE_QWEN3_PATH="${BASE_QWEN3_PATH:-Qwen/Qwen3-8B-Instruct}"
REFLECTOR_QWEN3_URL="${REFLECTOR_QWEN3_URL:-${BASE_QWEN3_URL}}"
REFLECTOR_QWEN3_MODEL="${REFLECTOR_QWEN3_MODEL:-Qwen3-8B-Instruct}"
REFLECTOR_QWEN3_PATH="${REFLECTOR_QWEN3_PATH:-${BASE_QWEN3_PATH}}"
REFLECTOR_DEEPSEEK_URL="${REFLECTOR_DEEPSEEK_URL:-http://127.0.0.1:8001/v1}"
REFLECTOR_DEEPSEEK_MODEL="${REFLECTOR_DEEPSEEK_MODEL:-DeepSeek-R1-Distill-Qwen-7B}"
REFLECTOR_DEEPSEEK_PATH="${REFLECTOR_DEEPSEEK_PATH:-deepseek-ai/DeepSeek-R1-Distill-Qwen-7B}"
BASE_ALT_URL="${BASE_ALT_URL:-}"
BASE_ALT_MODEL="${BASE_ALT_MODEL:-}"
BASE_ALT_PATH="${BASE_ALT_PATH:-}"
RESUME="${RESUME:-1}"
RESUME_ARGS=()
if [[ "${FORCE:-0}" != "1" && "${RESUME}" == "1" ]]; then
  RESUME_ARGS+=(--resume)
fi

TRAIN_LIMIT_ARGS=()
EVAL_LIMIT_ARGS=()
if [[ -n "${LIMIT}" ]]; then
  TRAIN_LIMIT_ARGS+=(--limit "${LIMIT}")
fi
if [[ -n "${EVAL_LIMIT}" ]]; then
  EVAL_LIMIT_ARGS+=(--limit "${EVAL_LIMIT}")
fi

[[ -s "${TRAIN_DATA}" && -s "${TEST_DATA}" ]] || {
  echo "Missing train/test data" >&2
  exit 1
}

mkdir -p "${OUTPUT_ROOT}"

run_case() {
  local name="$1" base_url="$2" base_model="$3" base_path="$4" reflector_url="$5" reflector_model="$6" reflector_path="$7"
  if [[ -z "${base_url}" || -z "${base_model}" ]]; then
    echo "[decouple] skip ${name}: base endpoint/model not configured"
    return 0
  fi
  local case_dir="${OUTPUT_ROOT}/${name}"
  local train_config="${case_dir}/train_config.json"
  local test_config="${case_dir}/test_config.json"
  mkdir -p "${case_dir}"

  python - "${CONFIG_TEMPLATE}" "${train_config}" "${TRAIN_DATA}" "${base_url}" "${base_model}" "${base_path}" "${reflector_url}" "${reflector_model}" "${reflector_path}" <<'PY'
import json, sys
src, dst, data_path, base_url, base_model, base_path, reflector_url, reflector_model, reflector_path = sys.argv[1:]
cfg=json.load(open(src, encoding='utf-8'))
cfg['data']['data_path']=data_path
cfg['data']['train_limit']=None
cfg['data']['eval_limit']=None
cfg['base_llm'].update({'base_url':base_url, 'model_name':base_model, 'local_model_path':base_path})
cfg['reflector_llm'].update({'base_url':reflector_url, 'model_name':reflector_model, 'local_model_path':reflector_path})
cfg['llm'].update({'base_url':base_url, 'model_name':base_model, 'local_model_path':base_path})
json.dump(cfg, open(dst,'w',encoding='utf-8'), ensure_ascii=False, indent=2)
PY
  cp "${train_config}" "${test_config}"
  python - "${test_config}" "${TEST_DATA}" <<'PY'
import json, sys
p, data_path=sys.argv[1:]
cfg=json.load(open(p, encoding='utf-8'))
cfg['data']['data_path']=data_path
json.dump(cfg, open(p,'w',encoding='utf-8'), ensure_ascii=False, indent=2)
PY

  local train_trace="${case_dir}/policy_train.jsonl"
  local test_trace="${case_dir}/policy_test.jsonl"
  if [[ "${FORCE:-0}" == "1" || "${RESUME}" == "1" || ! -s "${train_trace}" ]]; then
    echo "[decouple] ${name}: build train candidates"
    python -m src.build_policy_dataset --config "${train_config}" --split train "${TRAIN_LIMIT_ARGS[@]}" "${RESUME_ARGS[@]}" --output "${train_trace}" --actions direct,reflect_1,reflect_2
  fi
  if [[ "${FORCE:-0}" == "1" || "${RESUME}" == "1" || ! -s "${test_trace}" ]]; then
    echo "[decouple] ${name}: build test candidates"
    python -m src.build_policy_dataset --config "${test_config}" --split test "${EVAL_LIMIT_ARGS[@]}" "${RESUME_ARGS[@]}" --output "${test_trace}" --actions direct,reflect_1,reflect_2
  fi
  echo "[decouple] ${name}: traces ready under ${case_dir}"
}

run_case qwen3_base_qwen3_reflector "${BASE_QWEN3_URL}" "${BASE_QWEN3_MODEL}" "${BASE_QWEN3_PATH}" "${REFLECTOR_QWEN3_URL}" "${REFLECTOR_QWEN3_MODEL}" "${REFLECTOR_QWEN3_PATH}"
run_case qwen3_base_deepseek_reflector "${BASE_QWEN3_URL}" "${BASE_QWEN3_MODEL}" "${BASE_QWEN3_PATH}" "${REFLECTOR_DEEPSEEK_URL}" "${REFLECTOR_DEEPSEEK_MODEL}" "${REFLECTOR_DEEPSEEK_PATH}"
run_case alternate_base_deepseek_reflector "${BASE_ALT_URL}" "${BASE_ALT_MODEL}" "${BASE_ALT_PATH}" "${REFLECTOR_DEEPSEEK_URL}" "${REFLECTOR_DEEPSEEK_MODEL}" "${REFLECTOR_DEEPSEEK_PATH}"

echo "[decouple] candidate traces completed under ${OUTPUT_ROOT}"
echo "Next: train the identical frozen policy head separately for each case, then evaluate each case's own test trace."
