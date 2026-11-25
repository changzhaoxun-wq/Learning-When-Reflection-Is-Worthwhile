#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"
export PYTHONUNBUFFERED=1

ROOT="${ROOT:-outputs/decoupling_full_multiseed}"
CASE_NAME="alternate_base_deepseek_reflector"
BASE_ALT_URL="${BASE_ALT_URL:-http://127.0.0.1:8002/v1}"
BASE_ALT_MODEL="${BASE_ALT_MODEL:-Qwen3-4B-Instruct}"
BASE_ALT_PATH="${BASE_ALT_PATH:-Qwen/Qwen3-4B-Instruct}"
REFLECTOR_DEEPSEEK_URL="${REFLECTOR_DEEPSEEK_URL:-http://127.0.0.1:8001/v1}"
REFLECTOR_DEEPSEEK_MODEL="${REFLECTOR_DEEPSEEK_MODEL:-DeepSeek-R1-Distill-Qwen-7B}"
SEEDS="${SEEDS:-7 21 42}"
GPU="${GPU:-3}"
DEVICE="${DEVICE:-cuda}"
ENCODER_NAME="${ENCODER_NAME:-Qwen/Qwen3-0.6B}"
LAMBDA_COST="${LAMBDA_COST:-0.0001}"
EXPECTED_TRAIN_ROWS="${EXPECTED_TRAIN_ROWS:-7473}"
EXPECTED_TEST_ROWS="${EXPECTED_TEST_ROWS:-1319}"

trace_complete() {
  local path="$1"
  local expected="$2"
  [[ -s "${path}" ]] || return 1
  [[ "$(wc -l < "${path}")" -eq "${expected}" ]]
}

mkdir -p "${ROOT}"

TRAIN_TRACE="${ROOT}/${CASE_NAME}/policy_train.jsonl"
TEST_TRACE="${ROOT}/${CASE_NAME}/policy_test.jsonl"
if ! trace_complete "${TRAIN_TRACE}" "${EXPECTED_TRAIN_ROWS}" \
  || ! trace_complete "${TEST_TRACE}" "${EXPECTED_TEST_ROWS}"; then
  echo "[base-decouple] building Qwen3-4B Base + DeepSeek Reflector candidate traces"
  LIMIT="" \
  EVAL_LIMIT="" \
  OUTPUT_ROOT="${ROOT}" \
  BASE_ALT_URL="${BASE_ALT_URL}" \
  BASE_ALT_MODEL="${BASE_ALT_MODEL}" \
  BASE_ALT_PATH="${BASE_ALT_PATH}" \
  REFLECTOR_DEEPSEEK_URL="${REFLECTOR_DEEPSEEK_URL}" \
  REFLECTOR_DEEPSEEK_MODEL="${REFLECTOR_DEEPSEEK_MODEL}" \
  ./scripts/run_decoupled_ablation.sh
else
  echo "[base-decouple] full candidate traces already exist; reuse them"
fi

trace_complete "${TRAIN_TRACE}" "${EXPECTED_TRAIN_ROWS}" || {
  echo "Incomplete train trace: ${TRAIN_TRACE}" >&2
  exit 1
}
trace_complete "${TEST_TRACE}" "${EXPECTED_TEST_ROWS}" || {
  echo "Incomplete test trace: ${TEST_TRACE}" >&2
  exit 1
}

for SEED in ${SEEDS}; do
  echo "[base-decouple] training/evaluating ${CASE_NAME}, seed=${SEED}"
  ROOT="${ROOT}" \
  SEED="${SEED}" \
  RESULTS_BASENAME="eval_results_seed${SEED}.json" \
  GPU="${GPU}" \
  DEVICE="${DEVICE}" \
  ENCODER_NAME="${ENCODER_NAME}" \
  LAMBDA_COST="${LAMBDA_COST}" \
  SKIP_COMPLETED=1 \
  ./scripts/train_eval_decoupled_policy.sh
done

python -m src.summarize_multiseed \
  --input_dir "${ROOT}/${CASE_NAME}/policy" \
  --output "${ROOT}/${CASE_NAME}/policy/summary.json"

echo "[base-decouple] completed: ${ROOT}/${CASE_NAME}/policy/summary.json"
