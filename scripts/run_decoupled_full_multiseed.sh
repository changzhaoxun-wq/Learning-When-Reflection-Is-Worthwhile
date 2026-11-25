#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"
export PYTHONUNBUFFERED=1

# Keep the previous 200/100 smoke run intact. This script writes to a new root.
ROOT="${ROOT:-outputs/decoupling_full_multiseed}"
SEEDS="${SEEDS:-7 21 42}"
GPU="${GPU:-1}"
DEVICE="${DEVICE:-cuda}"
ENCODER_NAME="${ENCODER_NAME:-Qwen/Qwen3-0.6B}"
LAMBDA_COST="${LAMBDA_COST:-0.0001}"
EXPECTED_TRAIN_ROWS="${EXPECTED_TRAIN_ROWS:-7473}"
EXPECTED_TEST_ROWS="${EXPECTED_TEST_ROWS:-1319}"
TRACE_CASES=(
  qwen3_base_qwen3_reflector
  qwen3_base_deepseek_reflector
)

mkdir -p "${ROOT}"

trace_file_complete() {
  local path="$1"
  local expected_rows="$2"
  local actual_rows
  [[ -s "${path}" ]] || return 1
  actual_rows="$(wc -l < "${path}")"
  [[ "${actual_rows}" -eq "${expected_rows}" ]]
}

TRACES_READY=1
for CASE_NAME in "${TRACE_CASES[@]}"; do
  if ! trace_file_complete "${ROOT}/${CASE_NAME}/policy_train.jsonl" "${EXPECTED_TRAIN_ROWS}"; then
    TRACES_READY=0
  fi
  if ! trace_file_complete "${ROOT}/${CASE_NAME}/policy_test.jsonl" "${EXPECTED_TEST_ROWS}"; then
    TRACES_READY=0
  fi
done

if [[ "${FORCE:-0}" == "1" || "${TRACES_READY}" != "1" ]]; then
  if [[ "${FORCE:-0}" != "1" ]]; then
    for CASE_NAME in "${TRACE_CASES[@]}"; do
      for TRACE_SPEC in "policy_train.jsonl:${EXPECTED_TRAIN_ROWS}" "policy_test.jsonl:${EXPECTED_TEST_ROWS}"; do
        TRACE_NAME="${TRACE_SPEC%%:*}"
        EXPECTED_ROWS="${TRACE_SPEC##*:}"
        TRACE_PATH="${ROOT}/${CASE_NAME}/${TRACE_NAME}"
        if [[ -e "${TRACE_PATH}" ]] && ! trace_file_complete "${TRACE_PATH}" "${EXPECTED_ROWS}"; then
          BACKUP_PATH="${TRACE_PATH}.incomplete.$(date +%Y%m%d_%H%M%S)"
          mv -- "${TRACE_PATH}" "${BACKUP_PATH}"
          echo "[decouple-full] moved incomplete trace to ${BACKUP_PATH}"
        fi
      done
    done
  fi
  echo "[decouple-full] building missing full candidate traces"
  LIMIT="" \
  EVAL_LIMIT="" \
  OUTPUT_ROOT="${ROOT}" \
  FORCE="${FORCE:-0}" \
  ./scripts/run_decoupled_ablation.sh
else
  echo "[decouple-full] full candidate traces already exist; reuse them"
fi

for CASE_NAME in "${TRACE_CASES[@]}"; do
  trace_file_complete "${ROOT}/${CASE_NAME}/policy_train.jsonl" "${EXPECTED_TRAIN_ROWS}" || {
    echo "Incomplete train trace: ${ROOT}/${CASE_NAME}/policy_train.jsonl" >&2
    exit 1
  }
  trace_file_complete "${ROOT}/${CASE_NAME}/policy_test.jsonl" "${EXPECTED_TEST_ROWS}" || {
    echo "Incomplete test trace: ${ROOT}/${CASE_NAME}/policy_test.jsonl" >&2
    exit 1
  }
done

for SEED in ${SEEDS}; do
  echo "[decouple-full] training/evaluating seed=${SEED}"
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

for CASE_DIR in "${ROOT}"/*; do
  [[ -d "${CASE_DIR}/policy" ]] || continue
  PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}" python -m src.summarize_multiseed \
    --input_dir "${CASE_DIR}/policy" \
    --output "${CASE_DIR}/policy/summary.json"
done

echo "[decouple-full] completed; summaries are under ${ROOT}/*/policy/summary.json"
