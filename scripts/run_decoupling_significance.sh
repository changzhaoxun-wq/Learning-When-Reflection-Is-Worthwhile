#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"
export PYTHONUNBUFFERED=1

ROOT="${ROOT:-outputs/decoupling_full_multiseed}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/significance}"
GPU="${GPU:-3}"
DEVICE="${DEVICE:-cuda}"
ENCODER_BATCH_SIZE="${ENCODER_BATCH_SIZE:-8}"
BOOTSTRAP_ITERATIONS="${BOOTSTRAP_ITERATIONS:-10000}"
LAMBDA_COST="${LAMBDA_COST:-0.0001}"

CASE_SPECS=(
  "qwen3_base_deepseek_reflector|qwen3_base_deepseek_reflector"
  "qwen3_base_qwen3_reflector|qwen3_base_qwen3_reflector"
  "qwen3_4b_base_deepseek_reflector|alternate_base_deepseek_reflector"
)

ARGS=()
for CASE_SPEC in "${CASE_SPECS[@]}"; do
  DISPLAY_NAME="${CASE_SPEC%%|*}"
  CASE_NAME="${CASE_SPEC#*|}"
  CASE_DIR="${ROOT}/${CASE_NAME}"
  [[ -s "${CASE_DIR}/policy_test.jsonl" ]] || {
    echo "Missing test trace: ${CASE_DIR}/policy_test.jsonl" >&2
    exit 1
  }
  for SEED in 7 21 42; do
    [[ -s "${CASE_DIR}/policy/gate_calibrated_seed${SEED}.pkl" ]] || {
      echo "Missing gate model for ${CASE_NAME}, seed ${SEED}" >&2
      exit 1
    }
    [[ -s "${CASE_DIR}/policy/round_seed${SEED}.pkl" ]] || {
      echo "Missing round model for ${CASE_NAME}, seed ${SEED}" >&2
      exit 1
    }
  done
  ARGS+=(--case "${DISPLAY_NAME}" "${CASE_DIR}")
done

CUDA_VISIBLE_DEVICES="${GPU}" python -m src.analyze_decoupling_significance \
  "${ARGS[@]}" \
  --seeds 7 21 42 \
  --output_dir "${OUTPUT_DIR}" \
  --lambda_cost "${LAMBDA_COST}" \
  --device "${DEVICE}" \
  --encoder_batch_size "${ENCODER_BATCH_SIZE}" \
  --bootstrap_iterations "${BOOTSTRAP_ITERATIONS}"

echo "[significance] completed: ${OUTPUT_DIR}/SIGNIFICANCE_REPORT.md"
