#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
OUTPUT="${OUTPUT:-outputs/layer_ablation_gsm8k_budget_v1}"
ENCODER="${ENCODER:-Qwen/Qwen3-0.6B}"
DEVICE="${DEVICE:-cuda}"
PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}" python -m src.layer_ablation_budget_math \
  --encoder "${ENCODER}" --output "${OUTPUT}" --device "${DEVICE}" \
  --max-length "${MAX_LENGTH:-2048}" --encoder-batch-size "${ENCODER_BATCH_SIZE:-8}" \
  --test-trace "${TEST_TRACE:-outputs/math_full_test_failure_accounted_v1/policy_test_accounted.jsonl}" \
  --batch-size "${BATCH_SIZE:-128}" --epochs "${EPOCHS:-10}" \
  --seed "${SEED:-42}" --lambda-cost "${LAMBDA_COST:-0.0001}" "$@"
