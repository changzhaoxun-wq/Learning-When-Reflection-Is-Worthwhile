#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

OUTPUT="${OUTPUT:-outputs/layer_ablation_gsm8k_v1}"
ENCODER="${ENCODER:-Qwen/Qwen3-0.6B}"
DEVICE="${DEVICE:-cuda}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
ENCODER_BATCH_SIZE="${ENCODER_BATCH_SIZE:-8}"
BATCH_SIZE="${BATCH_SIZE:-128}"
EPOCHS="${EPOCHS:-10}"
SEED="${SEED:-42}"
LAMBDA_COST="${LAMBDA_COST:-0.0001}"

PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}" \
python -m src.layer_ablation_math \
  --encoder "${ENCODER}" \
  --output "${OUTPUT}" \
  --device "${DEVICE}" \
  --max-length "${MAX_LENGTH}" \
  --encoder-batch-size "${ENCODER_BATCH_SIZE}" \
  --batch-size "${BATCH_SIZE}" \
  --epochs "${EPOCHS}" \
  --seed "${SEED}" \
  --lambda-cost "${LAMBDA_COST}"
