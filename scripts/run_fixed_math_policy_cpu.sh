#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export CUDA_VISIBLE_DEVICES=""
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS="${THREADS:-4}"
export MKL_NUM_THREADS="${THREADS:-4}"
RUN_DIR="${RUN_DIR:-outputs/fixed_math_policy_cpu_$(date +%Y%m%d_%H%M%S)}"
exec "${PYTHON:-python}" -m src.train_fixed_math_policy \
  --root "${ROOT:-outputs/math_full_training_v1}" \
  --output "$RUN_DIR" --cache "${CACHE:-outputs/math_fixed_features}" \
  --encoder "${ENCODER:-Qwen/Qwen3-0.6B}" \
  --threads "${THREADS:-4}" --encoder-batch-size "${ENCODER_BATCH_SIZE:-2}" \
  --max-length "${MAX_LENGTH:-2048}" --epochs "${EPOCHS:-10}" \
  --seed "${SEED:-42}" --allow-partial "$@"
