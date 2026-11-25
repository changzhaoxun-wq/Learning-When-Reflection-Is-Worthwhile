#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export CUDA_VISIBLE_DEVICES=""
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${THREADS:-4}"
export MKL_NUM_THREADS="${THREADS:-4}"
exec "${PYTHON:-python}" -m src.math_gate_cpu_ablation \
  --source "${SOURCE_RUN:-outputs/fixed_math_policy_cpu_20260915_151132_seed42}" \
  --root "${ROOT:-outputs/math_full_training_v1}" \
  --output "${RUN_DIR:-outputs/math_gate_cpu_ablation_$(date +%Y%m%d_%H%M%S)}" \
  --threads "${THREADS:-4}" --epochs "${EPOCHS:-10}" --seed "${SEED:-42}" "$@"
