#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export NO_PROXY="127.0.0.1,localhost,${NO_PROXY:-}"
export no_proxy="${NO_PROXY}"
export PATH="${CUDA_HOME:-/usr/local/cuda}/bin:${PATH}"

# Local default: the closest available small Base plus the local Qwen3-8B Reflector.
# Set BASE_MODEL=Qwen2.5-1.5B or Qwen2.5-3B after placing that directory under
# Set BASE_MODEL/REFLECTOR_MODEL to local paths or Hugging Face model IDs.
BASE_MODEL="${BASE_MODEL:-Qwen3-4B-Instruct}"
REFLECTOR_MODEL="${REFLECTOR_MODEL:-Qwen3-8B-Instruct}"
BASE_GPU="${BASE_GPU:-1}"
REFLECTOR_GPU="${REFLECTOR_GPU:-2}"
REFLECTOR_THINKING="${REFLECTOR_THINKING:-false}"
ROOT="${ROOT:-outputs/model_scale_gsm8k_qwen3_4b_qwen3_8b}"
ENCODER="${ENCODER:-Qwen/Qwen3-0.6B}"
PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" -u -m src.model_scale_math_trace \
  --root "${ROOT}" --base-model "${BASE_MODEL}" --reflector-model "${REFLECTOR_MODEL}" \
  --base-gpus "${BASE_GPU}" --reflector-gpus "${REFLECTOR_GPU}" \
  --reflector-thinking "${REFLECTOR_THINKING}" --split all

for SEED in 7 21 42; do
  "${PYTHON_BIN}" -u -m src.train_fixed_math_policy \
    --root "${ROOT}" --output "${ROOT}/policy_seed${SEED}" \
    --cache "${ROOT}/policy_feature_cache" --encoder "${ENCODER}" \
    --seed "${SEED}" --epochs 10 --max-length 2048 --encoder-batch-size 2
done

for SEED in 7 21 42; do
  "${PYTHON_BIN}" -u -m src.evaluate_model_scale_math \
    --root "${ROOT}" --policy "${ROOT}/policy_seed${SEED}" \
    --encoder "${ENCODER}" --cache "${ROOT}/test_feature_cache"
done

printf '\nCompleted model-scale supplement under %s\n' "${ROOT}"
