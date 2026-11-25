#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export NO_PROXY="127.0.0.1,localhost,${NO_PROXY:-}"
export no_proxy="${NO_PROXY}"
export PATH="${CUDA_HOME:-/usr/local/cuda}/bin:${PATH}"

BASE_MODEL="${BASE_MODEL:-Qwen3-4B-Instruct}"
REFLECTOR_MODEL="${REFLECTOR_MODEL:-Qwen3-8B-Instruct}"
BASE_GPU="${BASE_GPU:-1}"
REFLECTOR_GPU="${REFLECTOR_GPU:-2}"
REFLECTOR_THINKING="${REFLECTOR_THINKING:-false}"
ROOT="${ROOT:-outputs/model_scale_qwen3_4b_qwen3_8b}"
PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" -u -m src.model_scale_resilient_math_trace \
  --root "${ROOT}" --base-model "${BASE_MODEL}" --reflector-model "${REFLECTOR_MODEL}" \
  --base-gpus "${BASE_GPU}" --reflector-gpus "${REFLECTOR_GPU}" \
  --reflector-thinking "${REFLECTOR_THINKING}" --split all

"${PYTHON_BIN}" -u -m src.model_scale_resilient_transfer_trace \
  --root "${ROOT}/transfer_traces" --base-model "${BASE_MODEL}" \
  --reflector-model "${REFLECTOR_MODEL}" --base-gpus "${BASE_GPU}" \
  --reflector-gpus "${REFLECTOR_GPU}" --reflector-thinking "${REFLECTOR_THINKING}"

printf '\nAll GSM8K and transfer traces are complete under %s\n' "${ROOT}"
