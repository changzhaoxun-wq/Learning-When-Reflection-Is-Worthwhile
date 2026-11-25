#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"
mkdir -p outputs

if [[ -n "${CONDA_SH:-}" && -f "${CONDA_SH}" ]]; then
  # Optional: set CONDA_SH and CONDA_ENV when using a dedicated vLLM env.
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV:-vllm_qwen3}"
fi

export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export NO_PROXY="127.0.0.1,localhost,${NO_PROXY:-}"
export no_proxy="127.0.0.1,localhost,${no_proxy:-}"


wait_for_api() {
  local port="$1"
  local name="$2"
  local timeout="${STARTUP_TIMEOUT:-600}"
  local waited=0

  echo "Waiting for ${name} API on port ${port} to become ready..."
  while [ "${waited}" -lt "${timeout}" ]; do
    if curl --noproxy '*' -fsS "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
      # Require two successful probes so a briefly opened port is not treated
      # as a healthy API server.
      sleep 2
      if curl --noproxy '*' -fsS "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
        echo "${name} API is ready on port ${port}."
        return 0
      fi
    fi
    sleep 5
    waited=$((waited + 5))
  done

  echo "ERROR: timed out waiting for ${name} API on port ${port} after ${timeout}s" >&2
  return 1
}

start_model() {
  local name="$1"
  local model_path="$2"
  local served_name="$3"
  local port="$4"
  local gpu="$5"
  local max_len="$6"
  local gpu_mem="$7"
  local log_file="outputs/vllm_${name}.log"
  local pid_file="outputs/vllm_${name}.pid"

  if ss -ltn | grep -q ":${port} "; then
    echo "Port ${port} is already in use; skip ${served_name}."
    return 0
  fi

  local extra_args=()
  if [ "${VLLM_ENFORCE_EAGER:-1}" = "1" ]; then
    extra_args+=(--enforce-eager)
  fi

  echo "Starting ${served_name} on 127.0.0.1:${port} using GPU ${gpu} max_len=${max_len} gpu_mem=${gpu_mem} enforce_eager=${VLLM_ENFORCE_EAGER:-1}"
  CUDA_VISIBLE_DEVICES="${gpu}" nohup "${VLLM_PYTHON:-python}" -m vllm.entrypoints.openai.api_server \
    --model "${model_path}" \
    --served-model-name "${served_name}" \
    --host 127.0.0.1 \
    --port "${port}" \
    --max-model-len "${max_len}" \
    --gpu-memory-utilization "${gpu_mem}" \
    "${extra_args[@]}" \
    > "${log_file}" 2>&1 &
  echo $! > "${pid_file}"
  echo "  pid=$(cat "${pid_file}") log=${log_file}"
}

# Default placement keeps the always-used base model separate and lets the
# smaller verifier share a GPU with the reflector. Override these from the
# shell when a different GPU layout is needed.
BASE_GPU="${BASE_GPU:-1}"
REFLECTOR_GPU="${REFLECTOR_GPU:-2}"
VERIFIER_GPU="${VERIFIER_GPU:-2}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
BASE_GPU_MEMORY="${BASE_GPU_MEMORY:-0.50}"
REFLECTOR_GPU_MEMORY="${REFLECTOR_GPU_MEMORY:-0.45}"
VERIFIER_GPU_MEMORY="${VERIFIER_GPU_MEMORY:-0.45}"
START_VERIFIER="${START_VERIFIER:-1}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-Qwen/Qwen3-8B-Instruct}"
REFLECTOR_MODEL_PATH="${REFLECTOR_MODEL_PATH:-deepseek-ai/DeepSeek-R1-Distill-Qwen-7B}"
VERIFIER_MODEL_PATH="${VERIFIER_MODEL_PATH:-Qwen/Qwen3-4B-Instruct}"

start_model "base_qwen3_8b"   "${BASE_MODEL_PATH}"   "${BASE_MODEL:-Qwen3-8B-Instruct}"   8000 "${BASE_GPU}" "${MAX_MODEL_LEN}" "${BASE_GPU_MEMORY}"

# Avoid concurrent weight loading when Base and Reflector share a GPU.
if [ "${BASE_GPU}" = "${REFLECTOR_GPU}" ]; then
  wait_for_api 8000 "base_qwen3_8b"
fi

start_model "reflector_deepseek_7b"   "${REFLECTOR_MODEL_PATH}"   "${REFLECTOR_MODEL:-DeepSeek-R1-Distill-Qwen-7B}"   8001 "${REFLECTOR_GPU}" "${MAX_MODEL_LEN}" "${REFLECTOR_GPU_MEMORY}"

# Models on different GPUs are started concurrently. Always verify both API
# ports before reporting success so a background engine failure is visible.
wait_for_api 8000 "base_qwen3_8b"
wait_for_api 8001 "reflector_deepseek_7b"

if [ "${START_VERIFIER}" = "1" ]; then
  start_model "verifier_qwen3_4b"   "${VERIFIER_MODEL_PATH}"   "${VERIFIER_MODEL:-Qwen3-4B-Instruct}"   8002 "${VERIFIER_GPU}" "${MAX_MODEL_LEN}" "${VERIFIER_GPU_MEMORY}"
  wait_for_api 8002 "verifier_qwen3_4b"
  echo "Use: tail -f outputs/vllm_base_qwen3_8b.log outputs/vllm_reflector_deepseek_7b.log outputs/vllm_verifier_qwen3_4b.log"
else
  echo "Verifier startup skipped (START_VERIFIER=${START_VERIFIER})."
  echo "Use: tail -f outputs/vllm_base_qwen3_8b.log outputs/vllm_reflector_deepseek_7b.log"
fi
