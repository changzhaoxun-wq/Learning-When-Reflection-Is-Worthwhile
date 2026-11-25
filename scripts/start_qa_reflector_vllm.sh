#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"
mkdir -p outputs

if [[ -n "${CONDA_SH:-}" && -f "${CONDA_SH}" ]]; then
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV:-vllm_qwen3}"
fi

GPU="${GPU:-0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEMORY="${GPU_MEMORY:-0.65}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-600}"
PORT="${PORT:-8001}"
MODEL_PATH="${MODEL_PATH:-deepseek-ai/DeepSeek-R1-Distill-Qwen-7B}"
SERVED_NAME="${SERVED_NAME:-DeepSeek-R1-Distill-Qwen-7B}"
LOG="outputs/vllm_reflector_deepseek_7b.log"
PID_FILE="outputs/vllm_reflector_deepseek_7b.pid"

export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export NO_PROXY="127.0.0.1,localhost,${NO_PROXY:-}"
export no_proxy="127.0.0.1,localhost,${no_proxy:-}"

if curl --noproxy '*' -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
  echo "Reflector API already ready on port ${PORT}."
  exit 0
fi

extra_args=()
if [[ "${VLLM_ENFORCE_EAGER:-1}" == "1" ]]; then
  extra_args+=(--enforce-eager)
fi

echo "Starting ${SERVED_NAME} on GPU ${GPU}, port ${PORT}, max_len=${MAX_MODEL_LEN}, gpu_mem=${GPU_MEMORY}"
CUDA_VISIBLE_DEVICES="${GPU}" nohup python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_PATH}" \
  --served-model-name "${SERVED_NAME}" \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEMORY}" \
  "${extra_args[@]}" \
  > "${LOG}" 2>&1 &
echo $! > "${PID_FILE}"

for ((waited=0; waited<STARTUP_TIMEOUT; waited+=5)); do
  if curl --noproxy '*' -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    sleep 2
    if curl --noproxy '*' -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
      echo "Reflector API is ready; pid=$(cat "${PID_FILE}") log=${LOG}"
      exit 0
    fi
  fi
  if ! kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
    echo "Reflector process exited during startup. Last log lines:" >&2
    tail -80 "${LOG}" >&2
    exit 1
  fi
  sleep 5
done

echo "Timed out waiting for reflector API after ${STARTUP_TIMEOUT}s" >&2
tail -80 "${LOG}" >&2
exit 1
