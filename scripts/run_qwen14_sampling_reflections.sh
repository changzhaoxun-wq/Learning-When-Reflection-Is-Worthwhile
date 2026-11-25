#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export NO_PROXY="127.0.0.1,localhost"
export no_proxy="127.0.0.1,localhost"
export CUDA_HOME="/usr/local/cuda"
export PATH="${CUDA_HOME:-/usr/local/cuda}/bin:$PATH"
exec "${PYTHON_BIN:-python}" -u -m src.rerun_qwen14_sampling_reflections "$@"
