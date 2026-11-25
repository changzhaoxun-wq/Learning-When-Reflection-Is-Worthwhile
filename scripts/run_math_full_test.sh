#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export no_proxy="localhost,127.0.0.1,::1${no_proxy:+,${no_proxy}}"
export NO_PROXY="${no_proxy}${NO_PROXY:+,${NO_PROXY}}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="${CUDA_HOME}/bin:${PATH}"
command -v ninja >/dev/null || { printf 'Missing ninja in the vLLM environment\n' >&2; exit 1; }
command -v nvcc >/dev/null || { printf 'Missing nvcc in CUDA_HOME/bin\n' >&2; exit 1; }
exec "${PYTHON_BIN:-python}" -u -m src.math_full_test_trace "$@"
