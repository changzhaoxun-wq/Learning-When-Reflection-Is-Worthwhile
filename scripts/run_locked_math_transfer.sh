#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export no_proxy="localhost,127.0.0.1,::1${no_proxy:+,${no_proxy}}"
export NO_PROXY="$no_proxy"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="${CUDA_HOME}/bin:${PATH}"
command -v ninja >/dev/null || { echo 'Missing ninja' >&2; exit 1; }
command -v nvcc >/dev/null || { echo 'Missing nvcc' >&2; exit 1; }
exec "${PYTHON_BIN:-python}" -u -m src.run_locked_math_transfer "$@"
