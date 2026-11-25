#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export no_proxy="localhost,127.0.0.1,::1${no_proxy:+,${no_proxy}}"
export NO_PROXY="${no_proxy}${NO_PROXY:+,${NO_PROXY}}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="${CUDA_HOME}/bin:${PATH}"
exec "${PYTHON_BIN:-python}" -u -m src.account_math_test_failures "$@"
