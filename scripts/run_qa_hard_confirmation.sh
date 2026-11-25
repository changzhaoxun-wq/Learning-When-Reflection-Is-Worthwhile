#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"
export NO_PROXY="127.0.0.1,localhost,${NO_PROXY:-}"
export no_proxy="${NO_PROXY}"
export CUDA_VISIBLE_DEVICES="${GPU:-2}"
exec "${PYTHON_BIN:-python}" -u -m src.qa_hard_confirmation \
  --development-root "${DEVELOPMENT_ROOT:-outputs/qa_direct_first/hotpotqa_hard_development_v1}" \
  --root "${ROOT:-outputs/qa_direct_first/hotpotqa_hard_confirmation_v1}" \
  --phase "${PHASE:-all}" --workers "${MAX_WORKERS:-4}" "$@"
