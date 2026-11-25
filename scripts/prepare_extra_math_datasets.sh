#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"
export PYTHONUNBUFFERED=1

PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}" python -m src.prepare_extra_math_datasets "$@"
