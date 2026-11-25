#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

exec python -m src.qa_pipeline "$@"
