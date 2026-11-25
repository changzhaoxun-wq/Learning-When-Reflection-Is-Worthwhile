#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"
export PYTHONUNBUFFERED=1

RAW_ROOT="${RAW_ROOT:-data/raw}"
OUTPUT_ROOT="${OUTPUT_ROOT:-data/prepared}"
MAX_CONTEXT_WORDS="${MAX_CONTEXT_WORDS:-900}"

PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}" python -m src.prepare_reasoning_datasets \
  --raw_root "${RAW_ROOT}" \
  --output_root "${OUTPUT_ROOT}" \
  --max_context_words "${MAX_CONTEXT_WORDS}" \
  "$@"
