#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"
INPUT="${INPUT:-outputs/policy_dataset_gsm8k_three_action_accuracy.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/policy_datasets/gsm8k_hierarchical_accuracy}"
OBJECTIVE="${OBJECTIVE:-accuracy}"
LAMBDA_COST="${LAMBDA_COST:-0.0001}"

PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}" python -m src.prepare_hierarchical_policy_dataset \
  --input "${INPUT}" \
  --output_dir "${OUTPUT_DIR}" \
  --objective "${OBJECTIVE}" \
  --lambda_cost "${LAMBDA_COST}"
