#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"

RUN_ROOT="${RUN_ROOT:-outputs/qa_direct_first/hotpotqa_8000_v12}"
INPUT="${INPUT:-${RUN_ROOT}/policy_train_selective.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${RUN_ROOT}/relabel_f1}"
RAW_HOTPOT_ROOT="${RAW_HOTPOT_ROOT:-data/hotpotqa/fullwiki}"
QA_F1_THRESHOLD="${QA_F1_THRESHOLD:-0.8}"
LAMBDA_COST="${LAMBDA_COST:-0.0001}"

PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}" python -m src.relabel_qa_policy_dataset \
  --input "${INPUT}" \
  --output "${OUTPUT_DIR}/policy_train_f1_reward.jsonl" \
  --quarantine_output "${OUTPUT_DIR}/quarantined_gold.jsonl" \
  --summary_output "${OUTPUT_DIR}/relabel_summary.json" \
  --raw_hotpot_root "${RAW_HOTPOT_ROOT}" \
  --qa_f1_threshold "${QA_F1_THRESHOLD}" \
  --lambda_cost "${LAMBDA_COST}"

PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}" python -m src.prepare_hierarchical_policy_dataset \
  --input "${OUTPUT_DIR}/policy_train_f1_reward.jsonl" \
  --output_dir "${OUTPUT_DIR}/hierarchical" \
  --objective stored_reward \
  --lambda_cost "${LAMBDA_COST}"

printf "QA offline relabel complete: %s\n" "${OUTPUT_DIR}"
