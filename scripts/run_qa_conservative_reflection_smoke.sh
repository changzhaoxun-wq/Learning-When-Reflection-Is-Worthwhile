#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"

SOURCE="${SOURCE:-outputs/qa_direct_first/hotpotqa_counterfactual_train_v1/policy_train_counterfactual_augmented.jsonl}"
CONFIG="${CONFIG:-outputs/qa_direct_first/hotpotqa_distractor_8000_v13/train_config.json}"
RAW_HOTPOT_ROOT="${RAW_HOTPOT_ROOT:-data/hotpotqa/distractor}"
ROOT="${ROOT:-outputs/qa_direct_first/hotpotqa_target_aware_v7_smoke}"
PER_GROUP="${PER_GROUP:-50}"
SEED="${SEED:-42}"
GPU="${GPU:-1}"
MAX_WORKERS="${MAX_WORKERS:-4}"

CANONICAL="${ROOT}/canonical_balanced.jsonl"
RAW_TRACE="${ROOT}/policy_reflection_raw.jsonl"
RELABELLED="${ROOT}/policy_reflection_relabelled.jsonl"

mkdir -p "${ROOT}"

python -m src.prepare_qa_reflection_smoke \
  --input "${SOURCE}" \
  --output "${CANONICAL}" \
  --summary_output "${ROOT}/sample_summary.json" \
  --per_group "${PER_GROUP}" \
  --seed "${SEED}"

GPU="${GPU}" ./scripts/start_qa_reflector_vllm.sh

python -m src.build_frozen_base_reflector_trace \
  --canonical "${CANONICAL}" \
  --config "${CONFIG}" \
  --output "${RAW_TRACE}" \
  --resume \
  --max-workers "${MAX_WORKERS}"

python -m src.audit_trace_completion \
  --trace "${RAW_TRACE}" \
  --output "${ROOT}/completion_audit.json" \
  --require_finish_reason \
  --require_final_answer

python -m src.relabel_qa_policy_dataset \
  --input "${RAW_TRACE}" \
  --output "${RELABELLED}" \
  --quarantine_output "${ROOT}/quarantine.jsonl" \
  --summary_output "${ROOT}/relabel_summary.json" \
  --raw_hotpot_root "${RAW_HOTPOT_ROOT}" \
  --qa_f1_threshold 0.8 \
  --lambda_cost 0.0001

python -m src.analyze_qa_reflection_prompt \
  --old "${SOURCE}" \
  --new "${RELABELLED}" \
  --output "${ROOT}/comparison.json" \
  --markdown_output "${ROOT}/COMPARISON.md"

echo "Smoke comparison: ${ROOT}/COMPARISON.md"
