#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"

PHASE="${PHASE:-all}"
ROOT="${ROOT:-outputs/qa_direct_first/hotpotqa_target_aware_v7_full}"
TRAIN_CANONICAL="${TRAIN_CANONICAL:-outputs/qa_direct_first/hotpotqa_counterfactual_train_v1/policy_train_counterfactual_augmented.jsonl}"
EVAL_CANONICAL="${EVAL_CANONICAL:-outputs/qa_direct_first/hotpotqa_distractor_8000_v13/direct_eval.jsonl}"
TRAIN_CONFIG="${TRAIN_CONFIG:-outputs/qa_direct_first/hotpotqa_distractor_8000_v13/train_config.json}"
EVAL_CONFIG="${EVAL_CONFIG:-outputs/qa_direct_first/hotpotqa_distractor_8000_v13/eval_config.json}"
RAW_HOTPOT_ROOT="${RAW_HOTPOT_ROOT:-data/hotpotqa/distractor}"
REFLECTOR_GPU="${REFLECTOR_GPU:-1}"
POLICY_GPU="${POLICY_GPU:-3}"
MAX_WORKERS="${MAX_WORKERS:-8}"
FORCE_POLICY="${FORCE_POLICY:-0}"

TRAIN_RAW="${ROOT}/policy_train_selective_raw.jsonl"
EVAL_RAW="${ROOT}/policy_eval_full_raw.jsonl"
RELABEL_DIR="${ROOT}/relabel_f1"
TRAIN_RELABEL="${RELABEL_DIR}/policy_train_f1_reward.jsonl"
EVAL_RELABEL="${RELABEL_DIR}/policy_eval_f1_reward.jsonl"
POLICY_DIR="${ROOT}/accuracy_first_2048"

mkdir -p "${ROOT}/audits" "${RELABEL_DIR}"

start_reflector() {
  GPU="${REFLECTOR_GPU}" ./scripts/start_qa_reflector_vllm.sh
}

train_reflect() {
  start_reflector
  python -m src.build_frozen_base_reflector_trace \
    --canonical "${TRAIN_CANONICAL}" \
    --config "${TRAIN_CONFIG}" \
    --output "${TRAIN_RAW}" \
    --reflection-actions reflect_1 \
    --canonical-reflections-only \
    --resume \
    --max-workers "${MAX_WORKERS}"
}

eval_reflect() {
  start_reflector
  python -m src.build_frozen_base_reflector_trace \
    --canonical "${EVAL_CANONICAL}" \
    --config "${EVAL_CONFIG}" \
    --output "${EVAL_RAW}" \
    --resume \
    --reflection-actions reflect_1 \
    --max-workers "${MAX_WORKERS}"
}

audit() {
  python -m src.audit_trace_completion \
    --trace "${TRAIN_RAW}" \
    --output "${ROOT}/audits/policy_train_selective.json" \
    --require_finish_reason \
    --require_final_answer
  python -m src.audit_trace_completion \
    --trace "${EVAL_RAW}" \
    --output "${ROOT}/audits/policy_eval_full.json" \
    --require_finish_reason \
    --require_final_answer
}

relabel() {
  python -m src.relabel_qa_policy_dataset \
    --input "${TRAIN_RAW}" \
    --output "${TRAIN_RELABEL}" \
    --quarantine_output "${RELABEL_DIR}/quarantined_train_gold.jsonl" \
    --summary_output "${RELABEL_DIR}/train_relabel_summary.json" \
    --raw_hotpot_root "${RAW_HOTPOT_ROOT}" \
    --qa_f1_threshold 0.8 \
    --lambda_cost 0.0001
  python -m src.relabel_qa_policy_dataset \
    --input "${EVAL_RAW}" \
    --output "${EVAL_RELABEL}" \
    --quarantine_output "${RELABEL_DIR}/quarantined_eval_gold.jsonl" \
    --summary_output "${RELABEL_DIR}/eval_relabel_summary.json" \
    --raw_hotpot_root "${RAW_HOTPOT_ROOT}" \
    --qa_f1_threshold 0.8 \
    --lambda_cost 0.0001
}

train_eval_policy() {
  GPU="${POLICY_GPU}" \
  TRAIN_TRACE="${TRAIN_RELABEL}" \
  TEST_TRACE="${EVAL_RELABEL}" \
  OUTPUT_DIR="${POLICY_DIR}" \
  MAX_LENGTH=2048 \
  FORCE="${FORCE_POLICY}" \
  ./scripts/run_qa_accuracy_first.sh
}

case "${PHASE}" in
  train_reflect)
    train_reflect
    ;;
  eval_reflect)
    eval_reflect
    ;;
  audit)
    audit
    ;;
  relabel)
    relabel
    ;;
  train_eval_policy)
    train_eval_policy
    ;;
  all)
    train_reflect
    eval_reflect
    audit
    relabel
    train_eval_policy
    ;;
  *)
    echo "Unknown PHASE=${PHASE}. Use train_reflect, eval_reflect, audit, relabel, train_eval_policy, or all." >&2
    exit 2
    ;;
esac

echo "[qa-v7-full] phase=${PHASE} complete root=${ROOT}"
