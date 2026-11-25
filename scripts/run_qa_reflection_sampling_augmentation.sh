#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"

PHASE="${PHASE:-all}"
ROOT="${ROOT:-outputs/qa_direct_first/hotpotqa_reflection_sampling_v1}"
OLD_TRAIN_TRACE="${OLD_TRAIN_TRACE:-outputs/qa_direct_first/hotpotqa_counterfactual_train_v1/policy_train_counterfactual_augmented.jsonl}"
NEW_TRAIN_TRACE="${NEW_TRAIN_TRACE:-outputs/qa_direct_first/hotpotqa_target_aware_v7_full/relabel_f1/policy_train_f1_reward.jsonl}"
BASE_CONFIG="${BASE_CONFIG:-outputs/qa_direct_first/hotpotqa_distractor_8000_v13/train_config.json}"
RAW_HOTPOT_ROOT="${RAW_HOTPOT_ROOT:-data/hotpotqa/distractor}"
SEEDS="${SEEDS:-101,202,303}"
TEMPERATURE="${TEMPERATURE:-0.4}"
TOP_P="${TOP_P:-0.9}"
REFLECTOR_GPU="${REFLECTOR_GPU:-1}"
POLICY_GPU="${POLICY_GPU:-1}"
MAX_WORKERS="${MAX_WORKERS:-8}"
FORCE_POLICY="${FORCE_POLICY:-1}"
FILTERED_DATASET="${FILTERED_DATASET:-${ROOT}/relabel_f1/strict_positive_dedup.jsonl}"
FILTER_SUMMARY="${FILTER_SUMMARY:-${ROOT}/relabel_f1/strict_positive_dedup_summary.json}"
FILTERED_OUTPUT_DIR="${FILTERED_OUTPUT_DIR:-${ROOT}/top_layer_policy_filtered}"

SUBSET="${ROOT}/unrepaired_direct_errors.jsonl"
CONFIG_DIR="${ROOT}/configs"
SUMMARY="${ROOT}/prepare_summary.json"
mkdir -p "${ROOT}/raw" "${ROOT}/relabel_f1" "${ROOT}/audits"

IFS=',' read -r -a SEED_ARRAY <<< "${SEEDS}"

prepare() {
  python -m src.prepare_qa_reflection_sampling \
    --old_trace "${OLD_TRAIN_TRACE}" \
    --new_trace "${NEW_TRAIN_TRACE}" \
    --base_config "${BASE_CONFIG}" \
    --output_subset "${SUBSET}" \
    --config_dir "${CONFIG_DIR}" \
    --summary_output "${SUMMARY}" \
    --seeds "${SEEDS}" \
    --temperature "${TEMPERATURE}" \
    --top_p "${TOP_P}"
}

generate() {
  GPU="${REFLECTOR_GPU}" ./scripts/start_qa_reflector_vllm.sh
  for index in "${!SEED_ARRAY[@]}"; do
    pass=$((index + 1))
    echo "[qa-sampling] generate pass=${pass} seed=${SEED_ARRAY[$index]}"
    python -m src.build_frozen_base_reflector_trace \
      --canonical "${SUBSET}" \
      --config "${CONFIG_DIR}/sample_${pass}_config.json" \
      --output "${ROOT}/raw/sample_${pass}.jsonl" \
      --incorrect-only \
      --reflection-actions reflect_1 \
      --resume \
      --max-workers "${MAX_WORKERS}"
  done
}

audit() {
  for index in "${!SEED_ARRAY[@]}"; do
    pass=$((index + 1))
    python -m src.audit_trace_completion \
      --trace "${ROOT}/raw/sample_${pass}.jsonl" \
      --output "${ROOT}/audits/sample_${pass}.json" \
      --require_finish_reason \
      --require_final_answer
  done
}

relabel() {
  for index in "${!SEED_ARRAY[@]}"; do
    pass=$((index + 1))
    python -m src.relabel_qa_policy_dataset \
      --input "${ROOT}/raw/sample_${pass}.jsonl" \
      --output "${ROOT}/relabel_f1/sample_${pass}.jsonl" \
      --quarantine_output "${ROOT}/relabel_f1/quarantined_sample_${pass}.jsonl" \
      --summary_output "${ROOT}/relabel_f1/sample_${pass}_summary.json" \
      --raw_hotpot_root "${RAW_HOTPOT_ROOT}" \
      --qa_f1_threshold 0.8 \
      --lambda_cost 0.0001
  done
}

filter_positive() {
  input_args=()
  for index in "${!SEED_ARRAY[@]}"; do
    pass=$((index + 1))
    input_args+=(--input "${ROOT}/relabel_f1/sample_${pass}.jsonl")
  done
  python -m src.filter_qa_reflection_augmentations \
    "${input_args[@]}" \
    --output "${FILTERED_DATASET}" \
    --summary_output "${FILTER_SUMMARY}"
}

train_eval() {
  augmentation_paths=()
  for index in "${!SEED_ARRAY[@]}"; do
    pass=$((index + 1))
    augmentation_paths+=("${ROOT}/relabel_f1/sample_${pass}.jsonl")
  done
  augmentation_csv="$(IFS=,; echo "${augmentation_paths[*]}")"
  GPU="${POLICY_GPU}" \
  OUTPUT_DIR="${ROOT}/top_layer_policy" \
  AUGMENTATION_DATASETS="${augmentation_csv}" \
  FORCE="${FORCE_POLICY}" \
  ./scripts/run_qa_top_layer_policy.sh
}

train_filtered() {
  if [[ ! -f "${FILTERED_DATASET}" ]]; then
    filter_positive
  fi
  GPU="${POLICY_GPU}" \
  OUTPUT_DIR="${FILTERED_OUTPUT_DIR}" \
  AUGMENTATION_DATASETS="${FILTERED_DATASET}" \
  FORCE="${FORCE_POLICY}" \
  ./scripts/run_qa_top_layer_policy.sh
}

case "${PHASE}" in
  prepare) prepare ;;
  generate) generate ;;
  audit) audit ;;
  relabel) relabel ;;
  filter) filter_positive ;;
  train_eval) train_eval ;;
  train_filtered) train_filtered ;;
  filtered)
    filter_positive
    train_filtered
    ;;
  all)
    prepare
    generate
    audit
    relabel
    train_eval
    ;;
  *)
    echo "Unknown PHASE=${PHASE}; use prepare, generate, audit, relabel, filter, train_eval, train_filtered, filtered, or all." >&2
    exit 2
    ;;
esac

echo "[qa-sampling] phase=${PHASE} complete root=${ROOT}"
