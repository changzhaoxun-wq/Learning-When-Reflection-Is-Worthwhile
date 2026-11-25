#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"

PHASE="${PHASE:-all}"
ROOT="${ROOT:-outputs/qa_direct_first/hotpotqa_counterfactual_train_v1}"
SOURCE_TRACE="${SOURCE_TRACE:-outputs/qa_direct_first/hotpotqa_8000_v12/relabel_f1/policy_train_f1_reward.jsonl}"
CONFIG="${CONFIG:-outputs/qa_direct_first/hotpotqa_distractor_8000_v13/train_config.json}"
TEST_TRACE="${TEST_TRACE:-outputs/qa_direct_first/hotpotqa_distractor_8000_v13/relabel_f1/policy_eval_f1_reward.jsonl}"
RAW_HOTPOT_ROOT="${RAW_HOTPOT_ROOT:-data/hotpotqa/fullwiki}"
SAMPLE_SIZE="${SAMPLE_SIZE:-1663}"
SEED="${SEED:-42}"
REFLECT_WORKERS="${REFLECT_WORKERS:-4}"
GPU="${GPU:-0}"

SAMPLE="${ROOT}/direct_correct_sample.jsonl"
RAW_COUNTERFACTUAL="${ROOT}/direct_correct_counterfactual_raw.jsonl"
RELABEL_DIR="${ROOT}/relabel_f1"
RELABELLED="${RELABEL_DIR}/direct_correct_counterfactual_f1.jsonl"
QUARANTINE="${RELABEL_DIR}/quarantine.jsonl"
AUGMENTED="${ROOT}/policy_train_counterfactual_augmented.jsonl"
POLICY_DIR="${ROOT}/selective_advantage_policy"

mkdir -p "${ROOT}" "${RELABEL_DIR}"

run_prepare() {
  python -m src.prepare_qa_counterfactual_training sample \
    --source "${SOURCE_TRACE}" \
    --output "${SAMPLE}" \
    --manifest_output "${ROOT}/sample_manifest.json" \
    --sample_size "${SAMPLE_SIZE}" \
    --seed "${SEED}"
}

run_generate() {
  if ! curl --noproxy '*' -fsS http://127.0.0.1:8001/v1/models >/dev/null; then
    echo "Reflector API is not ready on port 8001." >&2
    echo "Start it with: GPU=${GPU} ./scripts/start_qa_reflector_vllm.sh" >&2
    exit 2
  fi
  python -m src.build_frozen_base_reflector_trace \
    --canonical "${SAMPLE}" \
    --config "${CONFIG}" \
    --output "${RAW_COUNTERFACTUAL}" \
    --resume \
    --max-workers "${REFLECT_WORKERS}" \
    2>&1 | tee "${ROOT}/generate.log"
}

run_audit() {
  python -m src.audit_trace_completion \
    --trace "${RAW_COUNTERFACTUAL}" \
    --output "${ROOT}/counterfactual_audit.json" \
    --require_finish_reason \
    --require_final_answer
}

run_relabel() {
  python -m src.relabel_qa_policy_dataset \
    --input "${RAW_COUNTERFACTUAL}" \
    --output "${RELABELLED}" \
    --quarantine_output "${QUARANTINE}" \
    --summary_output "${RELABEL_DIR}/summary.json" \
    --raw_hotpot_root "${RAW_HOTPOT_ROOT}" \
    --qa_f1_threshold 0.8 \
    --lambda_cost 0.0001
  python -c "import pathlib; n=sum(1 for x in open('${RELABELLED}') if x.strip()); q=sum(1 for x in open('${QUARANTINE}') if x.strip()); expected=int('${SAMPLE_SIZE}'); assert n == expected and q == 0, f'Expected {expected} clean rows and no quarantine, got clean={n} quarantine={q}'"
}

run_merge() {
  python -m src.prepare_qa_counterfactual_training merge \
    --source "${SOURCE_TRACE}" \
    --completed_sample "${RELABELLED}" \
    --output "${AUGMENTED}" \
    --manifest_output "${ROOT}/merge_manifest.json"
}

run_train_eval() {
  GPU="${GPU}" \
  TRAIN_TRACE="${AUGMENTED}" \
  TEST_TRACE="${TEST_TRACE}" \
  OUTPUT_DIR="${POLICY_DIR}" \
  ./scripts/run_qa_selective_advantage.sh
}

case "${PHASE}" in
  prepare) run_prepare ;;
  generate) run_generate ;;
  audit) run_audit ;;
  relabel) run_relabel ;;
  merge) run_merge ;;
  train_eval) run_train_eval ;;
  all)
    run_prepare
    run_generate
    run_audit
    run_relabel
    run_merge
    run_train_eval
    ;;
  *)
    echo "Unknown PHASE=${PHASE}; expected prepare|generate|audit|relabel|merge|train_eval|all" >&2
    exit 2
    ;;
esac

echo "[qa-counterfactual] phase=${PHASE} completed root=${ROOT}"
