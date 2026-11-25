#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"
export PYTHONUNBUFFERED=1

SOURCE_ROOT="${SOURCE_ROOT:-outputs/decoupling_full_multiseed}"
ROOT="${ROOT:-outputs/decoupling_strict_frozen_base}"
CANONICAL_CASE="${CANONICAL_CASE:-qwen3_base_deepseek_reflector}"
TARGET_CASE="${TARGET_CASE:-qwen3_base_qwen3_reflector}"
SEEDS="${SEEDS:-7 21 42}"
GPU="${GPU:-1}"
DEVICE="${DEVICE:-cuda}"
ENCODER_NAME="${ENCODER_NAME:-Qwen/Qwen3-0.6B}"
LAMBDA_COST="${LAMBDA_COST:-0.0001}"
BOOTSTRAP_ITERATIONS="${BOOTSTRAP_ITERATIONS:-10000}"
TRACES_ONLY="${TRACES_ONLY:-0}"

SOURCE_CANONICAL="${SOURCE_ROOT}/${CANONICAL_CASE}"
SOURCE_TARGET="${SOURCE_ROOT}/${TARGET_CASE}"
STRICT_CANONICAL="${ROOT}/${CANONICAL_CASE}"
STRICT_TARGET="${ROOT}/${TARGET_CASE}"

for required in \
  "${SOURCE_CANONICAL}/policy_train.jsonl" \
  "${SOURCE_CANONICAL}/policy_test.jsonl" \
  "${SOURCE_TARGET}/policy_train.jsonl" \
  "${SOURCE_TARGET}/policy_test.jsonl" \
  "${SOURCE_TARGET}/train_config.json" \
  "${SOURCE_TARGET}/test_config.json"; do
  [[ -s "${required}" ]] || {
    echo "Missing required source artifact: ${required}" >&2
    exit 1
  }
done

mkdir -p "${STRICT_CANONICAL}" "${STRICT_TARGET}" "${ROOT}/audits"

link_artifact() {
  local source="$1" destination="$2"
  if [[ -e "${destination}" && ! -L "${destination}" ]]; then
    echo "Refusing to replace non-symlink artifact: ${destination}" >&2
    exit 1
  fi
  ln -sfn "${PROJECT_DIR}/${source}" "${destination}"
}

for name in policy_train.jsonl policy_test.jsonl train_config.json test_config.json; do
  if [[ -e "${SOURCE_CANONICAL}/${name}" ]]; then
    link_artifact "${SOURCE_CANONICAL}/${name}" "${STRICT_CANONICAL}/${name}"
  fi
done
link_artifact "${SOURCE_TARGET}/train_config.json" "${STRICT_TARGET}/train_config.json"
link_artifact "${SOURCE_TARGET}/test_config.json" "${STRICT_TARGET}/test_config.json"

echo "[strict-decouple] dry-run train trace"
python -m src.build_frozen_base_reflector_trace \
  --canonical "${SOURCE_CANONICAL}/policy_train.jsonl" \
  --target "${SOURCE_TARGET}/policy_train.jsonl" \
  --config "${SOURCE_TARGET}/train_config.json" \
  --output "${STRICT_TARGET}/policy_train.jsonl" \
  --dry-run

echo "[strict-decouple] build/resume train trace"
python -m src.build_frozen_base_reflector_trace \
  --canonical "${SOURCE_CANONICAL}/policy_train.jsonl" \
  --target "${SOURCE_TARGET}/policy_train.jsonl" \
  --config "${SOURCE_TARGET}/train_config.json" \
  --output "${STRICT_TARGET}/policy_train.jsonl" \
  --resume

echo "[strict-decouple] dry-run test trace"
python -m src.build_frozen_base_reflector_trace \
  --canonical "${SOURCE_CANONICAL}/policy_test.jsonl" \
  --target "${SOURCE_TARGET}/policy_test.jsonl" \
  --config "${SOURCE_TARGET}/test_config.json" \
  --output "${STRICT_TARGET}/policy_test.jsonl" \
  --dry-run

echo "[strict-decouple] build/resume test trace"
python -m src.build_frozen_base_reflector_trace \
  --canonical "${SOURCE_CANONICAL}/policy_test.jsonl" \
  --target "${SOURCE_TARGET}/policy_test.jsonl" \
  --config "${SOURCE_TARGET}/test_config.json" \
  --output "${STRICT_TARGET}/policy_test.jsonl" \
  --resume

python -m src.audit_frozen_base_decoupling \
  --canonical "${SOURCE_CANONICAL}/policy_train.jsonl" \
  --candidate "${STRICT_TARGET}/policy_train.jsonl" \
  --output "${ROOT}/audits/train.json"
python -m src.audit_frozen_base_decoupling \
  --canonical "${SOURCE_CANONICAL}/policy_test.jsonl" \
  --candidate "${STRICT_TARGET}/policy_test.jsonl" \
  --output "${ROOT}/audits/test.json"

if [[ "${TRACES_ONLY}" == "1" ]]; then
  echo "[strict-decouple] traces and audits completed under ${ROOT}"
  exit 0
fi

for name in hierarchical policy; do
  if [[ -e "${SOURCE_CANONICAL}/${name}" ]]; then
    link_artifact "${SOURCE_CANONICAL}/${name}" "${STRICT_CANONICAL}/${name}"
  fi
done

for seed in ${SEEDS}; do
  echo "[strict-decouple] train/evaluate target reflector policy seed=${seed}"
  ROOT="${ROOT}" \
  SEED="${seed}" \
  RESULTS_BASENAME="eval_results_seed${seed}.json" \
  GPU="${GPU}" \
  DEVICE="${DEVICE}" \
  ENCODER_NAME="${ENCODER_NAME}" \
  LAMBDA_COST="${LAMBDA_COST}" \
  SKIP_COMPLETED=1 \
  ./scripts/train_eval_decoupled_policy.sh
done

python -m src.summarize_multiseed \
  --input_dir "${STRICT_TARGET}/policy" \
  --output "${STRICT_TARGET}/policy/summary.json"

CUDA_VISIBLE_DEVICES="${GPU}" python -m src.analyze_decoupling_significance \
  --case "${CANONICAL_CASE}" "${STRICT_CANONICAL}" \
  --case "${TARGET_CASE}" "${STRICT_TARGET}" \
  --seeds 7 21 42 \
  --output_dir "${ROOT}/significance" \
  --lambda_cost "${LAMBDA_COST}" \
  --device "${DEVICE}" \
  --encoder_batch_size 1 \
  --bootstrap_iterations "${BOOTSTRAP_ITERATIONS}" \
  --shared_base_trace

echo "[strict-decouple] completed: ${ROOT}"
