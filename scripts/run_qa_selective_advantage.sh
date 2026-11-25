#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"

GPU="${GPU:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
TRAIN_TRACE="${TRAIN_TRACE:-outputs/qa_direct_first/hotpotqa_8000_v12/relabel_f1/policy_train_f1_reward.jsonl}"
TEST_TRACE="${TEST_TRACE:-outputs/qa_direct_first/hotpotqa_distractor_8000_v13/relabel_f1/policy_eval_f1_reward.jsonl}"
ENCODER_NAME="${ENCODER_NAME:-Qwen/Qwen3-0.6B}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/qa_direct_first/hotpotqa_distractor_8000_v13/selective_advantage}"
LAMBDA_COST="${LAMBDA_COST:-0.0001}"
LAYERS="${LAYERS:--4,-3,-2,-1}"
POOLING="${POOLING:-mean}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
MAX_TEXT_CHARS="${MAX_TEXT_CHARS:-8000}"
ENCODER_BATCH_SIZE="${ENCODER_BATCH_SIZE:-8}"
HIDDEN_SIZES="${HIDDEN_SIZES:-128,32}"
EPOCHS="${EPOCHS:-40}"
FOLDS="${FOLDS:-5}"
GATE_BETA="${GATE_BETA:-1.5}"
FORCE="${FORCE:-0}"

MODEL="${OUTPUT_DIR}/selective_advantage_policy.pkl"
TRAIN_CACHE="${OUTPUT_DIR}/train_hidden_features.npz"
TEST_CACHE="${OUTPUT_DIR}/test_hidden_features.npz"
RESULT="${OUTPUT_DIR}/offline_test_eval.json"

mkdir -p "${OUTPUT_DIR}"

if [[ "${FORCE}" == "1" || ! -f "${MODEL}" ]]; then
  echo "[selective-advantage] training from train trace only"
  CUDA_VISIBLE_DEVICES="${GPU}" PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}" \
    "${PYTHON_BIN}" -m src.train_selective_advantage_policy \
      --dataset "${TRAIN_TRACE}" \
      --output "${MODEL}" \
      --encoder_name "${ENCODER_NAME}" \
      --feature_cache "${TRAIN_CACHE}" \
      --curve_output "${OUTPUT_DIR}/oof_training_curve.png" \
      --history_output "${OUTPUT_DIR}/oof_training_history.json" \
      --lambda_cost "${LAMBDA_COST}" \
      --layers="${LAYERS}" \
      --pooling "${POOLING}" \
      --max_length "${MAX_LENGTH}" \
      --max_text_chars "${MAX_TEXT_CHARS}" \
      --encoder_batch_size "${ENCODER_BATCH_SIZE}" \
      --hidden_sizes "${HIDDEN_SIZES}" \
      --epochs "${EPOCHS}" \
      --folds "${FOLDS}" \
      --gate_beta "${GATE_BETA}" \
      --device cuda \
    2>&1 | tee "${OUTPUT_DIR}/train.log"
else
  echo "[selective-advantage] reuse model ${MODEL}"
fi

if [[ "${FORCE}" == "1" || ! -f "${RESULT}" ]]; then
  echo "[selective-advantage] one-shot frozen test evaluation"
  CUDA_VISIBLE_DEVICES="${GPU}" PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}" \
    "${PYTHON_BIN}" -m src.evaluate_selective_advantage_offline \
      --trace "${TEST_TRACE}" \
      --policy_model "${MODEL}" \
      --lambda_cost "${LAMBDA_COST}" \
      --feature_cache "${TEST_CACHE}" \
      --output "${RESULT}" \
      --difficulty_output "${OUTPUT_DIR}/offline_test_by_difficulty.json" \
      --source_output "${OUTPUT_DIR}/offline_test_by_source.json" \
      --analysis_output "${OUTPUT_DIR}/offline_test_analysis.json" \
      --decisions_output "${OUTPUT_DIR}/offline_test_decisions.jsonl" \
    2>&1 | tee "${OUTPUT_DIR}/eval.log"
else
  echo "[selective-advantage] reuse frozen evaluation ${RESULT}"
fi

echo "[selective-advantage] done: ${OUTPUT_DIR}"
