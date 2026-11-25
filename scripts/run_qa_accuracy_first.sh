#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"

GPU="${GPU:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
TRAIN_TRACE="${TRAIN_TRACE:-outputs/qa_direct_first/hotpotqa_counterfactual_train_v1/policy_train_counterfactual_augmented.jsonl}"
TEST_TRACE="${TEST_TRACE:-outputs/qa_direct_first/hotpotqa_distractor_8000_v13/relabel_f1/policy_eval_f1_reward.jsonl}"
ENCODER_NAME="${ENCODER_NAME:-Qwen/Qwen3-0.6B}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/qa_direct_first/hotpotqa_counterfactual_train_v1/accuracy_first_2048}"
LAYERS="${LAYERS:--4,-3,-2,-1}"
POOLING="${POOLING:-mean}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
MAX_TEXT_CHARS="${MAX_TEXT_CHARS:-8000}"
ENCODER_BATCH_SIZE="${ENCODER_BATCH_SIZE:-4}"
HIDDEN_SIZES="${HIDDEN_SIZES:-128,32}"
DROPOUT="${DROPOUT:-0.3}"
EPOCHS="${EPOCHS:-30}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
LEARNING_RATE="${LEARNING_RATE:-0.0005}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.001}"
FOLDS="${FOLDS:-5}"
SEED="${SEED:-42}"
MAX_REFLECTION_RATE="${MAX_REFLECTION_RATE:-0.15}"
LAMBDA_COST="${LAMBDA_COST:-0.0001}"
FORCE="${FORCE:-0}"

MODEL="${OUTPUT_DIR}/accuracy_first_policy.pkl"
TRAIN_GATE_CACHE="${OUTPUT_DIR}/train_gate_hidden_features.npz"
TRAIN_ACCEPT_CACHE="${OUTPUT_DIR}/train_acceptance_hidden_features.npz"
TEST_GATE_CACHE="${OUTPUT_DIR}/test_gate_hidden_features.npz"
TEST_ACCEPT_CACHE="${OUTPUT_DIR}/test_acceptance_hidden_features.npz"
RESULT="${OUTPUT_DIR}/offline_test_eval.json"

mkdir -p "${OUTPUT_DIR}"

if [[ "${FORCE}" == "1" || ! -f "${MODEL}" ]]; then
  echo "[qa-accuracy-first] train 2048-token OOF gate and acceptance ensemble"
  CUDA_VISIBLE_DEVICES="${GPU}" PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}" \
    "${PYTHON_BIN}" -m src.train_accuracy_first_qa_policy \
      --dataset "${TRAIN_TRACE}" \
      --output "${MODEL}" \
      --encoder_name "${ENCODER_NAME}" \
      --gate_cache "${TRAIN_GATE_CACHE}" \
      --acceptance_cache "${TRAIN_ACCEPT_CACHE}" \
      --curve_output "${OUTPUT_DIR}/oof_training_curve.png" \
      --history_output "${OUTPUT_DIR}/oof_training_history.json" \
      --layers="${LAYERS}" \
      --pooling "${POOLING}" \
      --max_length "${MAX_LENGTH}" \
      --max_text_chars "${MAX_TEXT_CHARS}" \
      --encoder_batch_size "${ENCODER_BATCH_SIZE}" \
      --hidden_sizes "${HIDDEN_SIZES}" \
      --dropout "${DROPOUT}" \
      --epochs "${EPOCHS}" \
      --train_batch_size "${TRAIN_BATCH_SIZE}" \
      --learning_rate "${LEARNING_RATE}" \
      --weight_decay "${WEIGHT_DECAY}" \
      --folds "${FOLDS}" \
      --random_state "${SEED}" \
      --device cuda \
      --max_reflection_rate "${MAX_REFLECTION_RATE}" \
    2>&1 | tee "${OUTPUT_DIR}/train.log"
else
  echo "[qa-accuracy-first] reuse model ${MODEL}"
fi

if [[ "${FORCE}" == "1" || ! -f "${RESULT}" ]]; then
  echo "[qa-accuracy-first] evaluate once on saved test candidates"
  CUDA_VISIBLE_DEVICES="${GPU}" PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}" \
    "${PYTHON_BIN}" -m src.evaluate_qa_accuracy_first_offline \
      --trace "${TEST_TRACE}" \
      --policy_model "${MODEL}" \
      --gate_cache "${TEST_GATE_CACHE}" \
      --acceptance_cache "${TEST_ACCEPT_CACHE}" \
      --lambda_cost "${LAMBDA_COST}" \
      --output "${RESULT}" \
      --difficulty_output "${OUTPUT_DIR}/offline_test_by_difficulty.json" \
      --source_output "${OUTPUT_DIR}/offline_test_by_source.json" \
      --analysis_output "${OUTPUT_DIR}/offline_test_analysis.json" \
      --decisions_output "${OUTPUT_DIR}/offline_test_decisions.jsonl" \
    2>&1 | tee "${OUTPUT_DIR}/eval.log"
else
  echo "[qa-accuracy-first] reuse evaluation ${RESULT}"
fi

echo "[qa-accuracy-first] done: ${OUTPUT_DIR}"

