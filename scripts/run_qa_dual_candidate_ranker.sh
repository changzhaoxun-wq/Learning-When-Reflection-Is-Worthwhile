#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"

GPU="${GPU:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OLD_TRAIN_TRACE="${OLD_TRAIN_TRACE:-outputs/qa_direct_first/hotpotqa_counterfactual_train_v1/policy_train_counterfactual_augmented.jsonl}"
NEW_TRAIN_TRACE="${NEW_TRAIN_TRACE:-outputs/qa_direct_first/hotpotqa_target_aware_v7_full/relabel_f1/policy_train_f1_reward.jsonl}"
OLD_TEST_TRACE="${OLD_TEST_TRACE:-outputs/qa_direct_first/hotpotqa_distractor_8000_v13/relabel_f1/policy_eval_f1_reward.jsonl}"
NEW_TEST_TRACE="${NEW_TEST_TRACE:-outputs/qa_direct_first/hotpotqa_target_aware_v7_full/relabel_f1/policy_eval_f1_reward.jsonl}"
ENCODER_NAME="${ENCODER_NAME:-Qwen/Qwen3-0.6B}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/qa_direct_first/hotpotqa_dual_candidate_ranker_v2}"
FEATURE_CACHE_DIR="${FEATURE_CACHE_DIR:-outputs/qa_direct_first/hotpotqa_dual_candidate_ranker_v1}"
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
MAX_INVOCATION_RATE="${MAX_INVOCATION_RATE:-0.15}"
LAMBDA_COST="${LAMBDA_COST:-0.0001}"
THRESHOLD_GRID_POINTS="${THRESHOLD_GRID_POINTS:-81}"
GATE_TARGET="${GATE_TARGET:-direct_error}"
RANKER_TARGET="${RANKER_TARGET:-repair_advantage}"
FORCE="${FORCE:-0}"

MODEL="${OUTPUT_DIR}/dual_candidate_policy.pkl"
TRAIN_GATE_CACHE="${TRAIN_GATE_CACHE:-${FEATURE_CACHE_DIR}/train_gate_hidden_features.npz}"
TRAIN_CANDIDATE_CACHE="${TRAIN_CANDIDATE_CACHE:-${FEATURE_CACHE_DIR}/train_candidate_hidden_features.npz}"
TEST_GATE_CACHE="${TEST_GATE_CACHE:-${FEATURE_CACHE_DIR}/test_gate_hidden_features.npz}"
TEST_CANDIDATE_CACHE="${TEST_CANDIDATE_CACHE:-${FEATURE_CACHE_DIR}/test_candidate_hidden_features.npz}"
RESULT="${OUTPUT_DIR}/offline_test_eval.json"

mkdir -p "${OUTPUT_DIR}"

if [[ "${FORCE}" == "1" || ! -f "${MODEL}" ]]; then
  echo "[qa-dual] train OOF gate and v5/v7 candidate ranker"
  CUDA_VISIBLE_DEVICES="${GPU}" PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}" \
    "${PYTHON_BIN}" -m src.train_qa_dual_candidate_policy \
      --old_dataset "${OLD_TRAIN_TRACE}" \
      --new_dataset "${NEW_TRAIN_TRACE}" \
      --output "${MODEL}" \
      --encoder_name "${ENCODER_NAME}" \
      --gate_cache "${TRAIN_GATE_CACHE}" \
      --candidate_cache "${TRAIN_CANDIDATE_CACHE}" \
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
      --max_invocation_rate "${MAX_INVOCATION_RATE}" \
      --lambda_cost "${LAMBDA_COST}" \
      --threshold_grid_points "${THRESHOLD_GRID_POINTS}" \
      --gate_target "${GATE_TARGET}" \
      --ranker_target "${RANKER_TARGET}" \
    2>&1 | tee "${OUTPUT_DIR}/train.log"
else
  echo "[qa-dual] reuse model ${MODEL}"
fi

if [[ "${FORCE}" == "1" || ! -f "${RESULT}" ]]; then
  echo "[qa-dual] evaluate aligned saved candidates; no LLM calls"
  CUDA_VISIBLE_DEVICES="${GPU}" PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}" \
    "${PYTHON_BIN}" -m src.evaluate_qa_dual_candidate_offline \
      --old_trace "${OLD_TEST_TRACE}" \
      --new_trace "${NEW_TEST_TRACE}" \
      --policy_model "${MODEL}" \
      --gate_cache "${TEST_GATE_CACHE}" \
      --candidate_cache "${TEST_CANDIDATE_CACHE}" \
      --lambda_cost "${LAMBDA_COST}" \
      --output "${RESULT}" \
      --difficulty_output "${OUTPUT_DIR}/offline_test_by_difficulty.json" \
      --source_output "${OUTPUT_DIR}/offline_test_by_source.json" \
      --analysis_output "${OUTPUT_DIR}/offline_test_analysis.json" \
      --decisions_output "${OUTPUT_DIR}/offline_test_decisions.jsonl" \
    2>&1 | tee "${OUTPUT_DIR}/eval.log"
else
  echo "[qa-dual] reuse evaluation ${RESULT}"
fi

echo "[qa-dual] done: ${OUTPUT_DIR}"
