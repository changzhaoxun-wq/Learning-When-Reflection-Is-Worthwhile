#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"

GPU="${GPU:-1}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OLD_TRAIN_TRACE="${OLD_TRAIN_TRACE:-outputs/qa_direct_first/hotpotqa_counterfactual_train_v1/policy_train_counterfactual_augmented.jsonl}"
NEW_TRAIN_TRACE="${NEW_TRAIN_TRACE:-outputs/qa_direct_first/hotpotqa_target_aware_v7_full/relabel_f1/policy_train_f1_reward.jsonl}"
OLD_TEST_TRACE="${OLD_TEST_TRACE:-outputs/qa_direct_first/hotpotqa_distractor_8000_v13/relabel_f1/policy_eval_f1_reward.jsonl}"
NEW_TEST_TRACE="${NEW_TEST_TRACE:-outputs/qa_direct_first/hotpotqa_target_aware_v7_full/relabel_f1/policy_eval_f1_reward.jsonl}"
ENCODER_NAME="${ENCODER_NAME:-Qwen/Qwen3-0.6B}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/qa_direct_first/hotpotqa_top_layer_policy_v3_r2_aug}"
TRAINABLE_BLOCKS="${TRAINABLE_BLOCKS:-2}"
LAYER_INDICES="${LAYER_INDICES:--2,-1}"
POOLING="${POOLING:-mean_last}"
HIDDEN_SIZES="${HIDDEN_SIZES:-128,32}"
DROPOUT="${DROPOUT:-0.2}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
MAX_TEXT_CHARS="${MAX_TEXT_CHARS:-8000}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
GRADIENT_ACCUMULATION="${GRADIENT_ACCUMULATION:-4}"
EPOCHS="${EPOCHS:-5}"
LEARNING_RATE="${LEARNING_RATE:-0.00002}"
HEAD_LEARNING_RATE="${HEAD_LEARNING_RATE:-0.0002}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
VALIDATION_SIZE="${VALIDATION_SIZE:-0.2}"
NEGATIVE_RATIO="${NEGATIVE_RATIO:-8}"
GATE_TARGET="${GATE_TARGET:-direct_error}"
RANKER_TARGET="${RANKER_TARGET:-candidate_correctness}"
SELECTION_SCORE_MODE="${SELECTION_SCORE_MODE:-correctness_margin}"
MIN_UTILITY_GAIN="${MIN_UTILITY_GAIN:-0.05}"
PATIENCE="${PATIENCE:-2}"
SEED="${SEED:-42}"
MAX_INVOCATION_RATE="${MAX_INVOCATION_RATE:-0.15}"
LAMBDA_COST="${LAMBDA_COST:-0.0001}"
THRESHOLD_GRID_POINTS="${THRESHOLD_GRID_POINTS:-81}"
LIMIT="${LIMIT:-}"
AUGMENTATION_DATASETS="${AUGMENTATION_DATASETS:-}"
FORCE="${FORCE:-0}"

MODEL="${OUTPUT_DIR}/top_layer_policy.pkl"
RESULT="${OUTPUT_DIR}/offline_test_eval.json"
mkdir -p "${OUTPUT_DIR}"

LIMIT_ARGS=()
if [[ -n "${LIMIT}" ]]; then
  LIMIT_ARGS=(--limit "${LIMIT}")
fi

AUGMENTATION_ARGS=()
if [[ -n "${AUGMENTATION_DATASETS}" ]]; then
  IFS=',' read -r -a augmentation_paths <<< "${AUGMENTATION_DATASETS}"
  for path in "${augmentation_paths[@]}"; do
    [[ -n "${path}" ]] && AUGMENTATION_ARGS+=(--augmentation_dataset "${path}")
  done
fi

if [[ "${FORCE}" == "1" || ! -f "${MODEL}" ]]; then
  echo "[qa-top-layer] train final ${TRAINABLE_BLOCKS} Transformer blocks and two heads"
  CUDA_VISIBLE_DEVICES="${GPU}" PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}" \
    "${PYTHON_BIN}" -m src.train_qa_top_layer_policy \
      --old_dataset "${OLD_TRAIN_TRACE}" \
      --new_dataset "${NEW_TRAIN_TRACE}" \
      --output "${MODEL}" \
      --encoder_name "${ENCODER_NAME}" \
      --curve_output "${OUTPUT_DIR}/training_curve.png" \
      --history_output "${OUTPUT_DIR}/training_history.json" \
      --trainable_blocks "${TRAINABLE_BLOCKS}" \
      --layer_indices="${LAYER_INDICES}" \
      --pooling "${POOLING}" \
      --hidden_sizes "${HIDDEN_SIZES}" \
      --dropout "${DROPOUT}" \
      --max_length "${MAX_LENGTH}" \
      --max_text_chars "${MAX_TEXT_CHARS}" \
      --train_batch_size "${TRAIN_BATCH_SIZE}" \
      --eval_batch_size "${EVAL_BATCH_SIZE}" \
      --gradient_accumulation "${GRADIENT_ACCUMULATION}" \
      --epochs "${EPOCHS}" \
      --learning_rate "${LEARNING_RATE}" \
      --head_learning_rate "${HEAD_LEARNING_RATE}" \
      --weight_decay "${WEIGHT_DECAY}" \
      --validation_size "${VALIDATION_SIZE}" \
      --negative_ratio "${NEGATIVE_RATIO}" \
      --gate_target "${GATE_TARGET}" \
      --ranker_target "${RANKER_TARGET}" \
      --selection_score_mode "${SELECTION_SCORE_MODE}" \
      --min_utility_gain "${MIN_UTILITY_GAIN}" \
      --patience "${PATIENCE}" \
      --random_state "${SEED}" \
      --device cuda \
      --max_invocation_rate "${MAX_INVOCATION_RATE}" \
      --lambda_cost "${LAMBDA_COST}" \
      --threshold_grid_points "${THRESHOLD_GRID_POINTS}" \
      "${AUGMENTATION_ARGS[@]}" \
      "${LIMIT_ARGS[@]}" \
    2>&1 | tee "${OUTPUT_DIR}/train.log"
else
  echo "[qa-top-layer] reuse model ${MODEL}"
fi

if [[ -n "${LIMIT}" ]]; then
  echo "[qa-top-layer] smoke training completed; skip full-test evaluation when LIMIT is set"
  exit 0
fi

if [[ "${FORCE}" == "1" || ! -f "${RESULT}" ]]; then
  echo "[qa-top-layer] evaluate saved test candidates; no LLM calls"
  CUDA_VISIBLE_DEVICES="${GPU}" PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}" \
    "${PYTHON_BIN}" -m src.evaluate_qa_dual_candidate_offline \
      --old_trace "${OLD_TEST_TRACE}" \
      --new_trace "${NEW_TEST_TRACE}" \
      --policy_model "${MODEL}" \
      --lambda_cost "${LAMBDA_COST}" \
      --output "${RESULT}" \
      --difficulty_output "${OUTPUT_DIR}/offline_test_by_difficulty.json" \
      --source_output "${OUTPUT_DIR}/offline_test_by_source.json" \
      --analysis_output "${OUTPUT_DIR}/offline_test_analysis.json" \
      --decisions_output "${OUTPUT_DIR}/offline_test_decisions.jsonl" \
    2>&1 | tee "${OUTPUT_DIR}/eval.log"
else
  echo "[qa-top-layer] reuse evaluation ${RESULT}"
fi

echo "[qa-top-layer] done: ${OUTPUT_DIR}"
