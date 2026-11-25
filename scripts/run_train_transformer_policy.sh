#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"
export PYTHONUNBUFFERED=1

DATASET="${DATASET:-outputs/policy_dataset_gsm8k_binary_direct_reflect1.jsonl}"
OUTPUT="${OUTPUT:-outputs/transformer_policy_gsm8k_binary_qwen3_layers.pkl}"
ENCODER_NAME="${ENCODER_NAME:-Qwen/Qwen3-0.6B}"
LAYERS="${LAYERS:--4,-3,-2,-1}"
POOLING="${POOLING:-mean}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
MAX_TEXT_CHARS="${MAX_TEXT_CHARS:-8000}"
ENCODER_BATCH_SIZE="${ENCODER_BATCH_SIZE:-8}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
HIDDEN_SIZES="${HIDDEN_SIZES:-128,32}"
EPOCHS="${EPOCHS:-80}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
TEST_SIZE="${TEST_SIZE:-0.2}"
LAMBDA_COST="${LAMBDA_COST:-0.0001}"
DEVICE="${DEVICE:-cuda}"
CURVE_OUTPUT="${CURVE_OUTPUT:-outputs/training_curves/transformer_policy_gsm8k_binary.png}"
HISTORY_OUTPUT="${HISTORY_OUTPUT:-outputs/training_curves/transformer_policy_gsm8k_binary_history.json}"
THRESHOLD_METRIC="${THRESHOLD_METRIC:-answer_accuracy}"
SELECTION_METRIC="${SELECTION_METRIC:-val_answer_accuracy}"
EPOCH_PLOT_DIR="${EPOCH_PLOT_DIR:-outputs/training_curves/transformer_policy_gsm8k_binary_epochs}"

PURE_TEXT_ARGS=()
if [[ "${PURE_TEXT:-1}" == "1" ]]; then
  PURE_TEXT_ARGS+=(--pure_text)
fi
CLASS_BALANCE_ARGS=()
if [[ "${BALANCE_CLASSES:-1}" == "0" ]]; then
  CLASS_BALANCE_ARGS+=(--no_class_balance)
fi
EPOCH_PLOT_ARGS=()
if [[ "${SAVE_EPOCH_PLOTS:-1}" == "0" ]]; then
  EPOCH_PLOT_ARGS+=(--no_epoch_plots)
fi

PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}" python -m src.train_transformer_policy   --dataset "${DATASET}"   --output "${OUTPUT}"   --encoder_name "${ENCODER_NAME}"   --layers="${LAYERS}"   --pooling "${POOLING}"   --max_length "${MAX_LENGTH}"   --max_text_chars "${MAX_TEXT_CHARS}"   --encoder_batch_size "${ENCODER_BATCH_SIZE}"   --train_batch_size "${TRAIN_BATCH_SIZE}"   --hidden_sizes "${HIDDEN_SIZES}"   --epochs "${EPOCHS}"   --learning_rate "${LEARNING_RATE}"   --weight_decay "${WEIGHT_DECAY}"   --test_size "${TEST_SIZE}"   --lambda_cost "${LAMBDA_COST}"   --device "${DEVICE}"   --curve_output "${CURVE_OUTPUT}"   --history_output "${HISTORY_OUTPUT}"   --epoch_plot_dir "${EPOCH_PLOT_DIR}"   --selection_metric "${SELECTION_METRIC}"   --threshold_metric "${THRESHOLD_METRIC}"   "${PURE_TEXT_ARGS[@]}" "${CLASS_BALANCE_ARGS[@]}" "${EPOCH_PLOT_ARGS[@]}"
