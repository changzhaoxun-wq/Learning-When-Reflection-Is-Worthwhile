#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"
export PYTHONUNBUFFERED=1

DATASET_DIR="${DATASET_DIR:-outputs/policy_datasets/gsm8k_hierarchical_accuracy}"
TRACE="${TRACE:-outputs/traces/gsm8k_test_policy_trace_detailed.jsonl}"
ENCODER_NAME="${ENCODER_NAME:-Qwen/Qwen3-0.6B}"
GPU="${GPU:-1}"
DEVICE="${DEVICE:-cuda}"
SEEDS="${SEEDS:-7 21 42}"
RUN_DIR="${RUN_DIR:-outputs/multiseed/gsm8k_hierarchical_accuracy}"

mkdir -p "${RUN_DIR}" "outputs/training_curves/multiseed"

for SEED in ${SEEDS}; do
  echo "[multiseed] seed=${SEED}: train gate"
  CUDA_VISIBLE_DEVICES="${GPU}" PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}" python -m src.train_transformer_policy \
    --dataset "${DATASET_DIR}/gate.jsonl" \
    --output "${RUN_DIR}/gate_seed${SEED}.pkl" \
    --encoder_name "${ENCODER_NAME}" \
    --layers=-2,-1 \
    --pooling mean_last \
    --pure_text \
    --max_length 1024 \
    --encoder_batch_size 8 \
    --train_batch_size 128 \
    --hidden_sizes 64,16 \
    --epochs 10 \
    --learning_rate 0.001 \
    --weight_decay 0.001 \
    --test_size 0.2 \
    --random_state "${SEED}" \
    --lambda_cost 0.0001 \
    --no_class_balance \
    --device "${DEVICE}" \
    --curve_output "outputs/training_curves/multiseed/gate_seed${SEED}.png" \
    --history_output "outputs/training_curves/multiseed/gate_seed${SEED}_history.json" \
    --epoch_plot_dir "outputs/training_curves/multiseed/gate_seed${SEED}_epochs" \
    --selection_metric val_answer_accuracy \
    --threshold_metric answer_accuracy

  echo "[multiseed] seed=${SEED}: train round"
  CUDA_VISIBLE_DEVICES="${GPU}" PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}" python -m src.train_transformer_policy \
    --dataset "${DATASET_DIR}/round.jsonl" \
    --output "${RUN_DIR}/round_seed${SEED}.pkl" \
    --encoder_name "${ENCODER_NAME}" \
    --layers=-2,-1 \
    --pooling mean_last \
    --pure_text \
    --max_length 1024 \
    --encoder_batch_size 4 \
    --train_batch_size 128 \
    --hidden_sizes 64,16 \
    --epochs 10 \
    --learning_rate 0.001 \
    --weight_decay 0.001 \
    --test_size 0.2 \
    --random_state "${SEED}" \
    --lambda_cost 0.0001 \
    --device "${DEVICE}" \
    --curve_output "outputs/training_curves/multiseed/round_seed${SEED}.png" \
    --history_output "outputs/training_curves/multiseed/round_seed${SEED}_history.json" \
    --epoch_plot_dir "outputs/training_curves/multiseed/round_seed${SEED}_epochs" \
    --selection_metric val_answer_accuracy \
    --threshold_metric answer_accuracy

  echo "[multiseed] seed=${SEED}: calibrate gate on its validation split"
  CUDA_VISIBLE_DEVICES="${GPU}" PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}" python -m src.scan_gate_cost_threshold \
    --dataset outputs/policy_dataset_gsm8k_three_action_accuracy.jsonl \
    --gate_model "${RUN_DIR}/gate_seed${SEED}.pkl" \
    --round_model "${RUN_DIR}/round_seed${SEED}.pkl" \
    --output_gate_model "${RUN_DIR}/gate_calibrated_seed${SEED}.pkl" \
    --device "${DEVICE}" \
    --test_size 0.2 \
    --random_state "${SEED}"

  echo "[multiseed] seed=${SEED}: offline test evaluation"
  CUDA_VISIBLE_DEVICES="${GPU}" PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}" python -m src.evaluate_hierarchical_policy_offline \
    --trace "${TRACE}" \
    --gate_model "${RUN_DIR}/gate_calibrated_seed${SEED}.pkl" \
    --round_model "${RUN_DIR}/round_seed${SEED}.pkl" \
    --output "${RUN_DIR}/results_seed${SEED}.json"
done

PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}" python -m src.summarize_multiseed \
  --input_dir "${RUN_DIR}" \
  --output "${RUN_DIR}/summary.json"

echo "[multiseed] completed; summary=${RUN_DIR}/summary.json"
