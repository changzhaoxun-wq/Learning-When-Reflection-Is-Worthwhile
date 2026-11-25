#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"
export PYTHONUNBUFFERED=1

ROOT="${ROOT:-outputs/decoupling_ablation}"
ENCODER_NAME="${ENCODER_NAME:-Qwen/Qwen3-0.6B}"
GPU="${GPU:-1}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-42}"
LAMBDA_COST="${LAMBDA_COST:-0.0001}"
HIDDEN_SIZES="${HIDDEN_SIZES:-64,16}"
RESULTS_BASENAME="${RESULTS_BASENAME:-eval_results.json}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"

for CASE_DIR in "${ROOT}"/*; do
  [[ -d "${CASE_DIR}" ]] || continue
  [[ -s "${CASE_DIR}/policy_train.jsonl" && -s "${CASE_DIR}/policy_test.jsonl" ]] || continue
  NAME="$(basename "${CASE_DIR}")"
  DATA_DIR="${CASE_DIR}/hierarchical"
  POLICY_DIR="${CASE_DIR}/policy"
  mkdir -p "${DATA_DIR}" "${POLICY_DIR}"

  GATE_MODEL="${POLICY_DIR}/gate_seed${SEED}.pkl"
  ROUND_MODEL="${POLICY_DIR}/round_seed${SEED}.pkl"
  CALIBRATED_GATE_MODEL="${POLICY_DIR}/gate_calibrated_seed${SEED}.pkl"
  RESULTS_PATH="${POLICY_DIR}/${RESULTS_BASENAME}"

  if [[ "${SKIP_COMPLETED}" == "1" \
    && -s "${GATE_MODEL}" \
    && -s "${ROUND_MODEL}" \
    && -s "${CALIBRATED_GATE_MODEL}" \
    && -s "${RESULTS_PATH}" ]]; then
    echo "[decouple-policy] skip completed ${NAME} seed=${SEED}"
    continue
  fi

  python -m src.prepare_hierarchical_policy_dataset \
    --input "${CASE_DIR}/policy_train.jsonl" \
    --output_dir "${DATA_DIR}" \
    --objective accuracy \
    --lambda_cost "${LAMBDA_COST}"

  if [[ "${SKIP_COMPLETED}" == "1" && -s "${GATE_MODEL}" ]]; then
    echo "[decouple-policy] reuse gate ${NAME} seed=${SEED}"
  else
    CUDA_VISIBLE_DEVICES="${GPU}" python -m src.train_transformer_policy \
      --dataset "${DATA_DIR}/gate.jsonl" \
      --output "${GATE_MODEL}" \
      --encoder_name "${ENCODER_NAME}" \
      --layers=-2,-1 \
      --pooling mean_last \
      --pure_text \
      --max_length 1024 \
      --encoder_batch_size 8 \
      --train_batch_size 128 \
      --hidden_sizes "${HIDDEN_SIZES}" \
      --epochs 10 \
      --learning_rate 0.001 \
      --weight_decay 0.001 \
      --test_size 0.2 \
      --random_state "${SEED}" \
      --lambda_cost "${LAMBDA_COST}" \
      --no_class_balance \
      --device "${DEVICE}" \
      --curve_output "${POLICY_DIR}/gate_seed${SEED}.png" \
      --history_output "${POLICY_DIR}/gate_seed${SEED}_history.json" \
      --epoch_plot_dir "${POLICY_DIR}/gate_seed${SEED}_epochs" \
      --selection_metric val_answer_accuracy \
      --threshold_metric answer_accuracy
  fi

  if [[ "${SKIP_COMPLETED}" == "1" && -s "${ROUND_MODEL}" ]]; then
    echo "[decouple-policy] reuse round ${NAME} seed=${SEED}"
  else
    CUDA_VISIBLE_DEVICES="${GPU}" python -m src.train_transformer_policy \
      --dataset "${DATA_DIR}/round.jsonl" \
      --output "${ROUND_MODEL}" \
      --encoder_name "${ENCODER_NAME}" \
      --layers=-2,-1 \
      --pooling mean_last \
      --pure_text \
      --max_length 1024 \
      --encoder_batch_size 4 \
      --train_batch_size 128 \
      --hidden_sizes "${HIDDEN_SIZES}" \
      --epochs 10 \
      --learning_rate 0.001 \
      --weight_decay 0.001 \
      --test_size 0.2 \
      --random_state "${SEED}" \
      --lambda_cost "${LAMBDA_COST}" \
      --device "${DEVICE}" \
      --curve_output "${POLICY_DIR}/round_seed${SEED}.png" \
      --history_output "${POLICY_DIR}/round_seed${SEED}_history.json" \
      --epoch_plot_dir "${POLICY_DIR}/round_seed${SEED}_epochs" \
      --selection_metric val_answer_accuracy \
      --threshold_metric answer_accuracy
  fi

  if [[ "${SKIP_COMPLETED}" == "1" && -s "${CALIBRATED_GATE_MODEL}" ]]; then
    echo "[decouple-policy] reuse calibrated gate ${NAME} seed=${SEED}"
  else
    CUDA_VISIBLE_DEVICES="${GPU}" python -m src.scan_gate_cost_threshold \
      --dataset "${CASE_DIR}/policy_train.jsonl" \
      --gate_model "${GATE_MODEL}" \
      --round_model "${ROUND_MODEL}" \
      --output_gate_model "${CALIBRATED_GATE_MODEL}" \
      --device "${DEVICE}" \
      --test_size 0.2 \
      --random_state "${SEED}"
  fi

  if [[ "${SKIP_COMPLETED}" == "1" && -s "${RESULTS_PATH}" ]]; then
    echo "[decouple-policy] reuse evaluation ${NAME} seed=${SEED}"
  else
    CUDA_VISIBLE_DEVICES="${GPU}" python -m src.evaluate_hierarchical_policy_offline \
      --trace "${CASE_DIR}/policy_test.jsonl" \
      --gate_model "${CALIBRATED_GATE_MODEL}" \
      --round_model "${ROUND_MODEL}" \
      --lambda_cost "${LAMBDA_COST}" \
      --output "${RESULTS_PATH}"
  fi
  echo "[decouple-policy] completed ${NAME} seed=${SEED}"
done
