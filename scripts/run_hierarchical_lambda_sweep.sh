#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"
export PYTHONUNBUFFERED=1

INPUT="${INPUT:-outputs/policy_dataset_gsm8k_full_policy_compat_no_verify.jsonl}"
TRACE="${TRACE:-outputs/traces/gsm8k_test_policy_trace_detailed.jsonl}"
ENCODER_NAME="${ENCODER_NAME:-Qwen/Qwen3-0.6B}"
GPU="${GPU:-1}"
DEVICE="${DEVICE:-cuda}"
LAMBDAS="${LAMBDAS:-0 0.00005 0.0001 0.0002 0.0005}"
SEEDS="${SEEDS:-42}"
ROOT="${ROOT:-outputs/lambda_sweep_hierarchical/gsm8k}"

mkdir -p "${ROOT}"

lambda_tag() {
  echo "$1" | tr '.' 'p'
}

for LAMBDA in ${LAMBDAS}; do
  TAG="lambda_$(lambda_tag "${LAMBDA}")"
  RUN_DIR="${ROOT}/${TAG}"
  DATA_DIR="${RUN_DIR}/policy_datasets"
  mkdir -p "${RUN_DIR}" "${DATA_DIR}"

  POLICY_DATA="${DATA_DIR}/policy.jsonl"
  if [[ "${FORCE:-0}" == "1" || ! -s "${POLICY_DATA}" ]]; then
    PYTHONPATH="${PWD}:${PYTHONPATH:-}" python -m src.prepare_policy_dataset \
      --input "${INPUT}" \
      --output "${POLICY_DATA}" \
      --lambda_cost "${LAMBDA}" \
      --actions direct,reflect_1,reflect_2 \
      --objective reward
  fi

  HIER_DIR="${DATA_DIR}/hierarchical"
  if [[ "${FORCE:-0}" == "1" || ! -s "${HIER_DIR}/gate.jsonl" || ! -s "${HIER_DIR}/round.jsonl" ]]; then
    PYTHONPATH="${PWD}:${PYTHONPATH:-}" python -m src.prepare_hierarchical_policy_dataset \
      --input "${POLICY_DATA}" \
      --output_dir "${HIER_DIR}" \
      --objective reward \
      --lambda_cost "${LAMBDA}"
  fi

  for SEED in ${SEEDS}; do
    GATE="${RUN_DIR}/gate_seed${SEED}.pkl"
    ROUND="${RUN_DIR}/round_seed${SEED}.pkl"

    echo "[lambda] lambda=${LAMBDA} seed=${SEED}: train gate"
    CUDA_VISIBLE_DEVICES="${GPU}" PYTHONPATH="${PWD}:${PYTHONPATH:-}" python -m src.train_transformer_policy \
      --dataset "${HIER_DIR}/gate.jsonl" \
      --output "${GATE}" \
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
      --lambda_cost "${LAMBDA}" \
      --no_class_balance \
      --device "${DEVICE}" \
      --curve_output "${RUN_DIR}/gate_seed${SEED}.png" \
      --history_output "${RUN_DIR}/gate_seed${SEED}_history.json" \
      --epoch_plot_dir "${RUN_DIR}/gate_seed${SEED}_epochs" \
      --selection_metric val_reward \
      --threshold_metric reward

    echo "[lambda] lambda=${LAMBDA} seed=${SEED}: train round"
    CUDA_VISIBLE_DEVICES="${GPU}" PYTHONPATH="${PWD}:${PYTHONPATH:-}" python -m src.train_transformer_policy \
      --dataset "${HIER_DIR}/round.jsonl" \
      --output "${ROUND}" \
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
      --lambda_cost "${LAMBDA}" \
      --device "${DEVICE}" \
      --curve_output "${RUN_DIR}/round_seed${SEED}.png" \
      --history_output "${RUN_DIR}/round_seed${SEED}_history.json" \
      --epoch_plot_dir "${RUN_DIR}/round_seed${SEED}_epochs" \
      --selection_metric val_reward \
      --threshold_metric reward

    CALIBRATED="${RUN_DIR}/gate_calibrated_seed${SEED}.pkl"
    CUDA_VISIBLE_DEVICES="${GPU}" PYTHONPATH="${PWD}:${PYTHONPATH:-}" python -m src.scan_gate_cost_threshold \
      --dataset "${POLICY_DATA}" \
      --gate_model "${GATE}" \
      --round_model "${ROUND}" \
      --output_gate_model "${CALIBRATED}" \
      --device "${DEVICE}" \
      --test_size 0.2 \
      --random_state "${SEED}" \
      --lambda_cost "${LAMBDA}" \
      --objective reward

    CUDA_VISIBLE_DEVICES="${GPU}" PYTHONPATH="${PWD}:${PYTHONPATH:-}" python -m src.evaluate_hierarchical_policy_offline \
      --trace "${TRACE}" \
      --gate_model "${CALIBRATED}" \
      --round_model "${ROUND}" \
      --lambda_cost "${LAMBDA}" \
      --output "${RUN_DIR}/results_seed${SEED}.json"
  done
done

PYTHONPATH="${PWD}:${PYTHONPATH:-}" python -m src.summarize_lambda_sweep \
  --root "${ROOT}" \
  --output "${ROOT}/summary.json"
echo "[lambda] completed; summary=${ROOT}/summary.json"
