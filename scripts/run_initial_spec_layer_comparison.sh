#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"
export PYTHONUNBUFFERED=1

GPU="${GPU:-0}"
ENCODER="${ENCODER:-Qwen/Qwen3-0.6B}"
TEST_TRACE="${TEST_TRACE:-outputs/math_full_test_failure_accounted_v1/policy_test_accounted.jsonl}"
INITIAL_ROOT="${INITIAL_ROOT:-outputs/layer_ablation_initial_selection_multiseed_v1}"
BUDGET_ROOT="${BUDGET_ROOT:-outputs/layer_ablation_budget_initial_spec_multiseed_v1}"

mkdir -p "${INITIAL_ROOT}" "${BUDGET_ROOT}"

for seed in 7 21 42; do
  CUDA_VISIBLE_DEVICES="${GPU}" PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}" \
    python -m src.layer_ablation_math \
      --encoder "${ENCODER}" \
      --output "${INITIAL_ROOT}/seed${seed}" \
      --device cuda \
      --max-length 2048 \
      --encoder-batch-size 8 \
      --batch-size 128 \
      --epochs 10 \
      --seed "${seed}" \
      --lambda-cost 0.0001 \
      --test-trace "${TEST_TRACE}" \
      > "${INITIAL_ROOT}/seed${seed}.log" 2>&1
done

for seed in 7 21 42; do
  CUDA_VISIBLE_DEVICES="${GPU}" PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}" \
    python -m src.layer_ablation_budget_math \
      --encoder "${ENCODER}" \
      --output "${BUDGET_ROOT}/seed${seed}" \
      --device cuda \
      --max-length 2048 \
      --encoder-batch-size 8 \
      --batch-size 128 \
      --epochs 10 \
      --seed "${seed}" \
      --lambda-cost 0.0001 \
      --test-trace "${TEST_TRACE}" \
      > "${BUDGET_ROOT}/seed${seed}.log" 2>&1
done

printf '%s\n' "Initial-selection runs: ${INITIAL_ROOT}" "Budget runs: ${BUDGET_ROOT}"
