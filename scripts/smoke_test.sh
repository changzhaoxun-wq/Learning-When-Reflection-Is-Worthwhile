#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="$(mktemp -d "${TMPDIR:-/tmp}/adaptive-reflection-smoke.XXXXXX")"
PYTHON_BIN="${PYTHON_BIN:-python}"

cleanup() {
  rm -rf "${RUN_DIR}"
}
trap cleanup EXIT

cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

echo "[smoke] run directory: ${RUN_DIR}"
echo "[smoke] building a tiny fake-model policy dataset"
"${PYTHON_BIN}" -m src.build_policy_dataset \
  --config configs/smoke_config.json \
  --split train \
  --limit 2 \
  --output "${RUN_DIR}/policy_dataset.jsonl"

echo "[smoke] training the tabular policy"
"${PYTHON_BIN}" -m src.train_policy \
  --dataset "${RUN_DIR}/policy_dataset.jsonl" \
  --output "${RUN_DIR}/policy_model.pkl" \
  --model_type logistic_regression

echo "[smoke] evaluating direct/fixed-reflection/learned-policy paths"
"${PYTHON_BIN}" -m src.evaluate_policy \
  --config configs/smoke_config.json \
  --policy_model "${RUN_DIR}/policy_model.pkl" \
  --split test \
  --limit 2 \
  --output "${RUN_DIR}/eval_results.json" \
  --difficulty_output "${RUN_DIR}/eval_by_difficulty.json" \
  --source_output "${RUN_DIR}/eval_by_source.json"

test -s "${RUN_DIR}/policy_dataset.jsonl"
test -s "${RUN_DIR}/policy_model.pkl"
test -s "${RUN_DIR}/eval_results.json"
echo "[smoke] PASS: ${RUN_DIR}"
