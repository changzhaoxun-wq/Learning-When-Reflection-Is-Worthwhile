#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"
export PYTHONUNBUFFERED=1

TRACE="${TRACE:-outputs/traces/gsm8k_test_policy_trace.jsonl}"
RF_POLICY_MODEL="${RF_POLICY_MODEL:-outputs/policy_model_gsm8k_current_rf_o1.pkl}"
MLP_POLICY_MODEL="${MLP_POLICY_MODEL:-outputs/neural_policy_model_gsm8k_current.pkl}"
LAMBDA_COST="${LAMBDA_COST:-0.0001}"
ACTIONS="${ACTIONS:-direct,reflect_1,reflect_2}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/offline_eval_runs/gsm8k}"
mkdir -p "${OUTPUT_DIR}"

run_one() {
  local name="$1"
  local model="$2"
  echo "[offline] ${name}: ${model}"
  PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}" python -m src.evaluate_policy_offline \
    --trace "${TRACE}" \
    --policy_model "${model}" \
    --lambda_cost "${LAMBDA_COST}" \
    --actions "${ACTIONS}" \
    --output "${OUTPUT_DIR}/${name}.json" \
    --difficulty_output "${OUTPUT_DIR}/${name}_by_difficulty.json" \
    --source_output "${OUTPUT_DIR}/${name}_by_source.json"
}

run_one rf "${RF_POLICY_MODEL}"
run_one mlp "${MLP_POLICY_MODEL}"
echo "[offline] completed; no LLM calls were made"
