#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"
export PYTHONUNBUFFERED=1

TRACE="${TRACE:-outputs/traces/gsm8k_test_policy_trace_detailed.jsonl}"
POLICY_ROOT="${POLICY_ROOT:-outputs/multiseed/gsm8k_hierarchical_accuracy}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/final_paper_eval/gsm8k_accuracy_first}"
SEEDS="${SEEDS:-7 21 42}"
LAMBDA_COST="${LAMBDA_COST:-0.0001}"

if [[ ! -s "${TRACE}" ]]; then
  echo "Missing frozen test trace: ${TRACE}" >&2
  exit 1
fi

python - "${TRACE}" <<'PY'
import json, sys
path = sys.argv[1]
versions = set()
with open(path, encoding="utf-8") as handle:
    for line in handle:
        if line.strip():
            versions.add(json.loads(line).get("prompt_version"))
if versions != {"gsm8k_policy_compat_v1"}:
    raise SystemExit(f"Prompt version mismatch: {sorted(versions, key=str)}")
print("Verified frozen prompt_version=gsm8k_policy_compat_v1")
PY

mkdir -p "${OUTPUT_DIR}"
for SEED in ${SEEDS}; do
  GATE="${POLICY_ROOT}/gate_calibrated_seed${SEED}.pkl"
  ROUND="${POLICY_ROOT}/round_seed${SEED}.pkl"
  [[ -s "${GATE}" && -s "${ROUND}" ]] || {
    echo "Missing frozen policy bundle for seed ${SEED}" >&2
    exit 1
  }
  python -m src.evaluate_hierarchical_policy_offline \
    --trace "${TRACE}" \
    --gate_model "${GATE}" \
    --round_model "${ROUND}" \
    --lambda_cost "${LAMBDA_COST}" \
    --output "${OUTPUT_DIR}/results_seed${SEED}.json"
done

python -m src.summarize_multiseed \
  --input_dir "${OUTPUT_DIR}" \
  --output "${OUTPUT_DIR}/summary.json"

echo "Frozen accuracy-first evaluation completed: ${OUTPUT_DIR}"
