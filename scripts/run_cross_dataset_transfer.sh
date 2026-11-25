#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"
export PYTHONUNBUFFERED=1
export NO_PROXY="127.0.0.1,localhost,${NO_PROXY:-}"
export no_proxy="127.0.0.1,localhost,${no_proxy:-}"

CONFIG_TEMPLATE="${CONFIG_TEMPLATE:-configs/default_config.json}"
GPU="${GPU:-1}"
DEVICE="${DEVICE:-cuda}"
DATASETS="${DATASETS:-gsm1k svamp multiarith}"
SEEDS="${SEEDS:-7 21 42}"
LIMIT="${LIMIT:-}"
RUN_DIR="${RUN_DIR:-outputs/cross_dataset_transfer/gsm8k_hierarchical_accuracy}"
POLICY_DIR="${POLICY_DIR:-outputs/multiseed/gsm8k_hierarchical_accuracy}"

declare -A DATA_PATHS=(
  [gsm1k]="${GSM1K_DATA:-data/gsm1k_test.jsonl}"
  [svamp]="${SVAMP_DATA:-data/svamp_test.jsonl}"
  [multiarith]="${MULTIARITH_DATA:-data/multiarith_test.jsonl}"
)

LIMIT_ARGS=()
if [[ -n "${LIMIT}" ]]; then
  LIMIT_ARGS+=(--limit "${LIMIT}")
fi

for DATASET in ${DATASETS}; do
  DATA_PATH="${DATA_PATHS[${DATASET}]:-}"
  if [[ -z "${DATA_PATH}" || ! -s "${DATA_PATH}" ]]; then
    echo "Missing test dataset for ${DATASET}: ${DATA_PATH}" >&2
    exit 1
  fi

  DATASET_DIR="${RUN_DIR}/${DATASET}"
  TRACE="${DATASET_DIR}/test_candidates.jsonl"
  CONFIG="${DATASET_DIR}/eval_config.json"
  mkdir -p "${DATASET_DIR}"

  if [[ "${FORCE:-0}" == "1" || ! -s "${TRACE}" ]]; then
    python - "${CONFIG_TEMPLATE}" "${CONFIG}" "${DATA_PATH}" <<'PYCONFIG'
import json, sys
src, dst, data_path = sys.argv[1:4]
with open(src, encoding="utf-8") as handle:
    config = json.load(handle)
config.setdefault("data", {})["data_path"] = data_path
config["data"]["train_limit"] = None
config["data"]["eval_limit"] = None
with open(dst, "w", encoding="utf-8") as handle:
    json.dump(config, handle, ensure_ascii=False, indent=2)
PYCONFIG

    echo "[transfer] ${DATASET}: generating test candidates from ${DATA_PATH}"
    PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${GPU}" \
      python -m src.build_policy_dataset \
      --config "${CONFIG}" \
      --split test \
      --output "${TRACE}" \
      --actions direct,reflect_1,reflect_2 \
      "${LIMIT_ARGS[@]}"
  else
    echo "[transfer] ${DATASET}: reuse ${TRACE}"
  fi

  python - "${TRACE}" <<'PYTRACE'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
versions = set()
with path.open(encoding="utf-8") as handle:
    for line in handle:
        if line.strip():
            versions.add(json.loads(line).get("prompt_version"))
if versions != {"gsm8k_policy_compat_v1"}:
    raise SystemExit(f"Prompt version mismatch in {path}: {sorted(versions, key=str)}")
print(f"[transfer] prompt_version=gsm8k_policy_compat_v1 verified for {path}")
PYTRACE

  for SEED in ${SEEDS}; do
    OUTPUT="${DATASET_DIR}/results_seed${SEED}.json"
    echo "[transfer] ${DATASET}: evaluate policy seed=${SEED}"
    PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${GPU}" \
      python -m src.evaluate_hierarchical_policy_offline \
      --trace "${TRACE}" \
      --gate_model "${POLICY_DIR}/gate_calibrated_seed${SEED}.pkl" \
      --round_model "${POLICY_DIR}/round_seed${SEED}.pkl" \
      --output "${OUTPUT}"
  done

  PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}" python -m src.summarize_multiseed \
    --input_dir "${DATASET_DIR}" \
    --output "${DATASET_DIR}/summary.json"
done

echo "[transfer] completed under ${RUN_DIR}"
