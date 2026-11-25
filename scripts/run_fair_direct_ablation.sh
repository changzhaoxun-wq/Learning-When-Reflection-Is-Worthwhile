#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"
export PYTHONUNBUFFERED=1

CONFIG_TEMPLATE="${CONFIG_TEMPLATE:-configs/default_config.json}"
TEST_DATA="${TEST_DATA:-data/gsm8k_test.jsonl}"
ROOT="${ROOT:-outputs/base_direct_fairness}"
LIMIT="${LIMIT-}"
FORCE="${FORCE:-0}"
BASE_8B_URL="${BASE_8B_URL:-http://127.0.0.1:8000/v1}"
BASE_8B_MODEL="${BASE_8B_MODEL:-Qwen3-8B-Instruct}"
BASE_8B_PATH="${BASE_8B_PATH:-Qwen/Qwen3-8B-Instruct}"
BASE_4B_URL="${BASE_4B_URL:-http://127.0.0.1:8002/v1}"
BASE_4B_MODEL="${BASE_4B_MODEL:-Qwen3-4B-Instruct}"
BASE_4B_PATH="${BASE_4B_PATH:-Qwen/Qwen3-4B-Instruct}"

[[ -s "${TEST_DATA}" ]] || { echo "Missing test data: ${TEST_DATA}" >&2; exit 1; }
mkdir -p "${ROOT}"

make_config() {
  local output="$1" url="$2" model="$3" model_path="$4"
  python - "${CONFIG_TEMPLATE}" "${output}" "${TEST_DATA}" "${url}" "${model}" "${model_path}" <<'PY'
import json, sys
src, dst, data, url, model, model_path = sys.argv[1:]
cfg = json.load(open(src, encoding="utf-8"))
base = dict(cfg.get("base_llm") or cfg.get("llm") or {})
base.update({
    "mode": "openai_compatible",
    "base_url": url,
    "model_name": model,
    "local_model_path": model_path,
    "temperature": 0.0,
    "max_tokens": 512,
    "enable_thinking": False,
    "seed": 42,
})
for key in ("top_p", "top_k", "min_p"):
    base.pop(key, None)
cfg["base_llm"] = base
cfg["llm"] = dict(base)
cfg["verifier"] = {"mode": "rule_based"}
cfg["actions"] = ["direct"]
cfg["data"]["data_path"] = data
cfg["data"]["eval_data_path"] = data
cfg["data"]["eval_limit"] = None
json.dump(cfg, open(dst, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
PY
}

run_case() {
  local name="$1" url="$2" model="$3" model_path="$4"
  local case_dir="${ROOT}/${name}"
  local config="${case_dir}/config.json"
  local trace="${case_dir}/direct.jsonl"
  mkdir -p "${case_dir}"
  make_config "${config}" "${url}" "${model}" "${model_path}"
  local limit_args=()
  local resume_args=(--resume)
  [[ -n "${LIMIT}" ]] && limit_args+=(--limit "${LIMIT}")
  [[ "${FORCE}" == "1" ]] && resume_args=()
  echo "[fair-direct] ${name}: explicit enable_thinking=false, temperature=0, max_tokens=512"
  python -m src.build_policy_dataset \
    --config "${config}" \
    --split test \
    "${limit_args[@]}" \
    "${resume_args[@]}" \
    --actions direct \
    --output "${trace}"
}

run_case qwen3_8b_no_think "${BASE_8B_URL}" "${BASE_8B_MODEL}" "${BASE_8B_PATH}"
run_case qwen3_4b_no_think "${BASE_4B_URL}" "${BASE_4B_MODEL}" "${BASE_4B_PATH}"

python -m src.audit_direct_generation \
  --left "${ROOT}/qwen3_8b_no_think/direct.jsonl" \
  --right "${ROOT}/qwen3_4b_no_think/direct.jsonl" \
  --left_name qwen3_8b_no_think \
  --right_name qwen3_4b_no_think \
  --max_tokens 512 \
  --json_output "${ROOT}/FAIR_DIRECT_AUDIT.json" \
  --markdown_output "${ROOT}/FAIR_DIRECT_AUDIT.md"

echo "[fair-direct] complete: ${ROOT}/FAIR_DIRECT_AUDIT.md"
