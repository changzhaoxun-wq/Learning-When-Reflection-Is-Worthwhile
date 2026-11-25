#!/usr/bin/env bash
set -euo pipefail

# Dedicated-GPU stage2 launcher for long-context QA experiments.
# Layout by default:
#   base      Qwen3-8B                  -> GPU 1, port 8000
#   reflector DeepSeek-R1-Distill-Qwen7B -> GPU 0, port 8001
#   verifier  Qwen3-4B                  -> optional GPU 3, port 8002
# QA uses a rule-based verifier, so START_VERIFIER defaults to 0. Set it to 1
# only for an LLM-verifier ablation. Override any variable from the shell.

export BASE_GPU="${BASE_GPU:-1}"
export REFLECTOR_GPU="${REFLECTOR_GPU:-0}"
export VERIFIER_GPU="${VERIFIER_GPU:-3}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
export BASE_GPU_MEMORY="${BASE_GPU_MEMORY:-0.65}"
export REFLECTOR_GPU_MEMORY="${REFLECTOR_GPU_MEMORY:-0.65}"
export VERIFIER_GPU_MEMORY="${VERIFIER_GPU_MEMORY:-0.45}"
export START_VERIFIER="${START_VERIFIER:-0}"

exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/start_stage2_vllm.sh" "$@"
