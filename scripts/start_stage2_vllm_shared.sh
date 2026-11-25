#!/usr/bin/env bash
set -euo pipefail

# Shared-GPU stage2 launcher for short-context math experiments.
# Base and Reflector share the free GPU 2; the smaller Verifier uses GPU 3.
# This keeps a useful context length while avoiding the two largest models
# competing with the existing jobs on GPUs 0, 1, and 3.
#   base      Qwen3-8B                   -> GPU 2, port 8000
#   reflector DeepSeek-R1-Distill-Qwen7B -> GPU 2, port 8001
#   verifier  Qwen3-4B                   -> GPU 3, port 8002
# Override any variable from the shell if needed.

export BASE_GPU="${BASE_GPU:-1}"
export REFLECTOR_GPU="${REFLECTOR_GPU:-1}"
export VERIFIER_GPU="${VERIFIER_GPU:-2}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
export BASE_GPU_MEMORY="${BASE_GPU_MEMORY:-0.50}"
export REFLECTOR_GPU_MEMORY="${REFLECTOR_GPU_MEMORY:-0.42}"
export VERIFIER_GPU_MEMORY="${VERIFIER_GPU_MEMORY:-0.25}"

exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/start_stage2_vllm.sh" "$@"
