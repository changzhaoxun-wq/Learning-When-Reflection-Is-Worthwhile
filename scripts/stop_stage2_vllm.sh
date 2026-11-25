#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"

pid_for_port() {
  local port="$1"
  local line
  line="$(ss -ltnp 2>/dev/null | grep ":${port} " | head -1 || true)"
  if [[ "${line}" =~ pid=([0-9]+) ]]; then
    echo "${BASH_REMATCH[1]}"
  fi
}

port_is_used() {
  local port="$1"
  ss -ltn 2>/dev/null | grep -q ":${port} "
}

stop_pid() {
  local name="$1"
  local pid="$2"
  local port="$3"

  if ! ps -p "${pid}" > /dev/null 2>&1; then
    echo "Process ${pid} for ${name} is not running"
    return 0
  fi

  local pgid
  pgid="$(ps -o pgid= -p "${pid}" | tr -d ' ')"
  echo "Stopping ${name}: pid=${pid} pgid=${pgid} port=${port}"
  if [ -n "${pgid}" ]; then
    kill -TERM "-${pgid}" 2>/dev/null || true
  else
    kill -TERM "${pid}" 2>/dev/null || true
  fi

  for _ in $(seq 1 10); do
    if ! ps -p "${pid}" > /dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done

  if ps -p "${pid}" > /dev/null 2>&1; then
    echo "${name} did not exit after TERM; sending KILL"
    if [ -n "${pgid}" ]; then
      kill -KILL "-${pgid}" 2>/dev/null || true
    else
      kill -KILL "${pid}" 2>/dev/null || true
    fi
  fi
}

stop_one() {
  local name="$1"
  local port="$2"
  shift 2
  local pid_files=("$@")
  local found_pid_file=0

  for pid_file in "${pid_files[@]}"; do
    if [ ! -f "${pid_file}" ]; then
      continue
    fi
    found_pid_file=1
    local pid
    pid="$(cat "${pid_file}")"
    stop_pid "${name}" "${pid}" "${port}"
    rm -f "${pid_file}"
  done

  if [ "${found_pid_file}" = "0" ]; then
    echo "PID file not found for ${name}: ${pid_files[*]}"
  fi

  if port_is_used "${port}"; then
    local port_pid
    port_pid="$(pid_for_port "${port}")"
    if [ -n "${port_pid}" ]; then
      echo "Port ${port} is still in use; stopping pid discovered from port ${port}: ${port_pid}"
      stop_pid "${name}_port_${port}" "${port_pid}" "${port}"
    fi
  fi

  if port_is_used "${port}"; then
    echo "WARNING: port ${port} is still in use after stopping ${name}:"
    ss -ltnp | grep ":${port}" || true
  else
    echo "${name} stopped; port ${port} is free"
  fi
}

stop_one base_qwen3_8b 8000   outputs/vllm_base_qwen3_8b.pid   outputs/vllm_base_qwen3_8b_math.pid
stop_one reflector_deepseek_7b 8001 outputs/vllm_reflector_deepseek_7b.pid outputs/vllm_reflector_deepseek_7b_lora.pid
stop_one verifier_qwen3_4b 8002 outputs/vllm_verifier_qwen3_4b.pid
