#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
output="outputs/v18-sft-rebalance-r2"
gpu_log="v18-sft-rebalance-r2-gpu-memory.csv"

if [[ -f "$output/metrics.json" ]]; then
  echo "Already complete: $output/metrics.json"
  exit 0
fi
resume=()
if [[ -d "$output" ]]; then
  latest="$(find "$output" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' -print | sort -V | tail -n 1)"
  if [[ -n "$latest" ]]; then
    resume=(--resume-from "$latest")
    echo "Resuming from $latest"
  elif [[ -n "$(find "$output" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "STOP: non-empty output has no checkpoint: $output" >&2
    exit 2
  fi
fi

python scripts/train_sft_v18_rebalance.py preflight --data-only \
  --output outputs/v18-sft-rebalance-preflight-r2 > /dev/null
nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader,nounits -l 2 > "$gpu_log" 2>&1 &
monitor_pid=$!
cleanup() {
  kill "$monitor_pid" 2>/dev/null || true
  wait "$monitor_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM
python -u scripts/train_sft_v18_rebalance.py train --output "$output" "${resume[@]}"
