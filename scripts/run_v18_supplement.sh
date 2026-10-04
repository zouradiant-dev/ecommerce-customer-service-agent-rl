#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

phase="${1:-}"
if [[ "$phase" != "smoke" && "$phase" != "full" ]]; then
  echo "Usage: bash scripts/run_v18_supplement.sh smoke|full" >&2
  exit 2
fi

run_name="v18-sft-supplement-${phase}-r1"
output="outputs/$run_name"
gpu_log="$run_name-gpu-memory.csv"

if [[ -f "$output/metrics.json" ]] && python -c \
  "import json; m=json.load(open('$output/metrics.json', encoding='utf-8')); raise SystemExit(0 if m.get('optimizer_steps') else 1)"
then
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
    echo "STOP: output is non-empty but has no checkpoint: $output" >&2
    exit 2
  fi
fi

python scripts/train_sft_v18_supplement.py preflight --data-only \
  --output "outputs/v18-sft-supplement-preflight-r1" > /dev/null

nvidia-smi \
  --query-gpu=timestamp,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader,nounits \
  -l 2 > "$gpu_log" 2>&1 &
monitor_pid=$!
cleanup() {
  kill "$monitor_pid" 2>/dev/null || true
  wait "$monitor_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

python -u scripts/train_sft_v18_supplement.py "$phase" \
  --output "$output" "${resume[@]}"
