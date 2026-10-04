#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

output="outputs/v18-sft-dev-smoke-local-r1"
log="v18-sft-dev-smoke-local-r1.log"
pid_file="v18-sft-dev-smoke-local-r1.pid"

if [[ -f "$output/summary.json" ]] && python -c \
  "import json; raise SystemExit(0 if json.load(open('$output/summary.json', encoding='utf-8')).get('complete') else 1)"
then
  echo "Evaluation is already complete: $output/summary.json"
  exit 0
fi

resume=()
if [[ -f "$output/protocol.json" ]]; then
  resume=(--resume)
elif [[ -d "$output" ]] && [[ -n "$(find "$output" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "STOP: non-empty output has no protocol.json: $output" >&2
  exit 2
fi

if [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
  echo "Evaluation is already running with PID $(cat "$pid_file")"
  exit 0
fi

arguments=(
  --manifest data/v18-train-dev-r1/tasks.jsonl \
  --output "$output" \
  --stage final=outputs/v18-sft-full-newgpu-r1 \
  --backend local-hf \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --base-model-revision a09a35458c702b33eeacc393d103063234e8bc28 \
  --max-tasks-per-scenario 1 \
  --trials-per-task 1 \
  --temperature 0 \
  --max-steps 20 \
  --max-tokens 512 \
  --max-input-tokens 8192 \
  --correction-retries 0
)

# Validate every frozen dev task and SQLite before starting a billed GPU run.
python scripts/evaluate_sft_v18.py "${arguments[@]}" --validate-only > /dev/null
echo "Preflight passed: frozen dev task files and SQLite hashes verified."

nohup python -u scripts/evaluate_sft_v18.py "${arguments[@]}" "${resume[@]}" \
  > "$log" 2>&1 < /dev/null &

pid=$!
echo "$pid" > "$pid_file"
echo "Started PID $pid"
echo "Log: $root/$log"
echo "Result: $root/$output/summary.json"
echo "Monitor: tail -f $log"
