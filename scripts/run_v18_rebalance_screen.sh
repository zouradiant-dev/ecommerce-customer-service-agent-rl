#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
run="v18-sft-rebalance-screen-r2"
output="outputs/$run"
resume=()
if [[ -f "$output/protocol.json" ]]; then
  resume=(--resume)
elif [[ -d "$output" && -n "$(find "$output" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "STOP: non-empty output has no protocol.json: $output" >&2
  exit 2
fi

python -u scripts/evaluate_sft_v18.py \
  --manifest data/v18-train-dev-r1/tasks.jsonl \
  --output "$output" \
  --stage step12=outputs/v18-sft-rebalance-r2/checkpoint-12 \
  --stage step16=outputs/v18-sft-rebalance-r2/checkpoint-16 \
  --stage step20=outputs/v18-sft-rebalance-r2/checkpoint-20 \
  --stage step24=outputs/v18-sft-rebalance-r2/checkpoint-24 \
  --stage step28=outputs/v18-sft-rebalance-r2/checkpoint-28 \
  --backend local-hf \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --base-model-revision a09a35458c702b33eeacc393d103063234e8bc28 \
  --scenario contradictory_order \
  --scenario cross_order_item_mismatch \
  --scenario refund_quantity_exceeds \
  --max-tasks-per-scenario 5 \
  --trials-per-task 1 \
  --temperature 0 \
  --max-steps 20 \
  --max-tokens 512 \
  --max-input-tokens 8192 \
  --correction-retries 0 \
  "${resume[@]}"
