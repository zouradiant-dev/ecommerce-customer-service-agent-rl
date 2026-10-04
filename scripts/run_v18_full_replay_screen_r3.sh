#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
run="v18-sft-full-replay-screen-r3"
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
  --stage baseline=outputs/v18-sft-full-newgpu-r1 \
  --stage step40=outputs/v18-sft-full-replay-r3/checkpoint-40 \
  --stage step60=outputs/v18-sft-full-replay-r3/checkpoint-60 \
  --stage step80=outputs/v18-sft-full-replay-r3/checkpoint-80 \
  --stage step100=outputs/v18-sft-full-replay-r3/checkpoint-100 \
  --stage step120=outputs/v18-sft-full-replay-r3/checkpoint-120 \
  --backend local-hf \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --base-model-revision a09a35458c702b33eeacc393d103063234e8bc28 \
  --scenario contradictory_order \
  --scenario cross_order_item_mismatch \
  --scenario refund_quantity_exceeds \
  --scenario cancel_after_shipment \
  --scenario cancel_already_cancelled \
  --scenario cancel_unshipped \
  --scenario address_change_after_shipment \
  --max-tasks-per-scenario 5 \
  --trials-per-task 1 \
  --temperature 0 \
  --max-steps 20 \
  --max-tokens 512 \
  --max-input-tokens 8192 \
  --correction-retries 0 \
  "${resume[@]}"
