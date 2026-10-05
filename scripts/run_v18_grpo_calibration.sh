#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
run="v18-grpo-calibration-r1"
output="outputs/$run"
log="$run.log"

[[ ! -e "$output" ]] || { printf 'STOP: output already exists: %s\n' "$output"; exit 2; }
[[ -f outputs/v18-sft-full-newgpu-r1/adapter_model.safetensors ]] || {
  printf 'STOP: frozen v18 SFT adapter is missing.\n'; exit 2;
}
[[ -f data/v18-train-dev-r1/tasks/v18_train_cross_order_item_mismatch_0000/task.json ]] || {
  printf 'STOP: frozen v18 train task payloads are missing; upload v18-grpo-train-data-r1.tar.gz first.\n'; exit 2;
}
gpu_pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
[[ -z "${gpu_pids//[[:space:]]/}" ]] || {
  printf 'STOP: GPU has compute processes: %s\n' "$gpu_pids"; exit 2;
}

export HF_HOME="${HF_HOME:-/home/featurize/hf-cache}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 USE_TF=0 USE_TORCH=1
export TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1

python -u scripts/sample_grpo_v18_calibration.py \
  --manifest data/v18-train-dev-r1/tasks.jsonl \
  --adapter outputs/v18-sft-full-newgpu-r1 \
  --output "$output.preflight-unused" \
  --validate-only

nohup python -u scripts/sample_grpo_v18_calibration.py \
  --manifest data/v18-train-dev-r1/tasks.jsonl \
  --adapter outputs/v18-sft-full-newgpu-r1 \
  --output "$output" \
  >"$log" 2>&1 < /dev/null &
pid=$!
touch "$log"
printf 'PID=%s\n日志：%s\n结果：%s\n' "$pid" "$log" "$output"
printf '完成后先人工填写 %s/review-pending.jsonl，再生成校准收据。\n' "$output"
tail --pid="$pid" -n 30 -F "$log"
