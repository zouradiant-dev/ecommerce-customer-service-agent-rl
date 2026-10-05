#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
grpo_python="${GRPO_V18_PYTHON:-/home/featurize/grpo-env-v12/bin/python}"
export HF_HOME="${HF_HOME:-/home/featurize/hf-cache}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 USE_TF=0 USE_TORCH=1
export TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1

phase="${1:-}"
receipt="${2:-}"
rollouts="${3:-}"
protocol="${4:-}"
qualification="${5:-}"
if [[ "$phase" != qualification && "$phase" != experiment ]]; then
  printf 'usage: bash scripts/run_grpo_v18.sh qualification RECEIPT REVIEWED_ROLLOUTS PROTOCOL\n' >&2
  printf '   or: bash scripts/run_grpo_v18.sh experiment RECEIPT REVIEWED_ROLLOUTS PROTOCOL QUALIFICATION_DIR\n' >&2
  exit 2
fi
[[ -x "$grpo_python" && -f "$receipt" && -f "$rollouts" && -f "$protocol" ]] || {
  printf 'STOP: GRPO Python or calibration evidence is missing.\n' >&2; exit 2;
}
if [[ "$phase" == experiment && ! -f "$qualification/complete.json" ]]; then
  printf 'STOP: experiment requires a completed qualification directory.\n' >&2; exit 2
fi

check_gpu() {
  gpu_pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
  [[ -z "${gpu_pids//[[:space:]]/}" ]] || {
    printf 'STOP: GPU has compute processes: %s\n' "$gpu_pids"; return 2;
  }
  revision="$(cat "$HF_HOME/hub/models--Qwen--Qwen2.5-7B-Instruct/refs/main")"
  [[ "$revision" == a09a35458c702b33eeacc393d103063234e8bc28 ]] || {
    printf 'STOP: cached Qwen base revision differs.\n'; return 2;
  }
}

common=(scripts/train_grpo_v18.py
  --manifest data/v18-train-dev-r1/tasks.jsonl
  --adapter outputs/v18-sft-full-newgpu-r1
  --calibration-receipt "$receipt"
  --calibration-rollouts "$rollouts"
  --calibration-protocol "$protocol"
  --experimental-only --max-hours 3)
phase_args=(--phase "$phase")
if [[ "$phase" == experiment ]]; then
  phase_args+=(--qualification "$qualification")
fi

if [[ "${6:-}" == --worker ]]; then
  run_dir="${7:?missing run directory}"
  trap 'status=$?; printf "V18_GRPO_WORKER_EXIT_CODE=%s\n" "$status"' EXIT
  exec 9>"$root/outputs/grpo-v18-training.lock"
  flock -n 9 || { printf 'STOP: another GRPO launcher holds the lock.\n'; exit 2; }
  check_gpu
  "$grpo_python" -u "${common[@]}" "${phase_args[@]}" --output "$run_dir"
  exit 0
fi

preflight_output="$root/outputs/v18-grpo-preflight-unused"
"$grpo_python" -u "${common[@]}" "${phase_args[@]}" \
  --output "$preflight_output" --runtime-check
check_gpu
mkdir -p "$root/outputs"
run_root="$(mktemp -d "$root/outputs/grpo-v18-${phase}-20261005.XXXXXX")"
run_dir="$run_root/training"
log="$run_root/run.log"
touch "$log"
nohup bash "${BASH_SOURCE[0]}" "$phase" "$receipt" "$rollouts" "$protocol" "$qualification" \
  --worker "$run_dir" >"$log" 2>&1 < /dev/null &
pid=$!
printf 'PID=%s\n日志：%s\n结果：%s\n' "$pid" "$log" "$run_dir"
printf 'Ctrl+C 只退出日志查看；失败不会自动重试。\n'
tail --pid="$pid" -n 30 -f "$log"
