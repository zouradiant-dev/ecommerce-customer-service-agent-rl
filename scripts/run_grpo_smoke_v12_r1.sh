#!/usr/bin/env bash
set -euo pipefail
task_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$task_repo"
task_python=/home/featurize/grpo-env-v12/bin/python
task_adapter="$task_repo/outputs/sft/qwen25-7b-seed-7/checkpoint-150"
task_evidence=/home/featurize/work/train-readiness-v12-review-20261001-r1/pilot
export USE_TF=0 USE_TORCH=1 PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0
export HF_HOME=/home/featurize/hf-cache HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
task_args=(scripts/train_grpo_smoke_v12_r1.py
  --manifest data/qwen15b-v2/tasks.jsonl --adapter "$task_adapter"
  --readiness-dir "$task_evidence" --smoke-only --max-steps 10
  --batch-size 1 --gradient-accumulation 8 --num-generations 4
  --learning-rate 5e-6 --temperature 0.8 --beta 0.04
  --max-completion-length 2048 --max-tool-iterations 12 --save-steps 10 --seed 7)

if [[ "${1:-}" == "--worker" ]]; then
  task_run_dir="${2:?missing unique run directory}"
  exec 9>"$task_repo/outputs/grpo-v12-training.lock"
  flock -n 9 || { printf 'STOP: another GRPO launcher holds the lock.\n'; exit 2; }
  task_gpu_pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
  if [[ -n "${task_gpu_pids//[[:space:]]/}" ]]; then
    printf 'STOP: GPU already has compute processes: %s\n' "$task_gpu_pids"
    exit 2
  fi
  printf 'GRPO_SMOKE_OUTPUT=%s/training\n' "$task_run_dir"
  set +e
  "$task_python" -u "${task_args[@]}" --output "$task_run_dir/training"
  task_status=$?
  printf '\nGRPO_SMOKE_EXIT_CODE=%s\n' "$task_status"
  exit "$task_status"
fi

[[ -x "$task_python" && -d "$task_adapter" && -d "$task_evidence" ]] || {
  printf 'STOP: missing checked GRPO environment, SFT adapter, or review evidence.\n'; exit 2;
}
# Read-only CPU validation first. This output target is never created.
"$task_python" "${task_args[@]}" --output "$task_repo/outputs/grpo-smoke-preflight-unused" --preflight-only
task_gpu_pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
if [[ -n "${task_gpu_pids//[[:space:]]/}" ]]; then
  printf 'STOP: GPU already has compute processes: %s\n' "$task_gpu_pids"
  exit 2
fi
task_run_dir="$(mktemp -d "$task_repo/outputs/grpo-smoke-v12-r1.XXXXXX")"
nohup bash "${BASH_SOURCE[0]}" --worker "$task_run_dir" >"$task_run_dir/run.log" 2>&1 < /dev/null &
task_pid=$!
printf 'GRPO_SMOKE_PID=%s\n日志：%s/run.log\n结果：%s/training\n' "$task_pid" "$task_run_dir" "$task_run_dir"
printf '现在只跟踪日志；Ctrl+C仅退出日志查看，不停止后台训练。\n'
tail --pid="$task_pid" -n 30 -f "$task_run_dir/run.log"
