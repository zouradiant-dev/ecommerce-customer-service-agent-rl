#!/usr/bin/env bash
set -euo pipefail
task_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$task_repo"
task_python="${GRPO_V13_PYTHON:-/home/featurize/grpo-env-v12/bin/python}"
export USE_TF=0 USE_TORCH=1 PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0
export HF_HOME=/home/featurize/hf-cache HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false

if [[ "${1:-}" == "--worker" ]]; then
  trap 'task_worker_status=$?; printf "V13_WORKER_EXIT_CODE=%s\n" "$task_worker_status"' EXIT
  task_run_dir="${2:?missing run directory}"
  task_evidence="${3:?missing evidence}"
  task_review="${4:?missing human review}"
else
  task_evidence="${1:?usage: bash scripts/run_grpo_experiment_v13.sh EVIDENCE_DIR HUMAN_REVIEW_JSON}"
  task_review="${2:?missing HUMAN_REVIEW_JSON; unsigned agent audit cannot start training}"
fi
task_args=(scripts/train_grpo_experiment_v13.py --manifest data/qwen15b-v2/tasks.jsonl
  --adapter "$task_repo/outputs/sft/qwen25-7b-seed-7/checkpoint-150"
  --evidence "$task_evidence" --review "$task_review" --experimental-only --max-hours 3)

check_gpu() {
  task_gpu_pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
  [[ -z "${task_gpu_pids//[[:space:]]/}" ]] || { printf 'STOP: GPU has compute processes: %s\n' "$task_gpu_pids"; return 2; }
  task_revision="$(cat /home/featurize/hf-cache/hub/models--Qwen--Qwen2.5-7B-Instruct/refs/main)"
  [[ "$task_revision" == a09a35458c702b33eeacc393d103063234e8bc28 ]] || { printf 'STOP: base revision differs.\n'; return 2; }
}

if [[ "${1:-}" == "--worker" ]]; then
  # Share the lock used by the old launcher; concurrent jobs cannot overlap.
  exec 9>"$task_repo/outputs/grpo-v12-training.lock"
  flock -n 9 || { printf 'STOP: another GRPO launcher holds the lock.\n'; exit 2; }
  check_gpu
  printf 'V13_RUN_DIR=%s\n' "$task_run_dir"
  "$task_python" -u "${task_args[@]}" --phase qualification --output "$task_run_dir/qualification"
  # The second preflight CPU-replays all 80 online outcomes and checks saved
  # tensors, source hashes, review, and memory. Any failure stops the chain.
  "$task_python" -u "${task_args[@]}" --phase experiment --qualification "$task_run_dir/qualification" \
    --output "$task_run_dir/experiment" --preflight-only
  printf 'V13_QUALIFICATION_PASSED; starting separate 100-step experiment from original SFT.\n'
  check_gpu
  "$task_python" -u "${task_args[@]}" --phase experiment --qualification "$task_run_dir/qualification" \
    --output "$task_run_dir/experiment"
  printf 'V13_TRAINING_DONE; evaluation is a separate manual command, not automatically billed.\n'
  exit 0
fi

[[ -x "$task_python" && -d "$task_evidence" && -f "$task_review" ]] || { printf 'STOP: missing environment/evidence/review.\n'; exit 2; }
"$task_python" "${task_args[@]}" --phase qualification --output "$task_repo/outputs/v13-preflight-unused" --runtime-check
check_gpu
mkdir -p "$task_repo/outputs"
task_run_dir="$(mktemp -d "$task_repo/outputs/grpo-experiment-v13-20261002.XXXXXX")"
# Prepare log before launching so tail cannot race a nonexistent log file.
touch "$task_run_dir/run.log"
nohup bash "${BASH_SOURCE[0]}" --worker "$task_run_dir" "$task_evidence" "$task_review" \
  >"$task_run_dir/run.log" 2>&1 < /dev/null &
task_pid=$!
printf 'PID=%s\n日志：%s/run.log\n结果：%s\n' "$task_pid" "$task_run_dir" "$task_run_dir"
printf 'Ctrl+C only exits log viewing. No automatic failure retry or checkpoint overwrite.\n'
tail --pid="$task_pid" -n 30 -f "$task_run_dir/run.log"
