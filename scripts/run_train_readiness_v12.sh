#!/usr/bin/env bash
set -euo pipefail
task_repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$task_repo_root"
task_python="$task_repo_root/.venv-training/bin/python"
task_adapter="$task_repo_root/outputs/sft/qwen25-7b-seed-7/checkpoint-150"
task_archive="$task_repo_root/outputs/train-readiness-v12-result-20261001-r1.tar.gz"
export USE_TF=0 USE_TORCH=1 HF_HOME=/home/featurize/hf-cache HF_HUB_OFFLINE=1
task_args=(scripts/sample_train_readiness.py --manifest data/qwen15b-v2/tasks.jsonl
  --adapter "$task_adapter" --tasks-per-template 1 --trials-per-task 4
  --temperature 0.4 --max-steps 16 --max-new-tokens 512 --seed 7)
if [[ "${1:-}" == "--worker" ]]; then
  task_run_dir="$2"
  set +e
  "$task_python" -u "${task_args[@]}" --output "$task_run_dir/pilot"
  task_status=$?
  printf '\nTRAIN_PILOT_EXIT_CODE=%s\n' "$task_status"
  # Preserve partial failures as well; never overwrite an earlier result archive.
  if [[ ! -e "$task_archive" ]]; then
    if tar -czf "$task_archive" -C "$task_run_dir" .; then
      printf '结果归档：%s\n' "$task_archive"
    else
      printf '归档失败；原始结果仍保存在：%s\n' "$task_run_dir"
      exit 3
    fi
  fi
  exit "$task_status"
fi
[[ -x "$task_python" ]] || { printf '找不到原 SFT 环境：%s\n' "$task_python"; exit 2; }
[[ ! -e "$task_archive" ]] || { printf '已有结果，保留不覆盖：%s\n' "$task_archive"; exit 2; }
mkdir -p outputs
exec 9>outputs/train-readiness-v12.lock
flock -n 9 || { printf '已有本轮采样在运行，不重复启动。\n'; exit 2; }
task_gpu_pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
[[ -z "$task_gpu_pids" ]] || { printf 'GPU 已有进程，未启动新任务：%s\n' "$task_gpu_pids"; exit 2; }
task_run_dir="$(mktemp -d "$task_repo_root/outputs/train-readiness-v12.XXXXXX")"
"$task_python" -u "${task_args[@]}" --output "$task_run_dir/pilot" --preflight-only >"$task_run_dir/preflight.json"
nohup bash "$task_repo_root/scripts/run_train_readiness_v12.sh" --worker "$task_run_dir" \
  >"$task_run_dir/run.log" 2>&1 < /dev/null &
printf '进程 PID：%s\n本轮目录：%s\n' "$!" "$task_run_dir"
printf '只跑 SFT-150 的 train 8题×4次，不训练、不跑 Base、不调用 API。\n'
printf '完成后归档：%s\n' "$task_archive"
printf '下面开始查看日志；Ctrl+C 只退出日志查看，后台采样继续。\n'
tail --pid="$!" -f "$task_run_dir/run.log"
