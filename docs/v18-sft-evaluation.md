# v18 SFT 成品评测

本评测只读取冻结的 v18 dev，不创建或读取 test-v2。它可直接使用本地
Transformers/PEFT 4-bit 推理，也可连接 OpenAI-compatible 模型服务。每条 rollout
使用独立 SQLite 副本，并由
`deterministic_structured_v18` 同时计算严格奖励和训练用分解奖励。

## 本地协议预检

预检不连接模型服务，也不创建输出目录：

```powershell
python scripts\evaluate_sft_v18.py `
  --manifest data\v18-train-dev-r1\tasks.jsonl `
  --output outputs\v18-sft-dev-smoke `
  --stage final=sft-final `
  --max-tasks-per-scenario 1 `
  --validate-only
```

预期计划为600个 dev 任务、30个场景、smoke 选择30题、test行数为0。

## 30场景 GPU smoke

请使用 `v18-sft-evaluator-r2.tar.gz` 或更新版本。完整评测包包含原始任务 manifest
以及 600 道 dev 对应的 600 个 `task.json` 和 600 个 `state.sqlite`；仅包含训练
JSONL 的 SFT 执行包无法运行工具环境评测。600 个 dev 快照均已冻结，不重新生成。
启动脚本会先验证所有 dev 文件及其 manifest 哈希，通过后才启动后台推理。

训练镜像中推荐直接加载最终 LoRA，无需安装 vLLM。`--stage` 右侧是 adapter 目录：

补丁包解压到项目根目录后，smoke 可以直接在后台启动：

```bash
bash scripts/run_v18_sft_eval_smoke.sh
tail -f v18-sft-dev-smoke-local-r1.log
```

脚本会自动识别已有的合规输出并续跑。它实际执行的完整命令如下：

```bash
python -u scripts/evaluate_sft_v18.py \
  --manifest data/v18-train-dev-r1/tasks.jsonl \
  --output outputs/v18-sft-dev-smoke-r1 \
  --stage final=outputs/v18-sft-full-newgpu-r1 \
  --backend local-hf \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --base-model-revision a09a35458c702b33eeacc393d103063234e8bc28 \
  --max-tasks-per-scenario 1 \
  --trials-per-task 1 \
  --temperature 0 \
  --correction-retries 0
```

默认仅使用本机 Hugging Face 缓存；若镜像中没有该模型快照，显式增加
`--allow-download`。SSH 中断后使用完全相同的参数并追加 `--resume`，脚本会核对
协议和源码哈希，只补跑尚未成功落盘的 task/trial。基础设施错误会保存到
`last_worker_error.json` 并立即停止，修复后续跑即可。

也可以先启动能提供 OpenAI Chat Completions 和原生 tool calls 的服务：

把最终 LoRA 注册为模型别名 `sft-final` 后运行：

```bash
python -u scripts/evaluate_sft_v18.py \
  --manifest data/v18-train-dev-r1/tasks.jsonl \
  --output outputs/v18-sft-dev-smoke-r1 \
  --stage final=sft-final \
  --api-base http://127.0.0.1:8000/v1 \
  --trials-per-task 1 \
  --max-tasks-per-scenario 1 \
  --temperature 0 \
  --correction-retries 0
```

正式评测必须保持 `--correction-retries 0`，避免用额外系统纠偏抬高模型成绩。
输出目录必须为空。服务或 SSH 中断后使用完全相同的参数并追加 `--resume`；脚本会
核对协议和源码哈希，只补跑缺失的 task/trial。

smoke 至少满足：

- `complete=true`；
- `worker_errors=0`；
- `max_step_exhaustions=0`；
- `tool_format_stable_rate >= 0.98`；
- `hard_safety_failure_rollouts=0`；
- 三个关键写工具均有成功覆盖。

## checkpoint 平台期比较

同一个模型服务注册 `checkpoint-700`、`checkpoint-800`、`checkpoint-900` 和最终
adapter 四个别名后，以完全相同的任务、采样和工具协议运行：

```bash
python -u scripts/evaluate_sft_v18.py \
  --manifest data/v18-train-dev-r1/tasks.jsonl \
  --output outputs/v18-sft-dev-checkpoints-r1 \
  --stage ckpt700=sft-ckpt700 \
  --stage ckpt800=sft-ckpt800 \
  --stage ckpt900=sft-ckpt900 \
  --stage final=sft-final \
  --api-base http://127.0.0.1:8000/v1 \
  --trials-per-task 1 \
  --temperature 0 \
  --correction-retries 0
```

这轮用于选择 checkpoint 和检查最近三个点的严格通过率波动是否不超过0.02。
不能根据未来 test-v2 调整 checkpoint、提示、Verifier 或阈值。

## 多次采样与奖励校准材料

checkpoint 冻结后，再用同一 final adapter 对 dev 每题采样4次：

```bash
python -u scripts/evaluate_sft_v18.py \
  --manifest data/v18-train-dev-r1/tasks.jsonl \
  --output outputs/v18-sft-dev-final-sampling-r1 \
  --stage final=sft-final \
  --api-base http://127.0.0.1:8000/v1 \
  --trials-per-task 4 \
  --temperature 0.8 \
  --correction-retries 0
```

`mixed_group_rate` 至少达到0.20，才说明同题组有足够相对奖励信号。之后需要从正、
负奖励和安全关键场景中分层抽样，添加人工 `human_review`，再运行
`scripts/audit_v18_reward_calibration.py`。原始 rollout 永远保留，不要就地覆盖。

## 输出

每个 stage 包含：

- `rollouts.jsonl`：逐题消息、工具执行、终态、严格验证和奖励分解；
- `summary.json`：总体与逐场景指标；
- 顶层 `protocol.json`：manifest、参数、版本和源码哈希；
- 顶层 `summary.json`：多 stage 配对胜负与通过率差值。

`strict_pass_rate` 是逐 rollout 严格通过率；`pass_at_k` 是每个任务在 k 次采样中
至少成功一次的任务比例。两者不能互换。
