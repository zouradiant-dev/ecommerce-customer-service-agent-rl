# GPU 服务器实验手册

本手册把一次可信实验拆成 Base → SFT → GRPO 三段。单卡机器采用“推理服务和训练交替运行”；不要让 vLLM 与 QLoRA 同时占满同一张卡。

## 1. 环境验收

先安装与服务器 CUDA 匹配的 PyTorch，再安装项目和训练依赖：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m pip install -r requirements-training.txt
python scripts/check_training_env.py --output reports/training-env.json
```

检查报告必须至少满足：CUDA 可用、`bitsandbytes` 可导入、GPU 显存被正确识别、`pip check` 无冲突。训练前再执行：

```bash
python -m unittest discover -s tests -v
python scripts/train_sft.py \
  --train data/sft/train.jsonl \
  --dev data/sft/dev.jsonl \
  --output outputs/sft-preflight \
  --validate-data-only
```

## 2. 构造固定任务集

正式实验不要在不同阶段重新生成 test：

```bash
ecommerce-cs build-tasks --output data/v1 --replicas 20 --seed 7
ecommerce-cs build-pilot \
  --manifest data/v1/tasks.jsonl \
  --output data/v1/pilot-train-dev.jsonl \
  --splits train,dev
```

把 `data/v1/summary.json`、manifest SHA256、代码版本和服务器镜像版本一起保存。
`pilot-train-dev.jsonl` 每个 train/dev 模板只选一个任务，共 22 个任务；先用它检查教师模型、API 费用和 Verifier，再跑完整 440 条 train/dev。不要把 test 加入教师轨迹收集。

## 3. Base 与教师轨迹

启动 OpenAI-compatible 服务后，对基础模型跑固定协议：

```bash
ecommerce-cs run-model \
  --manifest data/v1/tasks.jsonl \
  --split test \
  --model Qwen/Qwen2.5-7B-Instruct \
  --output outputs/base/test.jsonl \
  --trials-per-task 4 \
  --temperature 0.8
```

教师模型只对 train/dev 收集轨迹。验证、去重后再生成 SFT：

```bash
ecommerce-cs run-model \
  --manifest data/v1/pilot-train-dev.jsonl \
  --model <teacher-model> \
  --output outputs/teacher/pilot.jsonl \
  --trials-per-task 1

ecommerce-cs curate \
  --rollouts outputs/teacher/raw.jsonl \
  --output outputs/teacher/curated.jsonl \
  --rejected outputs/teacher/rejected.jsonl \
  --max-per-scenario 100

ecommerce-cs build-sft --rollouts outputs/teacher/curated.jsonl --output data/sft
```

Windows + 百炼可在已设置 `OPENAI_API_KEY` 和 `BAILIAN_API_BASE` 的同一个 PowerShell 中直接运行：

```powershell
.\scripts\run_teacher_pilot.ps1
```

脚本默认使用 `qwen3.7-plus-2026-05-26`、关闭思考，并只读取 22 条 train/dev pilot；密钥不会写入项目文件。脚本会在请求前拒绝含 test 的 manifest，并在输出数量异常或出现 `worker_error` 时停止，不会把 API 故障当成一次成功的 pilot。

若 Verifier 的语义规则或任务约束后来被修正，不要覆盖教师模型的原始轨迹。使用带版本号的派生文件重新评分：

```powershell
.\.venv-api\Scripts\ecommerce-cs.exe rescore `
  --rollouts outputs\teacher-qwen37\pilot-one-per-template.jsonl `
  --manifest data\qwen15b-v2\pilot-train-dev.jsonl `
  --output outputs\teacher-qwen37\pilot-one-per-template-rescored-v6.jsonl

.\.venv-api\Scripts\ecommerce-cs.exe build-sft `
  --rollouts outputs\teacher-qwen37\pilot-one-per-template-rescored-v6.jsonl `
  --output outputs\teacher-qwen37\pilot-sft-rescored-v6 `
  --tool-scope all
```

`rescore` 会从保存的轨迹重算沟通、策略、工具恢复和必要工具检查；只有依赖一次性任务数据库的状态结果沿用原 rollout。输出中的 `verifier_version` 和 `rescore_metadata` 用于审计规则变化。

当前状态：`persistent_tool_error_escalation` 的终态验证已经达到 3/3。300 条 train 原始轨迹保存在 `outputs/teacher-qwen37/full-train-v4.jsonl`；Verifier v6 离线重评分后为 264/300（88%），派生文件为 `outputs/teacher-qwen37/full-train-rescored-v6.jsonl`。原始文件必须保留，不要覆盖。

剩余 36 条中，11 条是 API 网络超时，另外 25 条主要来自旧运行时没有识别“您确认要……”等确认问句。运行时和语义规则均已修正，精确重试清单位于 `data/qwen15b-v2/retry-full-train-v6.jsonl`。只重跑这 36 条：

```powershell
.\scripts\run_teacher_subset.ps1 `
  -Manifest data\qwen15b-v2\retry-full-train-v6.jsonl `
  -Output outputs\teacher-qwen37\retry-full-train-v6.jsonl `
  -ExpectedTasks 36 `
  -TrialsPerTask 1
```

第一轮精确重试结果为 26/36，且没有 `worker_error`。与原 264 条成功轨迹合并计算，当前可用成功覆盖为 290/300（96.67%）。剩余 10 条暴露了两类运行时约束问题：模型把确认前隐藏的写工具误认为系统不支持，以及用户已经确认时仍重复询问。工具状态机现已收紧为“身份失败或同一工具连续失败后才暴露人工升级”，纠偏也会要求先完成缺失的必要读取工具。

第二轮只运行剩余 10 条：

```powershell
.\scripts\run_teacher_subset.ps1 `
  -Manifest data\qwen15b-v2\retry-full-train-round2.jsonl `
  -Output outputs\teacher-qwen37\retry-full-train-round2.jsonl `
  -ExpectedTasks 10 `
  -TrialsPerTask 1
```

第二轮结果为 10/10，且没有 `worker_error`。两轮成功重试已经通过 `merge-retries` 按 task id 合并：共 300 条，其中第一轮替换 26 条、第二轮替换 10 条。统一使用 Verifier v8 重评分后的文件为 `outputs/teacher-qwen37/full-train-verified-v8.jsonl`，最终 300/300 通过，15 个 train 场景均为 100%，失败分类为空。原始及各轮重试文件均保持不变。

训练集使用与部署时工具可见性一致的逐 assistant turn 格式：

```powershell
.\.venv-api\Scripts\ecommerce-cs.exe build-sft `
  --rollouts outputs\teacher-qwen37\train-dev-verified-v8.jsonl `
  --output data\qwen15b-v2\sft-teacher-v8 `
  --tool-scope stateful_turn
```

完整 dev 清单含 7 个独立模板、140 条任务，位于 `data/qwen15b-v2/dev-full.jsonl`。原始收集结果为 95/140（67.86%），没有 worker、状态、策略或工具失败；45 条失败全部来自严格字符串规则遗漏了正确的同义表达。Verifier v8 补充回归用例后离线重评分为 140/140，无需付费重跑，派生文件为 `outputs/teacher-qwen37/full-dev-verified-v8.jsonl`。

最终 train/dev 轨迹共 440 条且全部通过。`stateful_turn` SFT 共有 1653 条 train、616 条 dev，合计 2269 条，格式校验错误为 0。train 与 dev 模板保持完全隔离。

## 4. QLoRA SFT

```bash
python scripts/train_sft.py \
  --train data/qwen15b-v2/sft-teacher-v8/train.jsonl \
  --dev data/qwen15b-v2/sft-teacher-v8/dev.jsonl \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --output outputs/sft/seed-7 \
  --max-steps 200 \
  --seed 7
```

脚本使用 4-bit NF4、LoRA r16/alpha32、assistant-only loss，并保存 `checkpoint-N`、优化器状态、dev loss 和峰值显存。中断后可恢复：

```bash
python scripts/train_sft.py \
  --train data/sft/train.jsonl \
  --dev data/sft/dev.jsonl \
  --output outputs/sft/seed-7 \
  --resume-from outputs/sft/seed-7/checkpoint-100 \
  --max-steps 200
```

## 5. 服务和统一评测

vLLM 官方支持在同一基础模型服务中注册多个 LoRA，并把 LoRA 名称当作请求中的 model。具体参数以服务器安装版本的 `vllm serve --help` 为准；典型启动方式是：

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --enable-lora \
  --max-lora-rank 16 \
  --lora-modules sft=outputs/sft/seed-7 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes
```

参考：[vLLM LoRA 官方文档](https://docs.vllm.ai/en/latest/features/lora/)。只在隔离实验机中启用运行时动态加载；官方文档明确提示动态 LoRA 接口存在安全风险。

同一服务暴露 Base、SFT 和 GRPO 后，可一次运行相同 test 协议：

```bash
python scripts/evaluate_stages.py \
  --manifest data/v1/tasks.jsonl \
  --output outputs/final-eval \
  --stage base=Qwen/Qwen2.5-7B-Instruct \
  --stage sft=sft \
  --stage grpo=grpo \
  --trials-per-task 4 \
  --temperature 0.8
```

## 6. 在线 GRPO

在线 GRPO 使用较新的 TRL 状态环境接口，建议与 legacy QLoRA SFT 依赖分开安装：

```bash
python -m venv .venv-grpo
source .venv-grpo/bin/activate
python -m pip install -e .
python -m pip install -r requirements-online-grpo.txt

python scripts/train_grpo.py \
  --manifest data/v1/tasks.jsonl \
  --model outputs/sft/seed-7-merged \
  --output outputs/grpo/strict-seed-7 \
  --reward-mode strict_binary \
  --num-generations 4 \
  --max-steps 200 \
  --seed 7
```

先加 `--preflight-only` 可在 CPU 环境检查任务路由。`train_grpo.py` 会拒绝 test；每个 completion 都创建独立环境和 SQLite 副本，模型在 episode 内调用真实工具，最终奖励由状态、沟通、政策和必要工具联合决定。需要脚本追加用户轮次的任务暂不进入通用 TRL 工具循环，preflight 会报告排除数量和 task id。

若要保留原静态 rollout 算法作为对照，使用 `scripts/train_offline_group_awr.py`。它是 offline group-AWR，不是 GRPO。严格奖励、安全塑形、group size 和 KL 消融按 `configs/algorithm-experiments.json` 执行，完成后用 `scripts/summarize_algorithm_experiments.py` 统一生成配对报告。

## 7. 最低报告标准

- Base、SFT、GRPO 使用同一个冻结 test manifest；
- 至少 3 个训练 seed；
- 报告 pass^1、pass@2、pass@4、配对 bootstrap 95% CI；
- 单独报告政策违规、虚假成功声明、确认缺失和人工升级率；
- 报告零方差任务组比例、平均工具步数、超时率和峰值显存；
- 脚本 reference 的 100% 只能叫环境自检，不能写成模型结果。
