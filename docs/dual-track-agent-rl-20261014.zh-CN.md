# Agent 工程与 RL 研究：并行准备，分别验证

两条主线共用 Qwen2.5-7B-Instruct 与原 v18 SFT，但分别冻结任务、工具协议和指标。代码与 CPU 准备可以并行推进；一张 4090 上的评测和采样顺序运行，避免推理服务与实验争抢显存。本阶段不训练、不晋升候选模型、不使用独立 test。

## 先纠正历史证据

10 月 5 日已经完成过原 v18 的奖励采样，而非“还没采过”：10 个训练场景各选排序最前的两题，每题 4 次，共 20 个任务、80 条真实模型轨迹。NF4 推理、temperature=0.8，无 policy controller。二值奖励与塑形奖励均为 1；20 个组的标准差都为 0。采样循环约 957 秒，不能当成含环境安装和加载的总耗时。

10 月 14 日校验保存的压缩包 SHA256、协议与轨迹文件哈希，并在冻结任务的临时数据库中重放所有工具调用，重新计算状态、奖励和验证结果：80/80 与历史记录一致。这个 CPU 重放没有重新调用大模型，也不是人工奖励审核。原记录 human_review_complete=false、reward_calibrated=false、formal_rl_gate_passed=false；不能用它直接启动当前 GRPO。

历史归档 SHA256：`3bf624cdc283d2d27baa2e056ff89bf5d8a1b051879a5ce3ba7c9bfda61a08fd`。本机审计在 `outputs/v18-historical-signal-audit-20261014/audit.json`。当前部分源码已与采样当时不同，报告保留差异，历史成绩不改写。

复核命令（全程 CPU，新输出目录）：

```powershell
python scripts/audit_v18_historical_signal.py --archive ../v18-grpo-calibration-zero-signal-r1-results.tar.gz --manifest data/v18-train-dev-r1/tasks.jsonl --output outputs/new-historical-audit
```

因此不重复采旧 20 题，也不通过惩罚正确行为制造奖励差异。新的固定种子选择只是扩展诊断覆盖，尚未证明题目更难；它们仍在原 SFT 训练分布内，不是未见测试题。

## 两条主线的冻结设计

| 项目 | Agent 工程线 | RL 研究线 |
|---|---|---|
| 模型 | 原 v18 与保存的 64-step 在线补训候选 | 仅原 v18，不使用在线补训候选 |
| 工具 | 取消预览等 4 工具；知识组另加 search_policies | 原离线 9 工具环境 |
| 任务 | 57 个业务 dev，另有 6 个知识 dev | 原 train 中 10 场景各 2 题，排除历史 20 题 |
| 推理 | BF16、greedy、每次最多 512 新 token | NF4、temperature=0.8、每题 4 次 |
| 核心问题 | 新守卫下的任务完成、边界失败、知识工具选择 | 二值/塑形奖励是否有有效组内差异、奖励是否正确 |
| 本阶段产物 | 两套分别配对的评测、Bad Case、调用追踪 | 80 条新采样、奖励分布、待人工复核轨迹 |
| 训练 | 不启动 SFT | 不启动 GRPO |

原 v18 权重 SHA256 为 `f41c98a583a6ca66e18f1d6f8743029cb3a363cf96f43d80dbf0d29b74876b6c`；候选为 `3a5dd3e29427d2bf5fc89e813ff16dbd30bdca6339ad2fd4169b6c2b6f024764`。本次不改变二者。

业务 57 题由既有 15 题、商品绑定 12 题和当前生成的 30 个 v4 dev 案例组成。它们已用于开发分析，共享合成场景，不能宣称独立测试。既有 57 题旧 compact 协议的 40→51 成绩也不能直接与新配置相减，必须在新协议下重新配对。

知识 6 题覆盖退款说明、发货取消说明、接管说明、两种无证据查询和重启后再次查询。检索来源仅是三份演示文档，逐轮检查来源、无证据拒答、会话状态与业务状态无写入。检查不能证明回答逐句蕴含、真实商家政策正确或知识泛化；语义正确性仍需人工复核。源码测试替身能通过，不意味着实际 v18 会选对新增工具。

## 本机准备与执行入口

准备器只读原权重与数据，记录源码、负载、权重、任务、数据库哈希；已存在输出不会覆盖。源码变更必须重新冻结。

```powershell
python scripts/prepare_dual_track_experiment.py --output outputs/new-dual-track-plan
python scripts/run_dual_track_checks.py --plan-dir outputs/new-dual-track-plan --track agent --output outputs/new-agent-preflight --validate-only
python scripts/run_dual_track_checks.py --plan-dir outputs/new-dual-track-plan --track rl --output outputs/new-rl-preflight --validate-only
python scripts/package_dual_track_checks.py --plan-dir outputs/new-dual-track-plan --output packages/new-dual-track-checks.tar.gz
```

默认路径指向本机已验证的原适配器、备份里的候选适配器、v4 dev 案例与历史审计。移到其他电脑可用对应参数覆盖路径。运行器默认只有 CPU 预检；`--execute` 才实际加载 GPU。没有自动训练命令，也不会关闭已有服务。GPU 0 有计算进程或剩余显存不足会停止；请自行停止自己的推理服务再执行，保留其启动命令以便恢复。

执行包包含冻结源码、两组 dev、演示知识索引、任务选择及冻结 train 数据中的所需场景数据库；保留原 tasks.jsonl 哈希，但不包含 dev/test 数据库、模型权重、密钥或网页会话数据库。其 JSON 收据列出全部文件哈希。通过包迁移可保持本机字节哈希，直接换一个 Git checkout 的行尾或修改代码会让旧协议校验失败。

## 下次 Featurize 的命令

本阶段没有租实例。需要 GPU 时上传生成的执行包，核对旁边 JSON 收据中的 bundle_sha256，再解压到一个新目录。以下以其解压目录为当前目录，`experiment/` 是冻结协议目录。镜像中的原权重和旧候选路径以实际文件为准；未保存候选时，从已下载的备份恢复，不能把原权重冒充候选。

```bash
task_python=/home/featurize/grpo-env-v12/bin/python
task_original=/home/featurize/work/v18-sft-48gb-r1-fp32-grad-r2/outputs/v18-sft-full-newgpu-r1
task_candidate=/home/featurize/work/ecommerce-repair.VyMdGD/outputs/online-repair-pilot-r1/pilot
task_base=/home/featurize/hf-cache/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28

"$task_python" scripts/run_dual_track_checks.py \
  --plan-dir experiment --track agent --output outputs/agent-dev-r1 \
  --original-adapter "$task_original" --candidate-adapter "$task_candidate" \
  --base-model-dir "$task_base" --execute

"$task_python" scripts/run_dual_track_checks.py \
  --plan-dir experiment --track rl --output outputs/rl-signal-r2 \
  --original-adapter "$task_original" \
  --manifest data/v18-train-dev-r1/tasks.jsonl \
  --hf-home /home/featurize/hf-cache --execute
```

Agent 顺序运行业务原模型、业务候选、知识原模型、知识候选，分别输出 comparison。RL 顺序采样并生成 reward-signal.json 和 review-pending.jsonl。某个阶段失败会保留日志与未完成收据，不覆盖历史目录；无自动重训或模型替换。缓存模型不完整时失败，不自动下载。

上述是执行方法，不是已经运行的结果。两线工具数、精度、任务定义不同，指标分别汇报；不能合成“项目成功率”。

## 如何根据结果继续

Agent：先区分输入绑定、工具选择、缺少新预览、知识缺证据、生成格式与接口故障。能由业务校验解决的先改工程，模型反复不选新工具或多轮澄清失败才补针对性 SFT。新源码与提示词已变化，旧 v4 数据的训练 manifest 不能直接当作匹配当前运行时的训练授权；需要重新生成并校验。候选晋升前还要做独立 test 和原九工具能力回归。

RL：若新采样仍全 1，先扩展未采样 train 覆盖并检查奖励盲区，不能仅调奖励系数启动 GRPO。若出现差异，人工同时复核正例、负例和关键写操作，校验奖励精确率/召回率、混合组比例；奖励校准通过后才跑 10-step qualification，再决定是否做 100-step 短实验。最后在冻结 dev 上与 SFT 配对，并单独检查工具、安全、沟通与退化；只有有可复核收益才继续放大。

当前没有新 RL 权重，没有本阶段的新真实模型成绩。MCP 只读网关、检索入口、会话记忆、追踪等工程能力与 RL 结果分别展示；多 Agent routing 仍未实现。
