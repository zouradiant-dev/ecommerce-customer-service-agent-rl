# 基于 SFT 与 GRPO 的业务规则约束型售后客服 Agent

**Project paper:** [English (primary)](PROJECT_PAPER_EN.md) | [中文](PROJECT_PAPER_ZH.md)

**多轮 Agent 迭代：** 已补充预算化上下文、结构化业务状态、十五个在线多轮开发回归用例和分层调用追踪。运行 `python scripts/evaluate_online_agent.py --mode smoke --output outputs/online-agent-smoke` 验证工程行为；真实模型、上下文对照与追踪报告用法见 [记忆、评测与追踪说明](docs/online-agent-memory-evaluation-tracing.zh-CN.md)。Smoke 使用测试替身，不代表 SFT 模型效果。

**在线工程原型：** 新增持续会话、共享订单后台、用户确认、幂等取消和重启恢复。运行 `python scripts/run_customer_service.py` 后访问 `http://127.0.0.1:8766`；默认手动业务验证模式无需 GPU，模型模式需配置实际推理接口。启动步骤、接口契约与验证边界见 [在线客服工程说明](docs/customer-service-online.zh-CN.md)。这是独立入口，不改变原有训练及评测路径，也不代表已经接入真实订单或支付系统。

这是一个面向“电商退款与订单修改”的可验证 Agent 项目。它不依赖下载好的对话数据，而是先定义业务数据库、状态、政策和任务模板，再生成隔离任务，收集工具调用轨迹，完成：

> 基线评测 → 数据构建 → SFT → Bad Case 诊断 → RL Gate → 在线 GRPO → 配对统计验证

当前仓库已经实现可离线验证的数据、工具、环境、奖励和评测闭环，提供 ABCD 外部对话转换入口，并区分两条训练路线：静态轨迹上的离线 group-AWR 基线，以及基于 TRL 状态环境、每个 completion 独立 SQLite 快照的在线 GRPO。脚本 Agent 只用来验证环境正确性，不能当作真实模型训练结果；真实基线、SFT 和 GRPO 数字仍必须上 GPU 服务器测量。

v18 的 CPU 改造和新 train/dev 数据阶段已作为独立 opt-in 入口落地：业务状态机、证据绑定的结构化终态、`strict_binary_v18` / `train_shaped_v18` 双轨奖励、3,600 个新任务和 17,760 条严格验证的参考 SFT 样本均不改变历史默认路径。48G QLoRA 已完成 20 步 smoke 和 925 步 full；最终 LoRA 在修正后的 600 题 dev 上为 530/600（88.33%）且有 13 条硬安全失败。冻结 LoRA 配合证据型 policy controller r6 后，完整 600 题 dev 为 589/600（98.17%）且 `hard=0`。该增益属于模型外控制器系统，不是 LoRA 学习提升，也不是独立 held-out test 或生产上线证据。完整过程见 [v18 项目问题、解决方案与交接记录](docs/v18-project-issues-solutions-and-handoff.md)。

## v18 当前状态

- 冻结基座：`Qwen/Qwen2.5-7B-Instruct@a09a35458c702b33eeacc393d103063234e8bc28`。
- 原始 v18 LoRA：925 个优化器步，adapter SHA256 `f41c98a583a6ca66e18f1d6f8743029cb3a363cf96f43d80dbf0d29b74876b6c`。
- 600 题 dev：原始 508/600；只修正评测器派生字段后为 530/600，模型回复和工具轨迹未重生成。
- 针对性补训、再平衡和 full replay 均出现能力干扰，没有替代原始 adapter。
- policy controller r4 在定向题通过，但 150 题宽测只有 111/150 且 `hard=6`，已淘汰。
- policy controller r6：70/70 风险筛查、149/150 宽测、589/600 完整 dev，三轮均 `hard=0`。
- 当前可运行 CPU 环境/网页审计演示；真实 7B LoRA 推理仍需要 GPU 或兼容的远程推理服务。

## 当前实验结论

2026-10-03 的冻结 v17 最终 test 中，原 SFT 与探索性 GRPO checkpoint-25 均为 48/160（30.0%）；配对差值为 0，95% 置信区间为 `[-0.05, 0.05]`，没有证明 GRPO 优于 SFT。正式 RL Gate 未通过，当前模型不具备生产上线证据。失败原因、证据边界以及不污染旧 test 的 v18 改进路线见 [v17 最终 Test 失败复盘与 v18 改进计划](docs/final-test-v17-postmortem-and-v18-plan.md)，下一版严格评测/训练双轨奖励、硬安全门控和校准要求见 [v18 奖励机制设计](docs/reward-v18-design.md)。

## 基座模型

项目统一使用 **`Qwen/Qwen2.5-7B-Instruct`** 作为基础策略模型：baseline 从原始 Qwen 7B 开始，SFT 使用验证通过的客服工具轨迹进行 QLoRA，GRPO 再从 SFT checkpoint 继续训练，避免三个阶段更换基座导致对比失真。

- 统一模型配置：[configs/qwen2.5-7b-qlora.json](configs/qwen2.5-7b-qlora.json)
- `run-model` 不传 `--model` 时默认使用该模型；
- 可运行 `ecommerce-cs model-info` 查看完整模型与训练配置；
- 配置中的学习率、LoRA rank 和 KL 系数是起始值，必须根据 dev 集而不是 test 集选择；
- 当前本机为 8 GB 显存，适合环境测试和量化推理，7B 的正式 SFT/GRPO 建议放到更大显存的训练机。

## 为什么这个项目适合做作品集

- **真实业务约束**：身份、发货状态、退款期限、优惠券、不可退款商品、用户确认、工具异常、矛盾信息和越权诱导都会影响决策。
- **自建数据库**：每个任务都有独立 SQLite 快照，轨迹是否成功由数据库最终状态和业务规则共同判断，不只看模型最后一句话。
- **无标签泄漏 ID**：模型可见的客户、订单和商品 ID 由 seed 确定性哈希生成，不包含场景名称；身份核验后由工具返回内部 customer_id，模型无需猜测。
- **真实多轮确认**：部分任务会先让 Agent 追问身份或二次确认；工具只接受真实用户消息中出现的明确授权，模型自行填写 `confirmed=true` 不算授权。
- **可验证工具使用**：读写工具分离，写操作需要身份、证据、确认和政策许可；所有调用都有审计记录，未执行却声称“已取消/已退款”会被判失败。
- **数据闭环**：教师轨迹先经状态验证，再做工具序列去重、场景均衡、失败保留和模板级切分隔离。
- **RL 有进入条件**：只有当 SFT 的工具格式和覆盖率达标、验证集进入平台期、剩余错误偏策略性，且同任务多次采样有组内奖励差异时才进入 GRPO。
- **可信对比**：支持按同一任务配对比较、pass@k、场景切片、失败分类和 bootstrap 置信区间。

## 项目结构

```text
ecommerce-customer-service-agent-rl/
├── configs/algorithm-experiments.json # SFT/AWR/GRPO 消融矩阵
├── docs/
│   ├── data-design.md               # 数据库、任务组合与防泄漏设计
│   └── experiment-plan.md           # 从基线到 GRPO 的执行标准
├── scripts/run_local_demo.py        # 无模型依赖的端到端环境自检
├── scripts/run_web_demo.py          # 轻量网页审计演示
├── scripts/train_sft.py             # QLoRA SFT、checkpoint 与断点恢复
├── scripts/train_sft_v18_48gb.py    # v18 48G 哈希/显存预检与 smoke/full 门禁
├── scripts/build_v18_sft_package.py # 构建便携 v18 SFT 训练包
├── scripts/train_grpo.py            # TRL 状态环境在线 GRPO
├── scripts/train_offline_group_awr.py # 静态轨迹离线对照，不冒充 GRPO
├── scripts/summarize_algorithm_experiments.py # 配对算法实验汇总
├── scripts/check_training_env.py    # CUDA/依赖/显存预检
├── scripts/evaluate_stages.py       # Base/SFT/GRPO 统一配对评测
├── src/ecommerce_customer_service/
│   ├── database.py                  # 自建 SQLite schema
│   ├── tasks.py                     # 组合任务与独立数据库快照
│   ├── tools.py                     # 9 个客服工具与业务保护
│   ├── tools_v18.py                 # v18 证据绑定业务状态机（独立 opt-in）
│   ├── online_env.py                # 每条在线 rollout 的隔离状态环境
│   ├── online_env_v18.py            # v18 双轨奖励环境（独立 opt-in）
│   ├── rewards.py                   # 严格/安全塑形奖励分解
│   ├── terminal_v18.py              # v18 结构化终态协议与工具证据核验
│   ├── verifier_v18.py              # v18 严格评测、训练奖励和硬安全门控
│   ├── tasks_v18.py                 # 新 v18 train/dev 生成与防泄漏验证
│   ├── reference_v18.py             # 严格通过的确定性 SFT 协议种子
│   ├── sft_v18_48gb.py              # 冻结输入、来源与 48G 硬件预检
│   ├── webapp.py                    # 零前端依赖演示服务
│   ├── simulator.py                 # 脚本 Agent + OpenAI 兼容模型 Agent
│   ├── verifier.py                  # 状态/沟通/政策/工具联合奖励
│   ├── data.py                      # 去重、筛选、SFT/离线对照数据构建
│   ├── external_data.py             # ABCD 语言 SFT 安全转换
│   ├── diagnostics.py               # Bad Case 分类与 RL Gate
│   ├── evaluation.py                # pass@k 与配对 bootstrap
│   └── cli.py                       # 统一命令行入口
└── tests/                            # 环境、规则、数据和评测测试
```

## 自建数据库

数据库包含：`customers`、`products`、`inventory`、`orders`、`order_items`、`payments`、`shipments`、`returns`、`refunds`、`support_tickets`。

每个任务目录内都有自己的 `state.sqlite`。运行 rollout 时还会复制到临时目录，因此多个试验不会互相污染。取消订单、修改地址和创建退款必须真实更新数据库；如果模型只是口头声称“已处理”，验证器会归类为 `state_not_updated`。

## 快速开始

要求 Python 3.10+，本地自检没有第三方运行时依赖。

```powershell
cd D:\桌面\llm-learning\ecommerce-customer-service-agent-rl
python scripts\run_local_demo.py --output outputs\local-demo --replicas 1
python -m unittest discover -s tests -v
```

启动轻量应用展示（只依赖 Python 标准库）：

```powershell
python scripts\run_web_demo.py --port 8765
```

浏览器访问 `http://127.0.0.1:8765`，可对同一场景切换 verified reference 与 weak baseline，查看奖励分解、Verifier 结论和工具审计轨迹。它演示的是环境与评测能力，不把脚本策略伪装成训练模型。

如需在 Windows 8GB 显存机器上用真实的 0.5B 小模型观察完整工具调用流程，请按 [本地小模型运行指南](docs/local-small-model.md) 操作。该流程使用 Transformers 的 OpenAI 兼容服务，并可用 `run-model --verbose` 逐轮查看模型、工具、SQLite 状态验证与评分。

安装为可编辑包后也可以使用统一命令：

```powershell
python -m pip install -e .
ecommerce-cs build-tasks --output data\v1 --replicas 20
```

## 完整实验链路

### 1. 构造任务数据库

```powershell
ecommerce-cs build-tasks --output data\v1 --replicas 20 --seed 7
ecommerce-cs build-pilot `
  --manifest data\v1\tasks.jsonl `
  --output data\v1\pilot-train-dev.jsonl `
  --splits train,dev
```

模板只属于一个 split；当前共 30 个模板，分配为 15 个 train、7 个 dev、8 个 test，从源头避免同一话术模板跨训练和测试。新增模板覆盖恢复、幂等、多订单选择、数量边界、支付状态、跨订单参数绑定和“部分可执行”的批量请求，而不是只做同义话术改写。
`build-pilot` 会为每个指定模板选一条固定任务；默认只取 train/dev 的 22 条，适合先验证教师 API 和语义 Verifier，避免把 test 暴露给教师数据构建。

### 2. 跑真实基线

先用 vLLM、SGLang 或其他 OpenAI-compatible 服务启动基础模型，然后：

```powershell
ecommerce-cs run-model `
  --manifest data\v1\tasks.jsonl `
  --output outputs\base\rollouts.jsonl `
  --api-base http://127.0.0.1:8000/v1 `
  --trials-per-task 4 `
  --temperature 0.8

ecommerce-cs evaluate --rollouts outputs\base\rollouts.jsonl
ecommerce-cs badcases --rollouts outputs\base\rollouts.jsonl
```

LangGraph 可作为同一套模型、工具和任务的可选编排运行时：

```powershell
python -m pip install -e ".[agent]"
ecommerce-cs run-model --manifest data\v1\tasks.jsonl --output outputs\base-langgraph\rollouts.jsonl --runtime langgraph --api-base http://127.0.0.1:8000/v1 --scenario cancel_unshipped --limit 1
```

LangGraph 图用模型回合、工具执行和响应处理三个节点编排对话；响应处理节点负责用户追问与状态纠错，逐轮使用当前状态允许的工具。输出与默认 `native` 运行时保持同一 rollout 格式；业务授权、独立 SQLite 副本和 Verifier 仍由原项目负责。配对阶段评测也支持 `python scripts/evaluate_stages.py ... --runtime langgraph`。在线 GRPO 继续使用 TRL 的 `environment_factory`，不受此运行时选项影响。评测图不启用持久化 checkpoint，以免恢复执行时重复触发业务写工具。

### 3. 生成和筛选教师轨迹

用更强模型执行同一 `run-model` 命令并更换 `--model` 和输出路径。轨迹随后经过严格验证和筛选：

```powershell
ecommerce-cs curate `
  --rollouts outputs\teacher\raw.jsonl `
  --output outputs\teacher\curated.jsonl `
  --rejected outputs\teacher\rejected.jsonl `
  --max-per-scenario 100

ecommerce-cs build-sft `
  --rollouts outputs\teacher\curated.jsonl `
  --output data\sft
```

`data\sft\train.jsonl` 和 `dev.jsonl` 是 messages 格式，并包含 assistant-only `loss_mask`。test 轨迹不会进入 SFT 数据。

如需加入开源对话，先从 ABCD 官方仓库取得 `abcd_v1.1.json` 或 `.json.gz`，再运行：

```powershell
ecommerce-cs import-abcd `
  --source data\external\abcd_v1.1.json.gz `
  --output data\abcd-sft
```

转换器默认使用脱敏的 `delexed` 文本，只读取 train/dev，并把 ABCD action 留在 metadata，而不是伪装成本项目的成功工具调用。它适合改善客服语言和追问能力；工具正确性仍由本项目的 SQLite 验证轨迹训练。

### 4. SFT 后诊断与 RL Gate

对每个 SFT checkpoint 用相同任务、采样次数和温度评测，再运行：

```powershell
ecommerce-cs rl-gate `
  --rollouts outputs\sft-final\rollouts.jsonl `
  --checkpoint-pass-rates 0.43,0.49,0.50,0.50
```

只有五项检查同时通过才建议进入 RL：工具格式稳定、关键工具覆盖完整、SFT 平台期、失败以策略问题为主、同任务多次 rollout 有成功也有失败。

### 5. 在线 GRPO 与离线算法对照

v18 使用独立的、失效即停止的训练链，不能调用历史 v13/v17 入口：

```bash
# 1. 冻结 SFT 在 train-only 任务上做 4 次采样，不使用 policy controller
bash scripts/run_v18_grpo_calibration.sh

# 2. 人工逐条填写输出目录中的 reviewed-rollouts.jsonl 后生成绑定源码和协议的收据
python scripts/audit_v18_reward_calibration.py \
  --rollouts outputs/v18-grpo-calibration-r1/reviewed-rollouts.jsonl \
  --protocol outputs/v18-grpo-calibration-r1/protocol.json \
  --output outputs/v18-grpo-calibration-r1/calibration-receipt.json

# 3. 先跑 10 个 optimizer step；只有组内方差、梯度、权重更新和显存门槛全过才保留
bash scripts/run_grpo_v18.sh qualification \
  outputs/v18-grpo-calibration-r1/calibration-receipt.json \
  outputs/v18-grpo-calibration-r1/reviewed-rollouts.jsonl \
  outputs/v18-grpo-calibration-r1/protocol.json
```

100 步实验必须显式传入成功的 qualification 目录，并从原始 SFT adapter 重新开始，
不会续训 10 步产物。训练使用 `train_shaped_v18`；最终晋升仍使用
`strict_binary_v18`，并要求同控制器、同评测协议下的 SFT/GRPO held-out 配对比较。
校准采样、10 步和 100 步本身都不会把 `formal_rl_gate_passed` 写成 true。

先做不加载模型的环境预检：

```powershell
python scripts\train_grpo.py `
  --manifest data\v1\tasks.jsonl `
  --model outputs\sft\merged `
  --output outputs\grpo\preflight `
  --preflight-only
```

正式训练需在独立 CUDA 环境安装 `requirements-online-grpo.txt`。`scripts/train_grpo.py` 使用 TRL `GRPOTrainer` 的 `environment_factory`：每个 completion 都重置独立 SQLite，工具调用真实改变状态，episode 完成后再由 Verifier 给奖励。test 被拒绝；TRL 通用工具循环无法自然注入额外用户轮次，因此需要脚本用户回复的任务会在预检报告中明确排除，仍保留作 SFT 和评测深度任务。

```powershell
python scripts\train_grpo.py `
  --manifest data\v1\tasks.jsonl `
  --model outputs\sft\merged `
  --output outputs\grpo\strict-seed7 `
  --reward-mode strict_binary `
  --num-generations 4 `
  --seed 7
```

严格二值奖励要求数据库状态、必要信息、政策和必要工具全部正确。`audit_shaped` 只用于消融：一旦政策失败即归零，避免“危险操作靠过程分补回来”。原静态轨迹训练已改名为 `train_offline_group_awr.py`，仅作为离线算法对照，不能报告成在线 GRPO。完整消融矩阵见 `configs/algorithm-experiments.json`。

### 6. 配对统计验证

```powershell
ecommerce-cs compare `
  --base outputs\sft-final\test.jsonl `
  --candidate outputs\grpo\test.jsonl
```

报告提供公共任务上的 pass^1 差值和配对 bootstrap 95% 置信区间。正式报告还应至少使用 3 个随机种子，并单独汇报各业务场景。

多个算法 rollout 可统一汇总：

```powershell
python scripts\summarize_algorithm_experiments.py `
  --run sft_control=outputs\sft-final\test.jsonl `
  --run grpo_strict=outputs\grpo\strict-test.jsonl `
  --baseline sft_control `
  --output outputs\algorithm-report.json
```

## Bad Case 分类

项目将失败归入：工具选择错误、参数绑定错误、未查询必要信息就写入、过早升级人工、未确认、无工具证据却声称成功、调用不存在工具、重复调用/死循环、工具错误恢复失败、违反业务政策、环境状态未更新，以及沟通失败和运行环境错误。

## 与现有两个项目的关系

- `ecommerce-code-agent-data` 提供了清晰的数据包、环境和奖励分层思路，但原任务是代码修复，不是客服。
- `tau2-agentic-rl` 提供了成熟的 SFT→诊断→GRPO→配对统计实验方法，但业务是航空客服。
- 本项目复用两者的方法论，重新实现为电商售后域，并将业务数据改为你自己生成和控制的 SQLite 状态。

更细的数据设计见 [docs/data-design.md](docs/data-design.md)，实验准入与结果口径见 [docs/experiment-plan.md](docs/experiment-plan.md)。
外部数据集选型、许可状态和接入边界见 [docs/open-datasets.md](docs/open-datasets.md)。
GPU 环境、SFT、GRPO 迭代、断点恢复和统一评测命令见 [docs/server-experiments.md](docs/server-experiments.md)。
