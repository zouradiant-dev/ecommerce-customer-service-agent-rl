# SFT-150 → train 采样 → 有界在线 GRPO

本轮不重训 SFT、不调用付费语义审计、不重复跑 Base/全量 dev，不覆盖旧报告或权重。

## 本轮提示修复

`business_boundary_v12` 是显式选择的实验提示。默认 `SYSTEM_PROMPT` 原文不变，旧 dev 入口默认行为不变。

- 订单不确定：列订单、确认目标；目标已知但用户拒绝：暂停并明确未执行，不转去列无关订单。
- 取消/地址仅 pending、paid 可执行；退货资格独立查询，不能跨业务推论。
- 已有明确授权则完成前置查询后执行，不反复问确认；最终答复必须对应真实结果。
- 不带 task ID、预期答案、verifier intent 等 oracle 信息。

旧 v11 communication/verifier 字节未改变。提示变化产生的新分数是新实验，不追溯替换原报告。

## 第一步：32 条 train 轨迹

`scripts/sample_train_readiness.py` 在原 `.venv-training` 环境运行，Qwen7B NF4 + SFT-150，温度 0.4。
八个 train 模板：取消、改地址、损坏退款、部分退款、查物流、身份重试、退款重试、持续错误升级。
每类一个真实 train 实例、每题四次；选择无额外用户回复的 inline 任务，和当前在线入口的资格规则一致。
清单中的 dev/test 不会入选，也不能通过修改清单 split 冒充 train。

结果保留 `protocol.json`、`model_info.json`、完整 `sft/rollouts.jsonl`、原始 `report.json`、
`readiness.json`、`reward_review.md`、待审核 `reward_review.json`、`summary.json`。
中断/运行时错误保留部分结果，不自动重试，不能把未完成任务记成成功。

工程采样条件：组完整、至少两次采样、无 worker/格式/截断/步数耗尽、三个关键写工具都有严格通过轨迹、
至少 20% train 任务组有 0/1 混合结果。每题四次只用于这轮诊断，不是可靠的泛化估计。

`reward_review.json` 默认全部 pending。审核判 1 和判 0 的原始工具事实、状态、政策、最终答复，
每条给出证据；误判/不确定不能直接确认。不存在自动全通过审核脚本。
只有明确错误才能在独立奖励版本中修复并回归；不得用 dev 答案制成训练目标、放宽授权或改变状态断言来抬分。

运行 `bash scripts/run_train_readiness_v12.sh` 即可先 CPU preflight，再启动本轮后台采样；
结果目录随机唯一，旧结果不覆盖，服务器已有 GPU 进程时拒绝启动。无包安装或模型下载。
完成/失败后可下载 `outputs/train-readiness-v12-result-20261001-r1.tar.gz`。
重复执行遇到既有归档会停止，避免重复耗费 GPU；需要再跑时另建版本与输出。

## 第二步：奖励核查后再进行 GRPO 工程试跑

当前正式 RL gate **未通过**。本轮只开放显式 `--smoke-only` 且最多 20 个 optimizer steps；
不是完整 GRPO 实验，不能报告 held-out 提升。正式 gate 的五项要求并未改写为这轮工程条件。

先建立独立 Python 3.11+ CUDA 环境并检查，不升级/删除 `.venv-training` 或 conda base。
`requirements-online-grpo.txt` 固定 TRL 1.14.1，要求 Transformers 5.13+ 与 jmespath，
来自该版本的状态环境/响应解析 API；其余精确依赖版本和 CUDA 可用性必须记录并服务器验证。
这份依赖文件是准备方案，不是已经在服务器安装成功的环境证明。

与旧入口不同，本轮可直接加载 SFT adapter 为 trainable policy：TRL 1.14.1 为现有 PEFT policy
建立 `ref` 副本，KL 对比 SFT 而非关掉 adapter 后的原模型。启动时校验策略/参考初值一致、
只有 policy LoRA 可训练，结束时校验参考未变化；不需要先 CPU 合并十几 GB 模型。
参考 API：[固定版本源码](https://github.com/huggingface/trl/blob/v1.14.1/trl/trainer/grpo_trainer.py)。

训练条件还包括：SFT 权重/清单/数据库/奖励代码与审核证据匹配；NF4、单张可见 GPU、
native HF、gradient checkpointing、正 KL 系数、严格 0/1 奖励。
新在线提示版本 `business_boundary_v12_static_tools`：TRL public-method 工具面是静态的，
不再承诺像原 sampler 一样隐藏工具；后端仍逐调用检查身份、顺序、授权及实际业务政策。
这与前面的 native dynamic-tool 采样不同，必须在短程试跑中单独验证，不能说 sampler 已证明训练器跑通。

每个 completion 使用新 SQLite 副本；同组 task 必须相同。全部在线奖励、工具和 verifier 结果写入
`online_rewards.jsonl`。连续三个生成 batch 都无组内奖励差异则停止；非有限训练指标也停止。
错误保留现场，不把故障当作能力结果，也不静默切换为宽松或去掉 communication 的奖励。
成功后检查 optimizer steps、保存 adapter 的重新加载张量一致、SFT 参考未变化，写 `smoke_complete.json`。
单张 4090 的实际显存、反向传播和解析兼容性仍需要服务器试跑验证，不能承诺已经可运行。

请先完成第一步并返回归档。确认真实写操作、有效奖励和通信标签后，再提供与审核目录匹配的
10 步命令，避免未审核便消耗服务器或误用旧采样证据。
