# v18 实施与 48G SFT 准备状态

更新时间：2026-10-03。

## 已完成的 opt-in 实现

v18 使用独立模块，不改变历史 `tools.py`、`online_env.py`、`verifier.py`、v13/v17 评分或冻结结果：

- `tools_v18.py` 强制身份、本人订单列表、订单读取、商品归属、退款政策、用户确认和写操作的状态顺序；身份失败两次后只允许人工升级。
- `terminal_v18.py` 要求最终回复仅包含 `decision`、`action_taken`、`action_id`、`message`。动作及动态 ID 必须来自本轮成功工具结果。
- `verifier_v18.py` 分开实现正式评测用 `strict_binary_v18` 和训练用 `train_shaped_v18`；危险写入、伪造成功或伪造动作 ID 直接硬门控为 `-1`。
- `online_env_v18.py` 为每个 completion 使用独立 SQLite 副本，只接受带 v18 终态元数据的新训练任务。
- `v18_readiness.py` 和 `audit_v18_reward_calibration.py` 根据人工审阅、关键写工具覆盖、组内奖励差异、沟通 precision/recall 和源码哈希生成 RL 校准收据。

## v18 train/dev 与参考 SFT

`tasks_v18.py` 使用 seed 1807、独立 ID/状态命名空间和互斥提示家族生成新数据：

- train：3,000 个任务，每个场景 100 个；
- dev：600 个任务，每个场景 20 个；
- 30 个场景均覆盖，train/dev 精确提示重叠为 0；
- test 行为 0，`test_v2_created=false`，未复用 v17 test；
- 任务清单 SHA256：`b2564f4ed0d28dab405e6a3cfa19cff85ec29f3451b4f204926f0ee5c2c17c30`。

`reference_v18.py` 用确定性规则控制器为每个任务生成至少一条严格通过的轨迹，并按助手回合构建 assistant-only loss 数据：

- train：14,800 条 SFT 样本，SHA256 `2fbaaa34a8618fb00d4ed90081ed0bd478d46b633af5eb9efc34a432ce955f4f`；
- dev：2,960 条 SFT 样本，SHA256 `a1b246662715aaa8f72a8cdb234d218a19ff2d94fbe2164572d5eb8d4efae5d0`；
- 3,600 个任务全部由 `strict_binary_v18` 验证通过；
- 所有行均标记 `learned_model_result=false`，这些是 SFT 协议种子，不是真实模型成绩。

数据位于 `data/v18-train-dev-r1` 和 `data/v18-sft-reference-r1`。现有 `train_sft.py --validate-data-only` 已验证全部 17,760 条训练格式。

## 48G 单卡执行门禁

`configs/sft-v18-48gb.json` 固定：

- `Qwen/Qwen2.5-7B-Instruct` revision `a09a35458c702b33eeacc393d103063234e8bc28`；
- NF4 QLoRA、BF16、序列长度 4096、micro-batch 1、gradient accumulation 16；
- smoke 为 20 个优化器步，最多评估 120 条 dev；
- full 为 925 个优化器步，约覆盖 14,800 条训练样本一遍，最多评估 300 条 dev；
- 训练、开发、任务清单、数据摘要和两个生成器均由配置中的 SHA256 固定；
- `full` 必须先读取相同基础模型 revision 的完整 smoke `metrics.json`，不能由手写完成标志放行。

启动器的三个阶段相互独立：

```bash
python scripts/train_sft_v18_48gb.py preflight
python scripts/train_sft_v18_48gb.py smoke
# 人工检查 outputs/v18-sft-smoke-r1/metrics.json
python scripts/train_sft_v18_48gb.py full
```

`preflight` 会检查训练依赖、CUDA、BF16、至少 45 GiB 的报告显存、冻结输入哈希、来源标签、场景覆盖和 test 泄漏。本地可用 `preflight --data-only` 只验证 CPU 可检查部分；该选项不能用于 smoke/full。

便携训练包为 `packages/v18-sft-48gb-r1-deferred-scale-r3.tar.gz`，SHA256 为 `83a4b5d85f27e37f877f58b891dbd0a81664f41045563b2c49c7dab1a183b077`。压缩包不含 test-v2、任务 SQLite 副本或参考 rollout。逐微批次诊断先在 4-bit 基座上定位到长上下文样本 `sft_v18_v18_train_mixed_order_bulk_cancel_0006_turn5`，随后又发现非量化 BF16 基座会在任何优化器更新之前被另一条轨迹触发。该轨迹为 2,142 tokens，其中 56 个监督 tokens；它单独重复累计 16 次时在 SDPA 和 eager 下均正常，峰值显存约 19.27 GiB。按原始顺序将前 16 个样本分别清空梯度运行也全部正常，单样本梯度范数为 0.098 到 22.676。这一对照最初让排查方向转向梯度累计；后续实验证明它没有确认根因。

第一次 FP32 LoRA 缓冲区验证以及随后延后 loss 缩放的验证仍在同一第 6 个微批次失败，说明仅修改叶子梯度 dtype 或缩放时机仍不足。与“原始顺序逐样本清空 `.grad` 运行全部正常”的对照相比，当时提出 autograd 向已有 `.grad` 写入可能触发故障的假设；独立缓冲区的后续试验仍失败，因此该假设未成立。修订后的稳定性配置使用非量化 BF16 基座、FP32 LoRA 可训练参数、SDPA 和非 reentrant 梯度检查点；每个微批次均在空 `.grad` 上用未缩放 loss 独立反向，随后把该批梯度加到独立 FP32 缓冲区并立即清空 `.grad`。累计满 16 次后再统一除以 16、裁剪并更新。这与平均 microbatch loss 数学等价，并完整保留已验证稳定的逐批反向路径。smoke 仍保留前 128 个微批次的逐批梯度检查。

独立 FP32 梯度缓冲仍在不同样本上触发非有限值，不能据此把 BF16 确认为根因。随后用于诊断的回退配置将 76.56 亿基础模型参数和 LoRA 参数全部置为 FP32，同时保留 SDPA、非 reentrant 梯度检查点和独立 FP32 累计缓冲。该模式预计增加约 14.3 GiB 基础权重显存并降低吞吐，但 48 GiB 实例仍有机会容纳；必须先用 1 个优化器步记录峰值显存，再决定是否启动完整 smoke。训练器会打印并记录模型 dtype、可训练参数 dtype、梯度缩放时机和累计存储模式，在首个非有限 loss 或梯度处写入 `failure.json` 并停止。

## 基础权重诊断的新证据

纯基础模型（不加载 LoRA、不训练）的 FP32 前向在 SDPA 与 eager 中均失败，第 27 个 hidden-state 记录首次出现 NaN。eager 试验在前向之前的 GPU 检查报告 `model.layers.26.mlp.up_proj.weight` 含 NaN/Inf，与首个异常层对齐；前一次 SDPA 检查报告的非有限参数数目却为 0。该结果还需要 CPU 拷回逐位比较，以区分权重内容异常与 GPU 有限性检查本身的异常。

此前对 BF16、dropout、梯度累计和 SDPA 内核的确定性归因均撤回。`scripts/audit_qwen_base_cache.py` 直接在 CPU 上读取 safetensors 的目标张量，并将四个分片的实际 SHA256 与固定 revision 的官方 LFS 元数据比较。它不加载模型、不占 GPU、不修改权重；官方元数据已在本地查证并固定到脚本。磁盘检查通过之前，不再启动 smoke/full。

服务器的磁盘核验已通过：四个 safetensors 分片 SHA256 均与官方一致，CPU 上 `model.layers.26.mlp.up_proj.weight` 为 BF16，NaN/Inf 数均为 0。无需重新下载权重。下一步由 `scripts/diagnose_qwen_weight_transfer.py` 将同一张量分别通过 CPU FP32 上传、原始 BF16 上传、GPU BF16 转 FP32 并拷回 CPU做逐位比较，另对一个小 FP32 linear 与 CPU 结果比较。这只能检验这些局部复制/转换/计算路径；通过不代表完整 GPU 地址范围或所有模型算子均正常。当前尚不能将硬件故障确认为根因，smoke/full 保持未放行。

服务器已完成目标张量的三轮复制与转换，共九项测试；所有拷回 CPU 的结果都逐位一致，NaN/Inf 均为 0。这排除了该小规模测试中的复制或转换异常，但用户输出没有包含 linear 计算对照，不能把它也计为通过。`scripts/diagnose_qwen_model_loading.py` 接着在两个独立进程中比较分片加载至 GPU 与常规 CPU 加载后上传；它逐位核对完整基础模型与磁盘参考，并在一次 eager 前向后再次核对。CPU 路径还在上传之前核对 CPU 模型；GPU 有限性检查与 CPU 拷回结果分别报告，避免只靠 GPU reduction 判定权重内容。该脚本已通过本地语法检查，完整 GPU 诊断尚待服务器执行。

2026-10-03 的完整模型诊断现已返回：CPU 常规加载的全部 7,615,616,512 个参数元素与磁盘逐位一致；两种路径的 GPU 往返检查均出现 3,072 个错误元素。直接分片加载的差异位于 `model.layers.26.mlp.up_proj.weight`，CPU 加载后上传的差异位于 `lm_head.weight`，而后一路径的第 26 层权重完全一致。坏值仍为有限数，例如正常的 `-0.010009765625` 变为约 `-2.0392e29`，所以有限性检查没有报错。两条路径均在前向前停止，退出码为 2，没有训练或优化器更新，也没有 OOM 报错。

当前已确认完整模型经过 GPU 路径后发生数据完整性异常；尚不能单凭日志确定是显存芯片、驱动、传输链路还是宿主机内存问题。应由平台检查并迁移至健康实例，在新实例重跑同一逐位核验后才恢复 smoke。继续调整学习率、dropout、注意力实现或梯度累计不能解决这项已观测的权重差异。原始输出与平台反馈说明保存在 `outputs/v18-gpu-weight-audit-20261003/` 和 `docs/v18-gpu-weight-corruption-2026-10-03.md`。

用户更换实例后以相同模型、torch 2.2.2+cu121 和 transformers 4.46.3 重跑诊断。直接 `device_map` 与普通 CPU 加载后上传两条路径均对 7,615,616,512 个参数元素完成逐位核对，前向前后全部一致；相同 672-token dev 样本的所有 hidden states 与 logits 均为有限值，selected CE 为 0.0053057，两个独立进程均 exit=0。这一对照确认旧实例是故障范围，新实例通过基础模型完整性门槛。训练配置因此从用于诊断的全 FP32 基座恢复为 BF16 NF4 QLoRA；下一步仍须完成独立 smoke，不能直接启动 full。

新实例随后完成 BF16 NF4 QLoRA smoke：20/20 个优化器步全部完成，训练 loss 和梯度范数均为有限值，未生成 `failure.json`。120 条固定 dev 子集的 loss 从 0.3772076 降至 0.2077779，峰值显存 14.0896 GiB，总耗时 778.57 秒；14,800 条 train 和 2,960 条 dev 来源行均通过冻结数据检查且无跳过。该结果满足 launcher 的 full 门禁，可以使用同一配置哈希和 smoke 收据启动 925 步 full。

## 回归结果

`tests/test_v18.py` 的 10 个测试覆盖状态机、证据绑定、结构化终态、安全硬门控、校准收据、独立在线环境、train/dev 防泄漏、参考 SFT 严格通过及 48G 预检篡改拦截。

```text
v18 contract tests: 10 tests, OK
combined unittest: 247 tests, OK (1 optional LangGraph test skipped)
portable package data-only preflight: passed
package manifest entries: 55, all hashes verified
```

## 当前边界

48G 实例已经运行多轮诊断，但 smoke 尚未通过，没有可用于正式训练放行的完整 SFT checkpoint 或真实模型 dev 结果。纯基础模型 FP32 诊断峰值为 32.064 GiB；这不是正式训练最坏长度的显存证明。test-v2 尚未生成或打开。人工奖励校准也未完成，因此当前不能启动 GRPO；SFT 完成后仍需真实模型采样、人工正负审阅、mixed-group rate 和非零梯度资格检查。
