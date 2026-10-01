# 开源客服数据集调研与接入计划

> 核验日期：2026-09-21。许可条款可能变更，正式发布数据或模型前需重新核对原始页面。

外部对话不能直接替代本项目的 SQLite 任务环境。正确的分工是：

- 外部对话数据：语言风格、意图覆盖、澄清与共情 SFT；
- 自建状态任务：工具调用 SFT、可验证奖励和 GRPO；
- 外部 Agent 基准：跨环境泛化对照，不与自建 test 混用。

## 推荐顺序

| 数据集 | 规模/形式 | 许可/获取 | 本项目用途 | 结论 |
|---|---|---|---|---|
| [ABCD](https://github.com/asappresearch/abcd) | 10K+ 英文人人对话，55 个意图，含政策、场景和操作序列 | MIT，数据随官方仓库发布 | 多轮客服 SFT、身份核验/政策/操作流迁移 | **首选** |
| [tau2-bench retail](https://github.com/sierra-research/tau2-bench) | 交互式 retail 环境，包含政策、工具、任务和用户模拟 | MIT | 外部基准、多轮用户模拟设计参考 | **强烈推荐** |
| [Bitext Retail eCommerce](https://huggingface.co/datasets/bitext/Bitext-retail-ecommerce-llm-chatbot-training-dataset) | 44,884 英文单轮 QA，46 意图/13 类，含语体标签 | CDLA-Sharing-1.0，派生数据需注意归属和同许可分享 | 意图覆盖、话术扩增、拒绝/礼貌/拼写噪声 | **适合辅助 SFT** |
| [Chinese Ambiguous Reference](https://github.com/ygan/Chinese-Ambiguous-Reference) | 1,000+ 中文真实零售澄清对话 | MIT | 中文模糊指代、追问和澄清任务 | **值得接入** |
| [JDDC 2.1](https://github.com/hrlinlp/jddc2.1) | 约 24.6 万中文多轮会话、300 万 utterance，含图像和知识库 | 需申请并使用机构邮箱，不是无条件下载 | 中文电商对话预训练、重写与摘要 | **申请后再用** |
| [CSDS](https://github.com/xiaolinAndy/CSDS) | 中文客服多轮对话与细粒度摘要 | 官方页面提供下载，但仓库未显式给出数据许可 | 工单摘要、对话状态压缩 | **授权确认后使用** |

## 不建议直接作为核心证据的数据

- 只有几十行、用数据页导流付费完整版的数据；
- 声称数十万条但无生成方法、无质量抽检、无原始引用的合成客服数据；
- 将多个不同许可的数据集重新打包，但没有逐来源说明的聚合数据；
- 只有“用户问题—客服回复”而没有状态、工具或政策标注的数据，不应用来证明 Agent 操作能力。

## 建议的三路数据管线

### A. 语言与澄清 SFT

ABCD + Chinese Ambiguous Reference + 经许可审核的 Bitext 子集：

1. 仅保留退款、取消、物流、改址、人工升级等相关意图；
2. 脱敏并将实体替换为项目生成的受控实体；
3. 首次导入只把外部 action 名称保留在 metadata，不直接伪装成本项目工具；只有人工审核过参数和语义的一对一映射才可进入工具训练；
4. 只训练表达和追问能力，不从外部文本伪造数据库成功奖励。

本仓库已实现 ABCD 本地转换入口：

```powershell
ecommerce-cs import-abcd --source data\external\abcd_v1.1.json.gz --output data\abcd-sft
```

默认只处理 train/dev、使用 `delexed` 文本并输出 `external_dialogue_sft_v1`；`test` 会被转换器拒绝。

### B. 可验证工具 SFT / GRPO

继续使用本项目的独立 SQLite 快照生成任务。外部话术只作为用户表达模板，每条轨迹仍必须经过 verifier 的状态、政策、工具和沟通联合验证。

### C. 外部泛化评测

使用 tau2-bench retail 独立评测同一 checkpoint。不将 tau2 test 轨迹反向加入电商项目训练集，避免把外部基准变成训练数据。

## 接入前检查清单

- 记录原始 URL、revision/commit、许可、下载日期和文件 SHA256；
- 保留原始 train/dev/test 边界，不把原 test 用于选参；
- 通过对话/场景 ID 去重，不只按文本行去重；
- 抽检政策合规性，外部客服回复不默认是正确教师；
- 外部数据不能进入自建的冻结 test manifest；
- 派生数据和模型发布必须符合上游许可与归属要求。
