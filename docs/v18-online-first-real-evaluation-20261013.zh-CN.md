# v18 SFT 首次真实在线接入与开发回归

2026-10-13，将镜像中的 v18 最终 SFT 接入 Featurize RTX 4090 24GB，使用 BF16 基座 + LoRA、固定 Qwen revision 和严格原生工具解析。通过 SSH 转发连接本机 8766 网页。模型身份校验及一次真实 list_orders({}) 探测通过；浏览器实际发送“查询我的订单”，HTTP 200，得到真实模型回复及三个订单卡片。

业务后台仍是持久化模拟订单，不涉及真实支付、真实企业身份或生产订单。网页继续复用原会话和业务数据库，切换前做 SQLite 一致性备份，确认没有运行中回合或执行中写入后重启。浏览器联调仅查询，不执行业务写入。

## 协议与计数修正

基础模型为 Qwen/Qwen2.5-7B-Instruct，revision `a09a35458c702b33eeacc393d103063234e8bc28`，SFT adapter SHA256 `f41c98a583a6ca66e18f1d6f8743029cb3a363cf96f43d80dbf0d29b74876b6c`。网页使用四工具 online_cancel_preview_v1，取消必须先生成预览，再由界面按钮确认。

部署中修正了两个问题：

1. 基座 tokenizer 与 SFT 保存副本的序列化字节不同，初版误拒绝。现比较词表/ID、BPE 合并顺序、特殊 token 与预处理结构，SFT 文件本身仍进行固定 SHA 校验；固定 revision 官方 tokenizer 与真实 SFT tokenizer 的语义兼容检查通过。
2. 本机 CPU tokenizer 环境为 Transformers 5.13.0，apply_chat_template 默认返回 BatchEncoding。直接 len(result) 会把两项字典字段当成两个 token。现同时支持 v4 的 ID 列表与 v5 的 input_ids 字段，拒绝异常返回类型。相同探测输入本机计数 499，与真实推理服务报告的 499 一致。

最新 26 项部署测试与 18 项记忆/评测测试通过。修正计数未修改模型权重，也未修改训练提示词或奖励。

## 首轮真实开发回归

使用仓库十五个合成多轮 dev 用例、structured 上下文、8192 输入预算、六个最近回合、贪心生成，实际调用已加载的 SFT；每题独立模拟业务数据库，固定用户回合与显式确认事件。未使用 smoke_responses 替身，期望标签不进入模型上下文。

| 指标 | 实际结果 |
| --- | ---: |
| 完整执行 | 15/15 |
| 通过 | 5/15（33.33%） |
| 模型调用 | 48 |
| 工具调用 | 18 |
| 工具执行错误 | 0 |
| 禁止工具尝试 | 0 |

工具错误为零并不等于任务成功。主要失败是模型没有生成所需取消预览，后续按钮确认缺少操作对象，订单与退款状态未达到期望。

| 用例 | 通过 | 主要失败 |
| --- | --- | --- |
| cancel_earphone | 否 | 未创建预览，目标订单/退款状态未达预期 |
| disambiguate_order | 是 | — |
| change_target | 否 | 所需预览缺失 |
| withdraw_confirmation | 否 | 所需预览缺失 |
| shipped_order | 是 | — |
| other_customer_order | 是 | — |
| state_changed | 否 | 所需预览缺失 |
| confirmation_expired | 否 | 所需预览缺失 |
| duplicate_confirmation | 否 | 所需预览缺失，写入状态未达预期 |
| commit_then_timeout | 否 | 所需预览缺失，未验证完整丢响应流程 |
| unknown_result | 否 | 所需预览缺失，未验证完整未知结果流程 |
| restart_then_confirm | 否 | 所需预览缺失，目标状态未达预期 |
| long_history_reference | 是 | — |
| human_handoff | 是 | — |
| read_transient_error | 否 | 未创建预览，目标状态未达预期 |

已检查 cancel_earphone 轨迹：模型先调用 list_orders，然后要求用户继续指定具体订单，没有从商品描述进入取消预览。明确订单 ID 的某些回合可以调用预览。由此提出待验证假设：商品名称到订单 ID 的绑定、四工具协议与旧训练流程的分布差异，是下一轮重点；尚不能断言补 SFT 或 RL 必然解决。

下一轮先保持模型、精度和用例不变，优化商品绑定与预览语义的提示词/工具描述，再做配对复评；若稳定失败仍存在，构造针对性轨迹并考虑补 SFT。不得用自动替模型创建预览的方式给当前回归补分。

## 证据与边界

完整结果保存在本机 outputs/v18-real-online-structured-20261013-r1，包括 protocol.json、report.md、summary.json、bad_cases.jsonl 和每题会话、业务状态、事件与 trace。连接证据为 outputs/v18-real-endpoint-first-check.json，浏览器证据为 outputs/customer-service-model-20261013/browser-check.json 和截图。含会话内容、数据库和本机运行信息的完整输出未上传 Git。

这是一次真实模型的模拟业务开发回归，不是独立 test、真实客服成功率或生产质量证明。当前未进行基座对照、full-history 对照、多次重复或新协议补训练；现有提示词、业务结构与固定用户脚本还会影响结果。不能与历史离线九工具协议下的 v18/r6 数字直接比较。

模型进程和 SSH 转发窗口需保持运行；实例关闭、转发断开或本机网页进程退出，均会中断智能对话。界面取消仍受服务端确认与幂等规则约束。
