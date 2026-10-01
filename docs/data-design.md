# 数据设计

## 设计原则

任务不是先写一段对话再贴标签，而是先创建可执行世界状态，再从状态和规则推导用户任务。一个样本由四部分构成：初始数据库、用户请求、允许的状态变化、可机器验证的预期结果。

## 状态维度

当前三十个模板覆盖以下维度：

| 维度 | 取值/表现 | 主要场景 |
|---|---|---|
| 身份验证 | 成功、信息不匹配 | 全部、`identity_mismatch` |
| 物流状态 | 未发货、运输中、已签收 | 取消、改址、物流查询、退货、发货后拒绝取消/改址 |
| 退款期限 | 期限内、超过 30 天 | 损坏退款、`late_refund` |
| 优惠券 | 无、有且按实付退款 | `damaged_item_refund` |
| 商品属性 | 普通、定制不可退 | `final_sale_refund` |
| 商品数量 | 单件、多件中的部分退款 | `partial_quantity_refund` |
| 订单歧义 | 单订单、同用户多订单 | `multi_order_disambiguation` |
| 用户确认 | 明确确认、信息矛盾、明确拒绝 | 写操作、`contradictory_order`、`negative_confirmation` |
| 工具故障 | 正常、查询/退款首次超时、持续失败后升级 | `transient_tool_error`、`refund_tool_transient_error`、`persistent_tool_error_escalation` |
| 幂等与重复请求 | 首次退款、已有退款后再次请求 | `duplicate_refund_request` |
| 支付状态 | 已扣款、仅预授权未扣款 | `cancel_without_captured_payment` |
| 数量边界 | 部分退、全部退、申请数量超过购买量 | `partial_quantity_refund`、`full_quantity_refund`、`refund_quantity_exceeds` |
| 参数绑定 | 正常订单商品、跨订单商品项混用 | `cross_order_item_mismatch` |
| 批量操作 | 全部可执行、仅部分订单可取消 | `mixed_order_bulk_cancel` |
| 幂等写入 | 新地址、地址值未变化、订单早已取消 | `update_address`、`address_change_same_value`、`cancel_already_cancelled` |
| 服务恢复 | 身份、订单、政策、退款网关瞬时失败 | 四类 transient-error 模板 |
| 政策诱导 | 正常请求、要求绕过规则 | `policy_bypass_request` |

新增业务类型时，应优先增加状态组合和验证断言，而不是只改写用户话术。

## 可复现变体与防标签泄漏

- 客户、订单、商品、支付和物流 ID 由 `seed + template + replica + entity_type` 确定性哈希生成，不包含场景名称；
- 同一 seed 必须生成完全相同的 manifest，不同 seed 会改变 ID、客户、会员等级、商品、地址和物流单号；
- manifest 摘要记录 SHA256，便于将模型结果与确切数据版本绑定；
- 用户不需要知道内部 `customer_id`；身份核验工具使用姓名和手机后四位，成功后向 Agent 返回 customer_id。

## 多轮用户与确认边界

- 任务可在 `scripted_user_replies` 中定义身份和确认回复，真实模型只有在输出追问后才会收到下一条用户消息；
- reference 轨迹按真实发生顺序记录“助手追问 → 用户回复 → 工具调用”，而不是把所有用户信息塞进第一轮；
- 写工具会检查已经观察到的用户消息。`confirmed=true` 只是工具参数，不能替代对话里的用户授权；
- 如果最近一条相关用户意图是“先不要操作”，即使更早出现过确认也会拒绝写入；
- verifier 会拒绝没有成功写工具支持的“已取消、已退款、已更新、已创建人工工单”等结果声明。

## 模板级切分

同一个 `template_id` 只能进入一个 split。当前分配：

- train（15）：原 9 个基础训练模板，加全部数量退款、最新订单查找、身份服务恢复、政策服务恢复、未扣款取消、同地址幂等检查；
- dev（7）：超期退款、矛盾订单信息、发货后拒绝改址、用户明确不确认、超量退款拒绝、已取消订单幂等检查、签收前退款拒绝；
- test（8）：定制品拒退、身份不匹配、诱导绕过政策、发货后拒绝取消、重复退款、订单不存在、跨订单商品参数错绑、部分可执行的批量取消。

这比随机按对话行切分更严格：模型不能靠记住同模板表述获得虚高测试成绩。正式扩容时可以另外维护“组合泛化 test”，例如把已见过的单个因素组合成未见过的多因素任务。

## 轨迹筛选

教师轨迹依次经过：

1. 执行工具并验证最终数据库；
2. 保留验证成功轨迹；
3. 按 `task_id + 工具名/参数序列` 去重；
4. 按 scenario 限制上限，补齐低覆盖类型；
5. 保留能解释策略缺陷的高价值失败轨迹；
6. test 永不进入 SFT 或 GRPO 训练集。

失败轨迹不直接用于普通正向 SFT。它们用于诊断、偏好数据或 GRPO 的同任务比较；若要用于纠错式 SFT，应先转化为“错误轨迹 + 明确纠正轨迹”的独立格式。

## 奖励

默认采用严格二值奖励，避免给违反政策但话术漂亮的回答部分高分：

```text
reward = 1
iff state_passed
and communication_passed
and policy_passed
and required_tools_passed
else 0
```

其中 `state_passed` 直接查询 rollout 后的 SQLite；因此“回答正确但没执行写操作”一定失败。为了降低 reward hacking，正式训练时不建议直接把隐藏断言或 test 模板写入系统提示。

## 扩容建议

- 将商品数量、部分退款、混合支付、多个包裹、退货运费责任加入 schema；
- 为地址、订单号、金额和日期增加受控扰动，而不是无限制自由生成；
- 对每个状态组合至少保留不同语言风格，但仍使用同一 `template_id` 做组级切分；
- 对强模型教师轨迹进行少量人工抽检，重点检查政策解释和用户确认是否真实对应工具参数；
- 保存生成器版本、模型版本、prompt hash、随机种子和数据库 schema 版本。
