# 多轮上下文、端到端回归与调用追踪

更新日期：2026-10-13。

本次补充三个可运行模块：`context_builder.py`、`online_evaluation.py`、`telemetry.py`，并接入已有在线服务。没有租用 GPU、没有重新训练、没有执行真实 SFT 推理；原有 v18 数据和训练评测入口不变。MCP、RAG、多 Agent routing 不在本次实现范围。

## 1. 上下文管理

默认 `structured` 策略构造：系统提示 + 结构化业务状态/有限历史摘录 + 最近六个完整用户回合。

- 历史摘录是最近若干旧用户请求的限长原文，不是模型生成的语义摘要，也不保证保留所有旧信息。
- 当前操作状态来自持久化 operation 表，包括操作 ID、目标订单、金额、版本、有效期；过期确认在上下文中标记 expired。
- 历史订单事实只从成功 list_orders/get_order 结果提取白名单字段，保留来源、观察时间，明确标注可能过时；最多保留十个订单观察。
- 裁剪按完整用户回合进行，工具调用和返回不能拆开。当前用户回合、有效操作状态无法容纳时返回错误，不静默截断。
- 每次模型调用前都重新构造并检查预算，而不是只检查第一次调用。
- 历史消息保留在数据库中；仅模型输入裁剪。失败回合的残缺工具协议不会进入下一轮模型输入。
- 不从摘要授予身份或操作权限；写操作仍须服务端确认及后台最新状态校验。
- 本次没有长期用户画像或跨会话向量记忆。

目前仍读取该会话历史后构造输入；更大规模需要进一步做数据库分页和摘要缓存。有限旧请求摘录不是完整语义记忆，应通过真实长对话实验验证信息遗漏。

无 tokenizer 时使用明确标注的 UTF-8 保守估算，不能当作精确 token 数；它可能较早拒绝中文长输入。提供本地 tokenizer 才按聊天模板和工具 schemas 计数，且应与推理服务使用的模板一致。不会自动下载 tokenizer。

```powershell
python scripts/run_customer_service.py --context-budget 8192 --recent-turns 6
```

真实模型配置示例：

```powershell
python scripts/run_customer_service.py --mode model --api-base http://127.0.0.1:8000/v1 --model YOUR_SFT_SERVED_NAME --tokenizer-path PATH_TO_LOCAL_TOKENIZER --context-budget 8192
```

输入预算须给新生成 token 留余量；本项目模型请求默认最多生成 512 token。模板、模型上下文上限和窗口大小需要在联调时核对。

`--context-policy full-history` 保留全历史，不注入结构化工作记忆，也不会自动压缩；超过同一预算则失败。该模式用于与 structured 对照，不应为了提升得分而随意改变其他参数。

## 2. 多轮端到端开发回归

`data/online-agent-cases-v1.jsonl` 包含十五个程序构造的 dev 用例：基础取消、澄清订单、改目标、撤回确认、已发货订单、跨用户订单、状态竞态、确认过期、重复确认、提交后丢响应、结果未知、重启、长历史指代、转人工、一次读取超时。

每个用例有用户回合、显式确认/查询事件、环境事件、期望数据库状态和禁止回复片段。确认事件点击的是 Agent 实际创建的操作；若 Agent 没创建预览，验证失败，不替 Agent 自动调用预览。期望标签与 smoke fixture 不进入真实模型上下文。

先在本地运行工程 smoke：

```powershell
python scripts/evaluate_online_agent.py --mode smoke --output outputs/online-agent-smoke-20261013
```

Smoke 使用明确标注的脚本测试替身，只验证运行时和验证器。十五题全部通过不能表述成 SFT 模型成功率 100%。

真实模型接好后，同一评测入口可以调用实际模型：

```powershell
$env:MODEL_API_KEY = '模型服务密钥'
python scripts/evaluate_online_agent.py --mode model --model YOUR_SFT_SERVED_NAME --api-base http://127.0.0.1:8000/v1 --tokenizer-path PATH_TO_LOCAL_TOKENIZER --output outputs/online-agent-sft-structured
```

对照时保持用例、模型、解码、确认语义及业务校验不变，只切换 `--context-policy full-history` 或 structured。基座与 SFT 的比较则保持上下文策略不变、只更换实际服务模型。当前脚本单次贪心执行；多随机种子、配对置信区间、独立冻结测试和自适应用户模拟器属于后续实验。

每次运行输出：

```text
protocol.json        模式、模型名、用例/提示词/工具/代码哈希、上下文配置
summary.json         是否完整、用例通过情况、分类失败与调用数
report.md            可读结果表
bad_cases.jsonl      失败用例和定位信息
cases/{case_id}/
  sessions.sqlite    独立会话、操作、审计、trace
  business.sqlite    独立共享业务后台状态
  events.json        每步输入事件对应的结果或错误
  history.json       完整会话和操作轨迹
  spans.json         调用追踪
  result.json        状态验证及失败原因
```

输出目录必须是新目录，不会覆盖历史实验。当前 runner 只接受 dev，拒绝混入 test 标签；十五题是人工可检查的合成回归集合，不是独立泛化证据。首次真实模型调用异常时会停止后续用例，并将结果标为 incomplete、通过率 null，避免把不完整评测当完整成绩。

回复评测采用状态断言和有限片段检查，不是完整语义判分；否定表达、同义表达和复杂订单归属描述仍可能漏判或误判，应抽样人工复核。用户对话采用固定回合脚本，不能声称已实现自适应模拟用户。

## 3. 调用追踪与诊断

`spans` 表自动创建在 sessions.sqlite，不需要修改已有业务表。层次包括 HTTP、用户回合、上下文构造、模型调用、工具调用、业务后台调用和确认操作。

trace_id/parent_id 关联一次请求内部调用；conversation_id、request_id、operation_id 关联跨回合业务流程。记录时长、状态、错误类别、模型名、提示词/工具哈希、上下文裁剪量及 provider 返回的 token 用量。缺失 token 用量记为未知，不编造为零。

追踪表使用字段白名单，不保存原始用户文本、工具 payload 或密钥。原有会话和业务审计仍保存完整内容以供回放，真实数据接入后应单独处理访问权限、保留期限和脱敏。

```powershell
python scripts/report_online_traces.py --database outputs/customer-service/sessions.sqlite --output outputs/online-traces-20261013.json
```

报告只读数据库，包括调用分布、P50/P95 耗时、分类错误、未完成 span 与 token 覆盖率。未完成 span 在服务启动恢复时标记 interrupted，不伪造完成耗时。观测写入失败会告警，不改变业务执行结果。

“工具调用被拒绝”在某些安全用例中是正确行为，所以 trace 错误率不能直接当任务失败率。任务成功须由独立期望状态判定。当前没有自动告警平台、模型价格计费、在线 A/B 发布或 Monitor Agent 自动判分。

## 4. 验证与继续迭代

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
python -m unittest discover -s tests -p test_customer_service_serving.py -v
python -m unittest discover -s tests -p test_online_agent_iteration.py -v
```

新增测试验证完整工具配对、当前回合溢出、tokenizer 计数接口、摘要不授权、上下文对照、长历史续接、token 未知值、错误分层、trace 脱敏、期望标签隔离、错误成功回复、不完整评测、输出保护和数据 split 约束。

本次本地验证：原有 32 项工程测试 + 新增 18 项测试，共 50 项通过；十五个 smoke 回归用例通过。端到端回归调用服务运行时，不经过浏览器，也不连接企业业务；HTTP 行为由原有工程测试另外覆盖。这些结果不能作为真实 SFT 模型准确率。

接真实 SFT 后，先从 bad_cases.jsonl 和 spans.json 判断问题属于上下文丢失、模型工具选择、参数错误、业务拒绝还是回复事实错误，再决定改数据、上下文、工具或训练。不要把本次 smoke 成绩、历史 v18 开发成绩和未来在线模型成绩混为一个指标。
