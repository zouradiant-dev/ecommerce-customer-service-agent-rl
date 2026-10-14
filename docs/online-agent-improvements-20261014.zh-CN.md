# v18 在线 Agent 工程改进

本次围绕原 v18 与在线补训版本增加工程能力；没有更新权重、租用 GPU、重新执行真实模型评测或修改历史成绩。

## 订单选择和预览状态

新配置 `binding_guard_v4` 在模型请求取消预览前，从后台读取当前账号订单，按实际用户消息匹配商品、已登记别名、订单号或基于上一条取消需求的金额澄清。无匹配、多匹配、模型订单号与匹配结果不一致、核验后订单版本变化均拒绝创建预览。目标核验结果写入审计。

该校验是有限语法，不是通用语义理解。未支持的复杂表达会要求用户明确商品或订单号，不能声称已经解决所有自然语言绑定问题。后续应以真实请求的覆盖率和错误拒绝率评估。读取接口需要完整本人订单列表、商品名称与版本；分页企业接口需要先扩展适配器。

预览仍不执行取消。每条新用户消息仍使旧确认失效；没有有效预览时不会沿用“已有预览，请点击确认”的回复。过期卡片按 expired 展示。账户归属、按钮确认、金额和版本绑定、幂等以及 unknown 查询保留。

网页启动入口默认选新配置；历史评测入口默认仍是 baseline_v1，原 binding_compact_v3 成绩不变。选择旧配置不会启用新增目标匹配校验。

## MCP：只读标准接口

新增官方 Python SDK 1.26.0 stdio gateway，提供 list_orders、get_order、search_policies。一个进程绑定一个启动配置里的账号令牌，工具参数不能指定 customer_id。参数集合严格校验，令牌不输出到标准输出，gateway 仅访问本机 HTTP 服务且不跟随重定向。

没有提供取消、确认或 preview_cancel 工具，防止外部 MCP 客户端绕过在线运行时的用户需求绑定。它不是网页模型已经改走 MCP，也不是企业 OAuth 或多租户远程 MCP 部署。真实协议测试使用官方 ClientSession 启动服务端、发现工具、查询本人订单、拒绝其他账号订单以及未知写工具和身份参数。

官方 API 参考：[Python SDK v1.26.0](https://github.com/modelcontextprotocol/python-sdk/blob/v1.26.0/README.md)。

```powershell
python -m pip install -e ".[mcp]"
python scripts/run_customer_service.py --data-dir outputs/customer-service-v4 --knowledge-file data/demo-policies.jsonl
```

上述启动为无 GPU 的手动业务模式，知识 HTTP/MCP 接口可用，聊天仍须实际模型。另一个终端从本机 auth 文件选择自己账号的令牌放到 CUSTOMER_SERVICE_MCP_TOKEN 环境变量，不把令牌发给模型或提交 Git：

```powershell
python scripts/run_customer_service_mcp.py --api-base http://127.0.0.1:8766
```

stdio 服务应由 MCP 客户端启动或连接；它不是普通网页，也不会打印操作成功的假聊天。

## 知识检索与模型回答入口

`--knowledge-file` 可加载带 ID、版本、有效期、来源和 demo_only 标签的 JSONL。重叠生效版本被拒绝，过期内容过滤。检索采用 BM25 与中文双字词项，返回引用 ID、来源、版本和索引哈希；不使用向量库、embedding、reranker 或长期记忆。

配置知识后，模型多一个 search_policies 工具。无检索证据时给出缺少依据的回复；模型缺少有效引用或引用不存在时，改为返回检索证据。demo_only 内容明确标为演示。引用检查只验证来源 ID 存在，不证明每句话都由来源蕴含，语义幻觉仍需人工复核或独立回答评测。

`data/demo-policies.jsonl` 是根据本仓库模拟后台行为编写的三份演示说明，不是商家真实政策。当前 v18 及补训数据仍是原四工具协议，新增知识工具尚未做真实模型质量验证。

```powershell
python scripts/evaluate_policy_retrieval.py --knowledge-file data/demo-policies.jsonl --cases data/policy-retrieval-dev-v1.jsonl --as-of 2026-10-14 --output outputs/policy-retrieval-dev.json
```

四个小型合成 dev 查询只验证检索入口、来源与无证据行为，不能把它的指标当成 RAG 回答准确率或检索泛化证据。

## 防重叠与训练长度检查

数据生成器按实际 Qwen 模板渲染最后监督目标之前的输入，比较 token ID 哈希。发现 train/dev 重叠会阻止发布数据；sealed test 只审计首条用户输入，不执行参考策略或监督 test。转人工话术已经按 split 分开，既有 r3 数据不重写。

新增最大训练序列长度检查，默认 2304，溢出立即报错，不允许生成 ready 的训练 manifest。`--policy-profile` 明确记录数据对应的运行时配置。新数据仍是规则参考轨迹，不能当成真实客户日志。完整输入去重也不消除共享场景族、共享后续表达等统计依赖。

```powershell
python scripts/build_online_repair_sft.py --output outputs/new-v4-data --tokenizer-path models/v18-sft-verified --policy-profile binding_guard_v4 --max-sequence-length 2304
```

首次长提示词草稿最长 2333，已标为 training_ready=false 并退役；采用缩短后的新提示词重新生成数据。实际生成记录保留在 ignored outputs 中。旧 pilot 的冻结提示词和数据检查不能直接复用到新配置，下一次 GPU 实验必须使用匹配的新协议和收据。

## 追踪与复核

检索追踪只记录索引哈希和命中数，不保存 query 文本到 span。现有模型、工具、后台与确认 trace 继续有效。

```powershell
python scripts/report_online_quality.py --run PATH_TO_DEV_RUN --output outputs/review-queue.json
python scripts/report_online_traces.py --database PATH_TO_SESSIONS_SQLITE --output outputs/traces.json
```

复核队列包含失败类别和原始证据文件位置，默认 reviewed=false；未进行的人工审核不会算完成。没有自动 Monitor Agent、线上告警、在线 A/B 或推断出的生产任务成功率。

## 验证结果与下一阶段

已核验原始备份并复核六题轨迹。对六个失败案例的录制回复做新运行时回放：三个无匹配案例的错误预览被阻止，两笔书桌首轮歧义被阻止；聊天确认和牙刷澄清仍缺少新预览，任务未通过，但失效预览回复不再冒充有效操作。这是工程反事实回放，模型不能响应新工具错误，不能当成新模型成功率或把历史 51/57 改成更高分数。

验证包括旧业务与确认测试、新目标校验、知识来源和有效期、输入重叠、官方 MCP 实际 stdio 通信，以及程序参考数据执行。具体本机运行收据见 outputs/online-guard-20261014；真实模型端到端、新知识回答、独立 test 与旧九工具回归尚未运行。

下一次使用 GPU 先固定协议、用例和数据版本，对原模型与补训模型分别做新配置评测。运行时、提示词与权重变更分开比较；先确认边界质量和新知识工具选择，再决定是否补 SFT。当前未进入 RL。

多 Agent routing 暂不引入：现阶段售后操作与知识工具由同一 Agent 调用。未来出现独立物流、政策和售后职责时，再定义结构化路由输入、权限、失败回退和任务级配对对照，不能把普通工具选择描述为多个 Agent 协作。
