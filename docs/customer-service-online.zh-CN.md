# 电商售后 Agent 在线工程原型

本次实现把独立任务评测扩展成持续会话、共享业务后台和可靠取消订单。原有 SFT、GRPO、v18 控制器与离线评测文件保持原样。

## 已实现与边界

- 会话、用户消息、模型工具轨迹、操作预览、确认和执行结果持久化到 SQLite。
- Bearer token 映射服务端用户身份；用户不能通过请求体指定 customer_id。
- 订单查询、取消预览、用户按钮确认、取消执行、原操作结果查询。
- 确认绑定用户、会话、订单、后台计算的退款金额、订单版本和五分钟有效期。
- 新用户消息或新预览会作废当前会话旧的待确认操作。自然语言“确认”不会执行写操作。
- 业务后台再次校验订单归属和版本；稳定 operation_id 实现幂等。
- 写操作超时后标记 unknown，只查询原操作。unknown/submitted 阻止同一订单在其他会话重复发起。
- 服务重启后保留会话和确认；中断回合标记失败，中断写操作进入查结果流程。
- 转人工会停用当前会话自动操作；这里只实现本地待人工状态和审计，未接企业工单系统。
- 支持 OpenAI-compatible chat completions 模型接口，以及有明确契约的 HTTP 业务适配器。

这是在线工程原型，不是生产上线结果。默认共享模拟订单；取消真的改变本地业务库，但没有调用支付机构。模拟退款保持 pending，不会伪称实际到账。

本次没有重新训练模型，没有真实订单/支付接口凭证，也没有运行真实模型质量评测。在线工具协议和身份来源与训练时不同，因此不能沿用 v18 的 98.17% 成功率。模型文字仍可能有事实错误；操作卡片和后台返回结果才是业务执行依据。

## 快速启动：无需 GPU 的业务验证模式

在项目根目录运行：

```powershell
python scripts/run_customer_service.py
```

浏览器打开 `http://127.0.0.1:8766`。首次启动创建：

```text
outputs/customer-service/
  demo-auth.json     本地演示访问令牌，切勿提交或分享
  sessions.sqlite   会话、消息、操作和审计
  business.sqlite   共享订单、取消结果、模拟退款
  service.lock      同一数据目录只能运行一个服务进程
```

网页自动连接“小林”演示账号，使用 HttpOnly/SameSite cookie 处理身份；不再要求用户复制令牌或输入会话 ID。右上角可切换“小陈”。自动演示登录仅在默认模拟后台、默认凭证及 loopback 监听时启用，不能用作企业登录。Bearer token 仍供开发接口测试使用，不能提交或分享 `demo-auth.json`。

1. 打开网页，系统自动恢复上一次会话；“重新开始”创建新的咨询。
2. 查看“我的订单”。小林（Alice）可见 order_A、order_B、order_C；小陈（Bob）可见 order_D。
3. 在 order_A 卡片点击“申请取消”。这时订单仍是 paid。
4. 核对订单和金额，点击处理进度卡片中的“确认取消”。
5. 订单变为 cancelled，退款记录为 pending。再次确认同一操作不会产生另一笔退款。
6. 停止并重启服务，刷新网页；系统重新连接演示账号并恢复会话。

order_C 已发货，不允许取消。输入 Bob 的 order_D，Alice 无权读取或取消。

没有配置模型时，页面明确提示“智能客服尚未连接”，聊天发送按钮停用；用户可继续通过订单卡片验证业务流程。没有脚本 Agent 冒充模型。若要重新做首次取消演示，指定新的 `--data-dir`；重启原目录不会重置已取消的订单。

## 接入实际模型

先在独立推理服务中加载基座或 SFT 适配器。这里不会自动下载模型、训练或加载 LoRA；`--model` 必须是推理服务实际暴露的模型/适配器名称。

```powershell
$env:MODEL_API_KEY = '你的模型服务密钥'
python scripts/run_customer_service.py --mode model --api-base http://127.0.0.1:8000/v1 --model YOUR_SERVED_MODEL
```

请求使用非流式 `/chat/completions`，发送真实多轮 messages 和四个工具的 schemas：

| 工具 | 参数 | 行为 |
|---|---|---|
| list_orders | 无 | 查询当前认证用户的订单 |
| get_order | order_id | 查询订单与退款状态 |
| preview_cancel | order_id | 生成待确认操作，不取消订单 |
| escalate_to_human | reason | 当前会话进入本地待人工状态 |

模型不接收身份令牌，不提供确认接口，也不能执行 cancel_order。实际写操作只能由经认证的确认 HTTP 请求触发。伪造工具名、添加 customer_id/confirmed 等参数会被拒绝。

模型故障会标记本回合失败；不会自动切到脚本 Agent，也不会替用户填写确认。使用新的 request_id 继续。同一 request_id 重试只返回已经保存的响应或“处理中”，不同文本复用 ID 被拒绝。

第一版不自动压缩长历史：超过上下文保护阈值应新建会话，避免截断工具配对和授权信息。

## 用户 API

业务 API 需要 Bearer token 或演示登录生成的有效 cookie。网页、`/healthz`、`/api/ui-config` 为公开入口；`POST /api/demo-login` 仅在明确启用的本地演示模式可用，入参 `{"account":"alice"}` 或 `{"account":"bob"}`。跨来源 POST 被拒绝，cookie 一小时有效；网页重启后重新登录并按演示账号恢复历史。

| 方法 | 路径 | 请求体 |
|---|---|---|
| POST | /api/conversations | `{}` |
| GET | /api/orders | 无 |
| GET | /api/conversations/{id} | 无 |
| POST | /api/conversations/{id}/messages | `{"request_id":"客户端稳定ID","text":"用户消息"}` |
| POST | /api/conversations/{id}/preview-cancel | `{"order_id":"order_A"}` |
| POST | /api/operations/{id}/confirm | `{}` |
| GET | /api/operations/{id} | 无；unknown/submitted 时查询原后台结果 |

operation_id 是服务端生成的业务操作号，与模型 tool-call ID 不同。预览请求本身不执行写操作；再次预览会生成新操作并作废当前会话旧预览。

操作状态：pending_confirmation → executing → succeeded/failed/submitted/unknown。过期或换目标的预览为 expired/superseded。succeeded 只表示取消操作成功，不必然表示退款到账。

## HTTP 业务后台契约

`HttpBusinessBackend` 是可替换接口适配器，不代表已经接入任何真实企业。企业接口必须支持服务端身份校验、稳定幂等键、订单版本校验和持久结果查询；已有接口不符合时，应修改这一适配器。

```powershell
$env:BUSINESS_API_KEY = '业务服务密钥'
python scripts/run_customer_service.py --backend http --business-base https://YOUR_BUSINESS_SERVICE --auth-file PATH_TO_TOKEN_MAPPING --mode model --model YOUR_SERVED_MODEL
```

所有业务请求携带服务密钥。客户 ID 是会话服务注入的数据，企业服务仍须验证服务调用权限、客户及订单归属，不能只信任请求体。

| 接口 | 约定 |
|---|---|
| GET /customers/{customer}/orders | `{"orders":[{"id":"...","customer_id":"...",...}]}` |
| GET /customers/{customer}/orders/{order} | `{"order":{"id":"...","customer_id":"...",...},"refund":...}` |
| POST /cancellations/preview | 入参 customer_id、order_id；返回 order_id、order_version、refund_cents |
| POST /cancellations | 入参 customer_id、order_id、operation_id、expected_version |
| GET /cancellations/{operation}?customer_id=... | 返回该操作持久结果；404 暂无记录不授权重试 |

取消成功回执示例：

```json
{
  "operation_id": "op_...",
  "order_id": "order_A",
  "status": "succeeded",
  "order_status": "cancelled",
  "refund_id": "refund_...",
  "refund_cents": 19900,
  "refund_status": "pending"
}
```

status 可为 pending、succeeded、failed。已受理未完成时使用 pending；业务明确拒绝应以 HTTP 400/403/404/409/422 返回 `{"status":"rejected","code":"..."}`。其他错误和无法验证的回执均保守视为结果未知。

同一 operation_id、同一参数重复请求必须返回原结果；同一 ID 不同参数必须拒绝。订单归属、状态、版本、金额及取消写入必须在后台可靠地校验和执行。本地事务不能替代远端幂等。

没有查回结果时不能重新创建操作。目前没有自动重试 unknown，也没有企业退款回调、支付对账和人工核查界面；这些是生产接入前的后续工作。

## 代码导航

```text
src/ecommerce_customer_service/serving/
  storage.py   持久化、事务、会话归属和中断回合恢复
  backend.py   共享模拟后台、HTTP 契约适配器
  model.py     模型接口、在线提示词和工具 schemas
  service.py   回合编排、确认、幂等操作及结果查询
  api.py       HTTP 路由、认证、请求大小及并发限制
  runtime.py   跨进程数据目录锁
  index.html   对话、订单、操作卡片和恢复界面
scripts/run_customer_service.py   独立启动入口
tests/test_customer_service_serving.py   工程行为测试
```

## 测试与下一步

无需安装 pytest，可以在项目根目录运行：

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
python -m unittest discover -s tests -p test_customer_service_serving.py -v
```

测试覆盖归属校验、请求复用、确认绑定与过期、订单状态竞态、并发确认、提交成功后丢响应、未知结果阻止跨会话重做、重启恢复、工具协议配对、模型写操作越权、人工接管、HTTP 认证和模型接口传输。模型接口测试使用本地模拟 provider，不能替代真实模型评测。

工程测试增加了演示 cookie 登录、过期、非演示服务禁用自动登录和跨来源请求校验。界面已移除令牌、会话 ID、订单 ID 输入框和原始 JSON；业务操作仍按用户身份与具体操作校验。测试数据库为临时或专用验证目录，与默认演示目录分开。

本次界面更新验证：32 项 unittest 通过；无头 Chrome 验证自动登录、无需凭证字段、模型未接入提示、取消确认、退款 pending、账号隔离、刷新恢复和手机布局，无页面 JavaScript 异常。

完成本原型后，按顺序补：真实模型端到端质量测试 → 企业身份和只读接口 → 测试环境写接口联调 → 支付异步状态/核查 → 人工审核试运行。再考虑独立测试、指标监控、TLS/网关、部署、备份、压测和多实例架构。

服务默认只监听 127.0.0.1。当前单进程、小规模 SQLite 原型不提供生产 SLA，也不应直接暴露公网处理真实资金。
