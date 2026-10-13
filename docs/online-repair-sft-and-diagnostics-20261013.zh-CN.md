# 在线四工具协议：失败诊断与定向 SFT 数据准备

2026-10-13。本轮完成诊断代码和数据准备，未训练模型、修改模型权重或执行正式 test。先前实际模型的 5/15 开发通过数保持原记录；程序参考轨迹通过不能算作模型提升。

## 为什么要做这一步

已有模型在旧九工具协议上训练，现在网页采用四工具及界面按钮确认协议。之前在线轨迹显示：模型查到多个订单后，经常把无关商品也当成目标候选，缺少必要的取消预览。提示词对照没有给出可靠改善；另外两组遇到通用 HTTP 422 中断，旧服务无法解释究竟是格式错误、截断还是运行异常。

本轮先补足故障证据，再构造与当前运行时一致的监督数据。暂不引入 RL；当前证据没有证明 SFT 已达到瓶颈。

## 推理诊断

| 错误码 | 含义 | 评测处理 |
|---|---|---|
| `invalid_tool_call_output` | 原生工具调用标签、JSON 或参数形态无法解析 | 该场景失败，继续后续场景 |
| `generation_truncated` | 达到生成上限，输出未完整结束 | 该场景失败，继续后续场景 |
| `context_budget_exceeded` | 实际模板输入超预算 | 停止，报告未完成 |
| `cuda_out_of_memory` | 显存不足 | 停止，报告未完成 |
| `inference_failed` | 其他推理执行异常 | 停止，报告未完成 |

分类由推理服务返回的已知结构化错误驱动。旧服务的通用 422、网络故障等未知错误仍按基础设施/未知故障停止，不能猜成模型格式问题。失败场景保留在分母中，不改旧业务断言来提高通过率。

服务端默认保存错误码、异常类型、解析原因和已知 token 数；不知道的数量为 null。通过 `--capture-invalid-output` 显式启用后，才保存失败生成原文，最多 16384 字符并保留完整原文 SHA256。不保存请求头、凭据或输入正文；原文可能复述业务信息，应留在私有输出目录。Linux 目录/文件权限为 0700/0600。原文不进入控制台、网页错误提示、普通 trace 或 Git。

错误响应携带 `diagnostic_id`；`GET /v1/diagnostics/<id>` 沿用推理服务鉴权。当前采用 loopback + SSH 转发；开启公网绑定仍要求原有 API key。普通 trace 只存关联 ID、状态和 token 数，网页显示“这次回复没有完成操作”，不会执行格式不完整的工具请求。

## 定向数据的来源与构造

全部来自程序构造的订单、用户表达和参考动作，是合成数据，不是实际客户日志，也不是训练模型采样。构造器通过当前 CustomerService、ContextBuilder、四工具 schema 和隔离 SQLite 业务环境执行参考轨迹，再抓取每次模型应收到的上下文与下一步目标。训练时不会把期望数据库状态、场景标签或后续参考答案塞进模型输入。

覆盖 15 类场景：精确商品名、简称、无匹配、同名多单按金额澄清、只查询、已发货、未付款、撤回、换目标、聊天确认、越权订单、只读失败重试、按钮后的结果查询、先查询再取消、转人工。

数据同时改变本人订单数量和目标匹配数量。干扰商品不会意外变成第二个目标；换目标时新商品也保持唯一。订单 ID 随机化，工具列表按 ID 排序，目标不固定在列表第一项；金额顺序轮换，歧义澄清既有高价目标也有低价目标。取消后的退款状态是 pending，不能标成到账。

采用 `binding_compact_v3` 作为明确记录的训练协议；这是候选训练协议选择，不是已证明该提示词能提高旧模型效果，也没有切换现有网页的默认 baseline。

| split | 场景数 | 监督样本数 | 校验/执行情况 |
|---|---:|---:|---|
| train | 120 | 512 | 参考轨迹的业务状态、工具、token 与 loss mask 校验通过 |
| dev | 30 | 128 | 同上，用于后续训练开发检查 |
| synthetic test | 30 | 0 | 先冻结案例文件，未回放、未推理、未用于监督 |

每条监督样本只计算最后一个 assistant 回答或工具请求的损失。用户输入、工具返回、历史回答及服务端按钮结果不计算损失。最后的工具请求可以没有未来结果；前面的工具请求必须配对完成，不允许借这个例外隐藏残缺历史或重复调用 ID。原有完整轨迹格式要求保持。

原生 Qwen 模板渲染后，逐条解码监督区域，并核对回答正文、工具名和参数；限制输入不超过 8192、监督目标不超过 512 token。训练和推理使用同一 tokenizer 文件指纹。兼容 Transformers 4 的 IDs 列表和 Transformers 5 的 BatchEncoding，避免把字典键当 token。

商品目录、订单 ID、主取消请求模板在 split 间隔离；但场景类型和部分查询/澄清/确认续话模板共享。它是一个小规模协议修复数据包，不能当作生产分布，也不能声称全部表达模板隔离。合成 test 未运行，不等于已经具备独立真实业务测试证据；训练前仍需复核覆盖面，并规划旧能力回归及真实表达留出集。

## 文件与复现

本机最终数据位于 `outputs/online-repair-sft-20261013-r3`。`train.jsonl`、`dev.jsonl` 是训练前缀；`*-cases.jsonl` 是业务案例；`manifest.json` 记录数量、文件 SHA256、tokenizer 与源码指纹、策略协议、未训练/未评测 test 的事实。生成 payload 位于 ignored 输出目录，不上传 Git。早期 r1/r2 留作构造过程记录，后续使用 r3。

```powershell
python scripts/build_online_repair_sft.py --output outputs/NEW_REPAIR_DATA --tokenizer-path models/v18-sft-verified
python scripts/train_sft.py --train outputs/NEW_REPAIR_DATA/train.jsonl --dev outputs/NEW_REPAIR_DATA/dev.jsonl --output outputs/NEW_VALIDATION --validate-data-only
```

第二条命令只检查数据，不加载 GPU 或训练。目录已存在时构造器拒绝覆盖。

单次失败诊断工具不创建 CustomerService，也不执行返回的业务工具：

```powershell
python scripts/replay_online_failure.py --case-dir outputs/online-policy-factorial-20261013-r2/binding_compact_v3/cases/disambiguate_order --output outputs/NEW_FAILURE_DIAGNOSTIC
```

历史请求没有保存完整 payload，因此这里明确标成“基于失败后快照重建”，不是证明字节完全相同的原始回放。有业务操作的案例拒绝重建，要求提供明确来源的请求 JSON。输出私有 `request.json` 后，可用 `--request-json` 再发送同一 payload。成功回复也只保存，不调用业务工具。

工具先检查 `/healthz` 的诊断版本。此次实际检查确认远端仍为旧服务，返回 `server_update_required`，新增 GPU 请求为 0、业务工具执行为 false。服务器更新并启动后才能取到实际失败原文；现在不能声称已查明历史 422 的具体根因。

## 部署及下一阶段

代码包保存在 `packages/v18-online-diagnostics-20261013-r1`，不包含模型权重、凭据、业务数据库或生成数据。该目录的部署说明提供准确的 SCP、解包校验和启动命令；继续复用实例原有基础权重、SFT adapter 和 Python 环境。需要在模型终端先 Ctrl+C，再启动新版；保留本机 SSH 转发窗口。

此次远端 SSH 公钥尚未获授权，自动登录不可用；因此没有宣称服务器已更新。本机网页当前进程也未因源码变化自动升级。远端启动后，应检查健康信息、重启本机服务加载新错误分类，再只回放一个已知失败请求。接着才决定是否补 SFT，以及采用哪种训练预算。

后续补 SFT 必须生成新的权重身份及协议校验，不能直接覆盖旧 v18 文件或绕过固定哈希验证。比较补训前后模型时固定运行协议、精度、解码和开发案例，检查新协议任务、格式失败及旧能力回退。只在选型完成后运行冻结 test；当前不报告模型质量提升。

验证：本轮相关工程回归共 122 项独立测试通过（包含原有训练数据兼容检查）；数据另经真实 tokenizer 全量校验与 trainer 的 validate-data-only。这些都是工程和参考数据证据，不是新模型 test 成绩。

## 实际部署与单次诊断更新

2026-10-13 后续：用户在 Featurize 上传并启动 `8ddaef5` 代码包、恢复 SSH 转发后，实际健康检查确认 `ready=true`、`diagnostics_version=inference_diagnostics_v1`、诊断及失败原文捕获均开启，仍为原 v18 adapter、BF16。之前“尚未部署”的段落保留为准备阶段记录，目前新版已经生效。

发送一次之前保存的重建请求（用户先说取消一笔订单，再明确取消键盘）后，实际得到 HTTP 422 `invalid_tool_call_output`，解析原因 `unbalanced_tool_call_tags`。输入 1026 token，输出 22 token，finish_reason=stop。原文包含 `preview_cancel` 和正确的键盘订单 ID，但只有结尾 `</tool_call>`、缺少开头 `<tool_call>`，前缀还有两个 Unicode replacement character。严格解析拒绝该输出；没有执行任何业务工具。完整原文仅在私有 `outputs/inference-failure-diagnostic-20261013-r2` 中保存。

因此这一例已能区分为原生工具格式失败，不能归因为输出达到 512 token 上限、上下文超长或显存不足。原始历史输入未精确捕获，仍只声称本次保存的重建 payload 复现了格式拒绝；不能据此证明全部历史失败具有相同根因，也不能推断异常字符的内部成因。不会自动补标签并执行未经完整解析的请求。

本机网页服务已备份既有两份业务/会话 SQLite 并更新客户端。重启前 running turns、executing operations 均为 0；重启 recovery 三项均为 0，继续使用原数据目录和 baseline_v1。启动阶段的只读真实工具探针通过，网页 HTTP 200，已加载新的格式失败提示。重启会使内存中的演示登录 cookie 失效，刷新页面后如需登录，重新选择演示账号即可；会话和订单数据保留。

本次增加一次失败诊断生成和一次启动探针，未训练、未换权重、未执行正式 test，也没有新的完整开发集成绩。下一步可围绕候选绑定与原生格式做小规模补 SFT，随后固定协议比较开发结果及旧能力回退。
