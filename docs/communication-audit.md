# Communication 解决方案：停止 GPU 空耗，再做离线语义审计

## 现在先做什么

这一步只读取已有轨迹，**不训练、不重新生成回答、不需要 4090**。

1. 确认本地 `D:\桌面\llm-learning\sft-backup-20261001-seed7-step150\checkpoint-150` 已备份。权重、训练状态、分词器和配置均存在，权重 SHA256 为 `19DEB58F61E459AC1A7345866E33A234B397E4B4C6998CAAF9CADBF621FDC622`。这是 LoRA adapter，不是完整基础模型。
2. 100/200 的评测报告在本地，但未确认其权重已下载；如需保留，请先下载再归还实例。
3. 到租赁平台按你所用套餐的规则停止租用，并确认账单停止计费。关闭 SSH、停止 Python、GPU 利用率为零都不等于退租；不要只看 `nvidia-smi`。未备份的实例本地文件可能在归还后丢失。

当前仍以 checkpoint-150 为候选，冻结 v11 报告，不延长训练、不进入 RL。独立裁判先做旁路诊断；它不能直接替代未经校准的正式 reward。

## 本地一条命令预检查（免费）

在 **Windows PowerShell**，不是云服务器终端，执行：

```powershell
& 'D:\桌面\llm-learning\ecommerce-customer-service-agent-rl\scripts\run_communication_audit.ps1'
```

预期 `dry_run: true`、`selected_requests: 80`、`affordable_requests: 80`、预留估算 `3.6099700` 元。此命令即使环境中已有密钥也不会联网、不会扣费，也不创建新的结果目录。

已准备的任务在 `D:\桌面\llm-learning\ecommerce-customer-service-agent-rl\outputs\communication-audit-v1\prepared`：840 条 Base/100/150/200 的 dev 记录，去重后 560 个请求，0 条缺标准记录。相同回答只审计一次；原记录索引保留。第一批 80 个请求按七类标准轮换选择，不只挑旧评分失败项。

## 确认预算、地域及数据传输后才执行

执行会把任务要求、模拟对话、工具记录和最终回答发送到阿里云百炼。不会发送权重、API 密钥到输出文件，裁判输入不含模型名、checkpoint、旧成绩或隐藏任务元数据。仍须确认数据允许发送给服务商。

固定裁判：`qwen3-max-2026-01-23`，非思考模式。默认费用按北京地域、输入不超过 32K 的档位计算：输入 2.5 元/百万 token，输出 10 元/百万 token。其他地域价格不同，必须显式修改。价格来自[百炼官方价格表](https://help.aliyun.com/en/model-studio/model-pricing)，执行前以账户控制台实际计费为准；模式说明见[官方文档](https://help.aliyun.com/zh/model-studio/deep-thinking)。

确认后，在本地 PowerShell 设置地域和进程环境密钥：

```powershell
$env:BAILIAN_API_BASE = 'https://dashscope.aliyuncs.com/compatible-mode/v1'
# 如果环境已有正确的 DASHSCOPE_API_KEY，可跳过下面两行；不要把密钥发到聊天中。
$taskAuditSecret = Read-Host '输入百炼 API Key' -AsSecureString
$env:DASHSCOPE_API_KEY = [System.Net.NetworkCredential]::new('', $taskAuditSecret).Password
# 不让其他服务商的 OPENAI_API_KEY 优先于百炼密钥；仅修改当前终端环境。
Remove-Item Env:OPENAI_API_KEY -ErrorAction SilentlyContinue
& 'D:\桌面\llm-learning\ecommerce-customer-service-agent-rl\scripts\run_communication_audit.ps1' -Execute -BudgetYuan 5 -Limit 80
Remove-Item Env:DASHSCOPE_API_KEY -ErrorAction SilentlyContinue
```

**5 元是客户端预算预留，不是服务商硬限额。** 输入按 UTF-8 字节数加余量、输出按上限 2048 token 保守预留；跨档位输入被拒绝。超预算不发送下一条请求；请求超时可能已经计费，所以不自动重试，并保留部分结果。API 返回型号不匹配、用量超预估时停止。不要直接重复运行整批，以免重复计费；先查看输出目录的 `summary.json`。如要严格账户限额，请同时使用服务商支持的账户预算控制。

结果生成在 `outputs\communication-audit-v1\api-时间戳`，包含原始裁判输出、费用用量、逐项证据、待审核项和旁路报告。旧 rollout、reward、v11 报告不会覆盖。

## 判定方式与新情境

代码继续检查状态、工具、政策和精确数值；独立语义裁判检查沟通。每条标准只能返回 `met/unmet/uncertain`；通过必须引用最终回答原文。缺项、伪造引用、无效 JSON、截断输出、信息不足都进入待审，不默认成功。

现有任务的 32 个 intent 已转换成语义标准，另有五项通用要求：回应请求、事实有依据、结果一致、操作表述诚实、理由有依据。不是继续为每种说法补关键词。

新业务情境无需先出现在模型训练集，但**必须有经审核的业务政策和任务要求**。在任务文件 metadata 中添加，例如：

```json
{
  "communication_contract": {
    "reviewed": true,
    "version": "exchange-stock-policy-v1",
    "policy_facts": ["库存为零时不能承诺完成换货；可说明缺货并提供政策允许的下一步。"],
    "requirements": [
      {"id": "explain_stock_shortage", "description": "说明库存不足、尚未完成换货，不虚构到货日期。"}
    ]
  }
}
```

`reviewed` 必须由维护任务标准的人确认，不能由被测模型自报。未知 intent、空要求、未审核契约标为 `not_evaluable`；这表示无法评分，不代表模型失败或通过。该接口解决新情境的评分表达，不保证模型具备该情境能力。

## 后续只按这一条路线推进

1. 第一批 80 条仅做旁路审计。人工先在看不到裁判结论的情况下标注一部分，再对照裁判，重点检查真实错误被误放行、正确改述被漏判、否定/数量/对象/历史事实混淆。证据格式检查通过不等于语义裁判正确。
2. 在独立保留的一组轨迹上验证裁判标准；明确误放行、漏判和待审率，冻结裁判版本后再决定是否正式采用。当前报告始终标记 `calibrated: false`，不是已完成校准的 benchmark。
3. 将真正的模型失败与评分错误分开。需要改模型时只补充 train 情境的正确示范和反例；不能把 dev/test 回答直接混入训练。
4. 数据、训练命令和评测标准都准备好以后才重新租 GPU，做短程实验和真实任务对比。沟通旁路高分不能绕过工具/政策失败，也不能单独作为 RL 开跑条件。

代码已完成 CPU 自动化回归。**尚未执行真实付费裁判调用，也尚未证明裁判泛化质量；没有生成新的正式模型成绩。**
