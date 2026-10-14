# 4 步冒烟后训练顺序验收失败的恢复

## 已确认的原因

服务器在 `current-instance-closeout-20261014.erdgpT` 完成 4 步冒烟，随后抛出 `training_gate_failed`，当时尚未启动 64 步 pilot。目录末尾以已下载的导出收据为准，先前从截图抄录的 `erdgp7` 不准确。

原训练脚本在编码后已经执行一次 `random.Random(args.seed).shuffle(train_examples)`。新增 `--shuffle-training-rows` 时又加了一次，实际顺序变成两次相同种子的排列，而冻结验收计划只定义一次排列。此前“原脚本不打散，64 步可能漏掉末尾知识样本”的解释不准确；原脚本已经打散，重复打散来自本次新增代码。

880 条样本的 ID 顺序哈希：

- 计划中的单次打散：`6fa073e0b76ebacfe51b75a006cff83ab80d321ab52eeab7ecb36a998a897284`。
- 旧执行包实际双次打散：`5546a8f04c458a6e4fce90f7258b4e9fa4bb666e77252bf47ac3e8f75a80e8f4`。

这是本机重现的代码缺陷。服务器保存的 `metrics.json`、`training-order.json` 和实际适配器还必须由恢复入口逐项校验，不能仅凭截图跳过训练检查。

## 修复和预算

1. 编码与训练顺序准备统一到 `prepare_training_examples`，只打散一次；旧调用默认行为保留。兼容参数不再重复打散。
2. 使用真实 Qwen tokenizer 对 880 train、148 dev 重新编码：无跳过；实际训练顺序与单次打散冻结计划一致，前 512 条类别计数逐项一致。
3. 保留原冻结源码、旧模型、4 步结果、原始失败收据。恢复包只增加独立的修正训练入口、恢复执行入口和 CPU 校验模块。
4. 恢复必须确认：原始源码匹配已知执行包；失败确实发生在 smoke 的顺序验收；两个 dev 基线完整且模型/用例/运行时匹配；步数为 4；无跳过；初始化仍为已保存在线适配器；梯度和 loss 有限；训练参数为 FP32；实际 ID 顺序确实为已诊断的双次排列；实际 LoRA 权重有限且发生更新。
5. 仅接纳这一明确的顺序偏差。冒烟的前 32 条实际包含 4 条知识样本，并覆盖唯一目标、切换目标、聊天确认等重点场景。它用于数值和训练链路检查，不用于模型选择。
6. 通过 CPU 校验后复用两个已完成基线和这次 4 步冒烟，不追加 GPU 冒烟。64 步 pilot 从旧在线权重重新初始化优化器，不继承冒烟权重或优化器；使用计划中的单次打散顺序。
7. 后续仍是同协议 dev 对照、候选选择、一次封存合成验收、HTTP 联调与备份。出现其他指标异常则停止并保留原因，不放宽安全或数值检查。

## 恢复入口

Windows PowerShell：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "D:\桌面\llm-learning\ecommerce-customer-service-agent-rl\packages\Resume-CurrentInstanceCloseout.ps1"
```

使用当前 Featurize 实例端口 46739，保持前台连接。恢复包 SHA256 为 `519843090531793affbff81b4d28465500ecd2b9e29ee555d34fb458026a3428`，12,673 字节，不包含权重。以上保留故障当时的恢复设计；恢复现已执行并完成 64-step pilot 和后续验收，不应重跑启动命令。最终证据见 [收尾结果](current-instance-closeout-results-20261014.zh-CN.md)。
