# 在线协议补 SFT：小规模实验执行方案

当前状态：服务器已完成补训并进入候选评测，在 tokenizer 导出格式校验处中断；兼容修复与零训练步续评测补丁已准备，最终效果对照尚未完成。以下保留首次执行方案，实际进展见文末。Featurize 复用已有 4090 24GB 和原 v18 权重，不重新下载权重或升级依赖；SSH 上传/启动仍由用户在终端输入密码。

## 解决的问题与实验变量

已有在线轨迹出现过度澄清、缺少取消预览；真实单次诊断还观察到原生工具调用开标签缺失。此次补训仅检验四工具、订单绑定、界面确认及原生输出格式，不能提前声称所有历史失败都是格式原因。

从固定哈希的 v18 SFT adapter 初始化新 adapter。训练前后均使用 `binding_compact_v3`、8192 输入预算、六个最近回合、贪心解码、512 生成上限、同一 tokenizer 的词表和模板，实际推理均为 BF16。旧模型的新对照成绩会重新测量，不能拿旧 baseline_v1 的 5/15 与新模型不同提示词成绩直接比较。

## 数据与固定预算

使用已验证 r3 的 512 条训练前缀、128 条开发前缀，只计算最后一个 assistant 目标的损失。数据包仅包含 train/dev；本地冻结的合成 test 案例不上传或执行。生成器仍含合成数据构造规则，因此不能把这些合成 test 宣称为隐藏的真实业务分布。

| 设置 | 本次值 | 理由 |
|---|---|---|
| 方法 | 在旧 adapter 上继续 QLoRA NF4 | 控制 24GB 训练显存，保留原权重作为对照 |
| 冒烟检查 | 4 个 optimizer step | 检查 loss、梯度、dtype、数据跳过和实际权重变化 |
| 正式小实验 | 64 个 optimizer step | micro-batch 1 × 累积 8，恰好遍历 512 条样本一次 |
| 初始化 | 每个阶段均从原 adapter 开始 | 小实验不继承冒烟优化器或权重 |
| 学习率 | 2e-5，warmup 0.1 | 使用低学习率检验协议修复，减少一次大幅覆盖的风险 |
| 长度 | 2304，禁止截断目标 | 当前最大样本 2066 token；要求 train/dev 跳过数为 0 |
| 稳定性 | adapter/梯度 FP32，累积后缩放，非 reentrant checkpointing | 复用既有训练稳定性措施，记录实际 dtype 与有限性 |
| loss 监控 | 开发 CE：初始、32 步、64 步 | 用于诊断；不把 loss 下降等同于任务完成率上升 |

64 步是本次试验预算，不代表已经证明最优 epoch、学习率或数据量。新模型文件保存在新目录，旧权重固定哈希验证不被放宽。包中不包含基础权重、旧 adapter、凭据或实时业务数据库。

## 自动执行顺序

入口 `scripts/run_online_repair_pilot.py`，各 GPU 阶段在独立子进程中顺序执行，退出后释放显存。

1. 预检数据与哈希，使用服务器原 adapter tokenizer 重做原生格式/mask 验证，检查库版本、CUDA、BF16、24GB 显存等级、空闲显存和磁盘；不安装或升级库。
2. 旧 adapter 在 BF16 下完成固定开发对照。低成功率也完整保留；基础设施失败会中止。
3. 4 步冒烟。loss、梯度或权重更新检查不通过就停止，不自动重试 OOM。
4. 从原 adapter 和新优化器重新开始 64 步小实验；保存训练指标、checkpoint 与最终 adapter。
5. 新 adapter 建立独立 SHA256 身份及文件清单，再以 BF16 测相同案例。
6. 校验案例、运行时源码、提示词、工具、上下文、模板、词表、精度和解码一致后，输出逐题改善/回退、分组通过数、输出格式失败数及对照结果。权重完全未变化或任一阶段未完成时拒绝配对结论。
7. 启用 `--restore-original-service` 时，实验结束后恢复原 v18 推理服务，候选不自动替换网页模型。训练阶段网页模型回复暂不可用，订单/会话数据不由实验访问。

## 57 个开发案例的组成

| 组 | 数量 | 用途 |
|---|---:|---|
| 原在线回归 | 15 | 检查原有持续会话、确认、越权及恢复边界 |
| 绑定 guards | 12 | 检查商品名、改称、多单和误取消 |
| 新协议合成 dev | 30 | 检查训练目录以外的商品与不同主取消请求表达 |

参考答案只供构造/验证，发送给模型的是实际运行时上下文和工具 schema。配对开发文件剥离 smoke_responses。每题使用隔离模拟 SQLite，与网页的真实持久会话库分开。

这 57 题仍是小规模开发证据。原在线回归不等于旧九工具离线 600 题回归；本轮报告会明确 `old_offline_nine_tool_regression_performed=false`。正式晋升前还需旧离线能力、真实表达和独立测试证据。一次贪心实验不能提供多 seed 稳健性结论。

## 启动和取结果

已准备好的代码与数据包位于 `packages/online-repair-pilot-20261013-r1`，该目录的 `启动命令.md` 包含准确上传路径、归档 SHA256 和一条服务器启动命令。服务器先 Ctrl+C 停止当前推理进程，再运行实验；本机 SSH 转发保留。服务器预检不通过时先处理记录的原因，不直接修改环境重跑。

实验保留 `preflight.json`、`baseline/`、`smoke/metrics.json`、`pilot/metrics.json`、`pilot/repair-artifact.json`、`candidate/` 和 `comparison.json`。失败有 failure.json，失败原文在私有 diagnostics；普通结果不包含原文或客户输入。

恢复原服务后，同一鉴权下只读 `GET /v1/repair-experiment` 可读取固定实验摘要，包括完成状态、实际训练步数、loss/显存记录和配对结果。该接口不执行训练或业务操作，未传报告时不存在。可以通过现有 SSH 转发读取，不用在聊天中提供 SSH 密码。

CPU 验证：实际 trainer 编码 512/128 条，0 跳过、目标标签与已验证 mask 一致；67 项相关工程测试通过，包含实际数值权重变化检查，避免仅因 safetensors 序列化变化误判已训练。GPU 环境、训练是否完成、耗时、显存和新模型提升均必须以服务器执行结果为准，当前不填入训练成绩。

## 首次执行后的 tokenizer 格式兼容修复

用户后续服务器截图显示，流程已进入 candidate 评测，但在模型加载前的校验中因 `pilot/vocab.json` 缺失失败，随后恢复原服务。按执行顺序，64 步训练及其数值权重更新门槛已通过；候选任务评测尚未完成。当前本机 SSH 转发未连接，实际 loss、显存和耗时仍待读取，不从截图编造指标。

本机使用同一保存 tokenizer 在 Transformers 5 下实际复现：只输出 `tokenizer.json`、`tokenizer_config.json` 和 `chat_template.jinja`；词表与合并规则已完整包含在 tokenizer.json，不应要求旧版的两个冗余导出文件必须存在。修复后，这两个文件若存在仍会交叉校验，完整词表/ID、合并顺序、特殊 token、normalizer、编码预处理及模板检查保留。

实际保存还改写了 ByteLevel 的角色未使用字段：pre-tokenizer 的 trim_offsets，以及 decoder 的三个构造字段。根据 [Tokenizers 官方 ByteLevel 实现](https://github.com/huggingface/tokenizers/blob/main/tokenizers/src/pre_tokenizers/byte_level.rs)，前者不参与预分词，decoder 的 decode_chain 不读取后三个字段。仅对这些字段归一化，pre-tokenizer 的 add_prefix_space/use_regex、未知字段及 postprocessor 均继续严格核对。不能整体忽略预处理或解码配置来使检查通过。

修复通过 39 项相关回归检查；真实 v18 tokenizer 保存后重新加载，640 条 train/dev 前缀的完整 token IDs 和 assistant loss mask 与原 tokenizer 一致。校验没有访问合成 test，没有训练或改写权重。

续评测入口为 `scripts/resume_online_repair_evaluation.py`。它只对原实验仓库中已知旧哈希的 artifacts.py 打补丁并保留原代码备份；其余业务运行时源码、prompt、工具、原 baseline 结果和 pilot 权重不改。检查原训练确实为 64 步、baseline 完整后，用新输出目录重新评测候选并严格配对。该恢复流程的训练步数为 0、旧模型评测不重复，仍然不使用正式 test。

补丁包及准确命令在 `packages/online-repair-eval-fix-20261013-r1`；此次原实验路径由截图确认是 `/home/featurize/work/ecommerce-repair.VyMdGD`，不另建一套新训练。评测结束仍恢复原 v18 服务，通过只读实验摘要接口获取最终指标。
