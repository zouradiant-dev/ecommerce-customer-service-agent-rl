# 本地 Transformers Base/SFT 对照评测

此入口使用已安装的 torch、transformers、PEFT、bitsandbytes；不安装 vLLM、
不升级训练环境，不启动 HTTP 服务，不重新训练。

## 协议

- 只读取 manifest 的 **dev** 任务，且再次验证每个 task.json 的 split 和 ID。
- 默认每个 dev 模板选择 ID 最小的 1 条任务。本项目共 7 条，Base/SFT 各 7 条轨迹。
- 同一个 NF4 double-quant 基座只加载一次。Base 通过 PEFT 的 `disable_adapter()`
  关闭 LoRA，SFT 启用 LoRA；仅接受 bias=none、没有其他训练模块的纯 LoRA 权重。
- 检查基座与 adapter 的词表和 chat template 完全一致，同样的温度、token 预算、
  动态工具、系统提示、逐任务逐 trial 随机种子。推理统一为 BF16（不支持时 FP16）。
- 沿用现有原生 Agent 用户确认/身份回复和状态工具环境；**关闭 runtime oracle correction**。
  这与带 correction_retries=3 的教师服务收集协议不同，不直接比较两个成功率。
- Qwen 原生 `<tool_call>` JSON 严格解析；不修补 JSON，不使用参考 Agent 替代模型。
  畸形调用记为模型输出失败，并停止该轨迹；CUDA/环境异常则停止整个评测，保留部分输出。
- 每条轨迹调用 `run_task`，复制源 SQLite 到临时目录。每个阶段后再次审计源文件 SHA256。
- 默认只用缓存；输入超预算会明确报错，不静默截断。输出截断次数记入报告。
- **7 条任务的 pilot 是流程检查，不是完整 dev/test 分数，也不能证明已通过 RL gate。**

实现依据：[Qwen 原生模板](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/tokenizer_config.json)、
[PEFT 0.15.2 disable_adapter](https://github.com/huggingface/peft/blob/v0.15.2/src/peft/peft_model.py)。

## GPU 服务器运行

命令示例使用当前本地 v11 评分源码；服务器必须先部署相同版本并选择未使用的输出目录。
本次 v11 校准只在本地完成，没有自动更新服务器源码或重新运行 GPU 评测。

```bash
cd /home/featurize/work/ecommerce-customer-service-agent-rl
source .venv-training/bin/activate
export USE_TF=0
export USE_TORCH=1
export HF_HOME=/home/featurize/hf-cache

python -u scripts/evaluate_local_hf.py \
  --manifest data/qwen15b-v2/tasks.jsonl \
  --adapter outputs/sft/qwen25-7b-seed-7/checkpoint-150 \
  --output outputs/local-dev-pilot-seed7-step150-v11 \
  --tasks-per-template 1 --trials-per-task 1 --temperature 0 --seed 7
```

先出现 `preflight_ok`、`loading_model`，然后打印每条任务的 `task_start/task_done`。
最终仅以 `LOCAL_DEV_EVALUATION_COMPLETE` 与 `summary.json` 的 `complete=true` 判断完成。
输出包含 `protocol.json`、`model_info.json`、Base/SFT 两份 `rollouts.jsonl/report.json`
和最终配对 `summary.json`。新输出目录必须不存在，不覆盖任何既有结果。

仅做文件/清单检查时加 `--preflight-only`，不会加载 GPU 模型或创建输出目录。
若运行中断或失败，修复后选一个新的输出目录，保留原部分结果用于诊断。

Pilot 确认后，完整 dev 只需要将 `--tasks-per-template 1` 改为 `--full-dev`
并指定新的输出目录；本清单 dev 共 140 条任务。
要估计多次采样，必须使用正温度并增大 `--trials-per-task`；greedy 模式仅允许 1 trial。
**不要用这个 pilot 结果宣布泛化成功，也不要使用 test 调超参/选检查点。**

## 本地验证

```powershell
& 'D:\桌面\llm-learning\.venv\Scripts\python.exe' -m unittest discover -s tests -p 'test_local_hf.py' -v
```

这些 CPU/mock 测试检查解析、提示规范化、同种子/LoRA 开关、数据库隔离、
失败分类、输出保护和预检查。它们不代替真实 GPU pilot。

## v11：三个检查点使用相同规则重新评分（2026-10-01）

在 100/150/200 的已保存完整 dev 轨迹中，确认了五类表达漏判：
明确退款已处理完成、只允许申请一件退款、暂停或尚未执行退款、询问准确目标订单，
以及列表中相邻段落表达的超期拒绝。v11 补充这些等价表达，并测试否定、假设、
错误数量、其他对象的状态、绕过确认及相互矛盾的陈述。

没有使用模型名称或任务 ID 白名单，也没有修改生成对话、工具检查、权限或状态断言。
不明确的业务解释、退款数量误读、漏工具与只说“尚未确认”而没有承诺停止，仍保留失败。
重评分另加一条安全约束：保留下来的其他失败不能因为沟通检查通过而变成整体成功。

本地派生报告位于 `D:\桌面\llm-learning\server-checkpoint-compare-v11-rescored-20261001`：

| 检查点 | v10 成功数 | v11 成功数 | v11 成功率 |
| --- | ---: | ---: | ---: |
| 100 | 89/140 | 102/140 | 72.9% |
| 150 | 116/140 | 116/140 | 82.9% |
| 200 | 88/140 | 101/140 | 72.1% |

三组 Base 均为 8/140；教师 v8 的 440 条轨迹仍为 440/440。相对 v10 共修改 27 条
沟通判定，恢复 26 条成功，另 1 条因为仍漏工具而保持失败。90 项 CPU/mock 测试通过。
34 个原始报告/轨迹及既有校准文件的 SHA256 与操作前一致。

150 的底层重评分输入是原始 v9 生成报告，所以其子目录 `changes.jsonl` 记录 v9→v11。
根目录 `comparison.json` 和 `v10-to-v11-changes.jsonl` 使用保留的 v10 派生报告作为基线，
避免将之前 v10 修复的收益重复计入本次 v11。

150 仍是当前 dev 上较好的候选检查点，但这是查看 dev 后的评分校准，不是模型又学习了。
dev 只有七个模板族的 140 个实例，且本轮主要考察拒绝/查询，不能据此证明正向写操作、
独立泛化或 RL gate 达标。后续独立评测前冻结 v11，不用 test 继续调规则或选择检查点。

## v10：原始 dev 轨迹的语义重评分

本节记录历史 v10 校准；以下历史命令需要 v10 源码，不能用当前 v11 源码写入 v10 命名目录。

完整 v9 dev 评测又确认了被动语态“不被执行”、不需要重复操作、退款记录中
明确标注的 completed/已完成状态、带日期的历史取消，以及询问目标订单的等价表达漏判。
deterministic_semantic_v10 仅补充这些有明确依据的表达，并添加否定、未来状态、
其他对象的 completed 状态及继续执行退款等反例。用户回复协议仍为 v2；
任务、必要工具、权限、数据库断言和模型权重均未改变。

没有为含糊的地址政策解释、误读退款数量、只说“尚未确认”而未承诺停止等
不确定记录扩展规则。修正拒绝理由的表达也不能弥补遗漏的退款政策工具。

~~~powershell
& 'D:\桌面\llm-learning\.venv\Scripts\python.exe' scripts/rescore_local_hf.py --source 'D:\桌面\llm-learning\server-dev-full-20261001-step150-v9' --manifest data/qwen15b-v2/tasks.jsonl --output 'D:\桌面\llm-learning\server-dev-full-20261001-step150-v10-rescored'
~~~

此入口不调用模型、不消耗 GPU、不重新执行工具或改动业务数据库。它：

- 只接受已完整结束的 full-dev 报告，核对原报告、任务/trial 集合和 manifest SHA256。
- 拒绝已存在的输出目录，也拒绝与原报告相同或互相嵌套的输出路径。
- 对 Base 和 SFT 同时应用相同新规则；对每行断言原始对话、工具结果和其他生成数据未变。
- 保留原隔离执行的 state 检查，不能重新验证已经删除的临时 post-state 数据库。
- 重算并断言 policy、required-tools 及所有非 intent 失败与原记录完全一致。
- 新写 protocol.json、两组 rollouts.jsonl/report.json、逐条 changes.jsonl
  和 summary.json；在新协议中保留原始生成协议，记录评分代码哈希。
- 对七个原文件评分前后核对 SHA256，不覆盖 v9 报告。

这是查看 dev 后校准评分规则的派生诊断结果，不是模型进一步学习或独立的新性能提升。
原始 v9 分数继续保留。v10 应在后续独立评测前冻结；本次重评分不代替 held-out test，
也不宣称通过 RL gate。

## v9 校准与旧结果保留（2026-10-01）

首次 v8 pilot 暴露了确定性规则的三个同义表达漏判，以及模拟用户没有识别
“请您提供一下”“请确认您的姓名”等身份请求。v9 不改任务、数据库断言、
模型权重、工具格式、必要工具或 oracle correction 设置：

- `deterministic_semantic_v9` 接受“已完成退款”“实际购买数量是1件”
  “不允许创建售后退货退款”等等价表达，并增加否定、未来时、错误数量、
  矛盾表述的反例。不是看到“退款”“1件”等关键词就通过。
- 成功的 `get_order` 可以证明**同一客户、目标订单**的历史已取消/已退款事实；
  未查询、退款未完成、错订单、错客户，或声称“本次/我已为您”执行新操作时，
  仍需成功写工具。历史退款不能为创建退货、改地址或升级工单作证。
- `scripted_user_replies_v2` 只在明确询问身份字段时回复，不在提及姓名、
  无关的退款确认或“请不要提供”时注入身份。确认身份不是确认退款。
- `protocol.json` 同时记录 Verifier 版本、用户回复协议版本及五个评测源码的
  SHA256，Base/SFT 在相同代码和协议下比较；这些规则须在下一轮 GPU pilot
  前冻结，不能逐条追着模型输出改评分来抬高分数。

原始 `outputs/local-dev-pilot-seed7-step150` 保持不动；新评测另写
`outputs/local-dev-pilot-seed7-step150-v9`。本地亦保存了旧评测源码包
`local-hf-eval-v8-sources-20261001.tar.gz`，可用于核对旧规则。

旧轨迹重新打分只能验证评分修复，**不能补出被漏识别身份请求后本应发生的
新对话轮次**，因此必须用 v9 同时重新运行 Base/SFT。重打分不是新模型性能结果，
也不能和 v8 教师收集成功率直接比较。没有改动已有 `sft-teacher-v8` 数据。

```powershell
& 'D:\桌面\llm-learning\.venv\Scripts\python.exe' -m unittest discover -s tests -p 'test_evaluation_v9.py' -v
```
