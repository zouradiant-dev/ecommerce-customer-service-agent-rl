# v13：10 步验证 → 100 步受控 GRPO

这是限定五类 train 场景的探索实验，不是正式 RL gate，不证明 held-out 泛化。用户已在聊天中确认 20 条逐条奖励判定，独立 `review-confirmed/review.json` 记录该确认；原报告仍保留。未确认的审阅文件不能启动。

## 训练范围和参数

验证阶段仅使用已审阅的 5 个 train 题目；实验阶段使用相同五类场景各 10 个 inline train 题目，共 50 个不同题目。排除依赖后续模拟用户回复的任务，不引入 dev/test，不向模型提供参考答案、评分意图或任务元数据。

两阶段分别从原始 SFT checkpoint-150 开始，参考策略是该 SFT 的冻结副本。10 步权重不是 100 步的初始权重或 KL 参考；这不是断点续训。旧 v11/v12 入口、默认评分版本和 smoke 上限保持不变。

- 基座：Qwen2.5-7B-Instruct，revision `a09a35458c702b33eeacc393d103063234e8bc28`。
- NF4 double-quant QLoRA、BF16、SDPA、梯度检查点，单 GPU，无 vLLM。
- micro-batch 1，梯度累积 8，生成批次明确固定为 8，每题 G=4，即每次更新 8 条轨迹/2 题。
- 学习率 5e-6，temperature 0.8，beta 0.04，epsilon 0.2，seed 7。
- completion 总预算 2048 token，最多 12 次工具循环；截断轨迹屏蔽训练。
- 实验每 25 步保存，保留 checkpoint-25/50/75/100；验证保存 checkpoint-10。
- 每阶段默认 3 小时软时间上限，在生成/优化步边界检查，不是强制实时终止，不代表平台自动停止计费。

TRL 的生成批次默认与梯度累积关联，这里显式固定生成批次，避免未经测试地增加 KV 显存。[TRL 1.14.1 配置源码](https://raw.githubusercontent.com/huggingface/trl/v1.14.1/trl/trainer/grpo_config.py)

## 前置检查与失败处理

安装器只接受已知旧文件 SHA256，备份后复制；未知修改或运行中的项目任务会阻止安装。补丁不含模型权重，不安装或更新任何依赖，不自动启动训练。

训练前 CPU 重放 20 条已确认轨迹，校验工具输出、实际状态、v13 奖励、数据/基座/SFT/评分源哈希和 train 范围。验证阶段必须完成 10 次优化更新，观察到有限且非零的真实梯度、策略张量变化、冻结 SFT 不变、保存重载一致，以及新版分阶段显存证据通过。旧失败轮次仍保留，但不能替代新版真实资格检查，不能补写成功标志或直接授权 100 步。

r3 分别记录初始化、训练、最终保存和额外 adapter 重载阶段；训练期间每次生成评分和每次优化完成都记录 CUDA 计数。各阶段峰值独立重置，边界同步后仅释放未使用缓存；不释放模型或优化器张量，不更改训练配置。`memory_audit.jsonl` 即时落盘，`memory_report.json` 在显存判定之前落盘，即使未通过也能看到具体数值；中途 OOM 保留此前采样及异常日志。

1 GiB 余量仍然必须满足。r4 策略 `stage_reserved_upper_bound_and_driver_free_v2` 按阶段计算“总容量 − 阶段 reserved 峰值 − 同阶段采样到的非 allocator 占用峰值”，同时检查所有采样点的实际设备空闲显存，包括阶段边界和训练中间。reserved 已包含 allocated、inactive split 和缓存，碎片只作诊断，不再重复相加。两个独立峰值之和明确叫做保守容量上界，不冒充同一时刻的真实占用。所有阶段，包括额外 adapter 重载，都必须保留至少 1 GiB；没有删除保护或改小训练配置。外部占用仍为采样值，不能保证更长/不同轨迹不会 OOM。[PyTorch reserved 峰值定义](https://docs.pytorch.org/docs/stable/generated/torch.cuda.memory.max_memory_reserved.html)、[allocator 计数定义](https://docs.pytorch.org/docs/main/generated/torch.cuda.memory.memory_stats.html)

进入 100 步前，CPU 从原始显存采样重新计算结论并校验文件 SHA256、阶段顺序和 10 个优化步；修改汇总中的通过标志不能放行。新增 `V13_PROGRESS` 日志直接给出已完成优化步，例如 qualification 7/10 或 experiment 23/100。

进入 100 步前，再 CPU 重放全部 80 条在线验证轨迹，并检查完整奖励分组、梯度事件、保存策略与冻结 SFT 张量、所有运行源哈希。不接受只写 `complete=true` 的空证明。

出现非有限指标/梯度、OOM、连续三批没有组内奖励差异、时间上限或任何验证失败时停止。不会自动降配置、修改奖励、重试或覆盖旧结果。保留 `failure.json`、日志和已保存 checkpoint；失败的资格验证不允许进入实验。没有相对奖励信号时停止属于预期保护，需要诊断，不能删除保护强跑。

原始 native-dynamic 采样与 TRL static-tools 训练不同；日志保留该运行差异。v13 只修复已经确认的 narrow 语义问题，不是通用语义审计器。

## 部署和启动

2026-10-02 r2 补齐部署测试依赖；r3 增加分阶段采样证据，但独立 allocated/碎片峰值相加会重复估算，不能视为真实峰值。r4 修正统计并按已校验 r3 哈希安全升级，训练超参数和评分规则不变。旧包及旧失败结果不改写，旧策略的资格记录不能放行新训练。专项测试的 GPU 计数是明确的 synthetic fixture，不表示真实显卡已通过；`V13_REWARD_BATCH` 测试输出也不是实际训练。

上传打包文件及 `.sha256` 到服务器 `/home/featurize/work`。以下命令在服务器 Bash 终端执行，**不是 Windows PowerShell**。只运行一次；安装完成并通过测试后才会由最后一行启动 GPU。

```bash
(
  set -euo pipefail
  cd /home/featurize/work
  sha256sum -c grpo-experiment-v13-source-20261002-r4.tar.gz.sha256
  task_deploy_dir=$(mktemp -d /home/featurize/work/grpo-experiment-v13-deploy.XXXXXX)
  tar -xzf grpo-experiment-v13-source-20261002-r4.tar.gz -C "$task_deploy_dir"
  task_package="$task_deploy_dir/grpo-experiment-v13-source-20261002-r4"
  task_python=/home/featurize/grpo-env-v12/bin/python
  "$task_python" "$task_package/install_grpo_experiment_v13.py" \
    --repo /home/featurize/work/ecommerce-customer-service-agent-rl
  cd /home/featurize/work/ecommerce-customer-service-agent-rl
  "$task_python" -m unittest discover -s tests -p 'test_grpo_experiment_v13*.py' -v
  bash scripts/run_grpo_experiment_v13.sh \
    "$task_package/evidence" "$task_package/review-confirmed/review.json"
)
```

启动器后台顺序执行两阶段，先准备日志，避免 `tail` 找不到日志的竞态。打印真实 PID、日志及结果路径。`Ctrl+C` 只退出日志查看，不停止后台训练；退租/关机前需确认训练进程已结束。不要删除临时部署包，运行期间它存放冻结证据。

未确认的审阅文件不可替换为已确认状态；此次包中的确认来自用户的明确逐条审阅回复，并绑定原轨迹 SHA256。后来改奖励或换数据时需重新审阅，不允许沿用签名。

## 训练后比较（单独手动启动，不自动增加评估费用）

先比较每类 2 个 dev 题目（14 个），同一 NF4 基座、同一 dynamic-tools 提示、greedy decoding、v13 评分，SFT 只评一次。候选依次加载后卸载，不同时驻留多个模型。使用启动器实际打印的结果目录替换下列示例路径，不能把“实际结果目录”原样粘贴。

```bash
(
  set -euo pipefail
  cd /home/featurize/work/ecommerce-customer-service-agent-rl
  task_run_dir=$(ls -td outputs/grpo-experiment-v13-20261002.* | head -n 1)
  test -f "$task_run_dir/experiment/complete.json"
  task_eval_dir=$(mktemp -d outputs/grpo-v13-dev-screen.XXXXXX)
  /home/featurize/grpo-env-v12/bin/python -u scripts/evaluate_grpo_v13.py \
    --manifest data/qwen15b-v2/tasks.jsonl \
    --sft outputs/sft/qwen25-7b-seed-7/checkpoint-150 \
    --candidate "$task_run_dir/experiment/checkpoint-25" \
    --candidate "$task_run_dir/experiment/checkpoint-50" \
    --candidate "$task_run_dir/experiment/checkpoint-100" \
    --output "$task_eval_dir/comparison"
)
```

若存在多次运行，不要依赖“最新目录”；明确选择已成功完成的那次。14 题仅用于筛查退化，不能证明总体提升。之后固定选中 checkpoint 再加 `--full-dev` 跑 140 题。held-out test 不参与训练或选模型，最后另行冻结方案后验证。

只看 loss 或训练奖励上升不能判定训练更好；比较成功率、policy/工具/communication 失败和配对差异。当前包不会声称正式 gate 已通过。
