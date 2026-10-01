# 实验计划与验收标准

## 阶段 0：环境验收

- 三十个模板都能生成独立数据库；
- reference 脚本全部通过，说明工具和验证器闭环可达；
- deliberately weak baseline 不能全部通过，说明验证器不是恒真；
- 重复 rollout 不修改源任务数据库；
- 模板切分检查无泄漏。

## 阶段 1：真实基线

固定模型、提示、工具 schema、温度、最大步数和随机种子。每个 test 任务至少采样 4 次，输出：pass^1、pass@2、pass@4、每场景 pass^1、Bad Case 占比。

基线的意义是定位能力缺口，不是追求一个低分。若绝大多数失败是 JSON 格式损坏或完全不会调用工具，应先修提示、chat template 或进行基础 SFT，不应直接 RL。

## 阶段 2：SFT

训练数据仅使用经状态验证的 train/dev 教师轨迹；loss 只计算 assistant 消息。至少保存 3 个中后期 checkpoint，用统一评测协议判断是否平台化。

SFT 后重点检查：

- 工具调用是否合法且稳定；
- 关键工具是否都有成功样本；
- 写操作前是否已获取订单/政策证据；
- 是否正确绑定用户、订单、商品和金额；
- 是否把口头承诺误当成环境更新。

服务器执行入口为 `scripts/train_sft.py`。训练脚本使用 assistant-only token loss，保存 LoRA、优化器、调度器、step、dev loss 和峰值显存，可从 `checkpoint-N` 恢复。

## 阶段 3：RL Gate

默认门槛：

| 检查 | 默认标准 |
|---|---|
| 工具格式稳定 | 非法/不存在工具调用低于 2% |
| 关键工具覆盖 | 7 个关键工具均有成功执行 |
| SFT 平台期 | 最近 3 个 checkpoint 的 pass^1 极差不超过 0.02 |
| 策略型失败 | 至少 60% 失败属于规划、政策、恢复或状态一致性问题 |
| 组内差异 | 至少 20% 任务的多次采样同时出现成功和失败 |

任意一项未过都先修数据、工具或 SFT。特别是组内奖励全相同，GRPO 的相对优势接近零，继续训练通常没有有效信号。

## 阶段 4：GRPO

- 从同一个 SFT checkpoint 启动；
- 每个任务同策略采样 6 条左右；
- 使用严格业务 verifier 返回的 0/1 奖励；
- 只用 train，dev 做早停，test 完全封存；
- 记录零方差组比例、平均工具步数、KL、奖励均值、超时率；
- 不为“说得像正确答案”单独给高奖励。

如果二值奖励过稀疏，可以先增加课程式任务覆盖或改善 SFT；过程奖励只能基于可审计事件（如成功身份核验），并且不得让过程分覆盖最终政策违规。

`scripts/train_grpo.py` 使用 TRL 状态环境在线采样：每个 completion 获得独立 SQLite 副本，工具调用进入真实状态机，episode 结束后由 Verifier 读取最终状态并给出严格奖励。test 会被拒绝。需要额外脚本用户回复的任务不进入当前通用 TRL 工具循环，并在 preflight 中显式列出，而不是静默改变任务语义。

`scripts/train_offline_group_awr.py` 保留原来的静态轨迹、组归一化优势加权 NLL 训练，只作为 offline group-AWR 对照。它没有从更新后的策略在线重采样，也没有 PPO/GRPO ratio clipping，因此不得标记为 GRPO。

算法实验至少包含 SFT control、offline group-AWR、严格二值在线 GRPO、安全塑形奖励、group size 和 KL 系数消融；矩阵固定在 `configs/algorithm-experiments.json`。没有足够 GPU 时只提交 preflight、配置和汇总工具，不编造训练数值。

## 阶段 5：统计验证

SFT 与 GRPO 必须使用同一 test 快照、同一提示和同一采样预算。比较时按 `task_id` 配对，报告：

- SFT 和 GRPO 的 pass^1 / pass@k；
- 配对差值及 bootstrap 95% CI；
- 每场景变化；
- Bad Case 迁移矩阵；
- 至少 3 个随机种子的均值和方差；
- 平均工具步数、超时率和人工升级率，防止成功率提升但成本失控。

只有当 CI、跨种子结果和关键政策切片都支持改进时，才把收益归因于 GRPO。
