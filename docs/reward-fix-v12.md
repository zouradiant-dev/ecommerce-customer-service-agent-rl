# GRPO 奖励修复 v12（2026-10-02）

本次只修复已经确认的奖励和工具结果格式问题，没有启动新训练，没有覆盖旧报告或权重。
默认 verifier 为 `deterministic_semantic_v12`；历史 v11 可通过
`verify_task(..., verifier_version="deterministic_semantic_v11")` 复现。

## 改动

- 原退款去向识别支持“原支付渠道/路径、原付款渠道/路径”；人工升级支持“已为您升级至人工支持”。补入否定、条件、未来与失败表达的负例。
- 新增显式事实检查：退款/退货/工单编号必须匹配实际成功工具返回的对应类型，不允许用退货 ID 冒充退款 ID。退款金额按元/分统一核对，并检查已知的支付宝/微信渠道及带标签的结果状态。
- 证据限定为当前用户、目标订单的成功工具结果；失败调用和其他订单结果不能为编号背书。历史已完成的退款允许用对应的订单读取结果举证。
- 在线工具返回值保留字典接口，但 `str(result)` 输出严格 JSON：`true/false/null`。只规范工具输出，不修复或执行模型非法 JSON 调用。
- 新事实模块纳入新评测、pilot 与训练 preflight 的源码哈希。旧 v11 pilot 不会被偷偷改名为新版证据；正式 RL 限制及 smoke 步数上限保持原样。

这是受限的确定性合同，不是任意自然语言的完整事实审计器：未覆盖所有转述、否定作用域、商品描述、隐含编号或未知支付渠道。不能据此宣称语义问题全部消失。

## CPU 重放结果

80 条在线轨迹及 32 条 train pilot 在独立 SQLite 副本中重放。
全部工具结果和历史 v11 奖励先逐条复现一致，再使用新版评分。
原文件与原数据库的 SHA256 均未变；没有把离线新分数写回过去真正用于梯度更新的奖励。

- 在线轨迹：原 52/80，新 59/80。11 条明确同义表达漏判被纠正，4 条原正分因编号错误转为 0。
- 转为 0 的在线行号：38（工单号少了 `cust_`）、62/79（把退货 ID 写成退款 ID）、73（退货项 ID 不是真实返回 ID）。59 的退款 ID 拼写错误继续为 0。
- 在线混合奖励组：9/20 → 12/20。它只是旧轨迹的离线信号诊断，不代表新版重新采样的信号。
- 原 pilot：30/32 → 30/32，混合组仍 2/8。
- 非法 `confirmed: True` 未被执行，未执行、越序、缺工具和失败后虚报成功等真实错误仍为 0。

不要把这些分数变动称为模型性能提升。旧 GRPO checkpoint-10 受旧奖励更新过，保留为对照；新的干净短试应从原 SFT checkpoint-150 开始。

## 服务器部署（不启动训练）

先把源代码压缩包传到新实例，确认没有本项目评测/训练进程在跑。
解压到新目录，不要用 `cp -r` 覆盖整个项目。

以下命令假定压缩包已经位于 `/home/featurize/work/`：

```bash
cd /home/featurize/work
task_fix_dir=$(mktemp -d /home/featurize/work/reward-fix-deploy.XXXXXX)
tar -xzf reward-fix-v12-source-20261002-r1.tar.gz -C "$task_fix_dir"
python "$task_fix_dir/reward-fix-v12-source-20261002-r1/install_reward_fix_v12.py" \
  --repo /home/featurize/work/ecommerce-customer-service-agent-rl
```

安装器在写入前逐个核对源文件、旧文件哈希和业务工具哈希；遇到额外改动会停止。
需要替换的旧代码会先存到 `outputs/code-backups/reward-fix-v12-*`。
不会改动已有报告、权重、数据、环境依赖，也不会启动 GPU。
该压缩包是修复包，不是完整项目，须先恢复昨天的项目或使用保留该项目的镜像。

CPU 回归可以使用安装了本项目基础依赖的 Python 环境：

```bash
cd /home/featurize/work/ecommerce-customer-service-agent-rl
python -m unittest discover -s tests -p 'test_reward*.py' -v
```

不要继续使用旧 `package_train_readiness.py` 打包这个新版本，它明确冻结 v11 源码。
下一步是按新版合同生成或严谨绑定新的已复核 readiness 证据、更新 runtime 检查，并从 SFT-150 进行新的有界短试。
本次离线报告不是已审核的新版训练 readiness；旧 evidence 的版本/哈希失配应停止，不要删掉检查或手改 JSON 绕过。
正式长程 GRPO 和 held-out 提升仍未验证。
