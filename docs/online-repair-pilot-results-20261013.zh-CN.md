# 在线协议补 SFT：首次真实训练与配对开发结果

2026-10-13，从恢复后的服务器只读 `/v1/repair-experiment` 获取完整报告。实验完成，候选未晋升，原 `ecommerce-v18-sft` 服务仍在线。以下指标来自实际 GPU 训练/模型生成，不是脚本参考模型成绩。

后续输入重叠审计发现两个开发案例的实际模型输入与补训输入完全相同。下方原 57 题结果保留为原始开发回归记录，不宣称 57 题全部未在训练出现；审计细节与排除后的辅助统计见文末。

## 主要结果

同一批 57 个合成在线开发案例、同一 `binding_compact_v3`、BF16、贪心解码、512 生成上限、8192 输入预算及同一业务运行时下，原 v18 adapter 为 40/57（70.18%），补训 adapter 为 51/57（89.47%），净增 11 题、19.30 个百分点。逐题有 14 题改善、3 题回退；候选仍有 6 题失败。已知生成格式失败调用数从 2 降到 0，不能据此声称所有工具/业务错误均已解决。

| 开发组 | 原模型 | 补训模型 | 净增 |
|---|---:|---:|---:|
| 原在线回归 | 12/15 | 15/15 | +3 |
| 商品绑定 guards | 6/12 | 10/12 | +4 |
| 新协议合成 dev | 22/30 | 26/30 | +4 |
| 合计 | 40/57 | 51/57 | +11 |

这批原在线回归没有净回退；它不是旧九工具离线 600 题回归。主比较两边都使用 compact 提示词，不能将旧 baseline_v1 的历史 5/15 当作本次补训基线。网页目前仍是原模型、baseline_v1，尚未验证候选在网页默认配置下的效果。

## 实际训练与资源记录

| 项目 | 实际记录 |
|---|---|
| 初始化 | 原 v18 SFT adapter，基座冻结 |
| 方法 | NF4 QLoRA，adapter/梯度 FP32，累积后缩放 |
| 冒烟 | 4 个 optimizer step；开发 CE 16 条；训练计时 47.70 秒 |
| 小规模补训 | 64 个 optimizer step，512 条 train、128 条 dev，跳过数均为 0 |
| 补训开发 CE | 1.081956 → 0.083386 |
| 补训阶段计时 | 558.39 秒，约 9.31 分钟 |
| 补训显存峰值 | PyTorch 最大已分配显存 13.486 GiB，不等同于整个进程/nvidia-smi 的占用 |
| 数值权重门槛 | 392/392 个 LoRA tensor 发生实际数值变化，共检查 40,370,176 个元素，全部有限 |
| 更新量 | delta L2=1.039726，最大绝对 delta=0.000595246 |

训练计时字段从训练循环开始计算，不含此前模型加载和初始开发 loss 评测；不能把 9.31 分钟报成全实验耗时。冒烟和补训的开发样本数不同，二者 loss 不直接比较；loss 下降也不替代端到端任务成绩。64 步阶段从原 adapter 与新优化器重新开始，没有继承冒烟权重。

## 改善、回退和尚未完成的诊断

14 个改善案例：read_transient_error、repair_dev_query_then_cancel_001、repair_dev_query_only_000、repair_dev_confirmed_result_001、product_paraphrase、change_target、repair_dev_change_target_000、repair_dev_chat_confirm_000、withdraw_new_product、withdraw_confirmation、repair_dev_read_retry_001、new_product_exact、unpaid_no_refund、unseen_order_id。

3 个回退案例：

| 案例 | 正确业务行为 |
|---|---|
| no_matching_product | 用户取消洗衣机，但本人订单无匹配商品；不创建预览或写入 |
| repair_dev_no_match_001 | 用户取消挂钟，无匹配商品；保留原订单，不创建待确认操作 |
| repair_dev_true_ambiguity_000 | 两笔同名书桌，第一轮先澄清；用户按金额指定后才预览及确认 |

分组结果还显示 no_match 两题均失败、true_ambiguity 两题只通过一题，商品绑定 guards 的同名澄清案例仍失败，聊天确认两个案例只改善其中一个。这说明边界处理需要继续检查，不能只看总体通过率就替换服务。

当前只读摘要提供通过/失败结果，没有六题完整事件及工具轨迹。上表来自冻结案例的业务期望；不能据此断言失败一定是“误取消”“过早预览”或哪个具体工具调用。下一阶段先读取回退与残余失败的 result/history/events，复核既有有限短语断言是否误判，再决定补负例、调整分布或修改上下文。先前网页的“拒绝后用户表示知晓仍重复、订单事实幻觉”已另存为真实会话 Bad Case，尚未在本批 57 题中验证。

## 验证边界与当前决策

本次是单 seed、单次贪心生成、小规模合成开发配对实验。商品及主取消表达与训练分开，但场景族和部分续话模板共享。运行时生成会话/操作 ID 和时间，不声称两阶段每一次输入字节完全相同，也不报告多 seed 稳健性。独立合成 test、真实业务分布及旧离线九工具回归均未执行。

开发效果提升已经有可复核证据，但保留原服务，候选作为单独版本。先复核 3 个回退及其他 3 个残余失败，再进行有针对性的下一轮实验；本轮不自动晋升、不启用 RL、不启动额外训练。

训练后的首轮评测被 tokenizer 旧格式检查中断，续评测修复仅改 artifacts.py，未修改权重、原 baseline 结果或 10 项业务运行时源码。实际原文件 SHA256=089e1f4c10097a577671febcaf9eecec902277e3652d3f394676ac2bca66af23，修复后=67646c595fe2fa02095562172c4c4c8f454548387c551bcc3143a9447adaef16；续评测训练步数为 0。严格配对检查通过，实验摘要 experiment_complete=true、error_type=null、candidate_promoted=false。

完整读取证据保存在本机 ignored 目录 `outputs/online-repair-pilot-server-result-20261013-r3`（health.json、experiment-report.json），原始客户会话不上传 Git。服务器原训练目录 `/home/featurize/work/ecommerce-repair.VyMdGD/outputs/online-repair-pilot-r1`，续评测目录 `/home/featurize/work/ecommerce-repair.VyMdGD/outputs/online-repair-eval-resume-3cc5e3c9`。

## 后续输入重叠审计：纠正未见样本口径

审计本次 512 条 train 与 128 条 dev，先比较消息/tools，再用实际 Qwen chat template 渲染最后目标之前的完整模型输入，确认 `repair_dev_handoff_000` 和 `repair_dev_handoff_001` 与 8 条 train handoff 前缀完全重复。转人工第一轮不查询订单，输入中没有不同商品/订单状态，所以只划分 case ID、订单 ID 或商品目录并不能隔离该场景。这是数据划分遗漏，不能称为全部未见样本验证。

其他新 dev 前缀未发现完整渲染输入重复，但金额澄清、撤回、聊天确认、结果追问等部分用户续话文本共享；场景族和参考策略也共享。原 15 个在线案例和 12 个 binding guards 未发现完整案例 ID 或相同用户话语出现在本次 512 条补训数据中。对可核查的本地旧 v18 reference/full-replay/supplement 训练文件扫描也未找到这些案例 ID 或相同用户话语；该扫描不等同于证明所有历史权重的完整训练/预训练语料均已审计。

两个直接重复的 handoff 案例原模型和候选均通过。保留原始 57 题报告，另提供排除这两题的事后描述性重算：38/55（69.09%）→49/55（89.09%），净增仍为 11 题、20.00 个百分点。它不是新一次模型评测，也不能把其余 55 题直接改称独立盲测；不据此报告正式 test 成绩。

下一轮数据质量控制必须比较真正送入模型的渲染输入，而不只比较案例命名/业务 ID。现有 r3 train/dev/test 和模型权重保持不变；生成器中不依赖订单的其他 split 表达同样需要在使用前做完整输入重叠审计。合成 test 仍未执行。审计证据保存为 ignored `training-overlap-audit.json`。
