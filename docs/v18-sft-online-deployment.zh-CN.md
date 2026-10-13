# v18 SFT 推理接入与下一轮实验

本次完成的是离线权重核验、推理服务代码、工具协议防护和连接检查。没有租实例、下载基础模型、安装 GPU 依赖或运行真实模型。当前网页仍可使用订单卡片验证业务流程；智能对话尚未连接。

## 1. 模型备份核验结果

2026-10-13 检查了本机保存的三个压缩包，实际 SHA256 与原备份清单一致：

| 文件 | SHA256 |
| --- | --- |
| `v18-sft-full-newgpu-r1-inference.tar.gz` | `24695aace5336e68fc7c947797339c2c19f423862f2360186872f95dfc299d6f` |
| `v18-sft-full-newgpu-r1-final.tar.gz` | `0f6d58c0d9c54204493151be318c3018bc38bd7642eaa82e93bf1849581d3667` |
| `v18-sft-full-newgpu-r1-records.tar.gz` | `373b44c1f076ac2eb21d91e3c16992d0dd6ec80f995b59020f75da5ce289ddf9` |

推理包中的 LoRA 权重校验为 `f41c98a583a6ca66e18f1d6f8743029cb3a363cf96f43d80dbf0d29b74876b6c`，适配器配置与六个 tokenizer 文件也逐一通过字节哈希校验。基础模型为 `Qwen/Qwen2.5-7B-Instruct`，固定 revision 是 `a09a35458c702b33eeacc393d103063234e8bc28`；LoRA 为 r=16、alpha=32。

LoRA 备份不包含 7B 基础模型权重，不能只凭推理压缩包启动推理。准备脚本只提取八个经过固定哈希核验的普通文件，不提取训练状态 pickle、链接或脚本，也不覆盖已有目标目录。模型目录和检查报告都被 Git 忽略。

```powershell
python scripts/prepare_v18_inference.py --archive PATH_TO/v18-sft-full-newgpu-r1-inference.tar.gz
# 已经解压过时，只做复核：
python scripts/prepare_v18_inference.py --adapter-dir models/v18-sft-verified
```

报告在 `outputs/v18-inference-preflight.json`。`adapter_verified=true` 只证明保存文件匹配；`environment_ready=false`、`gpu_inference_verified=false`、`online_quality_verified=false` 保留未验证项。

## 2. 训练协议与网页协议有什么不同

| 环节 | v18 离线 SFT | 网页 online_cancel_preview_v1 | 本次处理 |
| --- | --- | --- | --- |
| 身份 | 模型调用姓名/手机号核验工具 | 服务端认证并绑定会话 | 不允许模型改 customer_id，也不伪造身份核验结果 |
| 工具 | 九个业务工具 | 查询列表、查询详情、取消预览、转人工四工具 | 旧工具及多余参数被拒绝，返回错误供模型纠正 |
| 写操作 | cancel_order(..., confirmed) | preview_cancel(order_id) 创建申请，按钮确认才写入 | 不把旧取消调用直接执行，也不把聊天确认当按钮确认 |
| 返回字段 | 离线订单/支付/政策结构 | 在线后台订单及操作状态结构 | 新协议是部署分布变化，不能声称旧 SFT 已覆盖 |
| 工具编码 | Qwen 原生标签，模板内 arguments 是对象 | HTTP 对外 arguments 是 JSON 字符串 | 推理和计数复用训练的 normalize_messages，原生标签严格解析 |
| 最终回答 | decision/action_taken/action_id/message JSON | 简洁中文及操作卡片 | 合法且无写入声明的 JSON 提取 message；写入声明拒绝；预览由服务端生成 |

原离线训练提示词、九工具评测和历史分数不改写。网页使用独立提示词、工具哈希与新协议标识，可通过 healthz 和评测 protocol.json 查验。可定位“模型仍习惯旧工具”问题，不把兼容性修改混入算法提升。

此层保证执行边界，不保证自然语言回复全部正确。普通中文中的虚假成功、事实遗漏仍需状态验证、Bad Case 检查与人工复核。新网页只实现取消订单主链路；退款申请、改地址、政策检索不会因加载模型自动出现。

## 3. 后续有 GPU 时如何启动

### 3.1 准备独立环境与基础模型

建立独立虚拟环境，按 GPU 驱动安装适合的 CUDA PyTorch，再安装 `requirements-inference-v18.txt`。这是待 GPU 验证的候选配置，不是已验证的环境锁。不要混合仓库不同训练版本的约束文件。NF4 另需与 CUDA 相容的 bitsandbytes。

显式下载固定 revision，或复用同一 snapshot。此步骤消耗网络和磁盘，准备脚本不会自动执行。服务器需要推理包和完整基础权重，不需要训练状态。

```python
from huggingface_hub import snapshot_download
path = snapshot_download(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    revision="a09a35458c702b33eeacc393d103063234e8bc28",
    allow_patterns=["*.json", "*.safetensors", "merges.txt", "vocab.json"],
)
print(path)  # 将 snapshots/revision 路径作为 BASE_SNAPSHOT
```

启动复核 revision 标识、Qwen 配置、tokenizer 字节哈希和分片完整性；这不等于基础权重逐片官方 SHA 校验。先做无 GPU 检查：

```bash
python scripts/serve_v18_sft.py --base-model-dir BASE_SNAPSHOT --check-only
```

### 3.2 运行真实 SFT

```bash
# 在服务器执行，默认只绑定本机；加载基座与已核验 LoRA
python scripts/serve_v18_sft.py --base-model-dir BASE_SNAPSHOT --precision bf16
# 小显存可尝试 --precision nf4，须另测质量/显存，不保证任何 8GB 卡可跑
```

固定模型名 `ecommerce-v18-sft`。推理使用 eval + inference_mode、贪心生成、最多 512 个新 token、默认 8192 个输入 token；超预算拒绝，不截断历史。CUDA 不可用、基础目录/权重不对、输出截断或工具 JSON 不合法都会失败，不替换成脚本回答。单进程同时只执行一个推理请求，繁忙返回 429，尚无生产并发队列。

GPU 在远端时，在本机终端保持 SSH 转发：

```bash
ssh -N -L 8000:127.0.0.1:8000 YOUR_USER@YOUR_SERVER
```

如需远端绑定，脚本要求环境变量 MODEL_API_KEY 至少 24 字符；这不是 TLS/企业网关。将同一环境变量传给连接检查和网页进程，密钥不写进文档、日志或浏览器。

### 3.3 先检查，再接网页

```powershell
# 只检查模型/适配器/tokenizer/提示词/工具哈希，不生成
python scripts/check_v18_endpoint.py
# 真实生成一次，必须返回 list_orders({})，不执行业务工具
python scripts/check_v18_endpoint.py --generate
# 网页启动前再执行身份核验与只读生成探测
python scripts/run_customer_service.py --mode model --model ecommerce-v18-sft --verify-v18-endpoint --tokenizer-path models/v18-sft-verified
```

探测通过只说明服务及简单工具协议可用，不证明十五个任务通过。旧 customer_id 参数、cancel_order 或只返回客套文字会使探测失败；应保存失败、查输出，不自动伪造调用。--verify-v18-endpoint 专用于本项目服务；其他模型服务可使用原通用 --api-base/--model 入口。

## 4. 接通后跑什么实验

现在不额外训练。先用实际 SFT 跑已有十五个多轮 dev 用例，再定位问题。

```powershell
python scripts/evaluate_online_agent.py --mode model --model ecommerce-v18-sft --tokenizer-path models/v18-sft-verified --context-policy structured --output outputs/v18-online-structured-r1
python scripts/evaluate_online_agent.py --mode model --model ecommerce-v18-sft --tokenizer-path models/v18-sft-verified --context-policy full-history --output outputs/v18-online-full-history-r1
```

保持案例、模型、精度、解码、窗口与确认语义一致，只切换记忆策略；保存 protocol.json、bad_cases.jsonl、状态及 trace。报告任务通过数、未确认写入/越权次数、假成功、参数不兼容、循环和 P50/P95 耗时。十五题仅是开发回归证据。

根据失败决定下一步：

1. 旧工具/参数：先优化提示词和描述；若失败稳定存在，再构造在线四工具轨迹，覆盖预览、按钮后的状态续接及拒绝纠正，考虑补 SFT。
2. 忘记目标或把历史预览当授权：检查订单指代、操作失效状态与工具配对，对照 full-history，修上下文再评测。
3. 不知道政策或召回不全：真正接入政策问答后，再建知识库、检索标注集，测召回与回答证据。当前取消主链路不为凑技术栈强加 RAG。
4. 考虑 RL：先证明线上协议 SFT 存在稳定失败、数据/提示词优化进入瓶颈，且奖励能区分优劣；历史 offline r6 控制器效果不能替代此证据。

调优前另冻结独立测试，扩大真实表达与边界覆盖。基座 vs SFT 对照需同协议、同精度、同控制器，单独提供基座服务；本次启动脚本固定加载 SFT，不声称已完成基座实验。

## 5. 本次验证边界

本机缺少 torch/transformers/peft 等依赖，未加载 CUDA 模型。已验证真实备份文件及 tokenizer；工程测试覆盖 HTTP 鉴权、身份/协议校验、生成上限、繁忙/截断拒绝、旧参数拒绝和终止 JSON 的写入声明拒绝，复跑会话与幂等操作回归。HTTP 单测中的生成器是测试替身。

本次通过 20 项新增部署测试、32 项会话测试、18 项记忆/评测测试和 16 项原有本地模型解析测试，共 86 项相关测试；十五个工程 smoke 用例通过。实际 endpoint 检查记录 URLError，service_identity_checked/generation_checked 均为 false，没有伪造已接通结果。

这些验证不能填入“真实 SFT 在线成功率”。仍需配套基础权重、GPU 环境、真实生成、完整开发回归和独立测试，才能判断在线效果。

参考：[PEFT 加载](https://huggingface.co/docs/peft/main/en/package_reference/peft_model)、[Transformers 4.51.3 量化](https://huggingface.co/docs/transformers/v4.51.3/en/quantization/bitsandbytes)。
