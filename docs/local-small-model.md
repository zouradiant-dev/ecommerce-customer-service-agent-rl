# 在 Windows 本地用 0.5B 小模型跑完整 Agent 流程

本文使用 `Qwen2.5-0.5B-Instruct`、Transformers 自带的 OpenAI 兼容服务和项目原有的 SQLite 工具环境。0.5B 适合演示流程，不代表正式训练效果；正式基线仍应使用项目配置中的 7B 模型。

## 当前机器的环境

- Python：`D:\downloads\anaconda\envs\pytorch_new\python.exe`
- GPU：NVIDIA GeForce RTX 5060 Laptop GPU，8GB
- PyTorch：2.11.0 + CUDA 12.8
- 本地模型：`models\Qwen2.5-0.5B-Instruct`

注意：实际目录名是 `pytorch_new`，不是 `pytorch\_new`。

## 第一次准备

在项目目录打开 PowerShell：

```powershell
$py = 'D:\downloads\anaconda\envs\pytorch_new\python.exe'
& $py -m pip install "transformers[serving]==4.57.6"
& 'D:\downloads\anaconda\envs\pytorch_new\Scripts\hf.exe' download `
  Qwen/Qwen2.5-0.5B-Instruct `
  --local-dir models\Qwen2.5-0.5B-Instruct
& $py -m pip install -e .
```

本机已经完成前两项；以后不需要重复下载。

## 终端 1：启动本地模型服务

```powershell
& 'D:\downloads\anaconda\envs\pytorch_new\Scripts\transformers.exe' serve `
  --device cuda `
  --dtype float16 `
  --host 127.0.0.1 `
  --port 8000 `
  --force-model 'D:\桌面\llm-learning\ecommerce-customer-service-agent-rl\models\Qwen2.5-0.5B-Instruct' `
  --model-timeout 3600 `
  --log-level info
```

看到 `Uvicorn running on http://127.0.0.1:8000` 后保留此终端。第一次请求才会把模型加载进显卡。

## 终端 2：生成任务并跑一个真实模型任务

```powershell
cd D:\桌面\llm-learning\ecommerce-customer-service-agent-rl
$py = 'D:\downloads\anaconda\envs\pytorch_new\python.exe'
$env:PYTHONPATH = (Resolve-Path src).Path

& $py -m ecommerce_customer_service.cli build-tasks `
  --output outputs\small-model-demo\dataset `
  --replicas 1 `
  --seed 7

& $py -m ecommerce_customer_service.cli run-model `
  --manifest outputs\small-model-demo\dataset\tasks.jsonl `
  --output outputs\small-model-demo\rollouts.jsonl `
  --model local-qwen-0.5b `
  --api-base http://127.0.0.1:8000/v1 `
  --scenario shipment_status `
  --limit 1 `
  --trials-per-task 1 `
  --temperature 0 `
  --max-steps 8 `
  --max-tokens 256 `
  --verbose
```

`--verbose` 会依次打印用户消息、模型的工具调用、工具返回值、模型最终回复和 Verifier 失败原因。这里用 `--scenario shipment_status --limit 1` 做最短演示；去掉这两个参数就会跑整个任务集。完整机器可读轨迹保存在 `outputs\small-model-demo\rollouts.jsonl`。

## 评分和失败分析

```powershell
& $py -m ecommerce_customer_service.cli evaluate `
  --rollouts outputs\small-model-demo\rollouts.jsonl

& $py -m ecommerce_customer_service.cli badcases `
  --rollouts outputs\small-model-demo\rollouts.jsonl
```

流程是：用户任务进入模型，模型选择并填写工具，项目在临时 SQLite 副本中执行工具，模型读取工具结果并继续，最后 Verifier 同时检查数据库状态、必要事实、业务政策和工具覆盖。每个 rollout 使用独立数据库副本，不会污染原始任务。

## 停止服务

回到终端 1 按 `Ctrl+C`。下次只需重新启动服务并执行 `run-model`，无需重新下载模型。
