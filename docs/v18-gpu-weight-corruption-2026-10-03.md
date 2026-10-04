# v18 基础模型 GPU 数据完整性异常

日期：2026-10-03。实例：Featurize RTX 4090 48G，报告显存 47.399 GiB。

## 已观测事实

- 基础模型为 Qwen/Qwen2.5-7B-Instruct，固定 revision `a09a35458c702b33eeacc393d103063234e8bc28`。
- 环境为 torch 2.2.2+cu121、CUDA runtime 12.1、transformers 4.46.3。
- 用户此前核验四个 safetensors 分片的 SHA256 均与官方一致；CPU 上目标磁盘张量无 NaN/Inf。
- 单独目标张量的三轮 CPU/GPU 复制与转换共九项检查全部逐位一致。
- 最新检查覆盖整个基础模型，共 7,615,616,512 个参数元素。常规 CPU 加载的模型与磁盘全部逐位一致。

| 加载与检查路径 | 错误张量 | 错误元素数 | 第 26 层目标张量错误数 |
| --- | --- | ---: | ---: |
| CPU 常规加载，在上传前比较 | 无 | 0 | 0 |
| 分片通过 device_map 加载至 GPU，再拷回 CPU 比较 | model.layers.26.mlp.up_proj.weight | 3,072 | 3,072 |
| CPU 常规加载后整体上传 GPU，再拷回 CPU 比较 | lm_head.weight | 3,072 | 0 |

直接加载路径中，期望值 `-0.010009765625` 拷回后变为 `-2.0392160725259565e+29`。CPU 上传路径中，期望值 `-0.007232666015625` 拷回后变为 `-5.885824602408568e+29`。错误值仍是有限数，GPU 与 CPU 的 isfinite 都为 true；逐位核验能够检出有限性检查无法发现的数据差异。

两种路径各在独立进程中运行；未加载 LoRA，未启用量化，未做前向、反向或优化器更新，首次 GPU 权重核验即失败并以 exit=2 停止。日志没有 OOM 报错。权重总大小约 28.370 GiB；这是权重存储量，不能当作正式训练的峰值显存。

## 结论范围与处理

已确认：完整模型经过该实例的 GPU 路径后存在数据完整性异常。这项事实早于训练，可解释继续调参无法消除故障。现有证据不能独立确定是显存芯片、GPU 驱动、CPU/GPU 传输路径或宿主机内存的具体故障，也未通过纯 CUDA 程序排除 PyTorch 路径问题。

建议平台保留实例日志，检查底层主机和 GPU，并将任务迁移至健康实例。在新实例运行同一模型逐位核验；通过后再恢复 smoke。smoke 通过前不启动 full，也不能把本机任何已产生 NaN 的 checkpoint 当作训练结果。

平台若需排查 CUDA 程序的非法访问，可参考 NVIDIA [Compute Sanitizer memcheck 文档](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html#memcheck-tool)。该工具能检测非法内存访问和硬件报告的异常，但不能把单次无报错当作所有显存均健康的证明。

## 可直接发送给平台的说明

> RTX 4090 48G 实例出现 GPU 数据完整性异常。官方 Qwen2.5-7B-Instruct 四个 safetensors 分片 SHA256 全部一致，CPU 加载后的全部 7,615,616,512 个参数元素与磁盘逐位一致。但上传 GPU 再拷回 CPU 后，两种加载方式都出现 3,072 个错误元素：device_map 加载损坏第 26 层 up_proj，CPU 加载后上传损坏 lm_head；正常约 0.01 的数值变成约 10^29。尚未执行模型前向或训练，没有 OOM 报错。请检查该实例的 GPU/显存、驱动和传输链路，协助迁移至健康实例。

## 证据文件

- `outputs/v18-gpu-weight-audit-20261003/model-load-audit.log`：用户返回的完整原始输出。
- `outputs/v18-gpu-weight-audit-20261003/summary.json`：从原始输出提取的关键结果与日志 SHA256。
- `scripts/diagnose_qwen_model_loading.py`：本次两种加载方式的完整复现脚本。
