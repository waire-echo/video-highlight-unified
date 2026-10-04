# 环境与模型

CLI/CPU baseline 用基础环境，模型按需使用独立环境，不强行把所有依赖放到一起。

| 环节 | 历史观察/安装参考 | 模型 |
|---|---|---|
| ORA | vLLM0.19.1、Transformers5.5.4、qwen-vl-utils0.0.14，Linux NVIDIA | 4B revision `01850297d5ab2adaaf130f700ed7cec52993d956` |
| ORA9B | 共用客户端，历史显存约49,140MiB | revision `fc14c214ed1161ad05a7911740b0f6be89d1b6b9` |
| Qwen | 本机观察Torch2.10.0+cu128、Transformers5.5.4、bitsandbytes0.50.0 | 本地3B，revision待补 |
| Spatial | requirements-spatial.txt、Torch2.10.0/torchvision0.25.0 CUDA12.8 | YOLO11m/pose，下载核对hash |
| PGL | 旧记录Torch2.4.1、NumPy1.23.5；YouTube-8M Inception/PCA另需特征环境 | checkpoint、Inception、PCA都需要 |

本轮没有重装GPU环境或验证最新兼容性。PGL特征环境没有完整锁定，requirements-pgl.txt只列依赖；确认可用版本后保存freeze再正式实验。

```bash
PYTHON_BIN=python3.11 bash scripts/setup_temporal_gpu.sh "$PWD/.runtime"
PYTHON_BIN=python3.11 bash scripts/setup_spatial_gpu.sh "$PWD/.runtime"
```

脚本默认下载4B，不自动切换/下载9B。配置记录模型实际参数与来源；9B使用前核对赛事参数口径，不能只凭名称判断实际规模。视频与权重不进入Git，服务仅监听loopback，模型离线运行。
