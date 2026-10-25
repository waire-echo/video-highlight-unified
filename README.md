# AIC-2026-37717297 算法代码

方案：ORA-Highlight-T1 与 RGB 评分驱动的连续空间构图。

两个阶段在同一工程中运行。统一入口依次执行输入准备、时间高光选择、时间合并与原生时间映射、主体连续关联与基础构图、RGB评分＋漏检证据连续路径、逐帧输出与 JSONL 校验。

## 当前交接状态

本目录已整理两阶段推理源码和第二阶段训练库。第一阶段适配权重已经补入相邻模型目录。完整基础模型需按固定版本链接下载；第一阶段训练源码仍待补齐，真实 GPU 全流程验收未完成。当前不能标记为完整提交已验收。CPU 与真实媒体回归已通过，不能代替真实 GPU 全流程验收。

第一阶段模型放入相邻“ AIC-2026-37717297-模型文件与运行环境 ”目录内的 models/ora 和 models/adapter；具体必需文件见该目录说明。两个交付目录保持并列。主体检测器从相邻模型目录自动补入本目录 models/spatial/assets，保留原空间模块的资产寻址。Git 仓库只保存源码和配置，权重由模型交接包提供。

## 环境准备

运行平台为 Linux x86_64、Python 3.11、兼容 NVIDIA GPU 和驱动，并具备 FFmpeg、FFprobe。依赖和安装器已保留；本轮未执行安装，不包含完整离线运行时。

在模型齐备的机器上，从本目录执行一次环境准备：

```bash
python3.11 scripts/setup_submission_env.py \
  --model ../AIC-2026-37717297-模型文件与运行环境/models/ora \
  --adapter ../AIC-2026-37717297-模型文件与运行环境/models/adapter \
  --spatial-package models/spatial
```

安装器会获取依赖，需要网络及相应授权。安装与 CUDA 检查通过后再执行推理。

## 一键推理

输入目录按 video 和 jsonl 子目录组织。每个媒体文件对应一行输入说明，包含 video_id 和 targetRatioWH；程序自行完成窗口准备与两个阶段，无需手工生成中间文件。

```bash
bash scripts/run_submission.sh --input-root /path/to/input --inference-gpus 0,1 --spatial-gpu 0
```

每次启动均创建全新的输出目录与运行缓存目录，从原始输入开始。该入口不接收历史预处理目录、旧交接文件或旧运行目录。处理中的有限故障重试只属于本次新运行。每次修复后再次启动会完整重新运行。

输出位于 outputs 下该次运行目录内的 run-<id>/06-submission，包含 predictions_spatial.jsonl、submission.jsonl、submission.zip 和 FINAL.json。实际是否完成以及是否发生降级，以该次运行记录为准。

## 训练源码

src/video_highlight 内的 spatial_supervision、spatial_source_roles、spatial_feature_materials、spatial_rank_training 分别负责第二阶段监督对齐、来源划分、真实 RGB 特征准备和评分头训练。配置参考位于 configs/training。

这些库属于同一工程，没有独立的第二阶段启动程序。第一阶段训练源码和整队训练调度尚未补齐，不能把这些训练库当作完整训练流程已经复现。

## 模型名称与文件

对外名称沿用技术方案。模型文件和图像处理配置必须配套；推理兼容代码保留在 rgb_vision_compat。模型参数或显示名称不得替换实际权重核验。

本交接不包含任何媒体输入、标注、预测、历史中间文件或运行日志。技术方案原文未修改。
