# 通用视频高光剪辑流水线

输入原视频和目标画幅，程序先找出高光时间，再为高光内的每一帧生成裁剪窗口，最终输出比赛需要的 `predictions_spatial.jsonl` 和 `submission.zip`。

当前路线是 **Video-ORA-4B v5 → YOLO11m → YOLO11m-pose → BoT-SORT**。模型使用已有权重离线推理，整个流程不包含训练。视频数量从输入索引读取。

本仓库统一整理了已实际运行的时间与空间脚本。原接力实现完成过GPU全量推理和小样本对照；新增统一入口与资源自适应调度通过本地工程检查，尚未重新进行GPU端到端验证。请从本仓库目录使用 `pipeline.py` 运行。

```mermaid
flowchart LR
    A[原视频 + 目标画幅] --> B[ORA发现事件、复看筛选、定位时间]
    B --> C[原视频时间戳换算逐帧帧号]
    C --> D[YOLO找主体]
    D --> E[Pose调整人物构图]
    E --> F[BoT-SORT保持主体与位置连续]
    F --> G[校验帧号和边界、打包提交文件]
```

## 快速开始

模型推理建议放在 Linux NVIDIA GPU 服务器。轻量命令行和工程测试需要 Python 3.11+；`ffmpeg`、`ffprobe` 必须可调用。运行过的完整环境是 RTX 4090 24GB。

### 1. 准备环境和权重

在 GPU 服务器的仓库根目录运行：

```bash
PYTHON_BIN=python3.11 bash scripts/setup_temporal_gpu.sh "$PWD/.runtime"
PYTHON_BIN=python3.11 bash scripts/setup_spatial_gpu.sh "$PWD/.runtime"
```

如果镜像的 Python 3.11 位于 Conda，可将 `PYTHON_BIN` 换成 `/opt/conda/bin/python`。已有环境和权重可以直接复用，只需把真实路径填写到配置，不必再次安装。

两个 YOLO 权重下载后会核对固定 SHA256；ORA 权重固定到指定 revision。安装和目录说明见 [部署说明](docs/deployment.md)。

### 2. 建立输入索引，填写配置

假设视频放在 `data/videos/`，原始逐视频 JSONL 配置放在 `data/configs/`：

```bash
python3.11 tools/build_index.py --videos data/videos --configs data/configs --output data/test_index.json
cp configs/pipeline.example.json configs/pipeline.local.json
```

编辑 `pipeline.local.json` 中的视频、索引、Python 和模型路径。相对路径以配置文件所在目录为起点；示例配置中的 `../` 指向仓库根目录。

`run_root` 是本次实验的独立运行目录。换数据、算法代码或权重时使用新的目录，避免混用缓存。

### 3. 执行完整流程

```bash
python3.11 pipeline.py --config configs/pipeline.local.json all
```

这条命令会启动本地 ORA 服务，完成时间定位、格式修复和原视频 PTS 导出，再停止本次启动的模型服务以释放显存，接着执行画框、跟踪、严格校验和打包。

在 SSH 会话外持续运行可使用：

```bash
nohup python3.11 pipeline.py --config configs/pipeline.local.json all > pipeline-console.log 2>&1 &
```

模型服务仅监听 `127.0.0.1`。数据和模型权重留在自己的服务器；仓库只保存程序、配置示例和说明。

## 已有时间结果时，只跑画框

将第一阶段的 `temporal_handoff.jsonl` 路径传入即可：

```bash
python3.11 pipeline.py --config configs/pipeline.local.json check
python3.11 pipeline.py --config configs/pipeline.local.json prepare --handoff /data/temporal_handoff.jsonl
python3.11 pipeline.py --config configs/pipeline.local.json spatial --handoff /data/temporal_handoff.jsonl --smoke-videos 0,1,2
python3.11 pipeline.py --config configs/pipeline.local.json spatial --handoff /data/temporal_handoff.jsonl
```

Smoke 只处理指定样本，不产生正式提交 ZIP。样本应覆盖自己的横版、竖版、人物、动物和空预测情况。无卡时可以运行 `prepare` 校验输入并生成临时中心框；它不调用模型。

默认使用 `auto`：预读视频帧、选择适合显存的 batch、按视频并行跟踪，并根据运行时资源控制新任务并发。需要对照原慢速设置时，增加 `--profile compat`。

## 输出在哪里

以示例配置为例：

```text
runs/experiment_001/
├── config.resolved.json             本次真实配置
├── temporal/
│   ├── records/                     ORA原始事件与定位记录
│   ├── native_repair_report.json    格式修复统计
│   ├── timestamps/                  原视频逐帧时间戳
│   ├── export_grounded/             默认时间/帧号交接
│   └── export_recall/               备用时间导出
├── spatial/work/                    中心框、YOLO、Pose、跟踪缓存
├── logs/                            进度、硬件占用、调度计划与报告
└── output/
    ├── predictions_spatial.jsonl    最终逐帧 [x,y,w]
    ├── submission.zip              仅包含上述 JSONL
    └── stage2_report.json           输入指纹、环境、变化统计与校验
```

空预测保留该视频的一行。第二阶段不得改变第一阶段的帧集合、画幅或裁剪宽度。

## 常用工具

| 目的 | 命令 |
|---|---|
| 只检查将要运行的命令 | `pipeline.py --config ... --dry-run all` |
| 只检查视频时长与窗口数量 | `pipeline.py --config ... temporal --plan` |
| ORA按时长分位选5条短测 | `pipeline.py --config ... temporal --smoke 5` |
| 查看进度和调度计划 | `pipeline.py --config ... status` |
| 校验最终空间文件 | `pipeline.py --config ... validate` |
| 随机预览3个视频 | `pipeline.py --config ... preview --random 3` |
| 指定视频预览 | `pipeline.py --config ... preview --video-id 0,1` |
| 慢速/接力重新推理对照 | `pipeline.py --config ... verify-relay --video-ids 0,1,2` |
| 停止本次启动的ORA服务 | `pipeline.py --config ... stop-server` |

## 接力版是否改变效果

已完成的小样本重新推理对照覆盖5个视频、1084帧，没有复用检测缓存：1075帧最终框完全一致，9帧相差1像素，平均裁剪IoU为0.999959。慢速75.867秒，接力60.061秒，耗时减少约20.8%。这说明本轮样本的结果非常接近，不代表全部视频逐位相同，也不是比赛准确率。

接力保持同一视频内部的帧序和跟踪状态，按独立视频并行，最后按原顺序合并。不同 batch 的半精度计算可能产生小幅数值差异。完整说明见 [性能和对照](docs/performance.md)。

## 工程结构

- `src/highlight_pipeline/temporal/`：当前ORA v5时间流程。
- `src/highlight_pipeline/spatial/`：主体检测、姿态构图、主体跟踪和空间校验。
- `src/highlight_pipeline/runtime/`：有序预读、跨视频并行、资源监测与调度。
- `tools/`：索引、权重下载、结果对照、预览和仓库内容检查。
- `scripts/`：GPU环境准备和模型服务入口。
- `configs/`：一份运行配置示例和固定模型版本信息。
- `docs/`：[流程与模块](docs/pipeline.md)、[字段约定](docs/formats.md)、[部署](docs/deployment.md)、[性能](docs/performance.md)。

本地运行工程检查：

```bash
python3.11 -m unittest discover -s tests -v
python3.11 -m compileall -q src tools pipeline.py
```

算法来源和依赖许可见 [来源说明](docs/sources.md)。
