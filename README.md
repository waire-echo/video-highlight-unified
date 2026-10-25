# 视频高光剪辑算法

本项目实现视频高光时间定位、主体检测与空间裁剪，输出符合赛题格式的逐帧预测结果。
时间定位采用 Video-ORA-4B 基座及 ORA-Highlight-T1 LoRA 适配器；空间模块根据高光帧和目标画幅生成裁剪框。

## 环境与模型

运行平台为 Linux x86_64、Python 3.11 和 NVIDIA CUDA GPU，需安装 FFmpeg、FFprobe。
Python 依赖见 `requirements-submission.txt`，固定版本见 `requirements-submission.lock.txt`。
模型文件与运行环境随交付材料提供。源码部署时，指定各类资产位置：

```bash
python3.11 scripts/setup_submission_env.py \
  --model /path/to/base_model --adapter /path/to/adapter \
  --spatial-package /path/to/spatial_assets
```

安装器创建独立 `.venv`，检查依赖和 CUDA，并生成当前机器的运行配置。
完整模型包使用包内默认模型位置时，可省略上述三个路径参数。

## 输入

```text
input/
├── video/<video_id>.mp4
└── jsonl/<video_id>.jsonl
```

每个 JSONL 文件包含一行输入配置，例如：

```json
{"video_id":"sample","targetRatioWH":[9,16]}
```

视频编号与文件名对应，`targetRatioWH` 表示目标画幅。
其他支持的索引形式见[部署与运行说明](docs/submission_portability.md)。

## 运行

```bash
bash scripts/run_submission.sh \
  --input-root /path/to/input --output outputs/submission
```

多卡运行通过 `--inference-gpus 0,1 --spatial-gpu 0` 指定设备。
推理参数保存在 `configs/submission.example.json` 或完整模型包的 `configs/submission.json` 中。

程序依次完成输入检查、视频窗口准备、时间推理、区间合并、原帧映射、空间裁剪及结果校验。
相同配置和输入再次运行时接续已完成的检查点；输入、模型或配置变化时创建独立运行。

## 输出

输出目录的 `CURRENT.json` 指向当前运行。其 `06-submission/` 目录包含：

- `predictions_spatial.jsonl`：逐视频高光帧与裁剪框。
- `submission.jsonl`：与预测文件内容相同的兼容文件。
- `submission.zip`：仅包含 `predictions_spatial.jsonl` 的结果压缩包。
- `FINAL.json`：结果校验及文件摘要。

## 测试与交付

源码测试需要 pytest，可在环境安装命令中附加 `--with-tests`。

```bash
python -B -m pytest -q -p no:cacheprovider
```

完整模型包的构建、校验及独立环境运行步骤见[部署与运行说明](docs/submission_portability.md)。
第三方组件的许可说明保留于 `LICENSE`、`licenses/` 及对应模型目录。
