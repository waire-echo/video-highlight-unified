# 环境和模型准备

## 已验证的运行条件

完整双阶段在RTX4090 24GB上运行过。时间服务使用Python3.11、Torch2.10.0+cu128、vLLM0.19.1、Transformers5.5.4和qwen-vl-utils0.0.14；空间阶段使用Ultralytics8.4.106、OpenCV5.0.0、lap0.5.13和scipy1.17.1。

环境拆成 `temporal-env` 与 `spatial-env`。新安装脚本在空间环境只安装一种OpenCV发行包；不要同时安装GUI和headless变体，它们共用 `cv2` 命名空间。[OpenCV安装说明](https://pypi.org/project/opencv-python/5.0.0.93/)明确要求选择一种发行包。

脚本只在指定运行目录创建环境和权重文件。若机器已经有可工作的环境，在配置中填写其Python路径即可。虚拟环境Python路径不能替换成其符号链接所指向的基础Python。

## 目录与磁盘

视频、模型、环境、缓存和输出都建议放到数据盘。示例配置把它们放在仓库的 `data/`、`.runtime/`、`runs/` 下，这些目录均被Git忽略。

系统需有 `ffmpeg` 和 `ffprobe`。如果OpenCV导入提示缺少 `libGL.so.1`，在Ubuntu安装 `libgl1` 和 `libglib2.0-0` 后重新检查。模型服务由vLLM运行，推荐Linux环境；Windows可用于轻量工具和本地工程检查。

## 固定权重

时间模型为 `OraRL/Video-ORA-4B`，固定revision `01850297d5ab2adaaf130f700ed7cec52993d956`。该快照实际参数统计为5,174,964,736，符合本项目9B以内的模型规模约束。名称4B与统计参数数值不完全相同。

两个空间权重固定到官方YOLO11仓库revision `8b8ac7d1fae7468f85dbf89670dd66f41f485aab`：

| 权重 | 字节数 | SHA256 |
|---|---:|---|
| yolo11m.pt | 40,684,120 | d5ffc1a674953a08e11a8d21e022781b1b23a19b730afc309290bd9fb5305b95 |
| yolo11m-pose.pt | 42,459,307 | 29b17eaf3a3117cbea906090dbedf9159f7c6a49db58ec8b99ed2dfde1cf6eb2 |

下载工具拒绝覆盖SHA不一致的现有权重。官方HF主站不可访问时，可在命令前设置 `HF_ENDPOINT=https://hf-mirror.com`；版本和SHA校验保持相同。

## 配置中的主要路径

| 字段 | 内容 |
|---|---|
| run_root | 本次实验根目录 |
| video_dir | 原视频MP4目录 |
| test_index | 原始ID和画幅索引JSON |
| temporal_python | 时间阶段环境的Python |
| python_executable | 空间阶段环境的Python |
| temporal_model | ORA模型目录 |
| yolo_model / pose_model | 两个YOLO权重文件 |
| temporal_handoff | 可选：已有时间交接JSONL |
| profile | auto或compat |

只有视频索引、路径和运行目录需要按自己的机器调整。帧步长、阈值和构图参数保留已经使用的方案。

## 模型服务管理

`all` 自动管理自己启动的模型服务。若使用已有外部服务，它不会擅自停止该服务；执行空间步骤前需要确保显存已经释放。

`media_root` 可选，指定允许服务读取的目录，需同时覆盖视频目录和时间阶段运行目录。当这两个目录位于互不相关的根路径时，应调整目录或明确设置合适的父目录。

输入数据和预测文件不上传仓库。运行配置不保存SSH密码、GitHub token或其他凭据。
