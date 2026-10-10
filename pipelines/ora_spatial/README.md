# ORA spatial reframing

规范化开发与版本切换使用根目录 `vh.py workflow`，见 [统一运行说明](../../docs/scene_workflow.md)。固定 v3 源码继续保留，优化在外部适配层中显式选择。

状态：`experimental-runnable`。这是 2026-10-10 从授权 GPU 服务器取得并逐文件核对的源码快照。49 项测试通过，`scene_general_v3` 已在 RTX 4090 服务器完成 426 个视频的全量运行和合并；目前尚无官方成绩。

第一次接手服务器或准备重跑时，先读 [`SERVER_HANDOFF.md`](SERVER_HANDOFF.md)。它说明当前服务器状态、权威结果、目录用途、全量运行、续跑、验收和禁止事项。

`scene_general_v3` 当前服务器基准的源码、运行环境、模型和外部输入恢复方式见 [`REPRODUCE_SCENE_GENERAL_V3.md`](REPRODUCE_SCENE_GENERAL_V3.md)。后续线程、TUI、README 和性能优化均以该基准为起点。

两阶段控制、路径选择、进度/错误查看、文件识别、三路对比视频和 README 收口的实施顺序见 [`PRODUCTIZATION_PLAN.md`](PRODUCTIZATION_PLAN.md)。

## 输入、处理和输出

输入是已经完成高光时间选择的提交包、对应原视频、逐帧时间戳、YOLO 证据与视频哈希清单。该路线**不改变入选帧集合**，只重新计算每个入选帧的 `[x, y, w]` 空间裁剪框。

主流程如下：

1. TransNetV2 按镜头边界切分连续高光片段。
2. Video-ORA-4B 判断画面关注对象，并按稀疏时间点输出归一化框。
3. ORA 二次复核候选框，拒绝角标、字幕、错误主体和不可见目标。
4. 可选 SAM 2.1 从通过复核的锚点传播掩码；失败时退回显著性、光流或中心框。
5. 虚拟镜头在速度、缩放与画面边界约束下生成固定或平移裁剪框。
6. 导出前检查视频顺序、帧集合、目标比例、边界和来源哈希。

`experiments/scene_general_v1`、`v2`、`v3` 是后续统一场景框实验。v3 每秒重新定位，检查语义连续性，区分 `visible`、`partial`、`occluded`、`out_of_view` 和 `uncertain`；该实验不调用 SAM 或 Pose。

## 目录

```text
ora_spatial/
├── src/                 ORA、复核、SAM、镜头切分和虚拟镜头
├── scripts/             准备、检查、运行、打包和可视化入口
├── config/              主流程与自适应裁剪参数
├── experiments/         scene_general v1/v2/v3
├── tests/               49 项单元与契约测试的一部分
├── provenance/          模型、权重和基线来源指纹
├── vendor/              TransNetV2 适配、ORA Prompt 与上游许可
└── SERVER_README_20261010.md
```

服务器原始说明完整保存在 `SERVER_README_20261010.md`。源码布局保持原样，因此命令应在本目录执行。

## 环境

Python 3.11。先安装基础依赖：

```bash
cd pipelines/ora_spatial
python -m venv env
source env/bin/activate
python -m pip install -r requirements.txt
```

根据机器 CUDA 版本单独安装 PyTorch 和 vLLM，因此它们没有写死在 `requirements.txt`。SAM 2 使用固定上游提交：

```bash
git clone https://github.com/facebookresearch/sam2.git vendor/sam2
git -C vendor/sam2 checkout 2b90b9f5ceec907a1c18123530e92e794ad901a4
python -m pip install -e vendor/sam2
```

Video-ORA-4B 固定为 `OraRL/Video-ORA-4B@01850297d5ab2adaaf130f700ed7cec52993d956`。权重、SAM checkpoint、TransNet 权重、视频和提交包均不在 Git 中；把它们放在本机路径，并按 `scripts/prepare_sample_delivery.py` 或 `src/prepare.py` 生成 `inputs/`。

原服务器启动器要求 `OLD_ROOT` 下同时存在 `env/bin/vllm` 和 `models/Video-ORA-4B`。在新机器上可按相同布局准备运行时，或自行启动兼容的本机 OpenAI API 端点。不要把模型目录或认证信息提交到 Git。

## 验证和运行

不启动模型即可运行测试：

```bash
python -m unittest discover -s tests -v
python -m unittest discover -s experiments/scene_general_v1 -p "test_*.py" -v
python -m unittest discover -s experiments/scene_general_v2 -p "test_*.py" -v
python -m unittest discover -s experiments/scene_general_v3 -p "test_*.py" -v
```

完整主流程先准备外部输入并执行预检；`run_gpu.sh` 会自行启动只监听本机的 ORA OpenAI 兼容端点：

```bash
export OLD_ROOT=/data/ora4b_runtime
export VIDEO_DIR=/data/videos
python scripts/preflight.py --video-dir /data/videos
bash scripts/run_gpu.sh partial
```

`partial` 只处理提示为 `scene/object/unknown` 的条目，`all` 处理全部非空高光。长任务使用新输出目录；不要修改正在运行任务所引用的源码和配置。

scene_general_v3 的服务器脚本含原机器默认路径，运行前通过独立环境或本机副本调整。源码快照没有为了便携而改写这些路径，来源和限制见 `provenance/source_snapshot.json`。

## 当前结论

- 原 Stage 2 的 YOLO/Pose/BoT-SORT 完整实现已经在 `src/video_highlight/spatial/`，未重复覆盖。
- 本目录补齐了此前仓库中缺失的 ORA 语义画框、SAM 传播和 scene_general 实验实现。
- 快照迁移时共执行 49 项测试，全部通过；只验证代码和契约，没有验证 GPU 推理质量或榜单涨分。
- 服务器上的 scene_general_v3 分段任务在快照时仍在运行，其输出、日志和测试集内容没有下载或提交。
