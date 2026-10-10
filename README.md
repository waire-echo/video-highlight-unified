# 通用视频高光剪辑

当前服务器 `scene_general_v3` 整理从交接提交 `81f2257` 开始。统一入口 `python vh.py workflow` 支持固定兼容版、完整时间→空间流程和独立 I/O 实验，提供 [TUI 与 WebUI](docs/scene_workflow_ui.md)。先读 [START_HERE.md](START_HERE.md) 与 [版本化工作流](docs/scene_workflow.md)。原始基线及结果保留，新入口的 GPU 端到端验证待申请 GPU 后完成。

[功能索引](docs/feature_index.md) 登记推荐入口。下文 ORA v5→YOLO/Pose 与本次 v3 空间流程分别通过配置选择。

输入原视频和目标画幅，程序选择高光时间，并为入选的每一帧生成裁剪框。比赛交付是 JSONL：`video_id`、`targetRatioWH`、逐帧 `frame` 与 `[x,y,w]`。MP4 用于预览效果。

**唯一维护入口：[thekfjie/video-highlight-unified](https://github.com/thekfjie/video-highlight-unified)**。不同路线同时存在于 `main`，实验用配置区分。旧仓库暂时保留历史。

## 推荐主线与最好结果

工程主线：**Video-ORA-4B v5 → 原视频 PTS 选帧 → YOLO11m → Pose → BoT-SORT → 校验/打包**。ORA 发现事件、复看视频筛选、定位时间；这里没有文本 Reranker或大模型训练。

原流程完成过复赛 426 视频、189,844 帧空间输出，非法框为 0，保留 4 条空预测。没有已确认的复赛排行榜分数。重构后的入口、Prompt 加载和实验登记只通过本地工程验证，尚未重跑 GPU 全链路。

已知初赛最高报告分数是 **56.43**。新找到的 YOLO/Pose 报告记载 48,895 帧，现有 exp07 ZIP 为 45,234 帧，尚不能确认两者是同一版本。它们不是已确认分数的 ORA v5。完整生成代码、Prompt、commit 和结果对应关系仍待补齐，见 [历史结果](docs/history.md)。

## 目录结构

```text
video-highlight-unified/
├── vh.py                       实验与路线统一入口
├── pipeline.py                 ORA/Spatial 的分阶段入口
├── pipelines/
│   ├── ora_direct/             ORA v5，4B/9B 共用代码
│   ├── ora_proposal/           Prompt/schema，完整代码缺失
│   ├── ora_reranker/           结果/设计证据，完整代码缺失
│   ├── qwen_direct/            历史直接定位
│   ├── qwen_window/            窗口评分/融合/聚合
│   ├── reze/                   Yes/No 曲线实验
│   ├── pgl_sum/                重要度模型对照
│   ├── baselines/              center/motion、官方与旧流程参考
│   ├── spatial_crop/           YOLO/Pose/跟踪/裁剪
│   ├── ora_spatial/            ORA画框、复核、SAM传播和虚拟镜头
│   ├── editing/                预览/成片工具
│   └── sam_tracking/           旧独立入口；实现已在ora_spatial中恢复
├── src/video_highlight/        公共算法、IO、校验、运行和实验管理
├── configs/pipelines/          JSON 实验配置
├── prompts/                    按任务/版本保存文本/schema/构造器
├── datasets/manifests/         数据描述，不含真实视频与完整索引
├── experiments/records/        配置快照、commit、指标、来源指纹
├── experiments/registry.csv    实验索引，可重建
├── tools/                      索引、预览、权重下载、审计
├── scripts/                    共用 runner、部署与合并脚本
├── tests/                      回归测试
├── licenses/                   第三方源码许可
├── references/                 资料目录；原始本机资料放 local/，不提交Git
├── artifacts/                  本机历史结果/交付/预览/来源快照，不提交Git
└── docs/                       运行、历史、迁移和协作说明
```

Git 不保存权重、视频、缓存、大量回答与预测。它们留在 `runs/`、`datasets/local/` 或外部目录，通过路径与哈希定位。

## 安装与一次 Baseline

Python 3.11+，准备 `ffmpeg` / `ffprobe`，从仓库运行：

```bash
python -m pip install -e ".[dev]"
python tools/make_demo.py
python vh.py run --config configs/pipelines/baseline_center.json --experiment demo-center-yourname-001 --author yourname
```

生成两条合成视频，执行全片居中裁剪；结果在 `runs/demo-center-yourname-001/predictions.jsonl`，实验在 `experiments/records/demo-center-yourname-001/`。把yourname换成自己的名字，并为新实验选未使用的ID。这检查读视频、稠密输出和边界，不代表高光成绩。已有ID只用于原源码版本的续跑；历史 `demo-center-20261004` 绑定重构时的commit，不在当前源码上复用。

CPU baseline 不需要 Torch、CUDA 或大模型。模型路线用独立环境，见 [环境与模型](docs/environments.md)。回归测试还需要 PyTorch，因为它覆盖 PGL 网络结构。

## 真实数据与 ORA

```bash
python tools/build_index.py --videos /data/videos --configs /data/configs --output datasets/local/test_index.json
cp configs/paths.example.json configs/paths.local.json
```

填写自己的机器路径和解释器。相对路径以仓库根目录为起点。`paths.local.json` 被忽略，仅放路径，禁止放密码/token；算法参数放 `configs/pipelines/`。换数据时同步修改 dataset ID/manifest。

```bash
python vh.py run --config configs/pipelines/ora4b_v5.json --experiment 20261004-yourname-ora4b-v5 --local configs/paths.local.json --dry-run --action all
python vh.py run --config configs/pipelines/ora4b_v5.json --experiment 20261004-yourname-ora4b-v5 --author yourname --local configs/paths.local.json --action all
```

已有时间结果时，路径配置增加 `temporal_handoff`，使用 `spatial_crop.json` 和 `--action spatial`，无卡先用 `--action prepare`。详细操作见 [运行说明](docs/running.md)。

后续 ORA 语义画框、框复核、SAM 传播和自适应虚拟镜头位于 [`pipelines/ora_spatial/`](pipelines/ora_spatial/README.md)。该路线保持高光时间帧不变，49 项测试通过；`scene_general_v3` 已在 RTX 4090 上完成 426 个视频的全量运行和结果合并，目前尚无官方分数。

2026-10-10 至 2026-10-11 完成的 `scene_general_v3` 已固定为后续工程优化基准；源码、环境、模型、非 Git 输入、实际结果指纹和空服务器恢复方法见 [`pipelines/ora_spatial/REPRODUCE_SCENE_GENERAL_V3.md`](pipelines/ora_spatial/REPRODUCE_SCENE_GENERAL_V3.md)。

后续两阶段 TUI、文件识别、错误查看、三路对比视频和文档整理按 [`pipelines/ora_spatial/PRODUCTIZATION_PLAN.md`](pipelines/ora_spatial/PRODUCTIZATION_PLAN.md) 实施，所有功能先保持基准兼容模式。

服务器交接、当前运行状态、完整结果位置和重跑命令见 [`pipelines/ora_spatial/SERVER_HANDOFF.md`](pipelines/ora_spatial/SERVER_HANDOFF.md)。

## 新实验与历史实验

先提交代码、配置和 Prompt，再登记。模型填写确切 revision，作者填实际运行的人。

```bash
python vh.py experiment create --id 20261004-yourname-qwen-window-v2 --author yourname --config configs/pipelines/qwen_window_v1.json --notes "公开开发集，聚合阈值对比"
python vh.py run --experiment 20261004-yourname-qwen-window-v2 --local configs/paths.local.json
python vh.py experiment metrics --id 20261004-yourname-qwen-window-v2 --file /path/to/metrics.json
python vh.py experiment list
python vh.py experiment show --id hist-exp07-boundary-b
```

登记保存配置快照、Prompt 文件 hash、数据 manifest hash 与代码 commit；运行记录实际输入文件/权重 hash 和命令。`metric_type` 区分榜单、代理评测与工程统计。见 [实验规范](docs/experiments.md)。

Prompt 如 `prompts/ora/v5/`、`prompts/qwen/window-v1/`。修改语义时新建版本目录并在配置指定，保留旧版本；不使用 `final2` 命名。

## 协作与验证

```bash
python -m pytest -q
python -m compileall -q src scripts tools pipelines prompts pipeline.py vh.py
python tools/audit_repo.py
```

分支用于一次开发：`feat/xxx`、`fix/xxx`、`exp/xxx`。值得留下的实现、配置、Prompt 和结论合入 `main`，完成后删除开发分支；算法不使用永久分支隔离。

[迁移报告](docs/migration_report.md) 说明迁移/舍弃/Archive 建议与历史缺口；[来源许可](docs/sources.md) 说明第三方代码；[协作规范](docs/workflow.md) 说明团队流程。

[参考资料索引](references/README.md) 集中赛题、历史方案和论文；[本地整理说明](docs/workspace_cleanup.md) 记录原目录归位、重复删除、旧路径查找和保留原则。
