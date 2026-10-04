# 通用视频高光剪辑

输入原视频和目标画幅，程序选择高光时间，并为入选的每一帧生成裁剪框。比赛交付是 JSONL：`video_id`、`targetRatioWH`、逐帧 `frame` 与 `[x,y,w]`。MP4 用于预览效果。

**唯一维护入口：[thekfjie/video-highlight-unified](https://github.com/thekfjie/video-highlight-unified)**。不同路线同时存在于 `main`，实验用配置区分。旧仓库暂时保留历史。

## 推荐主线与最好结果

工程主线：**Video-ORA-4B v5 → 原视频 PTS 选帧 → YOLO11m → Pose → BoT-SORT → 校验/打包**。ORA 发现事件、复看视频筛选、定位时间；这里没有文本 Reranker或大模型训练。

原流程完成过复赛 426 视频、189,844 帧空间输出，非法框为 0，保留 4 条空预测。没有已确认的复赛排行榜分数。重构后的入口、Prompt 加载和实验登记只通过本地工程验证，尚未重跑 GPU 全链路。

已知初赛最好是历史报告中 **56.43 分的 exp07**。它不是已确认分数的 ORA v5。结果 ZIP 已定位，生成代码、配置、Prompt、commit 尚未找齐，不能声称完整可复现。见 [历史结果](docs/history.md)。

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
│   ├── editing/                预览/成片工具
│   └── sam_tracking/           暂无实现，明确标记
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
└── docs/                       运行、历史、迁移和协作说明
```

Git 不保存权重、视频、缓存、大量回答与预测。它们留在 `runs/`、`datasets/local/` 或外部目录，通过路径与哈希定位。

## 安装与一次 Baseline

Python 3.11+，准备 `ffmpeg` / `ffprobe`，从仓库运行：

```bash
python -m pip install -e ".[dev]"
python tools/make_demo.py
python vh.py run --config configs/pipelines/baseline_center.json --experiment demo-center-20261004 --author yourname
```

生成两条合成视频，执行全片居中裁剪；结果在 `runs/demo-center-20261004/predictions.jsonl`，实验在 `experiments/records/demo-center-20261004/`。这检查读视频、稠密输出和边界，不代表高光成绩。新实验换 ID，已有 ID 表示续跑。

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
