# 通用视频高光剪辑

**唯一维护主线是 `main`。当前整理和推荐入口是 `scene_general_v3`，统一命令为 `python vh.py workflow`。** 新人先读 [START_HERE.md](START_HERE.md)，再按 [功能索引](docs/feature_index.md) 找入口；不要从历史 E1/E2/E3 脚本开始。

当前默认版本为 `scene-general-v3-compat-20261010`：使用冻结的 baseline，只运行原空间算法和原 I/O。完整时间→空间与 I/O 优化属于明确命名的实验版本。**工程和 CPU 自动化已通过，新启动器的真实 GPU 推理仍待验收。** 历史 426 视频结果不等于新启动器已经完成 GPU 复跑。

## 主线包含哪些功能

| 功能 | 入口 | 状态 |
|---|---|---|
| 版本、配置、完整执行计划 | `vh.py workflow versions/plan` | 可用 |
| 时间→精确 PTS 交接→空间→校验→交付 | `workflow run` | 已接入；新 GPU 推理待验收 |
| 已有 baseline / 时间交接的输入准备 | `workflow run ... --until handoff` | CPU 实测通过 |
| 只执行时间阶段 | `workflow run ... --until temporal` | 已接入；GPU 待验收 |
| 断点续跑、阶段状态、错误与来源检查 | `workflow resume/status` | CPU 与故障恢复测试通过 |
| TUI、WebUI、资源与日志、正式产物下载 | [界面使用说明](docs/scene_workflow_ui.md) | 终端和浏览器实测通过 |
| 帧集合与提交 ZIP 校验 | `workflow validate` | 历史 426 视频通过 |
| 独立 I/O 优化与逐视频检查点 | `optimized.json` / `workflow verify-io` | CPU 像素和缓存重导出通过；新 GPU 等价性待验收 |
| 三路同步对比、预计剩余时间 | [后续计划](pipelines/ora_spatial/PRODUCTIZATION_PLAN.md) | 未实现 |

## 三个版本，四个运行配置

| 版本 ID | 配置 | 输入 | I/O |
|---|---|---|---|
| `scene-general-v3-compat-20261010` | `configs/workflows/compat.json` | 冻结 baseline | 原始 I/O，默认 |
| `scene-general-v3-unified-v1` | `spatial.json` / `full.json` | 已有精确交接 / 新时间推理 | 原始 I/O，实验 |
| `scene-general-v3-io-v1` | `optimized.json` | 配置中指定的输入 | 顺序解码与检查点，实验 |

代码主线、算法版本、一次运行是三个不同概念：`main` 保存维护代码；`configs/workflows/versions/` 选择算法版本；每次运行使用自己的 `run_root`。不为每个算法长期维护一个 Git 分支。

## 从哪里运行

本服务器的唯一维护工作区：

```bash
cd /root/rivermind-data/development/video-highlight-unified
cat START_HERE.md
/root/rivermind-data/development/vh workflow versions
/root/rivermind-data/development/vh workflow plan \
  --config configs/workflows/compat.json --local configs/workflows/server.local.json
```

其他机器先准备 Python 3.11+，从仓库运行：

```bash
python -m pip install -e ".[dev]"
cp configs/workflows/paths.example.json configs/workflows/paths.local.json
# 填写真实路径、解释器和全新的 run_root，再生成计划
python vh.py workflow plan --config configs/workflows/compat.json \
  --local configs/workflows/paths.local.json
```

`*.local.json` 只放机器路径，不进入 Git，不保存 SSH 密码或 token。视频、模型、输入和运行结果留在服务器外部数据目录。完整说明见 [统一工作流](docs/scene_workflow.md)；界面启动、SSH 转发、服务重启见 [TUI / WebUI](docs/scene_workflow_ui.md)。

正式提交来自新运行的 `final/submission.zip`。历史正式结果位于：

```text
/root/rivermind-data/aic_spatial_trials_20261008/output/scene_temporal_full_426_20261011_v3/
```

该目录及原视频、模型和基准输入保留。旧快照只提供数据与历史溯源；维护代码在上面的 Git 工作区。

## 怎样约束后续开发

- 一次改动建一个 `feat/`、`fix/` 或 `exp/` 分支，经过 PR 和自动检查合入 `main`，再删除已经合并的开发分支。
- 默认版本和功能入口由 [项目策略](configs/project_policy.json) 登记；README、唯一入口与界面使用同一个默认版本。
- core、原 v3、Prompt、FPS、帧号和 PTS 保持原样。CI 检查 41 个冻结文件及所有版本指纹。
- 新算法放独立源码目录、增加新清单；旧版本清单不覆盖。每次实验使用新的运行目录。
- 续跑核对提交、配置、输入、模型和已完成阶段产物，变化后拒绝混用旧缓存。
- GPU 未验收保持 pending，CPU 检查通过不自动改成 GPU 等价性通过。
- GitHub 当前私有仓库套餐不支持强制分支保护；自动检查能发现违规，但不能阻止有写权限者绕过检查。详见 [分支规则](docs/branch_policy.md)。

[项目管理说明](docs/project_management.md) 说明主线、版本、发布、服务器部署和临时文件清理规则。

## 其他研究路线与历史资料

以下路线保留在 `main`，通过 `vh.py run --config configs/pipelines/...` 选择。它们不是本次 v3 默认流程，新 GPU 推理状态按各自文档判断。

| 目录 | 用途 |
|---|---|
| `pipelines/ora_direct/` | ORA v5 时间选择，4B / 9B |
| `pipelines/spatial_crop/` | YOLO / Pose / BoT-SORT 空间路线 |
| `pipelines/ora_spatial/` | 本次 v3 源码、历史基准与早期 E1/E2/E3 |
| `pipelines/qwen_direct/`、`qwen_window/` | Qwen 定位 / 窗口评分 |
| `pipelines/reze/`、`pgl_sum/` | 其他时间模型研究 |
| `pipelines/ora_proposal/`、`ora_reranker/` | 历史设计与证据，完整性见各自 README |
| `pipelines/baselines/`、`editing/`、`sam_tracking/` | 基线、预览与历史跟踪入口 |

历史报告的初赛 56.43 分并非本次 v3 已确认成绩。当前 v3 没有已确认的官方排行榜分数。[历史结果](docs/history.md)、[实验登记](docs/experiments.md)、[环境](docs/environments.md)、[来源许可](docs/sources.md) 保存各自背景。

## 自动化检查

```bash
python tools/check_scene_baseline.py
python tools/check_project_policy.py
python -m pytest -q
python -m compileall -q src scripts tools pipelines prompts pipeline.py vh.py
python tools/audit_repo.py
```

GitHub 的 Engineering contracts 在 push / PR 时自动执行编译、常规 pytest（含冻结契约、项目策略、原 v3 测试）和仓库审计。回归还需 CPU Torch，CI 已安装。验证范围、历史产物 SHA256 和 GPU 待办见 [验收记录](docs/scene_workflow_validation.md)。
