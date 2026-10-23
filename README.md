# SFT-v2 可搬移比赛推理分支

本实验分支只维护：新测试集 → ORA + 指定 Adapter → 高光原帧 → 空间裁剪 → JSONL/ZIP。
从 [独立环境部署与运行](docs/submission_portability.md) 开始；环境安装入口为
`scripts/setup_submission_env.py`，后续统一执行 `bash scripts/run_submission.sh --input-root <测试集目录>`。
大权重、测试视频、答案、个人配置及运行结果不进入 Git。真实验收记录见
[部署验收记录](docs/submission_deployment_acceptance.md)，不以历史比赛分数代替新代码验收。

---

以下为保留的主线历史说明，不是本推理分支的部署入口；其中未包含的研究模块请查看 `main`。

# 通用视频高光剪辑（主线历史说明）

> 本 `video-highlight-unified-new` 目录当前的 SFT-v2 比赛推理/交付入口是
> `scripts/run_submission_pipeline.py`。先读 [可搬移输入、运行与打包说明](docs/submission_portability.md)
> 和 [容错审查采纳及待验项](docs/robustness_review_response.md)。下方 `scene_general_v3`
> 是继承主线说明，不是本次 new 推理包的默认启动方式。新入口 CPU/故障注入、两卡公开小样本全流程及真实中断续跑已验证；不代表官方成绩。

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
| 前半段三方选段播放、直接文件夹导出 | [导出说明](docs/temporal_result_export.md) | 可用；复制已有文件，不跑模型 |
| 整片一次调用、实际 2 FPS 直接选高光 | [实验入口与结果](docs/ora_whole_video_2fps.md) | 200–300 共 101 条跑完，95 有效、6 失败；仅时间定位，无官方成绩 |
| 时间边界与原帧号 | [公共时间规则](docs/time_contract.md) | 严格模式默认；诊断宽容模式只修复有记录的输出和至多 0.5 秒片尾超出 |
| 原始 MP4 直接输入 ORA 全量诊断 | [输入、回放与离线审计](pipelines/ora_native_video/README.md) | 426 条完成；严格 129 有效；宽容离线复核 393 有效、33 失败；无官方成绩 |
| ORA 正式实验与统一输出合同 | [实验标准](docs/experiment_standard.md) / [v2 模板](configs/templates/ora-native-video-experiment-v2.json) | 新实验默认 xgrammar 约束与严格校验；schema、Prompt、配置和源码哈希进入运行身份 |
| Video-ORA SFT 训练数据 | [人工标注与训练 JSONL 规范](docs/video_ora_sft_data.md) | 格式已确认；426 条 SFT-v1 train/val 已生成并通过结构验收，真实训练待执行 |
| Video-ORA-4B SFT-v1 训练方案 | [目标、模型、数据与验收门禁](docs/video_ora_sft_v1.md) | 方案已确定；ms-swift、Loss Mask、3060 显存和 Adapter 回载待验证 |
| Video-ORA SFT 服务器操作 | [连接、只读核查、转换、验收与排错](docs/video_ora_sft_server_guide.md) | `.env` SSH 连接及 426 条全量转换已实测；数据目录和收据见指南 |
| 训练服务器连接 | 专用无密码 SSH key | 2026-10-23 已实测 key 登录与 SCP；仍须先连接学校 VPN |

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

### 训练服务器 SSH key

Windows 先连接学校 VPN，再使用专用无密码 key：

```powershell
$Key = "$HOME\.ssh\codex-teacher1-20261023-v2"
ssh -i $Key teacher1@172.31.167.10
```

上传单个文件：

```powershell
scp -i $Key .\local-file.py teacher1@172.31.167.10:/目标/目录/
```

key 登录不能替代 VPN。私钥只保存在本机 `.ssh` 目录，不得复制到仓库、
聊天记录、公开文档或服务器项目目录。2026-10-23 已实测 SSH key 登录成功，
且 SCP 上传后的本地/远端 SHA256 一致。历史 `.env` 密码包装器
`scripts/teacher1_ssh.ps1` 仍可用于兼容旧脚本；新操作优先使用上述
专用 key。

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

本机旧工作区已经归档并清理，恢复方法见 [本地旧版本清理](docs/local_workspace_retirement.md)。GPT/AI编辑参考按5条示范、10条示范、全量426条、局部备选和JSONL导出登记，见 [参考版本说明](docs/ai_reference_versions.md)。本机播放和文件索引在 `references/local/ai_references/README.md`；从GitHub克隆后需另取本地成果。原始标注、视频、预测和归档包均不进入 Git。

## 自动化检查

```bash
python tools/check_scene_baseline.py
python tools/check_project_policy.py
python -m pytest -q
python -m compileall -q src scripts tools pipelines prompts pipeline.py vh.py
python tools/audit_repo.py
```

GitHub 的 Engineering contracts 在 push / PR 时自动执行编译、常规 pytest（含冻结契约、项目策略、原 v3 测试）和仓库审计。回归还需 CPU Torch，CI 已安装。验证范围、历史产物 SHA256 和 GPU 待办见 [验收记录](docs/scene_workflow_validation.md)。
