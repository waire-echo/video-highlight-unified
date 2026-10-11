# 新人唯一入口

本项目唯一维护主线是 **main**，当前推荐入口为 **scene_general_v3** 的 `vh workflow`。先读根目录 [README](README.md)；功能位置查 [功能索引](docs/feature_index.md)，项目管理规则查 [项目管理](docs/project_management.md) 和 [分支规则](docs/branch_policy.md)。

## 第一步：确认当前代码和默认版本

```bash
cd /root/rivermind-data/development/video-highlight-unified
git branch --show-current
git rev-parse HEAD
git status --short
/root/rivermind-data/development/vh workflow versions
```

维护工作区应检出 main。不同算法通过版本清单选择，不切到历史功能分支猜测版本。默认是 `scene-general-v3-compat-20261010`，原 I/O 与冻结 baseline；统一实验版及 I/O 优化版使用另一个全新运行目录。

`configs/project_policy.json` 登记主线和默认配置，界面从同一策略获取默认版本。core、Prompt、FPS、原始帧号和 PTS 不改。GPU 未分配时可看计划、准备已有输入和查看历史结果；完整推理仍需申请 GPU。

## 第二步：生成计划

```bash
/root/rivermind-data/development/vh workflow plan \
  --config configs/workflows/compat.json --local configs/workflows/server.local.json
```

本服务器 `server.local.json` 已填写既有路径，不含密码，受 Git 忽略。CLI 新运行前填写全新的 run_root；WebUI 自动生成唯一目录。小样本 IDs 按原 manifest 顺序填写，handoff / highlight_frames / test_index 必须与所选 IDs 一致；服务器现有全量交接使用 all。

- 原空间基准：compat.json。
- 已有时间交接：spatial.json。
- 完整时间→空间：full.json。
- I/O 优化实验：optimized.json。

完整命令与续跑规则见 [统一工作流](docs/scene_workflow.md)。

## 第三步：使用界面或 CLI

WebUI 自己电脑先建立 SSH 转发，保持连接，再打开 `http://127.0.0.1:8189`：

```bash
ssh -N -L 8189:127.0.0.1:8189 root@hn01-ssh.gpuhome.cc -p 30192
```

服务器重启后按 [界面说明](docs/scene_workflow_ui.md) 启动 WebUI。TUI 在服务器执行：

```bash
/root/rivermind-data/development/vh workflow tui --local configs/workflows/server.local.json
```

无 GPU 可执行已有输入准备：

```bash
/root/rivermind-data/development/vh workflow run \
  --config configs/workflows/compat.json --local configs/workflows/server.local.json --until handoff
```

申请 GPU 后省略 --until handoff 执行推理；状态与续跑：

```bash
/root/rivermind-data/development/vh workflow status --run-root /实际运行目录
/root/rivermind-data/development/vh workflow resume --run-root /实际运行目录
```

最终提交为新运行 `final/submission.zip`。GPU pending 代表新入口尚未完成真实模型复跑，不把 CPU 验收称为 GPU 已通过。

## 原始资料和验收

原快照：`/root/rivermind-data/aic_spatial_trials_20261008`。原始 426 视频正式结果：

```text
/root/rivermind-data/aic_spatial_trials_20261008/output/scene_temporal_full_426_20261011_v3/
```

历史交接见 [SERVER_HANDOFF](pipelines/ora_spatial/SERVER_HANDOFF.md)，验收与原结果 SHA256 见 [验证记录](docs/scene_workflow_validation.md)。旧 E1/E2/E3 与 YOLO/Pose 是各自的历史/研究路线，不是本次默认入口。
