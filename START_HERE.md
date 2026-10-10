# 当前服务器开发入口

本次整理的是 **scene_general_v3**，基准算法提交 `36695a2`，交接提交 `81f2257`。原目录保留为历史快照。新开发目录与统一入口：

```bash
cd /root/rivermind-data/development/video-highlight-unified
cat docs/scene_workflow.md
/root/rivermind-data/development/vh workflow versions
/root/rivermind-data/development/vh workflow plan \
  --config configs/workflows/compat.json --local configs/workflows/server.local.json
```

`server.local.json` 已填写本服务器路径，不含 SSH 密码。新运行前把 `run_root` 改成新的实验目录；小样本复制运行配置，填写按原顺序排列的 `ids`。原视频、模型和输入继续引用旧目录。

原始固定结果与交接说明：

```text
/root/rivermind-data/aic_spatial_trials_20261008/START_HERE.md
/root/rivermind-data/aic_spatial_trials_20261008/output/scene_temporal_full_426_20261011_v3/
```

只跑原空间基准用 `compat.json`；已有 PTS 时间交接用 `spatial.json`；完整时间→空间用 `full.json`；I/O 优化实验用 `optimized.json`。不同版本使用独立运行目录。旧 E1/E2/E3 不运行这套 v3。

```bash
# 先申请 GPU；plan 可在无 GPU 时查看
/root/rivermind-data/development/vh workflow run \
  --config configs/workflows/compat.json --local configs/workflows/server.local.json

/root/rivermind-data/development/vh workflow status --run-root /实际运行目录
/root/rivermind-data/development/vh workflow resume --run-root /实际运行目录
```

新运行的正式提交为 `final/submission.zip`。历史比赛 ZIP 保持原位置和 SHA256。

WebUI 已部署在服务器 `127.0.0.1:8189`。自己的电脑运行下面的 SSH 转发后，浏览器打开 `http://127.0.0.1:8189`：

```bash
ssh -N -L 8189:127.0.0.1:8189 root@hn01-ssh.gpuhome.cc -p 30192
```

终端界面：

```bash
/root/rivermind-data/development/vh workflow tui --local configs/workflows/server.local.json
```

界面支持版本选择、计划、输入准备、运行、续跑、日志、资源和正式产物查看；WebUI 可以下载历史与新运行的正式产物。无 GPU 时可操作 CPU 输入准备，完整推理按钮等待 GPU。[界面使用说明](docs/scene_workflow_ui.md) 包含服务重启和全部操作。

检查日期：**2026-10-10 UTC / 2026-10-11 香港时间**，当时 GPU 已释放。目前完成工程和 CPU 验证；新入口的 GPU 全链路验收待申请 GPU 后执行，I/O 样本耗时不代表全量推理提速。

[统一运行说明](docs/scene_workflow.md) · [功能索引](docs/feature_index.md) · [历史交接](pipelines/ora_spatial/SERVER_HANDOFF.md) · [后续计划](pipelines/ora_spatial/PRODUCTIZATION_PLAN.md)
