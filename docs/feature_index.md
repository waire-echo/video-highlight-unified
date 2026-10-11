# 功能入口索引

唯一维护主线为 main，默认版本与工作流配置见 [项目策略](../configs/project_policy.json)。默认 scene_general_v3 兼容版；主线、算法版本、运行目录关系见 [项目管理](project_management.md)。

| 功能 | 推荐入口 | 状态 |
|---|---|---|
| 服务器接手 | `START_HERE.md` | 可用 |
| v3 版本选择 | `vh.py workflow versions` | 可用 |
| 执行计划 | `vh.py workflow plan` | 可用，只读 |
| v3 空间基准 | `workflow run` + `compat.json` | 工程完成，新启动器 GPU 待验收 |
| 前半时间选择 | 同上 + `full.json --until temporal` | 已接入，GPU 待验收 |
| 显式时间→空间适配 | 同上 + `spatial.json --until handoff` | 426 视频 CPU 验收通过 |
| 完整时间→空间 | 同上 + `full.json` | 已接入，GPU 待验收 |
| 断点与状态 | `workflow resume/status` | CPU 与故障恢复测试通过 |
| v3 格式校验 | `workflow validate` | 历史 426 视频通过 |
| 解码性能与等价检查 | `workflow verify-io` | 实际样本通过 |
| 其他研究路线 | `vh.py run --config configs/pipelines/...` | 保留 |
| 旧 E1/E2/E3 | `pipelines/ora_spatial/scripts/run_gpu.sh` | 历史入口 |
| TUI | `workflow tui --local ...` | SSH 终端与 CPU 准备实测通过 |
| WebUI | `workflow webui --local ...` | 已部署，浏览器、CPU 准备/续跑、ZIP 下载通过 |
| 三路同步对比 / 预计剩余时间 | 产品化计划 | 待开发 |
| GPT / AI参考分类、播放与JSONL | [参考版本](ai_reference_versions.md) | 本机索引已整理；原成果Git忽略 |
| 旧本地代码版本与历史恢复 | [旧工作区清理](local_workspace_retirement.md) | 6个旧目录已归档清理 |

[README](../README.md) · [唯一入口](../START_HERE.md) · [分支约束](branch_policy.md) · [界面使用](scene_workflow_ui.md)
