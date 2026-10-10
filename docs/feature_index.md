# 功能入口索引

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
| TUI / 三路同步对比 | 产品化计划 | 待开发 |
