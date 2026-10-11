# 项目管理与服务器部署

## 三层身份

| 身份 | 唯一来源 | 用途 |
|---|---|---|
| 维护代码 | Git main 与发布标签 | CLI、界面、状态机、版本选择和工程工具 |
| 算法版本 | configs/workflows/versions/*.json | 冻结源码、Prompt、模型 revision、输入方式与 I/O |
| 一次运行 | 独立 run_root 内 plan.json / frozen.json | 实际提交、输入、权重、阶段、错误、检查点和最终产物 |

用版本清单切换实验，不用切 main 与实验分支来猜算法。main 可以演进；旧版本的源码和清单保留。需要复现旧运行时，按 frozen.json 检出当时提交。

## 当前路线

推荐路线为 ORA v5 时间选择 → 原 PTS / 精确帧集合交接 → scene_general_v3 空间 → 校验 → 正式交付。默认 compat 直接使用冻结 baseline，跳过新的时间选择。空间 v3 和 YOLO/Pose 是不同的路线，不能把两者写成同一默认流程。

`configs/project_policy.json` 登记维护分支、默认版本、四种工作流配置、文档入口和待办。界面从该策略读取默认版本；自动测试校验策略与配置一致。

CLI、TUI 和 WebUI 共用同一控制器和状态机，不分别维护算法。正式输出为 final/submission.zip，原生 spatial/results.zip 用于证据归档。

## 服务器目录

| 路径 | 角色 | 维护规则 |
|---|---|---|
| /root/rivermind-data/development/video-highlight-unified | 唯一维护工作区，检出 main | 只在这里部署维护代码 |
| /root/rivermind-data/development/vh | 启动快捷命令 | 调用该工作区和既有客户端解释器 |
| /root/rivermind-data/aic_spatial_trials_20261008 | 原 v3 快照、输入、镜头缓存和正式结果 | 保留原内容；START_HERE 仅指向新入口 |
| /root/rivermind-data/ora4b_v5_20261001 | 原视频、模型权重和推理环境 | 通过配置引用，不复制进 Git |
| 维护工作区 runs/<实验ID> | 今后的实际实验和产物 | 独立目录、来源冻结、可续跑 |
| 维护工作区 runs/ui | 界面自身的必要状态与服务日志 | 正常运行资料，不存本次验收样本 |

`server.local.json` 是本服务器长期使用的路径配置，受 Git 忽略；保留它才能直接启动。不得把密码/token 写入配置或持久化到服务器的 Git credential 中。服务器不能直连 GitHub 时，可从控制端转送 Git bundle，部署后立即删除 bundle。

本次任务的 CPU 测试运行、缓存重放副本、浏览器截图、临时验收报告、测试配置、测试依赖目录、部署包、测试 stash 和本地控制端副本在验收后删除。保留 Git 中的代码/文档/测试、原数据/权重/正式结果，以及实际服务必需的配置和日志。后续真实实验运行不按本次临时清单删除。

## 发布检查

1. 确认工作区无未提交修改，记录 main 的实际 commit。
2. 通过 Engineering contracts；冻结文件、默认兼容版、旧版本和文档入口检查通过。
3. 按机器路径生成计划，检查 TUI / WebUI 和已有输入准备；无 GPU 时标明无法验证的部分。
4. 部署同一 main commit，服务只监听 loopback，通过 SSH 转发访问。
5. 对照原始结果 SHA256，确认没有修改视频、模型、Prompt、帧号和 PTS。
6. 清理本次创建的临时文件；机器配置和生产服务日志留在明确位置。
7. 每次功能变化同步 README、功能索引和相应使用说明；已完成内容不继续写在“待开发”中。

## 完成与待办

工程自动化、TUI/WebUI、版本隔离、原帧交接、续跑、日志与产物下载已通过 CPU 验收。GPU 新推理、完整时间→空间新回答对比、优化版新推理等价性仍 pending。三路同步对比与预计剩余时间未实现，后续开发先复用现有状态机。

历史比赛成绩、原算法全量运行和新工程入口验收分别记录，见 docs/scene_workflow_validation.md，不互相代替。
