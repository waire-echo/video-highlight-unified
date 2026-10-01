# 活跃代码来源

仓库只保留当前双阶段流程和必要工具。

- 时间阶段来自队伍仓库 `waire-echo/video-highlight-editing`，提交 `65fa242f6fbdf47c9ffdadf3dbed39fc8dbc902c` 下的 `TempSamp-R1/baseline/orarl_halfday`。模型和工程版本为 `orarl-4b-v5`。这里沿用事件发现、主体问答、复看筛选、原生时间定位和修复流程，没有加入文本Reranker或大模型训练。
- 空间阶段来自队友提供的 `ORA_RUN_v2_TEAM_20261001`，包含YOLO、Pose和BoT-SORT。检测、姿态和跟踪核心保持原方案，工程入口改为动态索引和统一配置。
- 接力模块来自本项目实际运行过的有序预读和跨视频并行脚本，增加了硬件计划、运行时资源记录和新任务并发控制。
- 预览工具来自已有的预测预览脚本，支持裁剪、原画画框、左右对比、指定/随机视频和拼接。

相关官方来源：[Video-ORA模型](https://huggingface.co/OraRL/Video-ORA-4B)、[OraRL](https://github.com/HVision-NKU/OraRL)、[TempSamp-R1](https://github.com/HVision-NKU/TempSamp-R1)、[Ultralytics](https://github.com/ultralytics/ultralytics)。

保留的Apache-2.0文件对应上游代码来源。模型与依赖的许可证以各自发行文件为准；Ultralytics依赖许可见其官方仓库。权重、视频、真实预测、个人接手记录和认证信息均不属于源代码仓库内容。
