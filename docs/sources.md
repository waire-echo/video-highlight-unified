# 活跃代码来源

仓库保留当前流程与历史对照。源码按各自来源保留许可，根目录Apache许可不覆盖第三方单独许可的模块。

- 时间阶段来自队伍仓库 `waire-echo/video-highlight-editing`，提交 `65fa242f6fbdf47c9ffdadf3dbed39fc8dbc902c` 下的 `TempSamp-R1/baseline/orarl_halfday`。模型和工程版本为 `orarl-4b-v5`。这里沿用事件发现、主体问答、复看筛选、原生时间定位和修复流程，没有加入文本Reranker或大模型训练。
- 空间阶段来自队友提供的 `ORA_RUN_v2_TEAM_20261001`，包含YOLO、Pose和BoT-SORT。检测、姿态和跟踪核心保持原方案，工程入口改为动态索引和统一配置。
- 接力模块来自本项目实际运行过的有序预读和跨视频并行脚本，增加了硬件计划、运行时资源记录和新任务并发控制。
- 预览工具来自已有的预测预览脚本，支持裁剪、原画画框、左右对比、指定/随机视频和拼接。
- Qwen/window/REZE/PGL及公共几何/校验来自旧队伍仓库本地78065e4和未提交修改，原始hash与dirty标记在迁移清单。
- ORA9B在65fa242核对，与4B core只差VERSION，共用实现，以模型revision与served name区分；最终空间尚未验收。
- 官方center/qwen baseline来自TempSamp-R1的3fd4e85，作为参考保留。
- PGL沿用Mr.HiSum aeb667f71239887034c2974062a341d312762867的架构与权重键；学术非商业许可保留在licenses/MR.HiSum.txt，不改授Apache许可。
- partial ORA9B光流参考来自本机历史脚本，依赖缺失，仅保留可复用部分。weighted F1为题面参考，不是主办方评分脚本。

相关官方来源：[Video-ORA模型](https://huggingface.co/OraRL/Video-ORA-4B)、[OraRL](https://github.com/HVision-NKU/OraRL)、[TempSamp-R1](https://github.com/HVision-NKU/TempSamp-R1)、[Ultralytics](https://github.com/ultralytics/ultralytics)。

保留的Apache-2.0文件对应上游代码来源。模型与依赖的许可证以各自发行文件为准；Ultralytics依赖许可见其官方仓库。权重、视频、真实预测、个人接手记录和认证信息均不属于源代码仓库内容。
