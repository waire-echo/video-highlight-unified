# 参考资料索引

这里保存资料目录和阅读说明。原始赛题、个人历史分析、测试集相关截图、论文 PDF 和报告保存在 `local/`，被 Git 忽略。本机可以打开以下链接；从 GitHub 克隆的成员需要自行取得原文件，查看 [目录规范](../docs/workspace_cleanup.md)。

## 赛题与交付

| 资料 | 位置 |
|---|---|
| 赛题 PDF | [原文件](local/competition/基于视频大模型的运动场景高光剪辑.pdf) |
| 技术报告模板 | [原文件](local/competition/【技术报告】基于视频大模型的运动场景高光剪辑测.docx) |
| 逐帧稠密标注说明 | [原文件](local/competition/关于测试集.txt) |
| 模型规模补充 | [原文件](local/competition/补充说明.txt) |
| 承诺书 | [原文件](local/competition/参赛选手承诺书.pdf) |
| 最小提交样例 | [JSONL](local/competition/format_examples/submit_example.jsonl) |

## 历史设计与失败记录

以下原文保留当时观点、路径和限制，不作为当前推荐配置。当前状态以 `docs/history.md` 和实验记录为准。

| 路线或问题 | 资料 |
|---|---|
| 模块接口与分阶段验收 | [阶段说明](local/analysis/PIPELINE_MODULES_AND_STAGE_ISOLATION.md) |
| 时间选择的总体诊断 | [诊断与路线图](local/analysis/HIGHLIGHT_TEMPORAL_DIAGNOSIS_AND_ROADMAP.md) |
| ORA 新旧结果 | [诊断](local/analysis/ORA_DIAGNOSIS_AND_IMPROVEMENT_20260922.md)、[直接输出对照](local/analysis/ORA_DIRECT_OUTPUT_COMPARISON_20260923.md) |
| ORA 流程设计 | [流程计划](local/analysis/ORA_FULL_PIPELINE_PLAN_20260923.md) |
| PGL-SUM 负面实验 | [失败审计](local/analysis/PGL_SUM_FAILURE_AUDIT_20260820.md) |
| Window scoring / REZE | [后续计划](local/analysis/WINDOW_SCORE_REZE_NEXT_PLAN.md) |
| TimeLens / LLaVA-ST 储备 | [历史计划](local/analysis/PLAN_B_TIMELENS_AND_LLAVAST.md) |
| 数据时长统计 | [统计原文](local/analysis/视频时长统计.md) |
| 56.43 空间方案 | [原始 DOCX](local/reports/yolo_56_43_20260715.docx)、[证据与缺口](../pipelines/baselines/yolo_pose_5643_reference/README.md) |

ORA 逐视频对照、抽查结论和渲染说明与对应图像一起保存在 `artifacts/previews/ora35_vs_exp07_audit/`、`artifacts/previews/ora_prompt_comparison_20260924/`。这些是历史诊断材料，不是真值标注或正式评测。

## 论文与上游

以下链接从历史项目资料整理，本轮未联网重新核实内容。这里记录的是项目中使用或讨论过的来源，不代表已经复现训练。

| 资料 | 官方入口 | 项目用途 |
|---|---|---|
| Video-ORA | [OraRL](https://github.com/HVision-NKU/OraRL)、[4B 模型](https://huggingface.co/OraRL/Video-ORA-4B)、[9B 模型](https://huggingface.co/OraRL/Video-ORA-9B) | 当前时间定位流程 |
| TempSamp-R1 | [代码](https://github.com/HVision-NKU/TempSamp-R1)、[论文](https://arxiv.org/abs/2509.18056) | 官方 baseline、训练方法储备 |
| Qwen2.5-VL | [3B 模型](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)、[7B 模型](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) | 直接定位、窗口评分 |
| Qwen Text Reranker | [官方实现](https://github.com/QwenLM/Qwen3-Embedding) | 历史 reranker 路线，完整运行代码尚缺 |
| PGL-SUM | [代码](https://github.com/e-apostolidis/PGL-SUM)、[MR.HiSum](https://github.com/MRHiSum/MR.HiSum) | 查询无关的摘要 baseline；许可见 `licenses/` |
| DSNet / CA-SUM | [DSNet](https://github.com/li-plus/DSNet)、[CA-SUM](https://github.com/e-apostolidis/CA-SUM) | 历史候选，未并入已完成实验 |
| REZE | [论文](https://arxiv.org/abs/2608.04480)、[本机 PDF](local/literature/reze_2608.04480.pdf) | Yes/No 分数曲线参考 |
| TimeLens / TimeLens2 | [TimeLens](https://github.com/TencentARC/TimeLens)、[TimeLens2](https://github.com/MCG-NJU/TimeLens2) | 带查询的时间定位储备 |
| LLaVA-ST | [代码](https://github.com/appletea233/LLaVA-ST)、[论文](https://arxiv.org/abs/2501.08282) | 时空定位储备 |
| QVHighlights | [论文](https://arxiv.org/abs/2107.09609)、[Moment-DETR](https://github.com/jayleicn/moment_detr) | 带文字查询的任务和数据集参考 |

模型版本以配置中的 revision 为准。正式赛事限制以原始赛题及最新比赛通知为准，历史报告的模型建议可能已过时。

## 查找本机文件

```bash
python tools/local_assets.py list --kind historical-report
python tools/local_assets.py find --path tmp/ora4b_v5_new_dataset_20261001/code/core.py
python tools/local_assets.py verify
python tools/locate_artifacts.py --experiment hist-qwen-balanced-42
```

`catalog.json` 是可共享的目录索引；`assets.local.json` 是本机逐文件来源与 SHA256，不能提交 Git。旧位置映射和删除证据位于 `artifacts/workspace_cleanup/20261004/operations.local.json`。
