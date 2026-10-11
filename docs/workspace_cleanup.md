# 本地工作区整理

整理日期：2026-10-04。开发、运行和登记统一从 `video-highlight-unified/` 进入。原始资料按用途归位，历史结果按路线或交付阶段保存，移动前后逐文件核对 SHA256。旧 Git 仓库、分支、未提交修改和远端保持原有状态。

**2026-10-11更新：**按用户要求，6个旧本地工作区和解压目录已在保存独立历史、未提交修改和结果后删除；两个本地开发分支已清理。以下2026-10-04的“暂留旧仓库”描述属于当时状态。当前处理、保留路径与恢复方式以 [旧工作区清理](local_workspace_retirement.md) 为准。GPT/AI参考新增 [版本说明](ai_reference_versions.md) 和本机索引，标注、视频及预测保持原样。

## 当前目录

```text
video-highlight-unified/
├── src/ configs/ prompts/ pipelines/      维护中的代码与实验配置
├── docs/                                 当前说明与历史结论
├── references/
│   ├── README.md catalog.json             可共享的参考资料目录
│   ├── assets.local.json                 本机逐文件哈希与原位置
│   └── local/                            原始资料，Git 忽略
│       ├── competition/                  赛题、通知、报告模板、样例
│       ├── analysis/                     旧方案分析与失败记录
│       ├── reports/                      56.43 空间方案原文
│       └── literature/                   已下载论文
├── datasets/local/aic_initial174/         原始初赛视频与画幅配置
└── artifacts/                            全部 Git 忽略
    ├── historical/                       Qwen、PGL、baseline 等旧结果
    ├── deliveries/ora4b_v5_426/           复赛时间/空间正式交付及 ZIP
    ├── previews/                         预览与对应诊断材料
    ├── source_archives/                  少量原始源码快照与部署来源
    ├── verification/                     历史验收记录
    └── workspace_cleanup/20261004/        整理计划、移动核对、删除清单
```

权重、虚拟环境和外部来源仓库继续由 `E:/AI/models`、`E:/AI/envs`、`E:/AI/venvs`、`E:/AI/sources` 管理；原始数据 ZIP 与微信原始附件保留外部位置。上述外部目录中只有已明确识别的本项目临时资料参与整理。

## 主要来源与新位置

| 原位置 | 新位置 |
|---|---|
| 根目录赛题、承诺书、模板、补充通知 | `references/local/competition/` |
| 根目录 ORA、PGL、Window、TimeLens 等分析 | `references/local/analysis/` |
| `TempSamp-R1/docs/` 中唯一的高分方案报告 | `references/local/reports/yolo_56_43_20260715.docx` |
| `E:/AI/tmp/` 中两份相同 REZE PDF | 保留一份到 `references/local/literature/` |
| 原初赛数据文件夹 | `datasets/local/aic_initial174/` |
| `cloud_temporal_20260816/` | `artifacts/historical/qwen_window_20260816/` |
| `temporal_reaggregation_20260817/` | `artifacts/historical/qwen_reaggregation_20260817/` |
| `pgl_sum_temporal_20260819/` | `artifacts/historical/pgl_sum_20260819/` |
| `submit/`、exp07、居中/Qwen 命名结果 | `artifacts/historical/` 各自目录 |
| `results/` 中复赛交付 | `artifacts/deliveries/ora4b_v5_426/temporal/`、`spatial/` |
| `previews/`、`cloud_review_20260815/` | `artifacts/previews/` |
| balanced 交接包、旧 ORA 生成/部署源码 | `artifacts/source_archives/` |
| `tmp/` 的验收结果 | `artifacts/verification/` |
| 根目录 4 个旧预览脚本 | 原文留 `artifacts/source_archives/legacy_preview_tools/`；可维护入口为 `tools/preview_submission.py` 与 `tools/preview_timeline.py` |

完整计数、可共享的移动目录表见 [整理摘要](migration/workspace_cleanup.json)。详细日志、原文件路径、逐文件哈希和每一项重复文件的保留位置存于本机整理目录。

本次移动3,279个文件，另外复制5份队友原始附件到本机来源区，保留2,173个文件的来源索引。删除1,538个文件，释放97,386,382字节，约92.87 MiB。其中1,176个为哈希相同的重复文件，其余为系统元数据和可重建缓存；这些数字不包含被清理的空目录。

## 删除与保留判断

删除经过完整哈希匹配的重复交接解压目录、重复下载的服务器快照、重复窗口分数/配置/通知，以及第二份相同 REZE PDF。重复文件必须存在内容一致的保留副本，不能仅凭文件名判断。

删除 `__MACOSX` AppleDouble 元数据、`.DS_Store`、Python 字节码、赛题 PDF 的可重建页面渲染和空 LibreOffice/临时目录。初赛视频、历史预测、最终交付、原始模型回答和不同实验版本保留。

原 balanced 交接 README、旧源码快照和历史命令保持原文。交接包内被去重的文件可能已不在原相对位置，通过 `tools/local_assets.py find --path <旧路径>` 找到保留副本。新操作使用本仓库运行文档，不能直接执行旧部署脚本。

`E:/AI/tmp/` 中遗留的一组 Git 对象只有对象目录，缺完整引用和工作区。已集中到 `artifacts/source_archives/github_upload_objects_20260820/`，暂不判定为可删除；它不是第四个维护入口。

## 56.43 分证据补充

新发现的 2026-07-15 报告写明 YOLO11m、15 帧平滑、16:9 人物锚点 0.45、9:16 Pose 融合 0.50 与内部边界 trim 0.25 秒。它描述 48,895 帧的提交；现有 exp07 ZIP 实际包含 45,234 帧，不能将二者合并成一个已确认实验。报告登记为 `hist-yolo-pose-5643-report`，exp07 原记录保留并标明冲突。

报告列出的原始时间 hints、trim/最终合成脚本、完整最终 JSONL、模型权重哈希和原始 commit 未找齐。提取参数仅供复现研究与对照，不加入当前正式比赛默认配置，也不声称已经重新取得 56.43 分。

## 以后怎么放

源码放 `src/` 或相应 `pipelines/`；可运行配置放 `configs/pipelines/`；运行 Prompt 按版本放 `prompts/`；说明放 `docs/`；外部资料的介绍与来源链接放 `references/`。原始论文、历史个人记录、赛事附件放 `references/local/`。实验原始结果放 `runs/<experiment_id>/`，保留的指标和摘要登记进 `experiments/records/`。

已有成绩文件不要覆盖，也不要仅凭名字认定分数。先登记来源和哈希，不完整的作者、commit、模型版本明确写缺失。清理新重复文件时照样先核对保留副本。

## 本机使用示例

在新仓库根目录建立初赛索引：

```bash
python tools/build_index.py --videos datasets/local/aic_initial174/video --configs datasets/local/aic_initial174/jsonl --output datasets/local/aic_initial174/test_index.json
python tools/locate_artifacts.py --experiment hist-exp07-boundary-b
python tools/local_assets.py find --path handoff_balanced_42_20260816/artifacts/original_qwen_20260816/temporal_handoff.jsonl.raw.jsonl
```

查看整段原视频、入选框与裁剪，或比较两份提交结果：

```bash
python tools/preview_timeline.py --videos /path/to/videos --prediction /path/to/predictions.jsonl --ids 0,1 --out runs/preview-a
python tools/preview_timeline.py --videos /path/to/videos --prediction /path/to/a.jsonl --compare /path/to/b.jsonl --ids 0,1 --out runs/preview-comparison
```

预览是无声诊断视频，使用源FPS显示时间；不能代替原视频PTS校验、真值或比赛评测。默认拒绝覆盖已有MP4，确需重做时指定新的输出目录或使用 `--overwrite`。旧3个按特定实验命名的渲染脚本原文保留在来源区，可维护逻辑合为这个通用入口。

## 整理验收

保留文件与重复保留副本全部进行SHA256复查；历史实验的artifact指纹必须仍匹配。通用渲染入口用横/竖画幅合成视频分别验证单份提交、双份对照、空预测，实际生成6条视频，每条解码20帧、输出30帧，并检查截图。

源码编译、回归测试、Git内容审计及链接核对结果记录在整理摘要中。未连接模型服务器或重跑GPU推理，未提交比赛官网。

旧仓库的远端 Archive 建议见 [迁移报告](migration_report.md)。本次没有删除旧仓库或执行远端 Archive。
