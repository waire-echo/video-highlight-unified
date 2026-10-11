# 本地旧工作区清理与恢复

整理日期：2026-10-11。本轮按用户“优先删除多余本地版本”的要求，先核对与保存独立资产，再删除旧工作区。当前唯一维护工作区为 `video-highlight-unified/`，代码以 `main` 为准。

## 实际处理

| 已删除的本地目录 | 判断与保留位置 |
|---|---|
| `video-highlight-unified-server-import/` | 源码和提交已在 main；正式 ZIP、合并报告移入 `artifacts/deliveries/scene_general_v3_full_426_20261011/`，复现输入移入 `artifacts/source_archives/ora_spatial_v3_repro_20261011/` |
| `video-highlight-unified-v6-wip/` | 未经过真实模型效果验证，不合入默认流程；独立提交、stash和9个未提交文件保存为 bundle / ZIP |
| `video-highlight-pipeline/` | 全部本地分支、远端跟踪引用及标签的提交均已包含于统一 main，可直接删除重复工作区 |
| `video-highlight-editing/` | 有独立分支与历史；全部 Git 引用保存为 bundle，11个未提交文件保存为 ZIP；现用实现已迁入 main，旧实验输出另存压缩归档 |
| `TempSamp-R1/` | 干净上游克隆；历史存 bundle，项目使用的官方 baseline 已在 `pipelines/baselines/`，其余内容按上游链接取得 |
| `server-import-20261010/` | 两份展开源码逐文件与原始 tar 核对一致；原始 tar 保留于 `artifacts/source_archives/server_snapshots_20261010/`，删除解压副本 |

删除了两个本地开发分支：已合并的 `feat/import-ora-spatial-server-20261010`，以及完成归档的 `feat/ora-evidence-v6`。v6的两次独立提交 `3064747` / `8aa58f7` 仍由本地标签 `archive/ora-evidence-v6-20261011` 保留，原 stash `bef5b4e` 也保留。**没有删除或 Archive 任何远端仓库，没有删除远端分支。**

## 归档目录

以下全部被 Git 忽略，保存在本机；需要跨机器接手时单独备份或传输，Git clone 不会取得它们。

```text
artifacts/source_archives/retired_workspaces_20261011/
├── manifest.local.json                  来源、处理、每个资产指纹
├── ora_evidence_v6.bundle                v6 Git 历史及同步前 stash
├── ora_v6_uncommitted.zip                9个文件原文、binary patch、状态
├── video_highlight_editing.bundle        旧队伍仓库所有 Git 引用
├── old_editing_uncommitted.zip           11个文件原文、binary patch、状态
├── old_editing_unique_outputs.zip        唯一旧实验输出及原位置清单
└── TempSamp_R1.bundle                    上游克隆的 Git 历史
```

旧队伍仓库的640个输出文件全部有保留位置：627个唯一文件进入压缩包，13个重复文件指向相同哈希的现存文件。包括 Qwen 顺序推理、Prompt对照和 YOLO 诊断，不把失败实验视为垃圾，也不把这些文件认成 GPT 标答。`yolov8n.pt` 移到 `models/legacy/yolov8n.pt`，不进入 Git，不替代当前模型。

本轮删除6个目录、2,163个原位置文件，原占用57,573,289字节。这个数是被删除目录的占用，**不是净释放量**；Git历史、未提交文件、结果压缩包和权重已在新位置保留。

逐文件清单、删除收据和参考版本复核在 `artifacts/workspace_cleanup/20261011/`。保留副本逐文件做 SHA256；ZIP检查CRC及成员内容；Git bundle验证可独立恢复。删除前再次核对原文件没有改变。

## 需要恢复旧实验时

从统一仓库根目录执行，目标应为一个不存在的临时目录：

```bash
git bundle verify artifacts/source_archives/retired_workspaces_20261011/ora_evidence_v6.bundle
git clone --branch archive/ora-evidence-v6-20261011 \
  artifacts/source_archives/retired_workspaces_20261011/ora_evidence_v6.bundle \
  /path/to/new-temporary-v6-checkout

git clone --branch integration-main \
  artifacts/source_archives/retired_workspaces_20261011/video_highlight_editing.bundle \
  /path/to/new-temporary-editing-checkout
```

对应未提交ZIP中的 `working_tree/` 是原文件内容，按其相对路径还原；`snapshot.json` 记录状态与指纹，`tracked_changes.patch` 是另一种还原已跟踪修改的方式。原文还原和patch选一种，不叠加应用。未跟踪文件单独还原，删除项按快照处理。

临时恢复用于查看和比较。有价值的新改进按 [分支规则](branch_policy.md) 迁入统一仓库，不恢复长期多仓维护。

## 有用的历史与仍待确认的内容

主线已有 baseline、旧 ORA、PGL-SUM、Qwen window 等路线；历史成绩和原始资产仍在 `artifacts/`，没有重新推理或改动当前比赛流程。v6保留作为可选证据复核实验，工程测试不等于模型效果验证。

旧初赛174条 GPT 原始标注、56.43的完整生成链路、独立 reranker 等缺失证据沿用 [迁移报告](migration_report.md)。本次没有判定它们无价值，也没有假称已找齐。GPT/AI参考的清晰分类见 [版本说明](ai_reference_versions.md)。

## 远端建议

`thekfjie/video-highlight-pipeline` 的已知内容已迁入，可以将其远端设为历史 Archive 并指向新仓库。旧 editing 远端先核实队友未交接的分支与成果，再决定 Archive。TempSamp-R1 是独立上游。以上只是建议，本轮未执行远端变更。
