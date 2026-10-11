# 本地 GPT / AI 参考版本

整理日期：2026-10-11。这里登记独立编辑参考与它们的派生文件。原始标注、视频与完整预测只保存在本机，Git 中只保存说明。历史实验成绩仍查 [实验记录](experiments.md) 和 [历史结果](history.md)。

**本机统一入口：`references/local/ai_references/README.md`**。机器可读索引为 `references/local/ai_references/registry.local.json`，含路径、覆盖范围、父版本和 SHA256。所有路径相对于仓库根目录；从 GitHub 克隆后需要另外取得本地成果。

## 已找到的版本

| 参考 ID | 范围和用途 | 原成果位置 |
|---|---|---|
| `ai-scenery5-20261008-v1` | 5 条风景编辑示范，历史小样本 | `runs/manual-landscape-edit-20261008/` |
| `ai-scenery10-20261009-v1` | 10 条风景编辑示范，另一组小样本 | `runs/scenery10-manual-20261009-1400/` |
| **`ai-full426-20261011-v1`** | **当前全量参考：426 条、546 段** | `runs/ai-reference-426-20261009/` |
| `ai-scenery3-conservative-20261010-v1` | 266 / 268 / 365 的保守备选 | 上述全量目录的 `variants/conservative/` |
| `ai-scenery3-aggressive-20261010-v1` | 同 3 条的略激进备选，已用于当前全量默认 | 上述全量目录的 `variants/aggressive/` |
| `ai-full426-jsonl-20261011-v1` | 当前全量参考的 JSONL 导出，426 条、77,198 帧 | 上述全量目录的 `reference_jsonl/` |

共有 **3 组独立标注、2 个局部备选和1份格式导出**，不是6套全量标答。全量主目录与略激进备选的上述3条选段和裁剪关键帧一致。本轮只建索引、修正说明，没有修改标注、成片或导出预测。

## 怎么看、怎么比较

- 看当前全量：打开 `runs/ai-reference-426-20261009/exports/index.html`。
- 看具体理由：打开 `annotations/<ID>.json`；原帧轨迹在 `exports/<ID>/dense_crops.json`。
- 看3条风景备选：打开 `variants/aggressive/exports/index.html`；保守成片分别在 `variants/conservative/exports/<ID>/edit.mp4`。
- 查配置、原片、帧号和指纹：结合 `metadata/`、`timestamps/`、单条导出 `manifest.json` 与 `delivery_manifest.json`。
- 查对应 JSONL：`reference_jsonl/ai_reference_426.jsonl`，校验记录 `reference_jsonl/validation.json`。

全量 comparison 是本次原画加框与裁后画面的同步预览。5条、10条示范中的比较视频按各自说明使用，不能都当成与旧 ORA 的自动对照。

30 FPS 是**预览视频的播放设置**。标注、轨迹和 JSONL 使用原帧号及原始 PTS；预览映射在 `preview_frame_map.csv`。本次整理没有改变原片帧率，也没有增加推理步骤。

## JSONL 与成绩的关系

本机已存在 JSONL，旧说明中“没有总 JSONL”已修正。其来源是当前426条参考，时间和入选帧集合没有变化；四元组 `[x,y,w,h]` 转为三元组 `[x,y,w]` 时，787帧为精确比例边界取整而仅减少宽度，最多2像素，x/y不变。它与预览框存在这点已记录的差别。

当前 ZIP 叫 `43.9_ai_reference_426.zip`；原校验报告记录其改名前的 `ai_reference_426.zip`。已核对内容指纹一致：

```text
JSONL 5768c2177c807420122e67a7b09ec470383993d7843eb8aa3950085b9f13aa6e
ZIP   9408deb8fcc1b39fe02688756bbd735a3799145f4c340d3e30012241c3c7fb38
```

**43.9目前只由文件名提供，尚未找到独立榜单证据。** 不将它登记为已经核实的算法成绩。上述参考也不是官方真值。原始导出报告写明未上传；本轮没有向比赛官网提交。

## 哪些不是新标注版本

`preview-standard-scenery10-20261009` 与 `preview-standard-291-qa` 是同一批参考的播放或帧映射检查；`reference-audit-20261009` 是初稿抽查快照；`reference-server-kit-20261009` 是工具交接。它们保留为证据，不单独计作标答。

旧 Qwen 的 balanced / conservative / precision 是算法参数档位；`legacy_gpt_prompt` 是 Qwen 使用的 Prompt 对照名称。不能凭这些名字将 Qwen 输出归为 GPT 标注。

本机历史初赛174条 GPT 标注原文件仍未找到。具体 GPT 模型版本、统一 Prompt 版本和生成 commit 也未从现有成果确认，索引明确记录缺失，不能补写成已知值。

## 以后新增版本

1. 新建 `runs/<reference_id>/`，不覆盖当前默认参考。
2. 保存编辑计划或标注、原片指纹、原始 PTS、查看记录、导出配置和校验记录。
3. 在本地索引增加记录；区分独立标注、局部备选、格式导出、预览重编码，并填写父版本。
4. 记录作者、日期、模型、Prompt 版本、代码 commit；确实缺失的写明缺失。
5. 算法正式实验仍进 `experiments/`，若用了参考，写具体 `reference_id` 与内容指纹，不能只写“GPT版”。

这批参考没有删除：小样本示范和局部备选包含独立取舍，仍有比较价值。原路径也被现有元数据引用，集中索引能够避免再次复制一套大文件。
