# 实验登记

`experiments/records/<id>/experiment.json` 是事实记录，`config.json` 是配置快照，`metrics.json` 是小型指标。CSV 只是检索索引，`experiment list` 会重建它。不需要数据库。

## 正式实验

1. 配置明确 pipeline、模型 ID/revision、数据 ID/manifest、Prompt 版本和算法参数。
2. 将代码、配置、Prompt 提交 Git。机器路径放被忽略的 `configs/paths.local.json`。
3. ID 用 `日期-作者-路线-变更`，如 `20261004-echo-ora4b-fps2`。
4. 通过 `vh.py run` 执行，失败状态也留存。续跑使用相同代码、Prompt、配置和输入。
5. 登记 metrics/notes，把小型实验记录提交 Git。只改变元数据的提交允许续跑。

保存代码 commit/源文件指纹、配置快照、Prompt 文件 hash、数据 manifest hash。运行时补充索引/外部权重 hash 与实际命令，回答和输出留在被忽略的 `runs/<id>/`。

指标示例：

```json
{"metric_type":"leaderboard-user-reported","score":56.43,"dataset":"aic_initial174","evidence":"队长提供的成绩记录","notes":"原始生成代码仍缺失"}
```

代理分、与旧提交的重合度、覆盖率、帧数、合法率都不是比赛准确率。离线指标记录评测代码和标注来源。`video_highlight.evaluation.weighted_f1` 是题面参考实现，不是主办方评分程序。

## 历史导入与外部产物

缺失值保持 null/unknown，`reproduction_gaps` 列缺口。来源 commit 不表示当前实现能重现当时所有步骤。

`artifacts` 保存 label、大小、SHA256 与 `location_key`。被忽略的 `experiments/artifact_locations.local.json` 把 key 映射到本机原文件。成员获取自己的副本后核对 hash、填写本机路径。

本机定位并校验：`python tools/locate_artifacts.py --experiment hist-exp07-boundary-b`。

`hist-qwen-balanced-42` 可以从原 raw 分数复现时间选择，但最终空间链路不完整。`hist-exp07-boundary-b` 目前只能定位结果，不能完整重跑。

## Prompt

`prompts/<路线>/<版本>/` 为一个 bundle。静态文字用 txt、schema 用 JSON，时长/FPS/候选动态文字用 `builders.py`，只负责生成文字。

配置 `prompts` 是“默认 bundle → 实际版本”的映射，如 `{"qwen/window-v1":"qwen/window-v2"}`。新版本保留同名构造器函数。记录保存每个文件 hash。改变 Prompt 或代码时新建实验。

多人使用不同 ID。单个工作区登记/重建 CSV 串行执行，合并后可由 records 重建索引。
