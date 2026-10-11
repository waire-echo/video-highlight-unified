# 2026-10-12 前半段：查清 ORA 与 GPT 参考的选段差距

## 本次目的

相同视频送入旧 ORA 和新的临时前半段，跑完再与本地 GPT/AI 剪辑参考对照。
要区分：主体/事件看错、候选漏掉、剪辑价值判断、时间边界、重复去除。
参考不是官方真值；保留少不等于质量高，接近参考也不等于比赛分数提高。
本轮只验证前半段，默认 12 条；临时目录可以保留，后半段接入不作为本轮先决条件。

开发分支：`exp/temporal-reference-test-20261012`。当前默认生产 workflow 保持兼容版。

## 已核对的证据

原路径中的 `ai_reference_426.zip` 已改名为 `43.9_ai_reference_426.zip`，内容指纹为
`9408deb8fcc1b39fe02688756bbd735a3799145f4c340d3e30012241c3c7fb38`。
ZIP 文件名中的 43.9 不在本次报告中当作已验证榜单成绩。
426 条旧 ORA 记录与参考 metadata/PTS 的 source_sha256 全部一致，原帧总数 234159。

| 全量历史结果 | 保留帧 | 占原帧 | 保留时长占比 |
|---|---:|---:|---:|
| 旧 ORA | 189844 | 81.07% | 81.42% |
| GPT/AI 参考 | 77198 | 32.97% | 33.16% |

交集 68594 帧；ORA 多留 121250 帧，参考中有 8604 帧未被 ORA 保留，即参考帧的 11.15%。
旧 ORA 候选区间本来就覆盖约 80% 原帧，987/992 个事件在复核后仍被保留；不是边界 padding 把约 33% 推成约 81%。
历史审计的 padding 独有帧仅约占最终保留帧 2.19%。阶段统计使用本机原 PTS，和旧四舍五入时间缓存会有 1–2 帧差异。

| 案例 | 旧流程证据 | 参考取舍 | 判断 |
|---|---|---|---|
| 20 小猫 | 32 个事件全部保留，约 89% 原帧 | 4 个有区别的互动，约 4% | 主体基本正确；重复和铺垫大量留下 |
| 313 跳水 | 起跳/水花正确，却把等待、在水中也列为事件，最终全片 | 0.9–4.45 秒起跳和水花 | 核心价值与尾段取舍，不是完全没看懂 |
| 184 小马 | query 写狗；主体二次纠正没有同步纠正事件 query，仍保留全片 | 4.45–7.35 秒小马动作 | 识别错误会沿定位传播；已查看原画抽样页确认画面是马 |
| 21 人物剪影 | 保留前半多个段，遗漏后段 39.6–43 秒 | 三个动作/画面变化 | 多留与漏留同时存在 |
| 266 风景 | 候选 0–10 秒，定位压缩到 0–2，再补到 2.25 秒 | 1.5–7.8 秒太阳上升 | 只缩短可能截掉有价值的发展 |
| 28 / 39 / 80 短片 | 全片 | 同样全片 | 正常一致对照，不能强制剪到 33% |

上述因果判断来自实际记录与代码；184/313 的现有原画抽样页本轮已查看。
不是本轮逐帧观看全部 426 视频，也没有隔离证明“换模型”或单独某一参数会提升成绩。
GPT 参考的准确 checkpoint/统一 Prompt/生成 commit 缺失，模型能力差异与观看方式差异仍不能排除。

## 新试验的修改

1. 保留云端原型的发现→全局计划→核心→边界→最终确认→视觉去重流程。默认同一批 ID：
   `20,21,28,39,80,87,91,180,313,324,365,384`。
2. 输入 side 从 384 恢复为旧 ORA 的 448；单视频帧预算从 64 提到 128。
   原 27 秒的 4/6 FPS 两次都被压到约 2.30 FPS；现在分别为 4 和约 4.67 FPS。
   20 秒现在可以分别用 4 和 6 FPS。双视频去重每条仍最多 32 帧。
3. 候选核心先加一次不带候选描述的盲观察，记录 subject / observed_event / uncertainty。
   核心复核另外给出 corrected observed_event 与 hypothesis_match，后续边界/确认用纠正后的事件。
   看错候选不能直接等于删除实际有价值的事件；完整漏发现仍需单独查 discovery/empty_recheck。
4. 每条保存实际采样、原片 hash、请求与回答；失败视频写 errors 并继续其余 ID，整批最后明确失败。
   不把错误或未解决候选当成正常空片；也不靠参考时段强行保留。
5. 配置、Prompt、源码、Git commit、输入、模型指纹记录到 frozen.json；输出不一致拒绝续跑。
6. 先检查单/双视频协议与 27 秒较大帧预算；失败即不启动 12 条。

这是一版组合诊断试验，不能凭一次对照把提升归到单个改动。需要拆分时复制配置关闭 blind_observation，
分别用新输出目录测试；源码/Prompt 不复制。CPU 通过只说明工程合同成立，真实模型回答与质量仍需 GPU。

### 输入视频处理说清楚

原片不改。模型临时片段使用 FFmpeg：按区间解码、FPS 重采样、保持比例缩放到 side 上限、H264 CRF20 转码、去音轨。
模型处理器还会按请求 FPS/max_frames 采样；因此记录的是请求/编码预算，不能声称模型看了每个原帧。
log316 不支持时，仅临时缩放把 transfer metadata 标为 unknown，日志保留；这不是色彩质量已验证的结论。
最后选段按原 PTS 映射零起始原帧号，没有改原 FPS 或伪造裁剪框。音频没有参与本次理解。

## 服务器交接与运行

现有数据根：`/root/rivermind-data/aic_spatial_trials_20261008`。
试验部署根：`/root/rivermind-data/aic_spatial_trials_20261008/experiments/temporal_reference_test_v1/repo`。
其中 deployment.local.json 保存实际 Git commit 与逐文件 SHA256。原 temporal_editor_v1 保留。

在试验部署根运行（只添加 `--check-only` 为 CPU 验收；去掉后才会启动本地 ORA 与默认 12 条）：

```bash
/root/rivermind-data/aic_spatial_trials_20261008/env/bin/python \
  pipelines/ora_temporal_editor_test/launch.py \
  --data-root /root/rivermind-data/aic_spatial_trials_20261008 \
  --video-dir /root/rivermind-data/ora4b_v5_20261001/videos \
  --runtime-root /root/rivermind-data/ora4b_v5_20261001 \
  --model-path /root/rivermind-data/ora4b_v5_20261001/models/Video-ORA-4B \
  --output /root/rivermind-data/aic_spatial_trials_20261008/output/temporal_reference_test_v1_12 \
  --check-only
```

只指定几条可添加 `--ids 20,21,313`。184/78/266 可作为另外一批诊断样例；不要默认替换原定 12 条。
只有显式 `--all` 才跑全量；本次没有授权立即全量推理。没有自动租 GPU 或调度器。

检查 `status.json`、`report.json`、`errors/`；查看 `records/<ID>.json` 的 candidates、blind_observation、
core_review/core_recheck、boundary_reviews、verification、duplicate_review。不能只看选帧数量。

## 离线对照工具

在本地统一仓库运行，不把参考传给 ORA：

```powershell
py -3.11 -X utf8 tools/compare_temporal_reference.py `
  --reference runs/ai-reference-426-20261009/reference_jsonl/43.9_ai_reference_426.zip `
  --reference-root runs/ai-reference-426-20261009 `
  --ora-frames artifacts/deliveries/ora4b_v5_426/temporal/highlight_frames.jsonl `
  --ora-records artifacts/deliveries/ora4b_v5_426/temporal/records `
  --ids 20,21,28,39,80,87,91,180,313,324,365,384 `
  --output runs/temporal-reference-test-20261012/pilot12
```

新结果下载后加 `--new-handoff <新目录>/temporal_handoff.jsonl --new-records <新目录>/records`，并用新报告目录。
不写 ids 则对照全量历史结果，属于离线统计而非全量推理。输出 index.html、comparison.csv、comparison.json。
时间轴同时显示旧 ORA、参考、新试验，展开查看事件与理由；缺失新视频列为 missing_new_ids，不冒充完成。
全量历史参考加权保留约 33%，这 12 条参考仅约 12%，样本不能代表全量分布，也不是任何目标配额。

### 验证状态

本机针对选帧、漏留统计、纠正事件传播、独立盲观察、实际加密差异的 CPU 测试已通过。
Linux 批处理失败恢复合同由服务器验收；主线冻结文件/版本与项目策略检查通过。
本机 Windows 全套 pytest 被已有 curses 依赖阻止收集，不能记为全套通过；Linux CI 另行记录。
服务器暂未挂载 GPU，真实协议、实际推理、质量与分数均 pending。实际验收记录见试验目录 output 与 deployment 收据。
