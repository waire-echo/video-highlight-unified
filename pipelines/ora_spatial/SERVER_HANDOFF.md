# ORA Spatial 服务器接手手册

更新日期：2026-10-11（Asia/Hong_Kong）

这份文档写给第一次接触本项目的人。先读完“当前状态”和“不要做什么”，再执行命令。

## 1. 这套流程在做什么

输入是 426 个原视频，以及上一阶段已经选好的高光帧。`scene_general_v3` 不重新决定哪些时间段是高光，而是在保持高光帧集合不变的前提下，为每个入选帧重新生成空间裁剪框 `[x, y, w]`。

处理过程是：

1. 读取 `inputs/baseline.zip` 中的高光帧、原裁剪信息和目标画幅；
2. 使用已有镜头切分 `output/shots/<video_id>.json`；
3. 每秒采样原视频画面，让 Video-ORA-4B 定位主体并检查前后语义连续性；
4. 生成多个裁剪候选，由同一个模型比较候选；
5. 平滑中心和宽度，并插值回所有入选的原视频帧；
6. 输出比赛格式的 JSONL 和 ZIP，并校验帧集合、坐标和压缩包结构。

它不会改变入选帧集合。视频 `97`、`176`、`285`、`367` 在上游时间结果中本来就是空结果，必须继续保持为空。

## 2. 当前服务器状态

服务器项目目录：

```text
/root/rivermind-data/aic_spatial_trials_20261008
```

关联资源：

```text
原视频：/root/rivermind-data/ora4b_v5_20261001/videos
ORA 模型：/root/rivermind-data/ora4b_v5_20261001/models/Video-ORA-4B
vLLM 环境：/root/rivermind-data/ora4b_v5_20261001/env
v3 客户端环境：/root/rivermind-data/aic_spatial_trials_20261008/env
```

截至 2026-10-11：

- 服务器 GPU 已释放，当前只有 1 核 CPU、16 GB 内存；
- 没有 vLLM 或 v3 推理任务在后台运行；
- 426 个视频已经全部运行完成并合并；
- 完整结果包含 426 条视频记录和 189,844 个入选帧；
- 结果格式校验通过，`invalid_count = 0`；
- 暂无官方比赛分数，不能把本次结果描述成已经确认提分；
- 磁盘是临时数据盘。平台明确提示释放实例时 `/root/rivermind-data` 不保存，因此重要结果已另存到本机，仍不要随意删除服务器文件。

登录后先执行：

```bash
cd /root/rivermind-data/aic_spatial_trials_20261008
cat START_HERE.md
nvidia-smi || true
ps -ef | grep -E 'vllm|scene_general_v3/temporal.py' | grep -v grep || true
df -h /root/rivermind-data
```

如果 `nvidia-smi` 没有列出 GPU，不要启动推理。先在云平台重新申请至少 24 GB 显存的 GPU；本次基准实际使用 RTX 4090 24 GB。

## 3. 哪些文件是最终结果

本次实际合并结果位于：

```text
output/scene_temporal_full_426_20261011_v3/
├── SPATIAL_temporal_v3_full_426.jsonl
├── SPATIAL_temporal_v3_full_426.zip
├── SPATIAL_temporal_v3_full_426.validation.json
├── merge_report.json
└── scene_general_v3_full_426_results.zip
```

用于比赛格式提交的是：

```text
output/scene_temporal_full_426_20261011_v3/SPATIAL_temporal_v3_full_426.zip
```

`scene_general_v3_full_426_results.zip` 是归档包，包含 JSONL、提交 ZIP、校验和合并报告，适合交接和备份，不是比赛网站要求的内层提交文件。

固定 SHA256：

```text
138cc1fe848c23877f83acb0d9732a8531eafce1292f91ffbcb784a3c0c0e70a  SPATIAL_temporal_v3_full_426.jsonl
bc0aa160e96571be54e3406796804b013ceb53b889763f780e736c8eaafde284  SPATIAL_temporal_v3_full_426.zip
3094e804d4d5828da66e6e5588063d0be18f922b6e8716a004f1a666c9a657ba  SPATIAL_temporal_v3_full_426.validation.json
e995ff42423feecb1e64729ad3677cd8f7fa7c8b31d8df1499c598e963d19711  merge_report.json
163e28d6b6ac2cad3a64ae84e446626ecf44e9c1fdfb763e1faf27cad6b16f55  scene_general_v3_full_426_results.zip
```

核对命令：

```bash
cd /root/rivermind-data/aic_spatial_trials_20261008
sha256sum output/scene_temporal_full_426_20261011_v3/*
unzip -t output/scene_temporal_full_426_20261011_v3/SPATIAL_temporal_v3_full_426.zip
unzip -t output/scene_temporal_full_426_20261011_v3/scene_general_v3_full_426_results.zip
```

本次任务实际分两批运行：

```text
output/scene_temporal_240_330_20261010_v3        91 个视频
output/scene_temporal_remaining_335_20261010_v3 335 个视频
```

上面两个目录保存模型回答、候选面板、逐段记录和断点缓存。完整合并结果能用于提交，但若要审计模型为什么画出某个框，需要查看这两个源运行目录。

## 4. 代码的权威版本

统一 Git 仓库：

```text
https://github.com/thekfjie/video-highlight-unified.git
```

不可变基线标签：

```text
ora-spatial-v3-baseline-20261011
```

关键提交：

```text
36695a2ad796b21623a9f7f8e759f9b21cf4ea91  服务器实际运行源码快照
5d9edf2                                      基线校验、环境、结果指纹和产品化计划
```

服务器当前目录是历史运行快照，不是以后长期开发的唯一 Git 工作区。开发新功能时应从统一仓库的标签或后续 main 建短期分支。不要直接在已完成运行目录里改源码后继续复用旧缓存。

服务器实际依赖的 10 个源码、配置、Prompt 文件，已经与提交 `36695a2` 逐字节核对，结果为 `10/10 MATCH`。模型 5 个权重分片的大小和 SHA256 也全部通过核对。

## 5. 目录怎么理解

```text
experiments/scene_general_v3/  本次 v3 算法、Prompt、配置和测试
experiments/scene_general_v1/  v3 复用的基础候选与通用配置
src/                           共用读取、导出和 ORA 请求工具
inputs/                        baseline、视频 manifest、逐帧时间戳
output/shots/                  426 个视频的镜头切分缓存
output/scene_temporal_*        正式运行、断点、模型回答和结果
output/scene_temporal_full_*   本次合并后的最终结果
provenance/                    固定源码、输入、环境和模型指纹
scripts/                       校验、一键基准运行和历史工具
docs/                          复现说明与后续产品化计划
models/                        本项目其他实验的 SAM/TransNet 权重
vendor/                        第三方参考实现和许可证
```

根目录原有 `README.md` 主要描述更早的 E1/E2/E3 ORA+SAM 消融路线。`bash scripts/run_gpu.sh all` 运行的是那套历史路线，不是本次 `scene_general_v3` 基准。接手 v3 时以本文件、`docs/REPRODUCE_SCENE_GENERAL_V3.md` 和 `experiments/scene_general_v3/README.md` 为准。

## 6. 在当前服务器做只读检查

这一步不需要 GPU：

```bash
cd /root/rivermind-data/aic_spatial_trials_20261008

# 检查基准源码、输入、426 个镜头缓存、模型权重和关键环境版本
env/bin/python scripts/verify_scene_general_v3_baseline.py \
  --root "$PWD" \
  --model-dir /root/rivermind-data/ora4b_v5_20261001/models/Video-ORA-4B \
  --check-environment

# 测试 v3 本身
env/bin/python -m unittest discover \
  -s experiments/scene_general_v3 -p 'test_temporal.py' -v
```

校验器应输出 `"pass": true`。失败时先看它报告的具体文件或包版本，不要为了通过校验去改 provenance。

## 7. 重新申请 GPU 后怎样全量重跑

先确认 GPU 和端口：

```bash
nvidia-smi
curl -fsS http://127.0.0.1:8008/v1/models || true
ps -ef | grep -E 'vllm|scene_general_v3/temporal.py' | grep -v grep || true
```

端口 8008 不应已有别的模型服务。使用一个全新的输出目录，避免覆盖本次实际结果：

```bash
cd /root/rivermind-data/aic_spatial_trials_20261008

export ORA_RUNTIME_ROOT=/root/rivermind-data/ora4b_v5_20261001
export VIDEO_DIR=/root/rivermind-data/ora4b_v5_20261001/videos
export OUTPUT_DIR="$PWD/output/scene_temporal_v3_full_426_rerun_$(date +%Y%m%d)"

mkdir -p "$OUTPUT_DIR"
nohup bash scripts/run_scene_general_v3_baseline.sh \
  > "$OUTPUT_DIR/run.log" 2>&1 < /dev/null &
echo $! | tee "$OUTPUT_DIR/launcher.pid"
```

这个入口会依次：

1. 校验固定源码、输入、镜头缓存、环境和模型权重；
2. 用与本次相同的参数启动本机 vLLM；
3. 等待模型服务可用；
4. 对 `0.mp4` 到 `425.mp4` 做全量预检；
5. 按 `0..425` 顺序运行 v3；
6. 正常退出或收到中断时停止它自己启动的 vLLM。

不要同时启动第二个相同任务。不要把 `--max-num-seqs`、请求顺序、线程、Prompt 或采样参数当作普通性能开关修改；修改后只能登记为新实验，不能再称为本次兼容基准。

## 8. 怎样看进度和错误

先设置实际输出目录：

```bash
export OUTPUT_DIR=/root/rivermind-data/aic_spatial_trials_20261008/output/scene_temporal_v3_full_426_rerun_YYYYMMDD
```

查看日志：

```bash
tail -F "$OUTPUT_DIR/run.log"
```

另开一个终端查看结构化状态：

```bash
python - "$OUTPUT_DIR/status.json" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
d = json.loads(p.read_text())
print('status       =', d.get('status'))
print('stage        =', d.get('stage'))
print('current_video=', d.get('current_video'))
print('segment      =', d.get('segment'))
print('completed    =', len(d.get('completed_ids', [])), '/', len(d.get('requested_ids', [])))
print('updated_utc  =', d.get('updated_utc'))
print('error        =', d.get('error'))
PY
```

查看资源：

```bash
watch -n 2 nvidia-smi
```

判断是否结束要同时看：

- `status.json` 中是否为 `status=complete`；
- 日志是否仍在更新；
- 推理进程是否还存在；
- 最终 JSONL、ZIP、validation、report 和 `results.zip` 是否都存在。

如果 `status=failed`，保留整个输出目录和日志。先修复外部原因，再用完全相同的源码、输入、ID 顺序和 `OUTPUT_DIR` 重启；完整 segment 会按签名复用。`run.lock` 文件可以保留，它使用的是进程级 `flock`，只要旧进程确实不存在就不会因为文件本身存在而阻止重启。

## 9. 重跑完成后怎样验收

直接全量运行生成的文件名是：

```text
SPATIAL_temporal_v3.jsonl
SPATIAL_temporal_v3.zip
SPATIAL_temporal_v3.validation.json
report.json
frozen.json
results.zip
```

检查状态和压缩包：

```bash
python -m json.tool "$OUTPUT_DIR/status.json" >/dev/null
python -m json.tool "$OUTPUT_DIR/SPATIAL_temporal_v3.validation.json" >/dev/null
unzip -t "$OUTPUT_DIR/SPATIAL_temporal_v3.zip"
unzip -t "$OUTPUT_DIR/results.zip"
sha256sum "$OUTPUT_DIR"/SPATIAL_temporal_v3* "$OUTPUT_DIR/results.zip"
```

本次固定结果的 JSONL SHA256 是 `138cc1fe848c23877f83acb0d9732a8531eafce1292f91ffbcb784a3c0c0e70a`。新运行哈希完全相同才叫逐字节复现；如果内容不同，先保存新结果和完整 `frozen.json`，将其登记为另一次实验。GPU、驱动、CUDA、PyTorch 或 vLLM 不同可能使 BF16 模型回答产生差异，所以“同一套流程”不等于跨环境承诺逐字节相同。

## 10. 不要做什么

- 不要删除或覆盖 `output/scene_temporal_full_426_20261011_v3`；
- 不要删除两个源运行目录，它们保存审计证据和可复用的逐段缓存；
- 不要覆盖 `inputs/baseline.zip`、`manifest.json`、`timestamps.npz` 或 `output/shots`；
- 不要替换模型权重后仍沿用原基线名称；
- 不要隐藏修改 FPS、丢帧、补帧、帧号、PTS、分辨率、裁边或高光片段；
- 不要针对单个测试视频写特殊规则；
- 不要把测试视频上传到第三方 API、公开仓库或社交平台；
- 不要人工修改正式测试集输出；
- 不要直接运行旧 `scripts/run_gpu.sh all` 来代替 v3；
- 不要因为看到 CPU/显存没有吃满就直接提高 vLLM 并发，本次兼容模式固定 `max-num-seqs=1`；
- 不要把 TUI、三路对比视频或自动文件识别描述成已经完成，它们目前只有实施计划。

任何流程变化都要写明：输入是什么、增加了哪一步、是否改变帧集合或模型请求、输出是什么、与基准是逐字节一致还是新实验。

## 11. 后续开发顺序

当前完成的是 P0：冻结可复现基准。建议按以下顺序继续：

1. P1：统一 CLI 和 `plan.json/status.json/events.jsonl/failed_items.jsonl`；
2. P2：只读 TUI，先展示历史运行、进度、日志和错误；
3. P3：让 TUI 启动、停止和续跑同一套 CLI；
4. P4：原画、基准、新方案三路同步对比视频；
5. P5：前半时间选择与后半空间裁剪的完整两阶段控制；
6. P6：在结果对比保护下优化预读、解码、线程和并发；
7. P7：README、功能索引和废弃入口收口。

详细约束见：

```text
docs/PRODUCTIZATION_PLAN.md
docs/REPRODUCE_SCENE_GENERAL_V3.md
```

## 12. 接手完成检查表

- [ ] 能说清楚输入、处理和最终提交文件；
- [ ] 已确认当前是否有 GPU、是否有推理进程；
- [ ] 已运行基线校验器并看到 `pass=true`；
- [ ] 已核对本次最终结果 SHA256；
- [ ] 已知道旧 E1/E2/E3 与本次 v3 是不同路线；
- [ ] 新运行使用新输出目录；
- [ ] 新改动来自统一 Git 仓库短期分支；
- [ ] 任何参数或步骤变化均登记为新实验；
- [ ] 没有把测试集或大模型权重提交到 Git；
- [ ] 没有把计划中的 TUI/三路对比误报为已实现。
