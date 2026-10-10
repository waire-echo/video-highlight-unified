# scene_general_v3 统一运行与版本管理

本次整理对象是交接提交 `81f2257` 的 `scene_general_v3`。算法来自 `36695a2`，固定标签为 `ora-spatial-v3-baseline-20261011`。旧 E1/E2/E3 和 YOLO/Pose 空间路线保留各自入口，不能当作 v3。

## 一个入口，明确选择版本

```bash
python vh.py workflow versions
python vh.py workflow --help
```

| 版本 | 输入方式 | I/O 策略 | 验证范围 |
|---|---|---|---|
| `scene-general-v3-compat-20261010` | 固定 baseline | 原始解码与导出 | 保留冻结算法和请求；新启动器 GPU 复跑待验收 |
| `scene-general-v3-unified-v1` | baseline / PTS handoff / 新时间推理 | 原始 I/O | 新实验；已有交接完成 CPU 验收 |
| `scene-general-v3-io-v1` | 同上 | 连续解码、128 MiB 缓存、逐视频检查点 | 新优化实验；CPU 已核对，GPU 等价性待验收 |

版本清单在 `configs/workflows/versions/`，记录源码根目录、入口、逐文件 SHA256、模型 revision、输入方式与 I/O 策略。入口必须在源码指纹中。修改算法、配置或 Prompt 时新增源码版本与清单，保留旧清单。旧清单对应源码发生变化会被拒绝。

运行配置在 `configs/workflows/`；机器路径单独放在被 Git 忽略的 `*.local.json`。`compat` 只接受固定输入，不重选时间、不打开优化策略。资源占用不会自动改变模型请求、采样、Prompt、OpenCV 线程或 vLLM 并发。

## 路径与执行计划

```bash
cp configs/workflows/paths.example.json configs/workflows/paths.local.json
# 填写机器路径、解释器，以及尚未使用的独立 run_root
python vh.py workflow plan --config configs/workflows/compat.json \
  --local configs/workflows/paths.local.json
```

`data_root` 包含外部 `inputs/{baseline.zip,manifest.json,timestamps.npz}`、`output/shots/` 和 `output/parameter_ledger.json`。源码始终由 Git 工作区的版本清单选择，不从历史服务器目录猜测。原视频、输入和权重仅引用原路径，不进入 Git。

相对路径以对应配置文件所在目录为起点，虚拟环境解释器的符号链接保持原样。`run_root` 必须与历史数据、视频、模型和运行环境分开。`plan` 打印实际路径、完整命令、阶段、端口与资源策略，不创建运行目录、不启动模型。

`ids` 为 `all` 或按原 manifest 顺序的字符串列表。小样本复制运行配置并填写 IDs、新 run_root；前半阶段的 `test_index` 必须仅包含同一组、同一顺序的 IDs。已有 handoff 与 highlight_frames 必须恰好包含所选 IDs；现有全量交接配 `all`，小样本先准备对应的交接文件。

## 四种输入与运行方式

```bash
# 原空间基准：高光帧不变
python vh.py workflow run --config configs/workflows/compat.json --local configs/workflows/paths.local.json

# 已有时间交接 → 适配 → v3 → 校验 → 最终交付
python vh.py workflow run --config configs/workflows/spatial.json --local configs/workflows/paths.local.json

# 新时间推理 → 交接 → v3 → 校验 → 最终交付
python vh.py workflow run --config configs/workflows/full.json --local configs/workflows/paths.local.json

# I/O 优化实验，使用另一个新的 run_root
python vh.py workflow run --config configs/workflows/optimized.json --local configs/workflows/paths.local.json
```

每次新运行都选择尚未使用的 run_root。切换配置不会把旧缓存改成新版本；已有目录只能通过 `resume`，来源必须一致。

时间阶段复用现有 `video_highlight.ora` 与 `pipeline.py temporal`；空间阶段加载逐字节封存的 v3 源码。两个阶段使用各自多模态服务参数，串行启动、释放服务。工作流持有 GPU 锁，拒绝占用端口，清理自己启动的进程组。

交接必须同时提供 PTS handoff 和精确 `highlight_frames.jsonl`。适配器检查原 PTS、inclusive 帧范围、原帧数量、画幅、空结果及两个文件的帧集合，显式生成初始居中框，保存输入/输出哈希。该输入属于新实验，不能称为原 YOLO-only 兼容基准。阶段失败后不进入下一阶段。

## 分阶段执行和续跑

```bash
# 只跑前半时间选择
python vh.py workflow run --config configs/workflows/full.json \
  --local configs/workflows/paths.local.json --until temporal

# 无 GPU 时可以完成已有交接的检查与适配
python vh.py workflow run --config configs/workflows/spatial.json \
  --local configs/workflows/paths.local.json --until handoff

python vh.py workflow status --run-root /data/runs/your-unique-experiment
python vh.py workflow resume --run-root /data/runs/your-unique-experiment
```

分阶段结束是 `stage_complete`，整体结束才是 `complete`。续跑核对 Git commit、版本清单、维护源码与 Prompt、解释器包版本、权重、输入、原视频、镜头来源和完成阶段的产物哈希。来源不同则检出原提交或创建新运行。预检失败且还未冻结来源时，可以修复外部原因后恢复预检；已有完成阶段不能缺少 `frozen.json`。

查看 `status.json`、`events.jsonl`、`logs/`、`errors/failed_items.jsonl` 和 `spatial/status.json`。日志按阶段追加，错误历史保留。前台用 Ctrl+C 停止，后台向本次工作流 PID 发 SIGTERM。

## 运行目录和交付

```text
runs/<experiment>/
├── plan.json / frozen.json / inputs.json
├── status.json / events.jsonl
├── source/ora_spatial/       封存源码，外部数据只读引用
├── stages/                  完成阶段收据与产物 SHA256
├── temporal/                模型证据、PTS 与时间交接
├── handoff/                 输入 ZIP、manifest、adapter.json
├── spatial/                 v3 原始证据、segment 缓存与报告
├── logs/ / errors/
└── final/
    ├── predictions_spatial.jsonl
    ├── submission.zip       比赛格式，根内仅 predictions.jsonl
    ├── validation.json
    ├── spatial_report.json
    └── artifacts.json       版本、大小、SHA256 和提交说明
```

全量空间完成，帧集合与框边界校验通过，ZIP 内部字节与 JSONL 一致，才生成正式交付。原生 `spatial/results.zip` 保存审计证据，不是比赛提交文件。历史正式结果仍位于旧快照的 `output/scene_temporal_full_426_20261011_v3/`。

## 性能变化与验证

优化版同一视频的后续段继续顺序解码，保留原 resize 算术，不使用关键帧 seek。反向读取且缓存缺失时从零重读，缓存上限 128 MiB，源文件变化立即拒绝。不改变原帧号、PTS、模型请求顺序或并发。

原版每完成一条视频都重新导出累计结果。优化版逐视频验证、原子保存 `spatial/checkpoints/<id>.json`，每 25 条生成一次累计 partial，最终仍调用同一个导出函数。segment 缓存继续负责推理续跑；优化模式的 partial ZIP 可能落后最新视频，进度以状态和检查点为准。

```bash
python vh.py workflow verify-io --data-root /data/scene-general-inputs \
  --video-dir /data/videos --ids 0,1,4,12,97,176 \
  --output /data/runs/verification/io.json

python vh.py workflow validate --data-root /data/scene-general-inputs \
  --prediction /data/historical/SPATIAL_temporal_v3_full_426.jsonl
```

`spatial/performance.json` 记录解码量、decoder 打开次数、缓存内存和导出耗时。CPU 验证证明受检样本的解码字节及缓存重导出关系；申请 GPU 后仍需以新目录对比请求、模型回答和最终 JSONL。全量 GPU 耗时和比赛成绩尚无新结论。

TUI 与 WebUI 已通过同一个 Controller 接入此状态机。使用 `workflow tui --local ...` 或 `workflow webui --local ...`，详见 [界面使用说明](scene_workflow_ui.md)。三路同步对比、预计剩余时间和其他时间模型自动适配仍在后续计划中。

本次实际检查范围与证据见 [验证记录](scene_workflow_validation.md)。
