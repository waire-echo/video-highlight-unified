# scene_general_v3 基准版复现说明

这份说明固定 2026-10-10 至 2026-10-11 在 RTX 4090 服务器运行的 `scene_general_v3`。以后整理 README、增加 TUI、调整线程或优化性能，都应从这个基准版做可比较改动，不直接覆盖它。

## 已确认一致的部分

- Git 基准提交：`36695a2ad796b21623a9f7f8e759f9b21cf4ea91`。
- 分支：`feat/import-ora-spatial-server-20261010`，已推送到私有统一仓库。
- 服务器实际依赖的 10 个源码、配置和 Prompt 文件，与该提交中的 Git blob 逐字节 `10/10` 相同。
- Video-ORA-4B 固定为 `OraRL/Video-ORA-4B@01850297d5ab2adaaf130f700ed7cec52993d956`，五个权重分片的大小和 SHA256 记录在 `provenance/ora_model_provenance.json`。
- 模型请求固定 `temperature=0`、`seed=20261010`；OpenCV 固定 2 个线程和 RNG seed 0。
- vLLM 启动参数、Python 包版本、GPU 和驱动记录在 `provenance/scene_general_v3_baseline_20261010.json`。

源码一致不代表外部输入已经进入 Git。测试视频、基础预测、时间戳、镜头缓存和模型权重按规则不提交仓库。

## 本地保留的非 Git 输入包

本机保存：

```text
artifacts/source_archives/ora_spatial_v3_repro_20261011/
└── ora_v3_inputs_20261011.tar.gz
```

SHA256：

```text
4c17da08fc45a68dfd6022d9d04895bf5c5f1fd2da4f42e0e932a61898dfb249
```

压缩包包含：

- `inputs/baseline.zip`；
- `inputs/manifest.json`；
- `inputs/timestamps.npz`；
- 426 个 `output/shots/*.json`；
- `output/parameter_ledger.json`；
- 两次分段运行的 `frozen.json`；
- 包内逐文件 `SHA256SUMS`。

原视频不在包内，但 `manifest.json` 保存了 426 个视频的源文件 SHA256。恢复服务器时必须使用同一批原视频。

## 从空服务器恢复

### 1. 取固定源码

```bash
git clone --branch feat/import-ora-spatial-server-20261010 \
  https://github.com/thekfjie/video-highlight-unified.git
cd video-highlight-unified/pipelines/ora_spatial
git checkout 36695a2ad796b21623a9f7f8e759f9b21cf4ea91
```

如果要使用新增的校验器和一键入口，检出该分支最新提交，但不要修改 `scene_general_v3_baseline_20261010.json` 中列出的 10 个基准文件。

### 2. 恢复小型外部输入

把本地 `ora_v3_inputs_20261011.tar.gz` 上传到服务器，先核对压缩包 SHA256，再解压。将包内的 `inputs/` 和 `output/` 放进当前 `ora_spatial/` 根目录。

```bash
sha256sum ora_v3_inputs_20261011.tar.gz
tar -xzf ora_v3_inputs_20261011.tar.gz
cd ora_v3_inputs_20261011
sha256sum -c SHA256SUMS
```

### 3. 恢复运行环境和模型

使用 Python 3.11.13。完整环境快照：

- `provenance/scene_general_v3_client_pip_freeze_20261010.txt`；
- `provenance/scene_general_v3_vllm_pip_freeze_20261010.txt`；
- `provenance/scene_general_v3_system_20261010.txt`。

关键版本是 PyTorch `2.10.0+cu128`、vLLM `0.19.1`、Transformers `5.5.4`、NumPy `2.2.6`、OpenCV headless `5.0.0.93` 和 SciPy `1.17.1`。CUDA 匹配的 PyTorch/vLLM 应从对应 CUDA 源安装，随后再按 freeze 文件核对完整环境。

下载固定 revision 的 Video-ORA-4B，然后运行校验器核对五个权重分片。不要只看模型目录名称。

### 4. 放置原视频并预检

原视频目录必须包含 `0.mp4` 至 `425.mp4`。正式运行前执行：

```bash
python scripts/verify_scene_general_v3_baseline.py \
  --root "$PWD" --model-dir /data/ora-runtime/models/Video-ORA-4B \
  --check-environment
```

随后启动 vLLM，再执行原预检。原预检会逐个核对视频 SHA256、帧数、镜头缓存和本机模型服务。

### 5. 全量运行

兼容入口默认运行 `0-425` 全量，并保留这次的 vLLM 参数：

```bash
export ORA_RUNTIME_ROOT=/data/ora-runtime
export VIDEO_DIR=/data/videos
export OUTPUT_DIR="$PWD/output/scene_temporal_v3_full_426"
mkdir -p "$OUTPUT_DIR"
nohup bash scripts/run_scene_general_v3_baseline.sh \
  > "$OUTPUT_DIR/run.log" 2>&1 < /dev/null &
```

失败后使用相同源码、输入、配置、ID 和输出目录重启，完整 segment 缓存会被复用。不要在运行期间修改源码或配置。

## 对“同样成绩”的准确理解

同一代码、Prompt、输入、模型、运行参数和环境，可以保证是同一套流程。在同型号 RTX 4090 和记录环境上，结果通常应一致或非常接近。

BF16 GPU 推理可能受显卡型号、驱动、CUDA、PyTorch、vLLM 和并行调度影响，因此跨环境不能承诺模型回答逐字节完全一致。正式基准完成后，还要长期保存最终 JSONL/ZIP 和 SHA256；这样即使服务器清空，也能保留这次实际提交结果和对应成绩。

本次服务器任务曾拆成 `240-330` 和其余 335 条运行。两部分都完成后使用 `scripts/merge_scene_general_v3_outputs.py` 按基础预测的 426 条原始顺序合并，并再次执行完整格式、帧集合和边界校验。以后新的全量运行直接用单一 `0-425` 任务，不再依赖拆分结果。

## 后续升级规则

1. 基准文件只读保留，不在原地修改。
2. 每个优化从该基准提交新建短期分支。
3. README、TUI 和日志改造不能改变模型请求 payload 或结果顺序。
4. 线程、预读、批处理或 vLLM 并发变化必须先跑固定样本，对比请求哈希、原始回答和最终 JSONL。
5. 有任一 bbox、frame 或模型回答差异，就登记为新实验版本，不宣称与基准完全等价。
6. 只有验证通过的工程改进才合回 main，并保留基准入口作为 `compat` 模式。
