# 部署与运行说明

## 1. 运行环境与资产

使用 Linux x86_64、Python 3.11、NVIDIA GPU 及与 CUDA 12.8 PyTorch 匹配的驱动。
FFmpeg 和 FFprobe 须位于 PATH 中，或在安装时显式指定位置。
Python 固定依赖见 `requirements-submission.lock.txt`；使用其他平台或依赖版本时需重新验证。

完整模型包目录如下：

```text
configs/submission.json
models/ora/                基座权重、tokenizer、processor 和聊天模板
models/adapter/            adapter_config.json、adapter_model.safetensors
models/spatial/            空间模块源码、配置和 YOLO 权重
scripts/                   环境安装、推理与打包入口
src/video_highlight/        公共模块
prompts/                   时间定位提示词
requirements-submission.txt
requirements-submission.lock.txt
```

模型、适配器、提示词和空间配置须使用同一套交付资产，文件校验值由 `PACKAGE.json` 记录。
基础模型须包含完整分片、分片索引、tokenizer、processor 配置、
Jinja 聊天模板及其 `chat_templates/` 子目录；模型加载使用本地文件。

## 2. 创建独立环境

源码部署时指定已有资产位置：

```bash
python3.11 scripts/setup_submission_env.py \
  --model /path/to/base_model --adapter /path/to/adapter \
  --spatial-package /path/to/spatial_assets \
  --ffmpeg /path/to/ffmpeg --ffprobe /path/to/ffprobe --with-tests
```

完整模型包使用默认目录时可省略三个资产路径参数；媒体工具在 PATH 中时也可省略对应参数。

安装器创建独立 `.venv`，先安装指定 CUDA 源的 PyTorch，再按 wheel 元数据安装匹配的
NVIDIA/Triton 运行库及项目依赖，最后执行 `pip check` 和隔离模式下的导入/CUDA 检查。
存在依赖锁定文件时优先采用锁定版本，不继承其他环境的 site-packages。
目标驱动不匹配时可通过 `--torch-index-url` 指定相应源；`--requirements` 可指定依赖合同。

安装记录保存在 `.runtime/environment.json`，机器路径配置保存在
`configs/submission.local.json`。这些是部署后生成的本机文件，不属于通用交付配置。
环境解释器保留 `.venv/bin/python` 调用路径，主流程与子进程使用同一独立环境。

Decord 0.6.0 的部分 wheel 存在内部标签不一致问题。安装器仅在新虚拟环境中、
确认该标签且原生库可加载时修正 WHEEL/RECORD 元数据；记录位于
`.venv/.runtime/decord-wheel-tag.json`，不修改解码器代码。

更换适配器而不重装依赖：

```bash
.venv/bin/python -I -B scripts/setup_submission_env.py \
  --skip-install --adapter /path/to/adapter
```

仅检查环境使用 `--check-only`。之后通过统一启动脚本运行，无需手动激活环境。

## 3. 输入与配置

输入目录包含 `video/<video_id>.mp4` 和 `jsonl/<video_id>.jsonl`。
每份 JSONL 仅一行，例如：

```json
{"video_id":"sample","targetRatioWH":[9,16]}
```

也支持 `videos/`，以及单个 `test_index.json`（对象数组）或 `test_index.jsonl`。
不要同时提供多个索引。编号可以不连续、可以包含非数字字符；不需要高光标注或预切片。
视频应可解码，逐帧时间戳须可读取且严格递增；带非零旋转元数据的输入会被拒绝。

配置中的相对路径以配置文件所在目录为基准，而非 shell 当前目录。
源码默认配置为 `configs/submission.example.json`，完整模型包使用 `configs/submission.json`。
启动脚本优先使用安装器生成的 `configs/submission.local.json`。

只读计划与输入准备：

```bash
python scripts/run_submission_pipeline.py --config configs/submission.json --dry-run
python scripts/run_submission_pipeline.py --config configs/submission.json --prepare-only
```

在源码目录运行以上命令时，将配置名换为 `configs/submission.example.json` 或已生成的本机配置。

## 4. 推理与结果

```bash
bash scripts/run_submission.sh \
  --input-root /path/to/input --output outputs/submission \
  --inference-gpus 0,1 --spatial-gpu 0
```

也可直接指定配置运行：

```bash
.venv/bin/python -I -B scripts/run_submission_pipeline.py \
  --config configs/submission.json --input-root /path/to/input \
  --output outputs/submission --inference-gpus 0,1 --spatial-gpu 0
```

设备编号按目标机器选择；单卡运行可将两个设备参数都设为 `0`，实际容量需满足模型与输入配置。
可通过 `--python`、`--spatial-python`、`--ffmpeg`、`--ffprobe` 指定解释器和媒体工具。

流程为：媒体检查 → 视频窗口准备 → 适配器时间推理 → 区间合并与修复
→ 原视频逐帧时间戳映射 → 主体检测与空间裁剪 → 协议校验与压缩。
默认采用 32 秒窗口、24 秒步长和 2 FPS；最大帧数、视觉预算、生成长度等以配置为准。
时间推理使用 Decord 读取器；空间模块以当前时间交接文件确定待处理帧。

同一命令再次运行时复用已完成检查点。输入、模型、提示词或配置变化时创建独立运行；
代码指纹变化也会更新运行身份。不要将不同运行的缓存混合使用。

输出目录的 `CURRENT.json` 指向 `run-<id>`；其 `06-submission/` 包含
`predictions_spatial.jsonl`、`submission.jsonl`、`submission.zip`、`FINAL.json`。
两个 JSONL 内容相同；ZIP 内仅有 `predictions_spatial.jsonl`。
预测框采用 `[x,y,w]`，帧号属于原视频。

## 5. 异常恢复

- 有效时间回答保持不变。异常条目按合法时间交集处理；只解析可独立解码的完整条目，
  不猜时间单位、不交换起止点。未解决窗口最多按 `recovery_attempts` 重试，默认一次。
- 重试预算持久化，重启不会叠加预算。仍失败时隔离该窗口，保留其余窗口结果；
  `03-lenient/TERMINAL_STATES.json` 区分正常无高光、局部失败和推理失败，
  `03-lenient/FALLBACKS.json` 记录降级。
- 空间异常仅对受影响帧补目标比例下的合法中心框，不增加或删除时间阶段选定的帧。
  空间阶段的 `RECOVERY.json`、`TERMINAL_STATES.json` 及最终 `REPAIRS.json` 记录处理情况。
- 缺少模型、权重校验失败、CUDA 不可用、输入身份冲突或磁盘不可写时明确报错。
  最终 ID、比例、帧号、裁剪框和 ZIP 校验始终执行。

正式入口固定采用上述恢复策略。兼容字段 `recovery_policy=diagnostic`、
`unresolved_policy=error`、`spatial_failure_policy=error` 或空间入口的
`--failure-policy error` 不会关闭恢复；读取时会给出提示。
`recovery_attempts=0` 仅取消模型重试，不取消确定性修复和最终局部兜底。

窗口、空间视频和阶段命令的初始超时分别为 `600`、`1800`、`86400` 秒，
对应 `window_timeout_sec`、`spatial_video_timeout_sec`、`command_timeout_sec`。
窗口或空间 worker 超时后回收该进程并保留已完成结果；阶段命令预算不是整个任务总时限。
采样帧数与视觉预算不会因 OOM 自动调整；修改相关配置后应重新验证。

## 6. 构建与验证完整模型包

在配置中填写实际基座、适配器和空间资产目录，再构建到新的输出目录：

```bash
python scripts/package_submission.py --config configs/submission.local.json --output dist/release
python dist/release/scripts/package_submission.py --verify dist/release
```

构建器按白名单复制运行源码、提示词、基座、适配器、空间资产、依赖说明和许可证，
生成逐文件 SHA256。缺少必需资产会中止构建；适配器仅复制配置和推理权重，权重字节保持不变。
包中不包含输入视频、标注、预测缓存、凭据、训练状态或虚拟环境。

文件完整性校验完成后，将包移动到另一目录，在独立环境中使用允许复现的视频执行全流程，
检查 `FINAL.json`、结果 JSONL 与 ZIP，再验证中断续跑及完成后重复运行的一致性。
`PACKAGE.json` 中的 `gpu_validation` 记录该实体包的验证状态；仅完成文件校验不等于模型推理验证。
