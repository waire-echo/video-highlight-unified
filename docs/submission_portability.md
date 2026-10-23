# 可搬移的比赛推理包

此入口只推理、不训练、不上传比赛官网。代码不依赖学校服务器路径、训练集、
历史预测或固定的视频数量。旧实验目录不覆盖。GPU 全流程验证须另行执行，
CPU 测试及打包哈希通过不代表模型效果或官方成绩。

## 输入和运行

输入文件夹放 `video/<video_id>.mp4` 和 `jsonl/<video_id>.jsonl`，每份 JSONL
仅一行 `{"video_id":"7","targetRatioWH":[9,16]}`。也支持 `videos/` 和
单个 `test_index.json`（对象数组）/`test_index.jsonl`；不要同时放多个索引。
不要求连续编号或 426 条，不需要高光答案、训练标注或预切片。

包内配置 `configs/submission.json` 的相对路径全部以**配置文件所在目录**为基准。
开发仓库使用 `configs/submission.example.json`，填写自己机器的模型位置；
机器绝对路径只放个人配置，不写入执行代码或赛方包。

```bash
python scripts/run_submission_pipeline.py --config configs/submission.json \
  --input-root /path/to/new_testset --output outputs/experiment_a \
  --inference-gpus 0,1 --spatial-gpu 0
```

`/path/to/new_testset` 是使用者传入的示例值，不是程序默认目录。
不提供上述参数时，包内配置默认读 `input/`，写 `outputs/submission/`。
在其他工作目录调用也可；脚本、Prompt、模型均不依赖 shell 当前目录。

CPU 准备和只读计划：

```bash
python scripts/run_submission_pipeline.py --config configs/submission.json --dry-run
python scripts/run_submission_pipeline.py --config configs/submission.json --prepare-only
```

流程自动完成：媒体/画幅检查 → 32 秒窗口/24 秒步长准备 → 指定 Adapter 推理
→ 合并与有审计的格式修复 → 真实原视频 PTS 映射 → v7E 空间裁剪 → 校验和 ZIP。
窗口、FPS、视觉预算和生成参数均在配置中。空间阶段只调用 Stage 2，不调用
复现包的旧 Stage 1，也不读取旧视频元数据、参考高光或按测试视频哈希定制的检测参数。

同一命令重跑复用已完成的检查点；换视频、目标画幅、Prompt、模型、权重或参数，
自动创建新输入缓存/运行，不混用旧回答。不要复制整个旧运行冒充新实验。
异常保留原始回答和失败记录。正式入口固定为有审计的自动容错，不提供严格停跑模式：

- 正常有效回答原样保留；异常回答固定按合法时间交集处理，
  仅对可独立解析的完整条目取合法时间交集，坏条目单独审计，不猜单位、不交换起止。
- 先本地修复，再仅重试仍未解决的精确 `window_id`；`recovery_attempts=1`，
  预算与结果持久化，同一运行重启不叠加新重试。仍失败时自动
  只丢失败窗口，保留同视频其他窗口，记录 `03-lenient/FALLBACKS.json`。
- `03-lenient/TERMINAL_STATES.json` 区分 `ok/repaired/partial/no_highlights/inference_failed`。
  推理全失败产生空结果不等于模型判定无高光。
- 空间异常/缺失框或 worker 崩溃，自动仅对受影响帧使用目标比例下的
  最大合法中心框，遵守赛题 `[x,y,w]` 协议；不改变时间层已经选好的帧集合。
  记录 `05-spatial/work/RECOVERY.json`、`TERMINAL_STATES.json` 和最终 `REPAIRS.json`。
- 缺模型、错误哈希、无 CUDA、输入身份混淆、不可写磁盘等部署故障明确失败，不能伪造全体空结果。

旧配置/命令中的 `recovery_policy=diagnostic`、`unresolved_policy=error`、
`spatial_failure_policy=error`（单独空间入口为 `--failure-policy error`）仅兼容读取：
打印 `strict_runtime_disabled` 警告并固定使用交集修复、失败窗口隔离、空间中心框兜底，
不能关闭兜底。示例配置、发布包和命令帮助不再提供这些切换项。
可将 `recovery_attempts` 设为 0 省去模型重试，但确定性修复与最终兜底始终保留。
局部故障仍有错误详情与降级终态；最终提交协议校验和全局部署/身份检查不关闭。

`window_timeout_sec=600`、`spatial_video_timeout_sec=1800`、`command_timeout_sec=86400`
是可改的初始预算，不是速度承诺。时间窗口有父进程硬看门狗；超时回收本次 worker，
保存该窗口失败后重新启动 worker 补其余窗口。空间超时回收该 worker，只补其尚未完成的视频。
`command_timeout_sec` 是每个阶段命令/时间推理分片预算，**不是整场比赛总时限**；
达到此阶段预算会停止并保留检查点，不能继续运行时假称整批已完成。
尚未启用未经 GPU 验证的 OOM 自动减帧/视觉预算切换。

最终产物位于 `outputs/experiment_a/run-<id>/06-submission/`，包含
`predictions_spatial.jsonl`、`submission.jsonl`、`submission.zip` 和 `FINAL.json`。

## 环境

Linux/Python 3.11 首次部署可运行以下命令；权重、Adapter、空间资产和媒体工具位置
由使用者传入，不写死在代码。若使用含权重的完整交付包，三个资产路径可省略。

```bash
python3.11 scripts/setup_submission_env.py \
  --model /path/to/base_model --adapter /path/to/adapter \
  --spatial-package /path/to/stage2_assets \
  --ffmpeg /path/to/ffmpeg --ffprobe /path/to/ffprobe --with-tests
bash scripts/run_submission.sh --input-root /path/to/new_testset --output outputs/experiment_a
```

安装器在当前仓库/交付包内创建 `.venv`，不继承训练环境的 site-packages；安装固定
CUDA 12.8 PyTorch 及仓库依赖，执行 `pip check` 和隔离模式导入/CUDA 检查。
先从 PyTorch 官方源安装 GPU 版 torch/torchvision 本体，再从当前配置的 PyPI 源统一
解析安装其依赖与项目依赖；两步完成并通过检查前，不视为可运行环境。
提供 `requirements-submission.lock.txt` 时优先使用其中的实测 Linux/Python 3.11 依赖快照；
自定义平台可用 `--requirements` 指定另一份依赖合同，但不继承原验收结论。
驱动/硬件不匹配时须明确调整 `--torch-index-url` 并重新验收，不能自动改用 CPU。
FFmpeg/FFprobe 是系统媒体依赖：先安装，或通过上述参数指定已安装的工具。
安装结果和完整 Python 包版本写到被忽略的 `.runtime/environment.json`；个人路径
与 Adapter 校验值写到 `configs/submission.local.json`，不上传 GitHub、不混入赛方包。
之后只需放入新测试集并执行 `bash scripts/run_submission.sh`；默认读取包内 `input/`，
也可用 `--input-root` 指定新文件夹。不需手工激活环境。
解释器配置保留 `.venv/bin/python` 的调用路径，不把软链接还原成基础 Python；
主流程与子进程必须使用同一独立环境。
新实验改变 Adapter 时用 `--skip-install --adapter ...` 重新配置，不必重装所有包。
只读环境复查使用 `--check-only`。它只验证环境，不宣称模型推理或定位质量已通过。

建议 Linux、Python 3.11 和独立虚拟环境。安装与驱动匹配的 CUDA PyTorch，
再安装 `requirements-submission.txt`；确保 `ffmpeg`、`ffprobe` 在 PATH。
可用 `--python` / `--spatial-python` 指定分离环境，`--ffmpeg` / `--ffprobe`
指定工具位置，不需修改源码。单卡显存不足时显式传多卡；两张 12GB 卡可作为
现有模型配置的起点，但本次可搬移重构尚未重新做 GPU 容量验收。
运行离线加载本地模型，不能通过下载或闭源 API 偷换缺失权重。
时间推理固定 `decord` 视频读取器，与保留的已完成推理日志一致；依赖包含
`decord`、PyAV、accelerate 和 einops，避免只装权重加载库却漏视频读取/分卡依赖。
基础模型的独立 Jinja 聊天模板（含 `chat_templates/`）随包复制，不只复制 JSON 和权重。
逐帧时间戳必须可读、严格递增；带非零旋转元数据的输入目前明确拒绝，
避免自动旋转后裁剪坐标不再属于原视频。不能把这类输入标成正常无高光。

依赖文件沿用项目已有实测版本；新的干净环境安装、不同 CUDA/驱动组合和
打包后的真实 GPU 端到端仍需验收，不能仅凭依赖文件存在声称已通过。
包装器保留 Stage 2 主体检测/关联逻辑，但不携带按历史测试视频哈希选择的检测配置；
原来的比例估算帧号已改为真实 PTS。因此不能声称与历史 41.09 包的完整输出逐帧相同，
更不能把历史成绩算作此新包成绩。

## 从开发仓库制作赛方包

个人实验配置指定本机现有 `model`、`adapter`、`spatial_package`。在有模型的
服务器上打包，避免把大权重经开发电脑来回传：

```bash
python scripts/package_submission.py --config configs/my_experiment.local.json --output dist/release_v2
python dist/release_v2/scripts/package_submission.py --verify dist/release_v2
```

构建器使用白名单打包代码、Prompt、基础模型、LoRA、Stage 2 和 YOLO 权重、
依赖说明与许可证，生成逐文件 SHA256；缺模型立即失败，不生成伪“可运行”的代码空包。
**不带测试视频、标注、参考答案、历史预测、`.env`、SSH key、训练日志或虚拟环境**。
基础模型及 Adapter 元数据随包提供，Adapter 权重字节不改。空间包仅携带 Stage 2
需要的组件，不捆绑无关训练模型。解压后可移动到任意目录；赛方只需安装环境、
放入新输入并执行统一命令。`PACKAGE.json` 明确记录本次包的 GPU 验证仍为待验。

交付前必须：完整包哈希校验 → 搬到另一目录 → 在干净环境做 GPU 小样本全流程
→ 中断续跑与结果逐帧校验。只有这些验收实际完成，才能称“赛方可复现已验收”。
