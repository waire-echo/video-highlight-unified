# SFT-v2 独立分支与服务器重新部署验收

2026-10-23。本次只处理 `video-highlight-unified-new`，不改另一个训练工作区、旧运行结果或其他人的 GPU 任务。

## 验收目标

从已推送的新分支创建干净工作区，不继承训练环境的 Python 包；按仓库的
`requirements-submission.txt` 创建独立 Python 3.11 环境，检查 pip、视频读取器和 CUDA。
基础模型、第二轮 Adapter 和空间资产通过明确配置供给，不把服务器路径写入执行代码。
完成真实小样本“输入 → 推理 → 时间交接 → 空间 → 最终校验 → ZIP”、同命令续跑，
以及包含权重的白名单包搬目录验证。没有上传比赛官网，不消耗评测机会。

## 已完成的验收

新分支：[`deploy/sft-v2-portable-20261023`](https://github.com/thekfjie/video-highlight-unified/tree/deploy/sft-v2-portable-20261023)。
基础运行修复 `e650082`；交付包训练元数据隔离修复 `cfb262a`。代码与权重的逐文件哈希、
完整环境版本、子进程日志和验证收据保留在服务器私有部署收据目录；不公开预测答案或凭据。

| 检查 | 实测结果 |
|---|---|
| 旧 `new` 工作区重建 | 精确确认目录、无在跑任务，先归档校验再删除；从新分支对应 Git bundle 创建干净 checkout，origin 保留 GitHub 仓库地址 |
| 独立环境 | Python 3.11.13，新建 `.venv`，`include-system-site-packages=false`；未继承训练环境的第三方包 |
| 依赖和 CUDA | `pip check` 通过；实际导入推理、解码和 YOLO 依赖；CUDA 12.8、两张 RTX 3060 12GB 可用 |
| 版本快照 | 实装 152 个包的 `requirements-submission.lock.txt`；torch 2.10.0+cu128、Swift 4.6.0、transformers 5.19.0、PEFT 0.20.0、YOLO/ultralytics 8.4.106 |
| 回归测试 | 本地 Windows Python 3.10：164 项通过；服务器独立 Python 3.11 + 实际媒体工具：164 项通过；交付清理修复后再次 164 项通过（46.64 秒） |
| 搬目录运行 | 含基础模型、第二轮 Adapter、Stage 2/YOLO 的完整包先通过文件 SHA256 校验，再搬到另一目录，从无关工作目录调用统一 shell 入口 |
| 真实时间推理 | GPU 0、1 分担模型；Adapter 256 个张量逐项回载比对；2 段约 40 秒公开视频、4 个窗口，4 个有效、0 个失败 |
| 真实空间阶段 | YOLO 检测缓存与逐视频检查点实际生成；2 个视频、3 组高光、1,109 帧；无视频降级、无最终坏框修复 |
| 最终交付 | `FINAL.json` 为 `READY_TO_SUBMIT`；两行 JSONL，ZIP 仅包含 `predictions_spatial.jsonl`，CRC 和内容 SHA256 通过 |
| 真实中断后续跑 | 首个有效窗口落盘后，向本次控制进程发送 SIGTERM；退出、保留失败证据；同一命令续跑至完成，原窗口内容 SHA256 和 mtime 均未改变 |
| 完成后再次运行 | 无新增推理/空间 worker 命令；JSONL SHA256 保持一致；ZIP 可重新生成，其容器时间戳不要求逐字节一致 |
| 精简交付包复验 | 移出 3 份训练参数/日志元数据后，再从空输出目录真实跑完同一公开小样本；4/4 窗口、1,109 帧，无空间降级，最终 JSONL 与首轮中断续跑产物 SHA256 一致 |

GPU 小样本采用**未读任何高光标注的公开 SumMe 视频**，不是官方比赛测试集。
当前公开小样本上，时间推理峰值 allocated 显存分别约 **7.12 / 4.60 GiB**；
这是该输入和冻结配置下的实测值，不是所有视频的显存上限承诺。
三个验收阶段的墙钟约 135 / 156 / 61 秒，包含资产哈希、加载、受控中断、续跑和导出，
不可将其中某个数字当作整批比赛数据的推理耗时。完成后重跑仍执行身份/哈希检查，并非立即返回。

原 `new` 源码归档 `old-new-source.tgz` 已验证，SHA256：
`939e4fbee605624ffa8092c912eacbc68f894b33f14dc2b922d0d0fd7cb831fa`。
旧权重、历史运行结果和另一个训练工作区均未删除。交付包排除的训练参数/日志元数据
移到私有收据目录备查，不混入赛方包。

## 本次发现并修复的部署问题

- Linux venv 的 `bin/python` 是软链接；对解释器使用 `resolve()` 会错误跳回基础环境。
  配置、命令选择和子进程现保留 venv 调用路径，并通过真实 venv 的 `sys.prefix` 回归验证。
- 新建环境按 GPU wheel 元数据安装精确 NVIDIA/Triton 运行库，避免 CUDA 小版本混装。
- Decord 0.6.0 的公开 wheel 内部误标 cp36。只修正新 venv 的 WHEEL/RECORD 元数据并记录前后哈希；
  原生解码器字节不改，原生库加载、真实视频读取与完整 `pip check` 均通过。
- 服务器旧 FFprobe 不支持 `stream_side_data` 选择器。分开读取完整流元数据与逐帧 PTS，
  不丢失旋转信息；兼容旧工具，同时继续明确拒绝尚未支持原坐标映射的旋转视频。
- Adapter 发布仅取配置和推理权重，不复制 checkpoint 的 `args.json`、`trainer_state.json`、
  `additional_config.json` 等训练参数/日志；相应白名单断言纳入搬目录测试。

## 下一次实验怎么运行

服务器 checkout 的个人配置与独立环境已准备好。测试集只需提供 `video/` 和 `jsonl/`，
或说明文档支持的索引；不要放高光答案。进入该 checkout，运行：

```bash
bash scripts/run_submission.sh --input-root /path/to/new_testset --output outputs/experiment_a
```

默认个人配置使用 GPU 0、1 做时间推理，GPU 0 做随后执行的空间阶段；不用手动切环境。
有任务占卡时先选择空闲 GPU，别与其他任务共用。更换输出目录保留新实验；
同一输入和命令重跑则接续既有检查点。

更换已训练好的 Adapter 不需重装全部依赖：

```bash
.venv/bin/python -I -B scripts/setup_submission_env.py --skip-install --adapter /path/to/new_adapter
bash scripts/run_submission.sh --input-root /path/to/new_testset --output outputs/experiment_b
```

新机器需先具备 Linux x86_64、Python 3.11、NVIDIA 驱动、匹配 GPU 和 FFmpeg/FFprobe；
按照 [环境与打包说明](submission_portability.md) 从仓库创建独立环境，或使用含权重的发布包。
Git 分支不包含大权重；源码 checkout 首次部署必须明确供给基础模型、Adapter 和 Stage 2 资产。
完整包已包含三类资产，其路径均相对配置文件；系统媒体工具可放 PATH 或由参数传入。

结果读取输出目录下的 `CURRENT.json`，其 `run_root` 下 `06-submission/` 包含
`submission.zip`、`predictions_spatial.jsonl`、`submission.jsonl`、`FINAL.json`。
正式入口固定默认兜底，旧严格开关只警告并被覆盖；错误详情、修复与降级终态仍可检查。

## 验收边界

以上证明当前环境和小样本工程流程能跑，不证明高光语义质量、官方成绩、全部未见测试集
或比赛总时限一定满足；没有重新训练、人工修正测试答案、上传比赛官网或消耗评测次数。
通用打包器不自动继承本次验收，其他权重/依赖/采样配置和新构建仍须重新验证。
最终训练元数据清理后的完整包已完成独立复验。首次全流程及续跑收据、精简包复验收据
和最终文件哈希均保留，当前已验收实体包另附 `ACCEPTANCE.json`；通用构建默认仍为待验。
