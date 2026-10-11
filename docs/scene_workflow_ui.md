# TUI 与 WebUI 使用说明

两个界面调用同一个 `workflow` CLI 和 Controller。两个界面的默认版本均从 configs/project_policy.json 读取，当前是 `scene-general-v3-compat-20261010`。切换实验版本时生成新计划，使用全新的运行目录；已有任务通过原计划续跑。core、冻结的 v3、Prompt、FPS、帧号和 PTS 没有修改。

## WebUI

服务器已经部署在 `127.0.0.1:8189`。在自己的电脑执行端口转发，按 SSH 提示输入密码：

```bash
ssh -N -L 8189:127.0.0.1:8189 root@hn01-ssh.gpuhome.cc -p 30192
```

保持这个终端连接，在浏览器打开 `http://127.0.0.1:8189`。

操作顺序：

1. 选择版本和输入方式，核对视频、输入和新运行目录。默认生成唯一目录。
2. 点击“生成执行计划”，查看实际阶段、命令、目录和版本指纹。
3. 无 GPU 时，已有 baseline 或时间交接可以点击“准备输入 · CPU”。完整推理按钮等待 GPU。
4. 有 GPU 时运行全流程；时间模式也支持只完成时间阶段。
5. 在“运行与产物”查看状态、日志、错误和正式产物，下载提交 ZIP / JSONL，或者续跑。

视频 IDs 按原 manifest 顺序填写。**已有 handoff 和 highlight_frames 必须恰好包含所选 IDs**；本服务器现有全量交接文件应填写 `all`。做小样本时先准备对应的交接文件；完整时间模式的 test_index 同样需要对应的 IDs。失败检查不会悄悄丢弃额外视频。

服务只监听服务器本机，通过 SSH 转发访问。页面启动时生成临时操作 token；页面刷新后使用当前 token。下载限定于已登记运行的正式产物，没有任意文件浏览接口。界面只能停止自己启动的任务；历史任务和其他进程不提供停止按钮。

关闭界面页面不会停止任务。重启服务后，可查看运行目录、已保存日志和状态，再按来源一致的计划续跑。跨版本、代码提交、输入或权重变化时会拒绝续跑；先检出原提交，或新建实验目录。

如果服务器重启，先在 SSH 终端启动：

```bash
cd /root/rivermind-data/development/video-highlight-unified
/root/rivermind-data/development/vh workflow webui \
  --local configs/workflows/server.local.json --port 8189
```

前台服务用 Ctrl+C 关闭。需要后台服务时：

```bash
cd /root/rivermind-data/development/video-highlight-unified
mkdir -p runs/ui
nohup /root/rivermind-data/development/vh workflow webui \
  --local configs/workflows/server.local.json --port 8189 \
  > runs/ui/webui.log 2>&1 < /dev/null &
echo $! > runs/ui/webui.pid
```

服务状态与日志：`ps -p "$(cat runs/ui/webui.pid)" -o pid,args`、`tail -n 50 runs/ui/webui.log`。确认 PID 的命令是本次 WebUI 后，可用 `kill -TERM "$(cat runs/ui/webui.pid)"` 关闭服务。已有进程占用 8189 时，不要再启动第二个服务。

## TUI

```bash
ssh -t root@hn01-ssh.gpuhome.cc -p 30192 \
  '/root/rivermind-data/development/vh workflow tui --local configs/workflows/server.local.json'
```

| 按键 | 操作 |
|---|---|
| `n` | 选择版本、模式、路径和 IDs，保存新计划 |
| `p` | 查看完整计划，↑↓滚动，Enter 返回 |
| `h` | 准备已有输入，到 handoff 为止 |
| `g` | 完整推理，无 GPU 时给出提示 |
| ↑↓ / Enter | 选择运行，查看日志和产物路径 |
| `r` | 续跑选中运行，可选择 handoff / complete / temporal |
| `v` | 查看版本清单 |
| `x` | 停止该 TUI 启动的运行 |
| `q` | 退出界面，已启动任务继续运行 |

TUI 需要交互终端。终端显示中文建议使用 UTF-8；80×24 的窗口可用，长运行列表会滚动。TUI 查看产物路径，WebUI 可以直接下载。

## 已验证范围

服务器 1 核 CPU、16 GiB 容器内存、无 GPU。内存显示计入可回收的 inactive file cache。桌面 1440 px 与手机 390 px 页面均无横向溢出，浏览器无脚本错误。WebUI 实际生成计划、准备输入、续跑、切换版本、查看历史并下载原提交 ZIP；TUI 通过 SSH PTY 生成计划和完成 CPU 输入准备。

CPU 与历史结果的验收记录见 [验证记录](scene_workflow_validation.md)。新入口完整 GPU 推理、优化版新模型回答等价性仍需申请 GPU 后验收。三路同步对比、预计剩余时间和其他时间模型自动适配继续作为后续开发项。
