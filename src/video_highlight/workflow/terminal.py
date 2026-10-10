"""Curses TUI using exactly the same controller as the WebUI."""
from __future__ import annotations

import curses
import json
from pathlib import Path
import sys
import time
import unicodedata

from .control import Controller


def fit(text, columns):
    result, width = [], 0
    for char in str(text):
        size = 0 if unicodedata.combining(char) else 2 if unicodedata.east_asian_width(char) in {"W", "F"} else 1
        if width + size > columns:
            break
        result.append(char)
        width += size
    return "".join(result)


def display(controller, selected=0, max_runs=12):
    resource = controller.resources()
    runs = controller.runs()
    lines = ["ORA 工作台 / scene_general_v3 / core 保持原样", "",
             f"GPU: {resource['gpu_name'] or '未分配'}  CPU: {resource['cpu_count']} 核  内存: {resource['ram_available_gib']} GiB  磁盘: {resource['disk_free_gib']} GiB",
             "新 GPU 全链路待验收；无 GPU 可生成计划和准备已有输入。", "",
             "[n] 新建计划  [p] 查看计划  [h] 准备输入(CPU)  [g] 完整运行(GPU)",
             "[r] 续跑选中任务  [v] 版本  [Enter] 日志/详情",
             "[x] 停止本界面任务  [q] 退出", "",
             "运行记录（↑↓选择）"]
    start = max(0, selected - max_runs + 1)
    for index, run in enumerate(runs[start:start + max_runs], start):
        marker = ">" if index == selected else " "
        lines.append(f"{marker} {run['name']} | {run['status']} | {run['completed']}/{run['total']} | {run['stage']}")
    if len(runs) > max_runs:
        lines.append(f"  显示 {start + 1}–{min(len(runs), start + max_runs)} / {len(runs)} 条（↑↓滚动）")
    if not runs:
        lines.append("  尚无运行，按 n 生成新计划。")
    return lines, runs


def tui(local, runs_root=None, smoke=False):
    if not sys.stdin.isatty():
        raise ValueError("TUI 需要交互终端，请通过 ssh -t 连接，或使用 WebUI")
    controller = Controller(local, runs_root)
    def screen_main(screen):
        try:
            curses.curs_set(0)
        except curses.error:
            pass
        screen.timeout(1000)
        selected, plan, message = 0, None, ""
        def draw(lines):
            screen.erase()
            height, width = screen.getmaxyx()
            for y, line in enumerate(lines[:height - 1]):
                try:
                    screen.addstr(y, 0, fit(line, max(1, width - 2)))
                except curses.error:
                    pass
            screen.refresh()
        def prompt(label, default=""):
            labels = label if isinstance(label, list) else [label]
            lines = [*labels, "", "默认：" + default, "输入新值后回车，留空使用默认值："]
            draw(lines)
            screen.timeout(-1)
            curses.echo()
            try:
                answer = screen.getstr(min(len(lines) + 1, screen.getmaxyx()[0] - 2), 0, 4096).decode("utf-8").strip()
            finally:
                curses.noecho()
                screen.timeout(1000)
            return answer or default
        def inspect(lines):
            offset = 0
            while True:
                draw(["[Esc/q/Enter] 返回，↑↓滚动", "", *lines[offset:]])
                key = screen.getch()
                if key in (27, ord("q"), 10, 13):
                    return
                if key == curses.KEY_DOWN:
                    offset = min(max(0, len(lines) - 1), offset + 1)
                if key == curses.KEY_UP:
                    offset = max(0, offset - 1)
        while True:
            lines, runs = display(controller, selected, max_runs=max(1, screen.getmaxyx()[0] - 17))
            selected = min(selected, max(0, len(runs) - 1))
            draw([*lines, "", "当前计划：" + (plan["plan"]["version"] if plan else "未生成"), message])
            if smoke:
                time.sleep(0.1)
                return
            key = screen.getch()
            try:
                if key == ord("q"):
                    return
                if key == curses.KEY_DOWN:
                    selected = min(len(runs) - 1, selected + 1)
                elif key == curses.KEY_UP:
                    selected = max(0, selected - 1)
                elif key == ord("n"):
                    boot = controller.bootstrap()
                    choices = boot["versions"]
                    menu = ["选择版本：", *[f"{i+1} = {row['id']}" for i, row in enumerate(choices)]]
                    choice = int(prompt(menu, "1")) - 1
                    if not 0 <= choice < len(choices):
                        raise ValueError("无效版本编号")
                    release = choices[choice]
                    mode = prompt("输入模式：" + ", ".join(release["input_modes"]), "baseline")
                    paths = dict(boot["paths"])
                    paths["run_root"] = prompt("新运行目录", paths["run_root"])
                    paths["video_dir"] = prompt("原视频目录", paths["video_dir"])
                    paths["data_root"] = prompt("外部数据目录", paths["data_root"])
                    fields = {"baseline": ("baseline",), "handoff": ("temporal_handoff", "highlight_frames"),
                              "temporal": ("test_index",)}.get(mode, ())
                    for field in fields:
                        paths[field] = prompt(field + " 路径", paths.get(field, ""))
                    ids = prompt("视频 IDs（原顺序，逗号分隔）", "all")
                    paths.pop("client_python", None)
                    paths.pop("temporal_python", None)
                    plan = controller.plan({"version": release["id"], "input_mode": mode, "ids": ids, "paths": paths})
                    message = "计划已保存。按 p 查看，h 准备输入，g 执行完整流程。"
                elif key == ord("p"):
                    if not plan:
                        raise ValueError("请先按 n 新建计划")
                    inspect(json.dumps(plan["plan"], ensure_ascii=False, indent=2).splitlines())
                elif key in (ord("h"), ord("g")):
                    if not plan:
                        raise ValueError("请先按 n 新建计划")
                    job = controller.start(plan["plan_id"], "handoff" if key == ord("h") else "complete")
                    message = "任务已启动，PID " + str(job["pid"])
                elif key == ord("v"):
                    inspect(json.dumps(controller.bootstrap()["versions"], ensure_ascii=False, indent=2).splitlines())
                elif key in (10, 13) and runs:
                    run = controller.detail(runs[selected]["run_id"])
                    artifacts = [str(Path(run["path"]) / row["name"]) for row in run["artifacts"]]
                    inspect([run["path"], json.dumps(run.get("error"), ensure_ascii=False), "正式产物：", *artifacts,
                             "", *run.get("log", "没有日志").splitlines()[-150:]])
                elif key == ord("r") and runs:
                    until = prompt("续跑到哪一步：handoff / complete / temporal", "handoff" if not controller.resources()["gpu_available"] else "complete")
                    job = controller.resume(runs[selected]["run_id"], until)
                    message = "续跑已启动，PID " + str(job["pid"])
                elif key == ord("x") and runs:
                    jobs = [job for job in runs[selected]["jobs"] if job["exit_code"] is None]
                    if not jobs:
                        raise ValueError("此运行没有当前界面可停止的任务")
                    controller.stop(jobs[0]["job_id"])
                    message = "任务已停止，可按原来源续跑。"
            except (OSError, ValueError, KeyError, IndexError) as error:
                message = "提示：" + str(error)
    curses.wrapper(screen_main)
    return 0
