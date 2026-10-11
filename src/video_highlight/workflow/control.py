"""Shared UI controller: plan and launch the existing CLI, never reproduce its algorithms."""
from __future__ import annotations

from pathlib import Path
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid

from .contracts import fingerprint, read_json, stamp, write_json
from .registry import REPO, PATHS, resolve, versions, default_version
from .runner import make_plan, stop_process

EDITABLE_PATHS = PATHS - {"client_python", "temporal_python"}


class Controller:
    def __init__(self, local, runs_root=None):
        self.local = Path(local).resolve()
        self.defaults = read_json(self.local)
        for field, value in list(self.defaults.items()):
            if field in PATHS:
                path = Path(value).expanduser()
                self.defaults[field] = os.path.abspath(path if path.is_absolute() else self.local.parent / path)
        self.defaults.setdefault("model_dir", str(Path(self.defaults["runtime_root"]) / "models/Video-ORA-4B"))
        self.defaults.setdefault("baseline", str(Path(self.defaults["data_root"]) / "inputs/baseline.zip"))
        self.runs_root = Path(runs_root or REPO / "runs").resolve()
        self.ui_root = self.runs_root / "ui"
        self.jobs = {}
        self.lock = threading.RLock()
        self.resource_cache = None

    def resources(self):
        with self.lock:
            if self.resource_cache and time.monotonic() - self.resource_cache[0] < 2:
                return self.resource_cache[1]
            from video_highlight.runtime.resource_tuning import capture_resources
            value = capture_resources().to_dict()
            value["gpu_available"] = bool(value["gpu_name"])
            value["checked_utc"] = stamp()
            try:
                limit = Path("/sys/fs/cgroup/memory.max").read_text().strip()
                used = int(Path("/sys/fs/cgroup/memory.current").read_text())
                if limit != "max":
                    statistics = dict(line.split() for line in Path("/sys/fs/cgroup/memory.stat").read_text().splitlines())
                    reclaimable = int(statistics.get("inactive_file", "0"))
                    value["ram_total_gib"] = round(int(limit) / (1 << 30), 2)
                    value["ram_available_gib"] = round(min(int(limit), max(0, int(limit) - used + reclaimable)) / (1 << 30), 2)
            except (OSError, ValueError):
                pass
            disk = shutil.disk_usage(self.runs_root if self.runs_root.exists() else self.runs_root.parent)
            value["disk_free_gib"] = round(disk.free / (1 << 30), 2)
            self.resource_cache = time.monotonic(), value
            return value

    def bootstrap(self):
        paths = dict(self.defaults)
        paths["run_root"] = str(self.runs_root / ("scene-v3-" + time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]))
        return {"versions": versions(), "default_version": default_version(), "paths": paths, "resources": self.resources(),
                "core_policy": "冻结的 v3、ORA core、Prompt、帧号与 PTS 保持原样",
                "verification": "CPU 工程验证已通过；新入口 GPU 推理待验收"}

    def plan(self, request):
        if set(request) - {"version", "input_mode", "ids", "paths"}:
            raise ValueError("Unknown plan fields")
        updates = request.get("paths", {})
        if not isinstance(updates, dict) or set(updates) - EDITABLE_PATHS:
            raise ValueError("Only documented input/output paths can be edited")
        paths = {**self.defaults, **updates}
        for field, value in list(paths.items()):
            if not isinstance(value, str) or not value.strip():
                raise ValueError("Empty or invalid path: " + field)
            path = Path(value).expanduser()
            paths[field] = os.path.abspath(path if path.is_absolute() else self.local.parent / path)
        ids = request.get("ids", "all")
        if isinstance(ids, str) and ids != "all":
            ids = [value.strip() for value in ids.split(",") if value.strip()]
        if not paths.get("run_root"):
            paths["run_root"] = self.bootstrap()["paths"]["run_root"]
        root = Path(paths["run_root"])
        if root.exists() and (not root.is_dir() or any(root.iterdir())):
            raise ValueError("输出目录已被使用，请填写新目录；已有任务请用续跑")
        identifier = uuid.uuid4().hex
        folder = self.ui_root / "plans" / identifier
        config = {"schema_version": "scene-workflow/v1", "version": request["version"],
                  "input_mode": request.get("input_mode", "baseline"), "ids": ids, "paths": {}}
        write_json(folder / "config.json", config)
        write_json(folder / "paths.local.json", paths)
        resolved, descriptor = resolve(folder / "config.json", folder / "paths.local.json")
        plan = make_plan(resolved, descriptor)
        write_json(folder / "plan.json", plan)
        return {"plan_id": identifier, "plan": plan, "resources": self.resources()}

    def load_plan(self, identifier):
        if not re.fullmatch(r"[a-f0-9]{32}", identifier):
            raise ValueError("Invalid plan ID")
        folder = self.ui_root / "plans" / identifier
        plan = read_json(folder / "plan.json")
        config, descriptor = resolve(folder / "config.json", folder / "paths.local.json")
        if make_plan(config, descriptor) != plan:
            raise ValueError("执行计划的代码或版本已变化，请重新生成计划")
        return folder, plan

    def require_gpu(self, mode, until):
        if (until != "handoff" or mode == "temporal") and not self.resources()["gpu_available"]:
            raise ValueError("GPU 不可用。可先选择已有 baseline/交接并执行“准备输入”，推理需先申请 GPU")

    def launch(self, root, command):
        root = Path(root)
        with self.lock:
            for job in self.jobs.values():
                if job["process"].poll() is None and job["run_root"] == str(root):
                    raise ValueError("该运行已有任务在执行")
            identifier = uuid.uuid4().hex
            log = self.ui_root / "jobs" / (identifier + ".log")
            log.parent.mkdir(parents=True, exist_ok=True)
            env = os.environ.copy()
            env.update(PYTHONPATH=str(REPO / "src"), PYTHONUNBUFFERED="1", PYTHONUTF8="1")
            with log.open("ab", buffering=0) as stream:
                process = subprocess.Popen(command, cwd=REPO, env=env, start_new_session=True,
                                           stdout=stream, stderr=subprocess.STDOUT)
            self.jobs[identifier] = {"process": process, "run_root": str(root), "log": str(log),
                                     "command": command, "started_utc": stamp()}
            record = self.job(identifier)
            write_json(log.with_suffix(".json"), record)
            return record

    def start(self, identifier, until="complete"):
        if until not in {"complete", "handoff", "temporal"}:
            raise ValueError("Unknown stage boundary")
        folder, plan = self.load_plan(identifier)
        mode = plan["config"]["input_mode"]
        if until == "temporal" and mode != "temporal":
            raise ValueError("时间阶段模式不匹配")
        self.require_gpu(mode, until)
        root = Path(plan["config"]["paths"]["run_root"])
        with self.lock:
            if root.exists() and (not root.is_dir() or any(root.iterdir())):
                raise ValueError("该计划的运行目录已使用，请在运行记录中续跑")
            return self.launch(root, [plan["config"]["paths"]["client_python"], str(REPO / "vh.py"), "workflow", "run",
                                     "--config", str(folder / "config.json"), "--local", str(folder / "paths.local.json"), "--until", until])

    def job(self, identifier):
        with self.lock:
            item = self.jobs.get(identifier)
            if item is None:
                raise ValueError("当前界面没有启动该任务")
            record = {"job_id": identifier, **{k: v for k, v in item.items() if k not in {"process", "saved_exit_code"}},
                      "pid": item["process"].pid, "exit_code": item["process"].poll()}
            if record["exit_code"] is not None and item.get("saved_exit_code") != record["exit_code"]:
                write_json(Path(item["log"]).with_suffix(".json"), record)
                item["saved_exit_code"] = record["exit_code"]
            return record

    def stop(self, identifier):
        with self.lock:
            if identifier not in self.jobs:
                raise ValueError("只能停止当前界面启动的任务")
            stop_process(self.jobs[identifier]["process"])
            return self.job(identifier)

    def directories(self):
        found = set()
        if self.runs_root.is_dir():
            found.update(path.resolve() for path in self.runs_root.iterdir() if path.is_dir() and path.name != "ui")
        historical = Path(self.defaults["data_root"]) / "output"
        if historical.is_dir():
            found.update(path.resolve() for path in historical.glob("scene_temporal_*") if path.is_dir())
        with self.lock:
            found.update(Path(job["run_root"]).resolve() for job in self.jobs.values())
        return sorted(found, key=lambda path: path.stat().st_mtime if path.exists() else 0, reverse=True)

    def lookup_run(self, identifier):
        for path in self.directories():
            if fingerprint(str(path))[:24] == identifier:
                return path
        raise ValueError("Unknown run")

    def describe(self, path):
        path = Path(path)
        status_file = path / "status.json"
        state = read_json(status_file) if status_file.is_file() else {"status": "unknown"}
        plan_file = path / "plan.json"
        plan = read_json(plan_file) if plan_file.is_file() else {}
        version = plan.get("version", "历史 v3" if path.name.startswith("scene_temporal_") else "未登记")
        validation = next(iter(path.glob("*.validation.json")), None)
        if state["status"] == "unknown" and validation:
            result = read_json(validation)
            state.update(status="archived", completed_ids=list(result.get("changes_by_video", {})),
                         requested_ids=list(result.get("changes_by_video", {})))
        jobs = [self.job(key) for key, value in list(self.jobs.items()) if value["run_root"] == str(path)]
        return {"run_id": fingerprint(str(path))[:24], "name": path.name, "path": str(path), "version": version,
                "status": state.get("status", "unknown"), "stage": state.get("stage", "—"),
                "completed": len(state.get("completed_ids", [])), "total": len(state.get("requested_ids", [])),
                "current_video": state.get("current_video"), "error": state.get("error"), "jobs": jobs,
                "can_resume": bool(plan.get("schema_version") == "workflow-plan/v1"), "updated_utc": state.get("updated_utc")}

    def runs(self):
        result = []
        for path in self.directories():
            try:
                result.append(self.describe(path))
            except (OSError, ValueError, KeyError):
                continue
        return result

    def detail(self, identifier):
        path = self.lookup_run(identifier)
        result = self.describe(path)
        result["artifacts"] = [{"name": file.relative_to(path).as_posix(), "bytes": file.stat().st_size} for file in self.artifacts(path)]
        result["records"] = {name: read_json(path / name) for name in ("plan.json", "status.json", "final/artifacts.json", "spatial/performance.json") if (path / name).is_file()}
        logs = sorted((path / "logs").glob("*.log"), key=lambda file: file.stat().st_mtime, reverse=True)
        logs += sorted(path.glob("*.log"), key=lambda file: file.stat().st_mtime, reverse=True)
        if result["jobs"]:
            logs.insert(0, Path(result["jobs"][-1]["log"]))
        else:
            # Keep CLI stdout visible after restarting the interface, without taking process ownership.
            records = sorted((self.ui_root / "jobs").glob("*.json"), key=lambda file: file.stat().st_mtime, reverse=True)
            for file in records:
                item = read_json(file)
                log = Path(item.get("log", ""))
                if item.get("run_root") == str(path) and log.is_file() and log.parent == self.ui_root / "jobs":
                    logs.insert(0, log)
                    break
        if logs:
            with logs[0].open("rb") as stream:
                stream.seek(max(0, logs[0].stat().st_size - (32 << 10)))
                result["log"] = stream.read().decode("utf-8", errors="replace")
            result["log_path"] = str(logs[0])
        return result

    def artifacts(self, path):
        path = Path(path)
        files = [path / "final" / name for name in ("submission.zip", "predictions_spatial.jsonl", "validation.json", "artifacts.json")]
        if path.name.startswith("scene_temporal_"):
            files.extend(path.glob("SPATIAL_temporal_v3*.zip"))
            files.extend(path.glob("SPATIAL_temporal_v3*.jsonl"))
        return [file for file in files if file.is_file() and file.resolve().is_relative_to(path.resolve())]

    def artifact(self, identifier, name):
        path = self.lookup_run(identifier)
        for file in self.artifacts(path):
            if file.relative_to(path).as_posix() == name:
                return file
        raise ValueError("只能下载该运行登记的正式产物")

    def resume(self, identifier, until="complete"):
        path = self.lookup_run(identifier)
        if until not in {"complete", "handoff", "temporal"}:
            raise ValueError("Unknown stage boundary")
        plan = read_json(path / "plan.json")
        self.require_gpu(plan["config"]["input_mode"], until)
        return self.launch(path, [plan["config"]["paths"]["client_python"], str(REPO / "vh.py"), "workflow", "resume",
                                  "--run-root", str(path), "--until", until])
