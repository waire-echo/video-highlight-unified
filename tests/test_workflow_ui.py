"""UI actions share CLI contracts; interface isolation and process ownership are enforced."""
from contextlib import contextmanager
import http.client
import json
from pathlib import Path
import re
import subprocess
import sys
import threading

import pytest

from video_highlight.workflow import control, runner
from video_highlight.workflow.contracts import write_json
from video_highlight.workflow.terminal import display, fit
from video_highlight.workflow.web import server


@pytest.fixture
def controller(tmp_path, monkeypatch):
    data = tmp_path / "data"
    write_json(data / "inputs/manifest.json", {"videos": {"0": {}, "1": {}}})
    local = tmp_path / "paths.local.json"
    write_json(local, {"data_root": "data", "video_dir": "videos", "runtime_root": "runtime",
                       "run_root": "runs/new", "client_python": sys.executable,
                       "temporal_handoff": "handoff.jsonl", "highlight_frames": "frames.jsonl", "test_index": "index.json"})
    value = control.Controller(local, tmp_path / "runs")
    monkeypatch.setattr(value, "resources", lambda: {"gpu_available": False, "gpu_name": None,
                                                   "cpu_count": 1, "ram_available_gib": 12, "disk_free_gib": 10})
    return value


def request(version="scene-general-v3-compat-20261010", mode="baseline", **paths):
    return {"version": version, "input_mode": mode, "ids": "0,1", "paths": paths}


def test_plan_freezes_version_and_resolves_paths_at_machine_config(controller):
    result = controller.plan(request(run_root="runs/ui-relative"))
    assert result["plan"]["config"]["paths"]["run_root"] == str(controller.local.parent / "runs/ui-relative")
    assert result["plan"]["config"]["ids"] == ["0", "1"]
    assert not Path(result["plan"]["outputs"]["run"]).exists()
    assert controller.load_plan(result["plan_id"])[1] == result["plan"]
    with pytest.raises(ValueError):
        controller.load_plan("../plan")
    with pytest.raises(ValueError, match="Only documented"):
        controller.plan(request(client_python="/bin/sh"))
    with pytest.raises(ValueError):
        controller.plan(request(run_root=None))


def test_gpu_gating_and_cpu_commands_use_shared_cli(controller, monkeypatch):
    result = controller.plan(request())
    calls = []
    monkeypatch.setattr(controller, "launch", lambda root, command: calls.append((root, command)) or {"pid": 1})
    with pytest.raises(ValueError, match="GPU"):
        controller.start(result["plan_id"])
    assert not calls
    controller.start(result["plan_id"], "handoff")
    command = calls[-1][1]
    assert command[:4] == [sys.executable, str(control.REPO / "vh.py"), "workflow", "run"]
    assert command[-2:] == ["--until", "handoff"]
    write_json(Path(result["plan"]["outputs"]["run"]) / "plan.json", result["plan"])
    with pytest.raises(ValueError, match="已使用"):
        controller.start(result["plan_id"], "handoff")
    run = controller.runs()[0]
    controller.resume(run["run_id"], "handoff")
    assert calls[-1][1][2:4] == ["workflow", "resume"]
    temporal = controller.plan(request("scene-general-v3-unified-v1", "temporal", run_root="runs/temporal"))
    with pytest.raises(ValueError, match="GPU"):
        controller.start(temporal["plan_id"], "handoff")


def test_changed_release_or_plan_cannot_be_started(controller):
    result = controller.plan(request())
    folder, plan = controller.load_plan(result["plan_id"])
    plan["version"] = "scene-general-v3-io-v1"
    write_json(folder / "plan.json", plan)
    with pytest.raises(ValueError, match="已变化"):
        controller.load_plan(result["plan_id"])


def test_stop_targets_only_owned_process_group_and_logs_survive_restart(controller):
    root = controller.runs_root / "process"
    root.mkdir(parents=True)
    command = [sys.executable, "-u", "-c", "print('ready', flush=True); import time; time.sleep(30)"]
    job = controller.launch(root, command)
    with pytest.raises(ValueError, match="执行"):
        controller.launch(root, command)
    with pytest.raises(ValueError, match="只能停止"):
        controller.stop("unknown")
    stopped = controller.stop(job["job_id"])
    assert stopped["exit_code"] is not None
    again = control.Controller(controller.local, controller.runs_root)
    detail = again.detail(again.runs()[0]["run_id"])
    assert detail["log_path"] == job["log"]
    assert detail["jobs"] == []


@contextmanager
def running_server(controller):
    http = server(controller, 0)
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    try:
        yield http.server_port
    finally:
        http.shutdown()
        http.server_close()
        thread.join(timeout=3)


def call(port, path, data=None, headers=None):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    body = json.dumps(data) if data is not None else None
    connection.request("POST" if data is not None else "GET", path, body, headers or {})
    response = connection.getresponse()
    result = response.status, response.read().decode(), dict(response.getheaders())
    connection.close()
    return result


def test_webui_api_token_origin_host_assets_and_gpu_disabled(controller):
    with running_server(controller) as port:
        status, html, headers = call(port, "/")
        assert status == 200 and "frame-ancestors 'none'" in headers["Content-Security-Policy"]
        token = re.search(r'name="workflow-token" content="([^"]+)"', html)[1]
        auth = {"X-Workflow-Token": token, "Origin": f"http://127.0.0.1:{port}"}
        assert call(port, "/app.js")[0] == 200
        assert call(port, "/style.css")[0] == 200
        assert call(port, "/../control.py")[0] == 404
        assert call(port, "/api/bootstrap", headers={"Host": "evil.example"})[0] == 403
        assert call(port, "/api/plan", request())[0] == 403
        assert call(port, "/api/plan", request(), {**auth, "Origin": "https://evil.example"})[0] == 403
        status, body, _ = call(port, "/api/bootstrap")
        assert status == 200 and len(json.loads(body)["versions"]) == 3
        status, body, _ = call(port, "/api/plan", request(), auth)
        assert status == 200
        identifier = json.loads(body)["plan_id"]
        status, body, _ = call(port, "/api/start", {"plan_id": identifier}, auth)
        assert status == 400 and "GPU" in json.loads(body)["error"]
        assert call(port, "/api/stop", {"job_id": "unknown"}, auth)[0] == 400


def test_tui_selected_run_is_visible_in_small_window(controller, monkeypatch):
    runs = [{"name": f"run-{i}", "status": "complete", "completed": 1, "total": 1, "stage": "deliver"} for i in range(30)]
    monkeypatch.setattr(controller, "runs", lambda: runs)
    lines, _ = display(controller, 29, max_runs=4)
    assert any(line.startswith("> run-29") for line in lines)
    assert len(lines) < 18


def test_terminal_width_accounts_for_chinese_and_combining_characters():
    assert fit("中文abc", 5) == "中文a"
    assert fit("e\u0301中文", 3) == "e\u0301中"


def test_registered_artifact_download_rejects_arbitrary_and_symlink_paths(controller):
    root = controller.runs_root / "complete"
    write_json(root / "final/validation.json", {"pass": True})
    (root / "final/submission.zip").symlink_to(controller.local)
    run = controller.runs()[0]
    assert [row["name"] for row in controller.detail(run["run_id"])["artifacts"]] == ["final/validation.json"]
    with pytest.raises(ValueError):
        controller.artifact(run["run_id"], "../paths.local.json")
    with pytest.raises(ValueError):
        controller.artifact(run["run_id"], "final/submission.zip")
    with running_server(controller) as port:
        path = f"/api/artifact?id={run['run_id']}&name=final/validation.json"
        status, body, headers = call(port, path)
        assert status == 200 and json.loads(body)["pass"]
        assert headers["Content-Disposition"].startswith("attachment;")


@pytest.mark.parametrize("error", [OSError("Exec format error"), subprocess.TimeoutExpired("nvidia-smi", 15)])
def test_no_gpu_driver_fails_before_launch(monkeypatch, error):
    def probe(*args, **kwargs):
        raise error
    monkeypatch.setattr(runner.subprocess, "run", probe)
    with pytest.raises(ValueError, match="No GPU is available"):
        with runner.managed_model(["never-start"], 8008, None, {}):
            pytest.fail("Inference must not start without GPU")
