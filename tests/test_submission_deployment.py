"""Deployment plumbing tests do not install packages or claim GPU quality."""
import json
import os
from pathlib import Path
import subprocess
import sys
import venv

import pytest

import setup_submission_env as setup
from video_highlight import submission_inputs as inputs
from test_submission_portability import example, put


def test_generated_personal_config_is_relative_and_has_no_strict_switch(example):
    folder = example.root / "checkout/configs"
    config = setup.make_config(
        {"schema": "video-highlight-submission-config/v1", "unresolved_policy": "error"}, folder,
        python=example.root / "checkout/.venv/bin/python", model=example.model,
        adapter=example.adapter, spatial=example.spatial, ffmpeg=Path(sys.executable),
        ffprobe=Path(sys.executable), adapter_sha="fixture", inference_gpus="1,2", spatial_gpu="1")
    path = folder / "submission.local.json"
    put(path, config)
    resolved = inputs.resolve_config(path)
    assert resolved["model"] == example.model
    assert resolved["adapter"] == example.adapter
    assert resolved["python"] == str(example.root / "checkout/.venv/bin/python")
    assert not {"unresolved_policy", "recovery_policy", "spatial_failure_policy"} & config.keys()


def test_deployment_checks_actual_asset_hashes(example):
    assert setup.validate_assets(example.model, example.adapter, example.spatial) == inputs.digest(
        example.adapter / "adapter_model.safetensors")
    (example.spatial / "assets/yolo11m.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="detector weights"):
        setup.validate_assets(example.model, example.adapter, example.spatial)


def test_environment_probe_excludes_development_pythonpath(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "must-not-leak")
    monkeypatch.setenv("PYTHONHOME", "must-not-leak")
    seen = []

    def run(command, **options):
        seen.append((command, options))
        return subprocess.CompletedProcess(command, 0, json.dumps({"gpu_count": 2}), "")

    monkeypatch.setattr(setup.subprocess, "run", run)
    result = setup.probe_environment(Path(sys.executable), "1,2")
    assert result == {"gpu_count": 2}
    assert seen[0][0][1:3] == ["-I", "-B"]
    assert seen[0][1]["env"]["CUDA_VISIBLE_DEVICES"] == "1,2"
    assert "PYTHONPATH" not in seen[0][1]["env"] and "PYTHONHOME" not in seen[0][1]["env"]


def test_environment_probe_does_not_hide_import_failure(monkeypatch):
    monkeypatch.setattr(setup.subprocess, "run", lambda command, **options:
                        subprocess.CompletedProcess(command, 1, "", "missing CUDA dependency"))
    with pytest.raises(RuntimeError, match="missing CUDA dependency"):
        setup.probe_environment(Path(sys.executable), "0,1")


def test_real_venv_interpreter_survives_config_and_executable_resolution(tmp_path):
    folder = tmp_path / "isolated runtime"
    venv.EnvBuilder(with_pip=False, symlinks=os.name != "nt").create(folder)
    python = folder / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    config_path = tmp_path / "config/submission.local.json"
    put(config_path, {"schema": "video-highlight-submission-config/v1",
                      "python": setup.relative_value(python, config_path.parent)})
    configured = inputs.resolve_config(config_path)["python"]
    selected = inputs.executable(configured)
    assert selected == python.absolute()
    result = subprocess.run([str(selected), "-I", "-B", "-c", "import sys; print(sys.prefix)"],
                            check=True, text=True, capture_output=True)
    assert Path(result.stdout.strip()).resolve() == folder.resolve()


@pytest.mark.parametrize("existing_not_venv", [False, True])
def test_check_only_refuses_missing_or_invalid_venv_without_installing(example, monkeypatch, existing_not_venv):
    monkeypatch.setattr(setup, "REPO", example.root)
    put(example.root / "configs/submission.example.json", inputs.read_json(example.config))
    monkeypatch.setattr(setup, "checked_tool", lambda value: Path(sys.executable))
    venv = example.root / "isolated environment"
    if existing_not_venv:
        venv.mkdir()
    expected = ValueError if existing_not_venv else FileNotFoundError
    with pytest.raises(expected):
        setup.main(["--venv", str(venv), "--check-only"])
    assert not (example.root / "configs/submission.local.json").exists()
    assert not (example.root / ".runtime").exists()
