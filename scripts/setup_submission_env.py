#!/usr/bin/env python3
"""Install and verify the isolated submission runtime; never start inference."""
from __future__ import annotations

import argparse
import json
import os
import platform
from pathlib import Path
import shutil
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from video_highlight.submission_inputs import digest, read_json, resolve_config, write_json


def relative_value(path: Path, config_dir: Path) -> str:
    invocation = Path(os.path.abspath(path.expanduser()))
    try:
        return Path(os.path.relpath(invocation, config_dir.resolve())).as_posix()
    except ValueError:  # Different Windows drives: personal config only.
        return str(invocation)


def checked_tool(value: str) -> Path:
    found = shutil.which(value)
    if not found:
        raise FileNotFoundError(f"Install this media tool or pass its path explicitly: {value}")
    subprocess.run([found, "-version"], check=True, stdout=subprocess.DEVNULL, timeout=20)
    return Path(found).resolve()


def validate_assets(model: Path, adapter: Path, spatial: Path) -> str:
    for path in (model / "config.json", adapter / "adapter_config.json",
                 adapter / "adapter_model.safetensors", spatial / "configs/v7e_b_4109.json",
                 spatial / "modules/stage2_b.py", spatial / "pipeline/common.py"):
        if not path.is_file():
            raise FileNotFoundError(f"Supply local model/Adapter/Stage-2 assets: {path}")
    if not list(model.glob("*.safetensors")):
        raise ValueError("Base-model safetensors are missing")
    asset = read_json(spatial / "configs/v7e_b_4109.json")["assets"]["yolo"]
    from package_submission import inside
    if digest(inside(spatial, asset["path"])) != asset["sha256"]:
        raise ValueError("Stage-2 detector weights differ from the supplied manifest")
    return digest(adapter / "adapter_model.safetensors")


def probe_environment(python: Path, gpus: str) -> dict:
    # -I excludes user-site and development PYTHONPATH from verification.
    code = '''
import importlib.metadata as metadata, json
import torch, torchvision, transformers, peft, decord, av, cv2, accelerate, einops
from swift.infer_engine import RequestConfig, TransformersEngine
from swift.model import get_model_processor
from swift.template import get_template
from ultralytics import YOLO
assert torch.cuda.is_available(), "A usable NVIDIA CUDA GPU is required"
names = ("torch", "torchvision", "ms-swift", "transformers", "peft", "tokenizers",
         "qwen-vl-utils", "decord", "av", "accelerate", "einops", "ultralytics", "numpy")
print(json.dumps({"versions": {n: metadata.version(n) for n in names},
                  "cuda": torch.version.cuda, "gpu_count": torch.cuda.device_count(),
                  "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}))
'''
    environment = {key: value for key, value in os.environ.items()
                   if key not in ("PYTHONPATH", "PYTHONHOME")}
    environment.update(CUDA_VISIBLE_DEVICES=gpus, YOLO_AUTOINSTALL="false", HF_HUB_OFFLINE="1",
                       TRANSFORMERS_OFFLINE="1", YOLO_CONFIG_DIR=str(python.parent.parent / ".runtime/ultralytics"),
                       MPLCONFIGDIR=str(python.parent.parent / ".runtime/matplotlib"))
    result = subprocess.run([str(python), "-I", "-B", "-c", code], env=environment,
                            text=True, capture_output=True, timeout=180)
    if result.returncode:
        raise RuntimeError("Isolated environment probe failed:\n" + result.stdout + result.stderr)
    return json.loads(result.stdout.splitlines()[-1])


def make_config(template: dict, config_dir: Path, *, python: Path, model: Path,
                adapter: Path, spatial: Path, ffmpeg: Path, ffprobe: Path,
                adapter_sha: str, inference_gpus: str, spatial_gpu: str) -> dict:
    config = dict(template)
    config.update({name: relative_value(path, config_dir) for name, path in {
        "python": python, "spatial_python": python, "model": model, "adapter": adapter,
        "spatial_package": spatial, "ffmpeg": ffmpeg, "ffprobe": ffprobe}.items()})
    config.update(expected_adapter_sha256=adapter_sha, inference_gpus=inference_gpus, spatial_gpu=spatial_gpu)
    for name in ("recovery_policy", "unresolved_policy", "spatial_failure_policy"):
        config.pop(name, None)
    return config


def install_cuda_runtime(python: Path, index_url: str) -> None:
    """Use the GPU wheel's exact runtime requirements, not a second CUDA guess."""
    code = ("import importlib.metadata as m,json; print(json.dumps([r.split(';')[0].strip() "
            "for r in m.requires('torch') or [] if r.split(';')[0].strip().lower().startswith(('nvidia-','triton=='))]))")
    requirements = json.loads(subprocess.check_output([str(python), "-I", "-B", "-c", code], text=True))
    if requirements:
        subprocess.run([str(python), "-m", "pip", "install", "--no-deps", *requirements,
                        "--index-url", index_url], check=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, default=Path(sys.executable), help="Python 3.11 used to create the venv")
    parser.add_argument("--venv", type=Path, default=REPO / ".venv")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--spatial-package", type=Path)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--torch-index-url", default="https://download.pytorch.org/whl/cu128")
    parser.add_argument("--requirements", type=Path, help="Override the verified Linux lockfile; requires fresh validation")
    parser.add_argument("--inference-gpus", default="0,1")
    parser.add_argument("--spatial-gpu", default="0")
    parser.add_argument("--with-tests", action="store_true")
    parser.add_argument("--skip-install", action="store_true", help="Reconfigure/recheck an already installed isolated environment")
    parser.add_argument("--check-only", action="store_true", help="Verify without changing config or installing packages")
    args = parser.parse_args(argv)
    requirements = args.requirements or REPO / "requirements-submission.lock.txt"
    if args.requirements is None and not requirements.is_file():
        requirements = REPO / "requirements-submission.txt"
    venv = args.venv.expanduser().resolve()
    python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    config_path = REPO / "configs/submission.local.json"
    template_path = REPO / "configs/submission.json"
    if not template_path.exists():
        template_path = REPO / "configs/submission.example.json"
    source_config = config_path if config_path.exists() else template_path
    current = resolve_config(source_config)
    model = (args.model or current.get("model", REPO / "models/ora")).resolve()
    adapter = (args.adapter or current.get("adapter", REPO / "models/adapter")).resolve()
    spatial = (args.spatial_package or current.get("spatial_package", REPO / "models/spatial")).resolve()
    ffmpeg = checked_tool(str(current.get("ffmpeg", args.ffmpeg)) if args.ffmpeg == "ffmpeg" else args.ffmpeg)
    ffprobe = checked_tool(str(current.get("ffprobe", args.ffprobe)) if args.ffprobe == "ffprobe" else args.ffprobe)
    adapter_sha = validate_assets(model, adapter, spatial)
    if venv.exists() and not (venv / "pyvenv.cfg").is_file():
        raise ValueError("Refusing to overwrite a directory that is not a virtual environment")
    if args.skip_install or args.check_only:
        if not python.is_file():
            raise FileNotFoundError(f"Install the isolated environment first: {python}")
    else:
        subprocess.run([str(args.python), "-c", "import sys; assert sys.version_info[:2] == (3,11), 'Python 3.11 required'"], check=True)
        if not venv.exists():
            subprocess.run([str(args.python), "-m", "venv", str(venv)], check=True)
        subprocess.run([str(python), "-m", "pip", "install", "--upgrade", "pip"], check=True)
        subprocess.run([str(python), "-m", "pip", "install", "torch==2.10.0", "torchvision==0.25.0",
                        "--no-deps", "--index-url", args.torch_index_url], check=True)
        if sys.platform == "linux" and platform.machine().lower() in ("x86_64", "amd64"):
            install_cuda_runtime(python, args.torch_index_url)
        subprocess.run([str(python), "-m", "pip", "install", "-r", str(requirements)], check=True)
        if args.with_tests:
            subprocess.run([str(python), "-m", "pip", "install", "pytest>=8"], check=True)
    subprocess.run([str(python), "-m", "pip", "check"], check=True)
    probe = probe_environment(python, args.inference_gpus)
    if probe["gpu_count"] != len(args.inference_gpus.split(",")):
        raise RuntimeError("Requested CUDA GPU count does not match the isolated environment")
    if not args.check_only:
        config = make_config(read_json(template_path), config_path.parent, python=python, model=model,
                             adapter=adapter, spatial=spatial, ffmpeg=ffmpeg, ffprobe=ffprobe,
                             adapter_sha=adapter_sha, inference_gpus=args.inference_gpus, spatial_gpu=args.spatial_gpu)
        write_json(config_path, config)
        frozen = subprocess.run([str(python), "-m", "pip", "freeze", "--all"],
                                check=True, text=True, capture_output=True).stdout
        write_json(REPO / ".runtime/environment.json", {
            "schema": "video-highlight-submission-environment/v1", "environment_probe": probe,
            "requirements_sha256": digest(REPO / "requirements-submission.txt"),
            "installed_requirements_sha256": digest(requirements),
            "installed_requirements_file": requirements.name,
            "config": str(config_path), "adapter_sha256": adapter_sha,
            "pip_freeze": frozen.splitlines(), "gpu_pipeline_validation": "pending"})
        for folder in ("input/video", "input/jsonl", "outputs"):
            (REPO / folder).mkdir(parents=True, exist_ok=True)
    print(json.dumps({"state": "ENVIRONMENT_VERIFIED", "config": str(config_path),
                      "venv": str(venv), **probe, "inference_started": False}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
