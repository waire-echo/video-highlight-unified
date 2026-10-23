#!/usr/bin/env python3
"""Build an offline runnable release with an explicit file allowlist, never test data."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import uuid

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from video_highlight.submission_inputs import digest, read_json, resolve_config, write_json


SCRIPTS = ("run_submission_pipeline.py", "run_v7e_stage2_submission.py",
           "infer_video_ora_highlights_adapter_426.py", "merge_video_ora_highlights_inference.py",
           "merge_video_ora_highlights_lenient_diagnostic.py", "package_submission.py",
           "setup_submission_env.py", "run_submission.sh")
SPATIAL_CODE = ("pipeline/__init__.py", "pipeline/common.py", "modules/__init__.py",
                "modules/stage2_b.py", "vendor/__init__.py", "vendor/yolo_center.py", "vendor/subject_association.py")
MODEL_EXTENSIONS = {".json", ".safetensors", ".model", ".txt", ".tiktoken", ".vocab", ".jinja", ".jinja2"}
PACKAGE_MODULES = ("__init__.py", "submission_inputs.py", "submission_recovery.py", "validation.py", "io_utils.py", "highlight_output.py", "timing.py")


def inside(folder: Path, relative: str) -> Path:
    value = (folder / relative).resolve()
    if not value.is_relative_to(folder.resolve()):
        raise ValueError(f"Package path escapes its root: {relative}")
    return value


def verify_package(folder: Path) -> dict:
    manifest = read_json(folder / "PACKAGE.json")
    if manifest.get("schema") != "video-highlight-release/v1":
        raise ValueError("Unsupported release manifest")
    for name, info in manifest["files"].items():
        path = inside(folder, name)
        if path.stat().st_size != info["bytes"] or digest(path) != info["sha256"]:
            raise ValueError(f"Release integrity check failed: {name}")
    return {"state": "VERIFIED", "files": len(manifest["files"]), "weights_included": True,
            "test_data_included": False, "gpu_validation": manifest["gpu_validation"]}


def build_package(config_path: Path, output: Path) -> dict:
    options = resolve_config(config_path)
    model = Path(options.get("model", REPO / "models/ora"))
    adapter = Path(options.get("adapter", REPO / "models/adapter"))
    spatial = Path(options.get("spatial_package", REPO / "models/spatial"))
    required = [model / "config.json", model / "tokenizer_config.json", adapter / "adapter_config.json", adapter / "adapter_model.safetensors",
                spatial / "configs/v7e_b_4109.json", spatial / "assets/yolo11m.pt"]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    if not list(model.glob("*.safetensors")):
        raise ValueError("Base model weights must be included; a code-only package is not runnable")
    if not any((model / name).is_file() for name in ("tokenizer.json", "tokenizer.model", "vocab.json")):
        raise ValueError("Base model tokenizer files are missing")
    if not any((model / name).is_file() for name in ("processor_config.json", "preprocessor_config.json")):
        raise ValueError("Base model video/image processor config is missing")
    for index in model.glob("*.safetensors.index.json"):
        for shard in set(read_json(index)["weight_map"].values()):
            if not inside(model, shard).is_file():
                raise FileNotFoundError(f"Missing base-model shard: {shard}")
    adapter_sha = digest(adapter / "adapter_model.safetensors")
    if options.get("expected_adapter_sha256") not in (None, adapter_sha):
        raise ValueError("Configured adapter fingerprint differs from the supplied weights")
    spatial_config = read_json(spatial / "configs/v7e_b_4109.json")
    yolo_asset = spatial_config["assets"]["yolo"]
    yolo_source = inside(spatial, yolo_asset["path"])
    if digest(yolo_source) != yolo_asset["sha256"]:
        raise ValueError("Spatial model weight fingerprint differs from its config")
    output = output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(f".{output.name}.building-{uuid.uuid4().hex}")
    staging.mkdir()

    def copy(source, relative):
        target = inside(staging, str(relative))
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)

    for name in SCRIPTS:
        copy(REPO / "scripts" / name, Path("scripts") / name)
    for name in PACKAGE_MODULES:
        copy(REPO / "src/video_highlight" / name, Path("src/video_highlight") / name)
    prompt = Path(options.get("prompt", REPO / "prompts/ora/whole-video-highlights-v1/prompt.txt"))
    copy(prompt, "prompts/ora/whole-video-highlights-v1/prompt.txt")
    for name in ("LICENSE", "requirements-submission.txt"):
        copy(REPO / name, name)
    if (REPO / "requirements-submission.lock.txt").is_file():
        copy(REPO / "requirements-submission.lock.txt", "requirements-submission.lock.txt")
    copy(REPO / "docs/submission_portability.md", "README.md")
    for path in (REPO / "licenses").rglob("*"):
        if path.is_file():
            copy(path, path.relative_to(REPO))
    for source, target_dir in ((model, "models/ora"), (adapter, "models/adapter")):
        for path in source.iterdir():
            if path.is_file() and path.suffix in MODEL_EXTENSIONS:
                copy(path, Path(target_dir) / path.name)
        for path in (source / "chat_templates").rglob("*"):
            if path.is_file() and path.suffix in {".jinja", ".jinja2", ".json"}:
                copy(path, Path(target_dir) / path.relative_to(source))
        for name in ("LICENSE", "LICENSE.md", "LICENSE.txt", "NOTICE"):
            if (source / name).is_file():
                copy(source / name, Path(target_dir) / name)
    for index in model.glob("*.safetensors.index.json"):
        for shard in set(read_json(index)["weight_map"].values()):
            copy(inside(model, shard), Path("models/ora") / shard)
    # This metadata is not a weight: remove the old server's base-model pathname.
    adapter_config = read_json(staging / "models/adapter/adapter_config.json")
    adapter_config["base_model_name_or_path"] = "../ora"
    write_json(staging / "models/adapter/adapter_config.json", adapter_config)
    for name in SPATIAL_CODE:
        path = spatial / name
        if path.is_file():
            copy(path, Path("models/spatial") / name)
        elif name not in {"pipeline/__init__.py", "modules/__init__.py", "vendor/__init__.py"}:
            raise FileNotFoundError(path)
    for name in ("LICENSE", "NOTICE"):
        if (spatial / name).is_file():
            copy(spatial / name, Path("models/spatial") / name)
    copy(yolo_source, Path("models/spatial") / yolo_asset["path"])
    # Only Stage 2 is called; never ship historical videos, targets or Stage 1 models.
    spatial_config = {key: spatial_config[key] for key in ("schema", "pipeline_version", "stage2")}
    spatial_config["stage2"].pop("historical_detector_profile", None)
    spatial_config["assets"] = {"yolo": yolo_asset}
    write_json(staging / "models/spatial/configs/v7e_b_4109.json", spatial_config)
    write_json(staging / "models/spatial/RELEASE_MANIFEST.json",
               {"stage": "stage2-only", "reference_answers_included": False})
    release_config = read_json(REPO / "configs/submission.example.json")
    for name in ("fps", "max_frames", "window_seconds", "window_stride", "video_max_token_num",
                 "video_min_token_num", "inference_max_tokens", "per_gpu_gib", "end_tolerance_sec",
                 "recovery_attempts", "window_timeout_sec", "command_timeout_sec", "spatial_video_timeout_sec"):
        if name in options:
            release_config[name] = options[name]
    release_config["expected_adapter_sha256"] = adapter_sha
    write_json(staging / "configs/submission.json", release_config)
    for name in ("input/video", "input/jsonl", "outputs"):
        (staging / name).mkdir(parents=True)
    files = {path.relative_to(staging).as_posix(): {"sha256": digest(path), "bytes": path.stat().st_size}
             for path in sorted(staging.rglob("*")) if path.is_file()}
    write_json(staging / "PACKAGE.json", {"schema": "video-highlight-release/v1", "files": files,
               "adapter_sha256": adapter_sha, "test_data_included": False, "gpu_validation": "pending_for_this_build"})
    result = verify_package(staging)
    os.replace(staging, output)
    return {**result, "output": str(output)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify:
        result = verify_package(args.verify.resolve())
    elif args.config and args.output:
        result = build_package(args.config, args.output)
    else:
        parser.error("Pass --verify PACKAGE or --config EXPERIMENT.json --output NEW_RELEASE_DIR")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
