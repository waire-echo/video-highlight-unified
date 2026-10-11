"""Release manifests select code and resource policy without rewriting a baseline."""
from __future__ import annotations

import os
from pathlib import Path
import re
import sys

from .contracts import read_json, sha256

REPO = Path(__file__).resolve().parents[3]
VERSION_ROOT = REPO / "configs/workflows/versions"
PATHS = {"data_root", "video_dir", "runtime_root", "model_dir", "client_python", "temporal_python",
         "run_root", "baseline", "temporal_handoff", "highlight_frames", "test_index"}
INPUT_MODES = {"baseline", "handoff", "temporal"}


def versions():
    return [read_json(path) for path in sorted(VERSION_ROOT.glob("*.json"))]


def release(name):
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]+", name):
        raise ValueError("Invalid version name")
    path = VERSION_ROOT / (name + ".json")
    if not path.is_file():
        raise ValueError("Unknown version; use 'vh.py workflow versions'")
    value = read_json(path)
    if value.get("schema_version") != "workflow-release/v1" or value.get("id") != name:
        raise ValueError("Invalid release manifest")
    source = (REPO / value["source_root"]).resolve()
    if not source.is_relative_to(REPO):
        raise ValueError("Release source_root must be inside the repository")
    sealed_files = {**value["source_files"], **value["provenance_files"]}
    for relative, expected in sealed_files.items():
        file = (source / relative).resolve()
        if not file.is_relative_to(source) or sha256(file) != expected:
            raise ValueError("Release source changed: " + relative + "; register a new version")
    for key in ("entrypoint", "preflight_entrypoint"):
        if value.get(key) not in value["source_files"]:
            raise ValueError("Release entry points must be included in its source fingerprint")
    return value


def resolve(config_path, local_path=None):
    config_path = Path(config_path).expanduser().resolve()
    value = read_json(config_path)
    if set(value) - {"schema_version", "version", "input_mode", "ids", "paths"}:
        raise ValueError("Unknown workflow configuration fields")
    if value.get("schema_version") != "scene-workflow/v1":
        raise ValueError("Expected scene-workflow/v1 configuration")
    descriptor = release(value["version"])
    if value.get("input_mode") not in INPUT_MODES or value["input_mode"] not in descriptor["input_modes"]:
        raise ValueError("This version does not support the selected input mode")
    paths = {}
    for data, origin in [(value.get("paths", {}), config_path.parent),
                         (read_json(local_path) if local_path else {}, Path(local_path).resolve().parent if local_path else config_path.parent)]:
        if not isinstance(data, dict) or set(data) - PATHS:
            raise ValueError("Machine configuration must contain only documented paths")
        for key, item in data.items():
            if not isinstance(item, str) or not item.strip():
                raise ValueError("Empty path: " + key)
            path = Path(item).expanduser()
            # Preserve interpreter symlinks: resolving a venv python loses its environment.
            paths[key] = os.path.abspath(path if path.is_absolute() else origin / path)
    for key in ("data_root", "video_dir", "runtime_root", "run_root"):
        if key not in paths:
            raise ValueError("Missing machine path: " + key)
    paths.setdefault("model_dir", str(Path(paths["runtime_root"]) / "models/Video-ORA-4B"))
    paths.setdefault("client_python", sys.executable)
    paths.setdefault("temporal_python", str(Path(paths["runtime_root"]) / "env/bin/python"))
    paths.setdefault("baseline", str(Path(paths["data_root"]) / "inputs/baseline.zip"))
    if value["input_mode"] == "handoff" and not {"temporal_handoff", "highlight_frames"} <= paths.keys():
        raise ValueError("handoff mode requires temporal_handoff and highlight_frames")
    if value["input_mode"] == "temporal" and "test_index" not in paths:
        raise ValueError("temporal mode requires test_index")
    requested = value.get("ids", "all")
    manifest = read_json(Path(paths["data_root"]) / "inputs/manifest.json")
    ordered = list(manifest["videos"])
    if any(not re.fullmatch(r"[A-Za-z0-9_-]+", vid) for vid in ordered):
        raise ValueError("Manifest contains unsafe video IDs")
    if requested == "all":
        ids = ordered
    elif isinstance(requested, list) and requested and all(isinstance(x, str) for x in requested):
        ids = requested
    else:
        raise ValueError("ids must be 'all' or a nonempty list of video ID strings")
    if len(ids) != len(set(ids)) or any(x not in manifest["videos"] for x in ids):
        raise ValueError("Duplicate or unknown video IDs")
    if ids != [x for x in ordered if x in set(ids)]:
        raise ValueError("IDs must retain the manifest's original order")
    root = Path(paths["run_root"]).resolve()
    protected = [Path(paths[k]).resolve() for k in ("data_root", "video_dir", "runtime_root", "model_dir")]
    protected.append(REPO / "pipelines/ora_spatial")
    if any(root == p or root in p.parents or root.is_relative_to(p) for p in protected):
        raise ValueError("run_root must be separate from code, baseline data, videos and runtime")
    return {"schema_version": value["schema_version"], "version": descriptor["id"],
            "input_mode": value["input_mode"], "ids": ids, "paths": paths}, descriptor


def default_version():
    """One declared default for both interfaces; experiments remain opt-in."""
    policy = read_json(REPO / "configs/project_policy.json")
    if policy.get("schema_version") != "video-highlight-project/v1":
        raise ValueError("Invalid project policy")
    name = policy["default_version"]
    descriptor = release(name)
    if descriptor["mode"] != "compat" or descriptor["io_profile"] != "original":
        raise ValueError("Default UI version must preserve frozen compatibility")
    return name
