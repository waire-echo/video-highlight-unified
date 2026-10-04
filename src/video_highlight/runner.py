"""Dispatch configured research pipelines without copying experiment code."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo

from . import experiments as exp
from .prompts import bundle_hashes

REPO = exp.REPO
PIPELINES = {
    "baseline_center": "baseline", "baseline_motion": "baseline", "qwen_direct": "frontend",
    "qwen_window": "frontend", "reze": "frontend", "pgl_sum": "frontend",
    "ora_direct": "ora", "spatial_crop": "spatial", "ora_proposal": "reference",
    "ora_reranker": "reference", "sam_tracking": "reference",
}
PATH_FIELDS = {'run_root', 'test_index', 'video_dir', 'python_executable', 'temporal_python',
               'temporal_model', 'yolo_model', 'pose_model', 'temporal_handoff', 'media_root',
               'pgl_checkpoint_path', 'pgl_feature_model_dir', 'pgl_hints_jsonl'}


def load_config(path):
    config = exp.read_json(path)
    if config.get("schema_version") != "pipeline-config/v1" or config.get("pipeline") not in PIPELINES:
        raise ValueError("Unknown pipeline or configuration schema")
    if not isinstance(config.get("dataset"), dict) or not config["dataset"].get("id"):
        raise ValueError("dataset.id is required")
    manifest = config["dataset"].get("manifest")
    if not manifest or not (REPO / manifest).is_file():
        raise ValueError("dataset.manifest must name a tracked dataset description")
    config["dataset"]["manifest_sha256"] = exp.sha256(REPO / manifest)
    bundle_hashes(config.get("prompts", {}))
    return config


def absolute(value):
    path = Path(value).expanduser()
    return str(path if path.is_absolute() else (REPO / path).resolve())


def commands(config, experiment_id, local=None, action="temporal"):
    kind = PIPELINES[config["pipeline"]]
    if kind == "reference":
        raise ValueError("This route has reference assets only; implementation is missing. See pipelines/" + config["pipeline"] + "/README.md")
    paths = {**config.get("paths", {}), **(local or {})}
    if set(paths) - PATH_FIELDS:
        raise ValueError('Only machine paths belong in --local/paths: ' + ', '.join(sorted(set(paths) - PATH_FIELDS)))
    paths.setdefault("run_root", "runs/" + experiment_id)
    paths.setdefault("test_index", "datasets/local/test_index.json")
    paths.setdefault("video_dir", "datasets/local/videos")
    root = Path(absolute(paths["run_root"]))
    python = paths.get("python_executable", sys.executable)
    params = dict(config.get("parameters", {}))
    generated = {}
    if kind in {"ora", "spatial"}:
        if kind == 'spatial' and action == 'temporal':
            action = 'spatial'
        resolved = {**paths, **config.get('runtime', {}), "model_revision": config.get("model", {}).get("revision", ""),
                    "served_model_name": config.get("model", {}).get("served_name", "Video-ORA-4B"),
                    "temporal_parameters": params if kind == 'ora' else {},
                    "spatial_parameters": config.get('spatial_parameters', {}), "run_root": str(root)}
        for field in ("video_dir", "test_index", "temporal_model", "yolo_model", "pose_model", "temporal_handoff", "media_root"):
            if resolved.get(field):
                resolved[field] = absolute(resolved[field])
        for field in ("temporal_python", "python_executable"):
            resolved[field] = paths.get(field, sys.executable)
        file = root / "pipeline.resolved.json"
        generated[file] = resolved
        return [[sys.executable, str(REPO / "pipeline.py"), "--config", str(file), action]], generated, paths
    params.update(index=absolute(paths["test_index"]), video_dir=absolute(paths["video_dir"]))
    if action != 'temporal':
        raise ValueError('This route emits temporal/baseline results; action must be temporal')
    if kind == "baseline":
        params.update(out=str(root / "predictions.jsonl"))
        script = "run_modified_baseline.py"
    else:
        params.update(out=str(root / "temporal_handoff.jsonl"), window_cache_root=str(root / "cache"),
                      window_log_root=str(root / "logs"))
        if paths.get("temporal_model"):
            params["temporal_model_path"] = absolute(paths["temporal_model"])
        if config["pipeline"] == "pgl_sum":
            for field in ("pgl_checkpoint_path", "pgl_feature_model_dir", "pgl_hints_jsonl"):
                if paths.get(field):
                    params[field] = absolute(paths[field])
            params["pgl_feature_cache_dir"] = str(root / "feature_cache")
        script = "run_temporal_frontend.py"
    file = root / "runner.resolved.json"
    generated[file] = params
    return [[python, str(REPO / "scripts" / script), "--config", str(file)]], generated, paths


def execute(config, experiment_id, *, local=None, dry_run=False, action="temporal"):
    if config['pipeline'] == 'spatial_crop' and action == 'temporal':
        action = 'spatial'
    command_list, generated, paths = commands(config, experiment_id, local, action)
    if dry_run:
        print(json.dumps({"commands": command_list, "resolved": {str(k): v for k,v in generated.items()}}, ensure_ascii=False, indent=2))
        return
    record, saved = exp.verify(experiment_id)
    if saved != config:
        raise ValueError("Run configuration differs from the registered experiment")
    env = os.environ.copy()
    env.update(PYTHONPATH=str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", ""),
               PYTHONUTF8="1", PYTHONIOENCODING="utf-8", VIDEO_HIGHLIGHT_PROMPT_ROOT=str(REPO / "prompts"),
               VIDEO_HIGHLIGHT_PROMPT_OVERRIDES=json.dumps(config.get("prompts", {})),
               VIDEO_HIGHLIGHT_PORTABLE_RUNTIME="1")
    root = Path(absolute(paths["run_root"]))
    env.update(VIDEO_HIGHLIGHT_CACHE_DIR=str(root / 'cache'), VIDEO_HIGHLIGHT_TEMP_DIR=str(root / 'tmp'),
               HF_HOME=str(root / 'cache/huggingface'), TORCH_HOME=str(root / 'cache/torch'))
    inputs = {}
    for field in ("test_index", "temporal_handoff", "yolo_model", "pose_model", "pgl_checkpoint_path"):
        if paths.get(field):
            source = Path(absolute(paths[field]))
            if not source.is_file():
                raise FileNotFoundError(source)
            inputs[field] = {"path":str(source), "sha256":exp.sha256(source)}
    index = exp.read_json(Path(absolute(paths['test_index'])))
    media = {}
    for item in index:
        name = str(item['video_id']) + '.mp4'
        source = Path(absolute(paths['video_dir'])) / name
        media[name] = exp.sha256(source)
    import hashlib
    inputs['video_manifest'] = {'videos':len(media), 'sha256':hashlib.sha256(
        json.dumps(media, sort_keys=True).encode()).hexdigest()}
    run_manifest = {"experiment_id":experiment_id, "git_commit":record["git_commit"], "inputs":inputs,
                    "prompt_hashes":record["prompt_hashes"], "action":action, "commands":command_list}
    previous = root / "run_manifest.json"
    if previous.exists():
        previous_state = exp.read_json(previous)
        for key in ("experiment_id", "git_commit", "inputs", "prompt_hashes"):
            if previous_state.get(key) != run_manifest[key]:
                raise ValueError("Run directory has different inputs/code; choose a new experiment and run_root")
    for path, content in generated.items():
        exp.write_json(path, content)
    exp.write_json(previous, run_manifest)
    exp.write_json(root / 'video_hashes.json', media)
    metadata = exp.experiment_path(experiment_id) / "experiment.json"
    record["status"] = "running"
    record['runtime_inputs'] = {key:{k:v for k,v in value.items() if k != 'path'} for key,value in inputs.items()}
    exp.write_json(metadata, record)
    exp.registry()
    try:
        for command in command_list:
            subprocess.run(command, check=True, cwd=REPO, env=env)
    except BaseException:
        record["status"] = "failed"
        raise
    else:
        record["status"] = "completed"
    finally:
        record["last_action"] = action
        record["updated_at"] = datetime.now(ZoneInfo("Asia/Hong_Kong")).isoformat()
        exp.write_json(metadata, record)
        exp.registry()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Unified video highlight experiment entry")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--config", type=Path)
    run.add_argument("--experiment", required=True)
    run.add_argument("--author")
    run.add_argument("--local", type=Path)
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--action", default="temporal", choices=("temporal", "all", "spatial", "prepare", "check", "validate", "status"))
    experiments = sub.add_parser("experiment")
    commands_sub = experiments.add_subparsers(dest="operation", required=True)
    commands_sub.add_parser("list")
    create = commands_sub.add_parser("create")
    create.add_argument("--id", required=True)
    create.add_argument("--author", required=True)
    create.add_argument("--config", required=True, type=Path)
    create.add_argument("--notes", default="")
    metrics = commands_sub.add_parser("metrics")
    metrics.add_argument("--id", required=True)
    metrics.add_argument("--file", type=Path, required=True)
    show = commands_sub.add_parser("show")
    show.add_argument("--id", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "experiment":
            if args.operation == "list":
                for row in exp.registry():
                    print(f"{row['experiment_id']:40} {row['pipeline']:17} {str(row['score']):7} {row['status']}")
            elif args.operation == "create":
                print(json.dumps(exp.create(args.id, args.author, load_config(args.config), notes=args.notes), ensure_ascii=False, indent=2))
            elif args.operation == "metrics":
                exp.record_metrics(args.id, exp.read_json(args.file))
            elif args.operation == "show":
                print(json.dumps(exp.read_json(exp.experiment_path(args.id) / "experiment.json"), ensure_ascii=False, indent=2))
        else:
            if args.config:
                config = load_config(args.config)
            else:
                config = exp.read_json(exp.experiment_path(args.experiment) / "config.json")
            if not args.dry_run and not (exp.experiment_path(args.experiment) / "experiment.json").exists():
                if not args.author:
                    raise ValueError("--author is required for a new experiment")
                exp.create(args.experiment, args.author, config)
            execute(config, args.experiment, local=exp.read_json(args.local) if args.local else None,
                    dry_run=args.dry_run, action=args.action)
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        print(str(error), file=sys.stderr)
        return 1
    return 0
