"""One resumable state machine for temporal selection, handoff, spatial inference and delivery."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request

from .contracts import RunState, fingerprint, read_json, run_lock, sha256, write_json
from .handoff import build_spatial_input, common, validate_result
from .registry import REPO, release, resolve, versions

SCENE = REPO / "pipelines/ora_spatial"


def make_plan(config, descriptor):
    paths = config["paths"]
    root = Path(paths["run_root"])
    source = root / "source/ora_spatial"
    spatial = root / "spatial"
    python = paths["client_python"]
    ids = ",".join(config["ids"])
    baseline_record = read_json(SCENE / "provenance/scene_general_v3_baseline_20261010.json")
    spatial_server = [str(Path(paths["runtime_root"]) / "env/bin/vllm"), "serve", paths["model_dir"],
                      *baseline_record["vllm_flags"]]
    stages = []
    if config["input_mode"] == "temporal":
        from video_highlight.cli import server_command
        temporal_config = {"temporal_model": paths["model_dir"], "temporal_python": paths["temporal_python"],
                           "video_dir": paths["video_dir"], "temporal_run_dir": str(root / "temporal"),
                           "media_root": os.path.commonpath([paths["video_dir"], str(root)]), "model_port": 8000}
        stages.append({"stage": "temporal", "server_command": server_command(temporal_config),
                       "command": [python, str(REPO / "pipeline.py"), "--config", str(root / "temporal.resolved.json"), "temporal"]})
        stages.append({"stage": "build-spatial-input", "entry": "exact original PTS/frame-mask adapter"})
    elif config["input_mode"] == "handoff":
        stages.append({"stage": "build-spatial-input", "entry": "exact original PTS/frame-mask adapter"})
    else:
        stages.append({"stage": "import-baseline", "entry": "read-only baseline references"})
    if descriptor["mode"] == "compat":
        stages.append({"stage": "verify-baseline", "command": [python, str(SCENE / "scripts/verify_scene_general_v3_baseline.py"),
                       "--root", str(source), "--model-dir", paths["model_dir"], "--check-environment"]})
    stages.append({"stage": "spatial", "server_command": spatial_server,
                   "preflight_command": [python, str(source / descriptor["preflight_entrypoint"]),
                                         "--ids", ids, "--video-dir", paths["video_dir"]],
                   "command": [python, "-m", "video_highlight.workflow.scene_entry", "--source-root", str(source),
                               "--entrypoint", descriptor["entrypoint"],
                               "--io-profile", descriptor["io_profile"], "--checkpoint-interval", str(descriptor["checkpoint_interval"]),
                               "--ids", ids, "--video-dir", paths["video_dir"], "--output", str(spatial)]})
    stages.extend([{"stage": "validate", "entry": "shared scene contracts + ZIP payload equality"},
                   {"stage": "deliver", "entry": "atomic export to final/ + SHA256 manifest"}])
    return {"schema_version": "workflow-plan/v1", "version": descriptor["id"], "mode": descriptor["mode"],
            "config": config, "release_sha256": fingerprint(descriptor), "stages": stages,
            "resources": {"spatial_port": 8008, "temporal_port": 8000, "max_num_seqs": 1,
                          "opencv_threads": 2, "io_profile": descriptor["io_profile"]},
            "outputs": {"final": str(root / "final"), "submission": str(root / "final/submission.zip"),
                        "spatial_evidence": str(spatial), "run": str(root)},
            "equivalence": descriptor["equivalence"], "gpu_validation": descriptor["gpu_validation"]}


def frozen_inputs(plan):
    config = plan["config"]
    paths = config["paths"]
    data = Path(paths["data_root"])
    source = REPO / release(config["version"])["source_root"]
    files = [data / "inputs/manifest.json", data / "inputs/timestamps.npz", data / "output/parameter_ledger.json"]
    if config["input_mode"] == "baseline":
        files.append(Path(paths["baseline"]))
    elif config["input_mode"] == "handoff":
        files.extend(Path(paths[k]) for k in ("temporal_handoff", "highlight_frames"))
    else:
        files.append(Path(paths["test_index"]))
    metadata = read_json(data / "inputs/manifest.json")
    verified = {}
    api = common(source)
    if config["input_mode"] == "baseline":
        baseline = api.load_rows(paths["baseline"])
        rows = [row for row in baseline if str(row["video_id"]) in set(config["ids"])]
        if [str(row["video_id"]) for row in rows] != config["ids"]:
            raise ValueError("Baseline IDs/order do not match requested videos")
        api.validate(rows, rows, metadata, preserve_width=False)
        if plan["mode"] == "compat" and sha256(paths["baseline"]) != metadata["baseline_sha256"]:
            raise ValueError("Compatibility mode requires the frozen baseline")
    for vid in config["ids"]:
        video = Path(paths["video_dir"]) / (vid + ".mp4")
        actual = sha256(video)
        if actual != metadata["videos"][vid]["source_sha256"]:
            raise ValueError("Original video changed: " + vid)
        verified[str(video)] = actual
        shot = data / "output/shots" / (vid + ".json")
        if read_json(shot)["source_sha256"] != metadata["videos"][vid]["source_sha256"]:
            raise ValueError("Shot source changed: " + vid)
        files.extend([video, shot])
    # Freeze every weight shard and model/tokenizer configuration, not just a directory name.
    model_record = read_json(source / "provenance/ora_model_provenance.json")
    model = Path(paths["model_dir"])
    for item in model_record["weight_files"]:
        file = model / item["filename"]
        if sha256(file) != item["sha256"] or file.stat().st_size != item["size_bytes"]:
            raise ValueError("Model weight mismatch: " + item["filename"])
    weights = {str(model / item["filename"]): item["sha256"] for item in model_record["weight_files"]}
    files.extend(file for file in model.iterdir() if file.is_file() and file.suffix in {".json", ".txt", ".jinja"})
    source_files = {}
    for folder in (REPO / "src/video_highlight", REPO / "prompts/ora/v5"):
        for file in sorted(folder.rglob("*")):
            if file.is_file() and "__pycache__" not in file.parts and file.suffix not in {".pyc", ".pyo"}:
                source_files[file.relative_to(REPO).as_posix()] = sha256(file)
    for file in (REPO / "pipeline.py", REPO / "vh.py", SCENE / "scripts/verify_scene_general_v3_baseline.py"):
        source_files[file.relative_to(REPO).as_posix()] = sha256(file)
    descriptor = release(config["version"])
    source_files.update({str(Path(descriptor["source_root"]) / k): v for k, v in descriptor["source_files"].items()})
    runtime = {}
    check = "import importlib.metadata as m,json,sys\npackages={}\nfor x in ['numpy','opencv-python-headless','scipy','requests','Pillow','torch','vllm','transformers']:\n try: packages[x]=m.version(x)\n except m.PackageNotFoundError: packages[x]=None\nprint(json.dumps({'python':sys.version,'packages':packages}))"
    for role in ("client_python", "temporal_python"):
        runtime[role] = json.loads(subprocess.check_output([paths[role], "-c", check], text=True, timeout=30))
    commit = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    return {"schema_version": "workflow-frozen/v1", "plan_sha256": fingerprint(plan),
            "source_files": source_files, "files": {str(file): verified[str(file)] if str(file) in verified else sha256(file) for file in files},
            "weights": weights, "runtime": runtime, "git_commit": commit}


def stop_process(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def environment(plan):
    paths = plan["config"]["paths"]
    result = os.environ.copy()
    result.update(PYTHONPATH=str(REPO / "src"), PYTHONUNBUFFERED="1", PYTHONUTF8="1", PYTHONIOENCODING="utf-8",
                  VIDEO_HIGHLIGHT_PROMPT_ROOT=str(REPO / "prompts"), VIDEO_HIGHLIGHT_PROMPT_OVERRIDES="{}",
                  VLLM_NO_USAGE_STATS="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                  VLLM_VIDEO_LOADER_BACKEND="opencv",
                  LD_LIBRARY_PATH=str(Path(paths["runtime_root"]) / "python311/lib") + ":" + result.get("LD_LIBRARY_PATH", ""))
    # Explicitly undo inherited automatic tuning / prompt overrides from other experiments.
    result.pop("VIDEO_HIGHLIGHT_PORTABLE_RUNTIME", None)
    return result


def run_command(command, state, env):
    root = state.root
    log = root / "logs" / (state.state["stage"] + ".log")
    log.parent.mkdir(parents=True, exist_ok=True)
    state.update(command=command, log=str(log))
    state.event("command_started", stage=state.state["stage"], command=command, log=str(log))
    print(json.dumps({"stage": state.state["stage"], "command": command, "log": str(log)}, ensure_ascii=False), flush=True)
    with log.open("ab", buffering=0) as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, cwd=REPO,
                                   env=env, start_new_session=True)
        try:
            previous = None
            temporal_records = {}
            while process.poll() is None:
                child = root / "spatial/status.json"
                if state.state["stage"] == "spatial" and child.is_file():
                    progress = read_json(child)
                    if progress != previous:
                        fields = {k: progress[k] for k in ("current_video", "segment", "sample", "completed_ids", "requested_ids") if k in progress}
                        state.update(**fields, child_stage=progress.get("stage"))
                        state.event("progress", **fields, child_stage=progress.get("stage"))
                        previous = progress
                elif state.state["stage"] == "temporal":
                    changed = False
                    for file in (root / "temporal/records").glob("*.json"):
                        stat = file.stat()
                        marker = (stat.st_mtime_ns, stat.st_size)
                        if file.stem not in temporal_records or temporal_records[file.stem][0] != marker:
                            temporal_records[file.stem] = (marker, read_json(file).get("status"))
                            changed = True
                    if changed:
                        fields = {"completed_ids": [vid for vid in state.state["requested_ids"] if temporal_records.get(vid, (None, None))[1] == "complete"],
                                  "failed_ids": [vid for vid in state.state["requested_ids"] if temporal_records.get(vid, (None, None))[1] == "error"]}
                        state.update(**fields)
                        state.event("progress", **fields)
                time.sleep(0.5)
            if process.returncode:
                raise subprocess.CalledProcessError(process.returncode, command)
        finally:
            stop_process(process)


@contextmanager
def managed_model(command, port, state, env):
    import fcntl
    # Serialize GPU ownership across workflow directories; do not kill existing services.
    with Path("/tmp/video-highlight-workflow-gpu.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("Another workflow owns the GPU") from error
        probe = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                               text=True, capture_output=True, timeout=15)
        if probe.returncode or not probe.stdout.strip():
            raise ValueError("No GPU is available; allocate a GPU before inference")
        with socket.socket() as check:
            check.bind(("127.0.0.1", port))
        log = state.root / "logs" / (state.state["stage"] + "-model.log")
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("ab", buffering=0) as stream:
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                       env=env, start_new_session=True, cwd=REPO)
            state.event("model_started", pid=process.pid, command=command, port=port)
            try:
                deadline = time.monotonic() + 900
                while True:
                    if process.poll() is not None:
                        raise RuntimeError("Model service exited; see " + str(log))
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Model startup timed out; see " + str(log))
                    try:
                        with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=2) as response:
                            if any(row["id"] == "Video-ORA-4B" for row in json.load(response)["data"]):
                                break
                    except (OSError, ValueError, KeyError):
                        pass
                    time.sleep(1)
                yield
            finally:
                stop_process(process)
                state.event("model_stopped", pid=process.pid)


def link(source, target):
    source, target = Path(source).resolve(), Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        if target.resolve() != source:
            raise ValueError("Input reference changed: " + str(target))
    elif target.exists():
        raise ValueError("Refusing to overwrite " + str(target))
    else:
        target.symlink_to(source, target_is_directory=source.is_dir())


def prepare_source(plan, descriptor):
    paths = plan["config"]["paths"]
    root = Path(paths["run_root"])
    source = root / "source/ora_spatial"
    original = REPO / descriptor["source_root"]
    for relative, expected in descriptor["source_files"].items():
        target = source / relative
        if target.exists():
            if sha256(target) != expected:
                raise ValueError("Sealed run source changed: " + relative)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original / relative, target)
            target.chmod(0o444)
    data = Path(paths["data_root"])
    link(data / "inputs/timestamps.npz", source / "inputs/timestamps.npz")
    link(data / "output/shots", source / "output/shots")
    link(data / "output/parameter_ledger.json", source / "output/parameter_ledger.json")
    link(original / "provenance", source / "provenance")
    return source


def temporal_files(root):
    return root / "temporal/export_grounded/temporal_handoff.jsonl", root / "temporal/export_grounded/highlight_frames.jsonl"


def stage_action(stage, plan, descriptor, state, env):
    root = state.root
    paths = plan["config"]["paths"]
    data = Path(paths["data_root"])
    source = root / "source/ora_spatial"
    mode = plan["config"]["input_mode"]
    if stage["stage"] == "temporal":
        index = read_json(paths["test_index"])
        if [str(row["video_id"]) for row in index] != plan["config"]["ids"]:
            raise ValueError("Temporal test index must contain requested IDs in original order")
        config = {"run_root": str(root), "video_dir": paths["video_dir"], "test_index": paths["test_index"],
                  "temporal_model": paths["model_dir"], "temporal_python": paths["temporal_python"],
                  "python_executable": paths["client_python"], "temporal_run_dir": str(root / "temporal"),
                  "model_revision": descriptor["model_revision"], "model_port": 8000,
                  "media_root": os.path.commonpath([paths["video_dir"], str(root)])}
        write_json(root / "temporal.resolved.json", config)
        # Reuse the maintained temporal implementation, with our owned service lifecycle.
        with managed_model(stage["server_command"], 8000, state, env):
            run_command(stage["command"], state, env)
        return [*temporal_files(root), root / "temporal/temporal.complete"]
    if stage["stage"] in {"import-baseline", "build-spatial-input"}:
        if mode == "baseline":
            link(paths["baseline"], source / "inputs/baseline.zip")
            if descriptor["mode"] == "compat":
                (root / "handoff").mkdir(parents=True, exist_ok=True)
                shutil.copyfile(data / "inputs/manifest.json", root / "handoff/manifest.json")
            else:
                manifest = read_json(data / "inputs/manifest.json")
                manifest["baseline_sha256"] = sha256(paths["baseline"])
                write_json(root / "handoff/manifest.json", manifest)
            link(root / "handoff/manifest.json", source / "inputs/manifest.json")
            return [root / "handoff/manifest.json"]
        handoff, highlights = temporal_files(root) if mode == "temporal" else (paths["temporal_handoff"], paths["highlight_frames"])
        build_spatial_input(handoff, highlights, data / "inputs/manifest.json", data / "inputs/timestamps.npz",
                            root / "handoff", plan["config"]["ids"], source)
        link(root / "handoff/baseline.zip", source / "inputs/baseline.zip")
        link(root / "handoff/manifest.json", source / "inputs/manifest.json")
        return [root / "handoff/baseline.zip", root / "handoff/manifest.json", root / "handoff/adapter.json"]
    if stage["stage"] == "verify-baseline":
        run_command(stage["command"], state, env)
        return []
    if stage["stage"] == "spatial":
        with managed_model(stage["server_command"], 8008, state, env):
            run_command(stage["preflight_command"], state, env)
            run_command(stage["command"], state, env)
        child = read_json(root / "spatial/status.json")
        if child.get("status") != "complete":
            raise ValueError("Spatial stage did not finish completely")
        state.update(completed_ids=child["completed_ids"], requested_ids=child["requested_ids"])
        return [root / "spatial" / name for name in ("SPATIAL_temporal_v3.jsonl", "SPATIAL_temporal_v3.zip", "SPATIAL_temporal_v3.validation.json", "report.json", "frozen.json", "results.zip")]
    if stage["stage"] == "validate":
        report = validate_result(source, source / "inputs/baseline.zip", source / "inputs/manifest.json",
                                 root / "spatial/SPATIAL_temporal_v3.jsonl", plan["config"]["ids"])
        write_json(root / "final/validation.json", report)
        return [root / "final/validation.json"]
    if stage["stage"] == "deliver":
        final = root / "final"
        outputs = [final / "validation.json"]
        for old, new in [("SPATIAL_temporal_v3.jsonl", "predictions_spatial.jsonl"),
                         ("SPATIAL_temporal_v3.zip", "submission.zip"), ("report.json", "spatial_report.json")]:
            target = final / new
            temporary = target.with_suffix(target.suffix + ".tmp")
            shutil.copyfile(root / "spatial" / old, temporary)
            os.replace(temporary, target)
            outputs.append(target)
        manifest = {"schema_version": "workflow-artifacts/v1", "version": plan["version"],
                    "submission": "submission.zip", "frames_preserved": True,
                    "artifacts": {file.name: {"sha256": sha256(file), "size_bytes": file.stat().st_size} for file in outputs},
                    "equivalence": plan["equivalence"], "gpu_validation": plan["gpu_validation"]}
        write_json(final / "artifacts.json", manifest)
        return [*outputs, final / "artifacts.json"]
    raise ValueError("Unknown stage: " + stage["stage"])


def execute(plan, *, resume=False, until="complete"):
    root = Path(plan["config"]["paths"]["run_root"])
    if not resume and root.exists() and any(root.iterdir()):
        raise ValueError("Run directory is not empty; use resume or a new run_root")
    with run_lock(root):
        if resume and read_json(root / "plan.json") != plan:
            raise ValueError("Code, version, configuration or inputs changed; check out original version or create a new run")
        state = RunState(root, plan)
        descriptor = release(plan["version"])
        env = environment(plan)
        previous_handler = signal.getsignal(signal.SIGTERM)
        def interrupted(signum, frame):
            raise KeyboardInterrupt("Received termination signal")
        signal.signal(signal.SIGTERM, interrupted)
        record_failure = not resume
        try:
            if not resume:
                state.update(stage="preflight")
                state.event("started", version=plan["version"])
                write_json(root / "plan.json", plan)
            frozen = frozen_inputs(plan)
            if (root / "frozen.json").exists():
                if read_json(root / "frozen.json") != frozen:
                    raise ValueError("Code, version, configuration or inputs changed; check out original version or create a new run")
            else:
                if any((root / "stages").glob("*.json")):
                    raise ValueError("Missing frozen record for completed stages")
                write_json(root / "frozen.json", frozen)
                write_json(root / "inputs.json", {"paths": plan["config"]["paths"], "files": frozen["files"], "weights": frozen["weights"]})
            record_failure = True
            if resume:
                state.event("resumed", version=plan["version"])
            state.update(stage="prepare-source")
            prepare_source(plan, descriptor)
            for stage in plan["stages"]:
                receipt = root / "stages" / (stage["stage"] + ".json")
                state.update(stage=stage["stage"], current_video=None, segment=None, error=None)
                if receipt.is_file():
                    saved = read_json(receipt)
                    if saved.get("frozen_sha256") != fingerprint(frozen) or any(sha256(root / name) != expected for name, expected in saved["outputs"].items()):
                        raise ValueError("Completed stage artifacts changed: " + stage["stage"])
                    state.event("stage_reused", stage=stage["stage"])
                else:
                    outputs = stage_action(stage, plan, descriptor, state, env)
                    write_json(receipt, {"stage": stage["stage"], "frozen_sha256": fingerprint(frozen),
                                         "outputs": {file.relative_to(root).as_posix(): sha256(file) for file in outputs}})
                    state.event("stage_completed", stage=stage["stage"])
                if stage["stage"] == until or (until == "handoff" and stage["stage"] in {"import-baseline", "build-spatial-input"}):
                    state.update(status="stage_complete", stage=stage["stage"])
                    return read_json(root / "status.json")
            state.update(status="complete", stage="complete", final=plan["outputs"]["final"])
            state.event("completed", final=plan["outputs"]["final"])
        except BaseException as error:
            if record_failure:
                state.failure(error)
            raise
        finally:
            signal.signal(signal.SIGTERM, previous_handler)
    return read_json(root / "status.json")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Versioned temporal → handoff → scene-v3 → validation workflow")
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("versions", help="List explicit supported releases")
    for action in ("plan", "run"):
        command = sub.add_parser(action)
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--local", type=Path)
        if action == "run":
            command.add_argument("--until", choices=("temporal", "handoff", "complete"), default="complete")
    for action in ("resume", "status"):
        command = sub.add_parser(action)
        command.add_argument("--run-root", type=Path, required=True)
        if action == "resume":
            command.add_argument("--until", choices=("temporal", "handoff", "complete"), default="complete")
    validate = sub.add_parser("validate", help="Read-only validation of historical or current results")
    validate.add_argument("--data-root", type=Path, required=True)
    validate.add_argument("--prediction", type=Path, required=True)
    io = sub.add_parser("verify-io", help="Compare original and optimized decoded pixels on real videos; no GPU required")
    io.add_argument("--data-root", type=Path, required=True)
    io.add_argument("--video-dir", type=Path, required=True)
    io.add_argument("--ids", required=True)
    io.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.action == "versions":
            result = [{k: row[k] for k in ("id", "mode", "input_modes", "io_profile", "gpu_validation")} for row in versions()]
        elif args.action in {"plan", "run"}:
            config, descriptor = resolve(args.config, args.local)
            plan = make_plan(config, descriptor)
            if args.action == "run" and args.until == "temporal" and config["input_mode"] != "temporal":
                raise ValueError("--until temporal requires temporal input mode")
            result = plan if args.action == "plan" else execute(plan, until=args.until)
        elif args.action == "resume":
            previous = read_json(args.run_root / "plan.json")
            if Path(previous["config"]["paths"]["run_root"]).resolve() != args.run_root.resolve():
                raise ValueError("Resume path differs from the saved run directory")
            if args.until == "temporal" and previous["config"]["input_mode"] != "temporal":
                raise ValueError("--until temporal requires temporal input mode")
            result = execute(make_plan(previous["config"], release(previous["version"])), resume=True, until=args.until)
        elif args.action == "status":
            result = read_json(args.run_root / "status.json")
            child = args.run_root / "spatial/status.json"
            if child.is_file():
                result["spatial"] = read_json(child)
        elif args.action == "verify-io":
            from .verification import verify_io
            result = verify_io(SCENE, args.data_root, args.video_dir, args.ids.split(","), args.output)
            result = {key: value for key, value in result.items() if key != "cases"}
        else:
            manifest = args.data_root / "inputs/manifest.json"
            result = validate_result(SCENE, args.data_root / "inputs/baseline.zip", manifest,
                                     args.prediction, list(read_json(manifest)["videos"]))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as error:
        print("ERROR: " + str(error), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted; resume with the original run directory", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
