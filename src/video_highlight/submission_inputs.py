"""Prepare video inputs and timing metadata for inference."""
from __future__ import annotations

from bisect import bisect_left
from contextlib import contextmanager
from decimal import Decimal
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import uuid

from .submission_recovery import RUNTIME_POLICIES, runtime_policy


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def identity(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def sort_id(value):
    value = str(value)
    return (0, int(value), value) if value.isdecimal() else (1, 0, value)


def index_rows(path: Path) -> list[dict]:
    if path.is_dir():
        result = []
        for file in sorted(path.glob("*.jsonl")):
            if file.name.startswith("._"):
                continue
            rows = [json.loads(line) for line in file.read_text(encoding="utf-8-sig").splitlines()
                    if line.strip()]
            if len(rows) != 1:
                raise ValueError(f"Expected one input configuration: {file}")
            row = dict(rows[0])
            row.setdefault("video_id", file.stem)
            if str(row["video_id"]) != file.stem:
                raise ValueError(f"Input filename/video_id mismatch: {file}")
            result.append(row)
    else:
        body = path.read_text(encoding="utf-8-sig").strip()
        result = json.loads(body) if body.startswith("[") else [
            json.loads(line) for line in body.splitlines() if line.strip()]
    if not isinstance(result, list) or not result:
        raise ValueError("Input index must contain at least one video")
    normalized, seen = [], set()
    for row in result:
        video_id = str(row["video_id"])
        if not re.fullmatch(r"[A-Za-z0-9_-]+", video_id) or video_id in seen:
            raise ValueError(f"Invalid or duplicate video_id: {video_id!r}")
        ratio = row.get("targetRatioWH")
        if (not isinstance(ratio, list) or len(ratio) != 2
                or any(type(item) is not int or item <= 0 for item in ratio)):
            raise ValueError(f"Missing/invalid targetRatioWH: {video_id}")
        seen.add(video_id)
        normalized.append({"video_id": video_id, "targetRatioWH": ratio})
    return sorted(normalized, key=lambda row: sort_id(row["video_id"]))


def discover_input(root: Path) -> tuple[Path, Path]:
    root = root.resolve()
    video_dirs = [root / name for name in ("video", "videos") if (root / name).is_dir()]
    indexes = [root / name for name in ("test_index.json", "test_index.jsonl", "jsonl")
               if (root / name).exists()]
    if len(video_dirs) != 1 or len(indexes) != 1:
        raise ValueError("--input-root must contain exactly one video/ or videos/ and "
                         "one jsonl/ or test_index.json[ l ]; otherwise pass --videos and --index")
    return video_dirs[0], indexes[0]


PATH_OPTIONS = {"model", "adapter", "materialized", "videos", "index", "spatial_package",
                "pipeline_dir", "input_root", "prompt", "seed_strict_merged", "existing_run_root"}
EXEC_OPTIONS = {"python", "spatial_python", "ffmpeg", "ffprobe"}
VALUE_OPTIONS = {"expected_adapter_sha256", "expected_model_revision", "inference_gpus",
                 "spatial_gpu", "fps", "max_frames", "video_max_token_num", "video_min_token_num",
                 "inference_max_tokens", "per_gpu_gib", "end_tolerance_sec", "spatial_arm",
                 "window_seconds", "window_stride", "unresolved_policy", "recovery_policy",
                 "recovery_attempts", "command_timeout_sec", "window_timeout_sec",
                 "spatial_video_timeout_sec", "spatial_failure_policy"}


def resolve_config(path: Path) -> dict:
    config = read_json(path)
    if not isinstance(config, dict) or set(config) - PATH_OPTIONS - EXEC_OPTIONS - VALUE_OPTIONS - {"schema"}:
        raise ValueError("Unknown pipeline configuration key(s)")
    if config.get("schema") != "video-highlight-submission-config/v1":
        raise ValueError("Unsupported submission config schema")
    result = {key: value for key, value in config.items() if key != "schema"}
    for key in PATH_OPTIONS & result.keys():
        if result[key] is not None:
            value = Path(result[key]).expanduser()
            result[key] = (path.resolve().parent / value).resolve()
    for key in EXEC_OPTIONS & result.keys():
        value = result[key]
        if value and ("/" in value or "\\" in value):
            # Dereferencing venv/bin/python selects the base interpreter and
            # loses the virtual environment. Preserve its invocation pathname.
            result[key] = os.path.abspath(path.resolve().parent / Path(value).expanduser())
    for key in RUNTIME_POLICIES.keys() & result.keys():
        result[key] = runtime_policy(key, result[key])
    return result


def executable(value) -> Path:
    found = shutil.which(str(value))
    if not found:
        raise FileNotFoundError(f"Executable not available: {value}")
    return Path(os.path.abspath(found))


@contextmanager
def run_lock(folder: Path):
    """OS-owned lock: released on crashes; never steal another running process's work."""
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / ".pipeline.lock").open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError(f"Another process owns this run: {folder}") from error
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def probe_video(path: Path, ffprobe: Path) -> dict:
    # Use a separate unfiltered stream probe: old FFprobe builds reject the
    # stream_side_data selector, and frame-only filters on newer builds can
    # omit display matrices even with -show_streams. Never lose rotation data.
    raw = json.loads(subprocess.check_output([
        str(ffprobe), "-v", "error", "-select_streams", "v:0", "-show_streams",
        "-of", "json", str(path)], text=True, encoding="utf-8", timeout=900))
    stream = raw["streams"][0]
    rotations = [float(stream.get("tags", {}).get("rotate", 0)),
                 *[float(data.get("rotation", 0)) for data in stream.get("side_data_list", [])]]
    if any(not math.isfinite(value) or value % 360 != 0 for value in rotations):
        raise ValueError(f"Rotated source requires a verified original-coordinate mapping; refusing to guess: {path}")
    frame_data = json.loads(subprocess.check_output([
        str(ffprobe), "-v", "error", "-select_streams", "v:0", "-show_frames",
        "-show_entries", "frame=best_effort_timestamp_time,pkt_duration_time",
        "-of", "json", str(path)], text=True, encoding="utf-8", timeout=900))
    frames = frame_data.get("frames", [])
    if not frames or any("best_effort_timestamp_time" not in frame for frame in frames):
        raise ValueError(f"Video has missing decoded frame timestamps: {path}")
    origin = Decimal(frames[0]["best_effort_timestamp_time"])
    times = [float(Decimal(frame["best_effort_timestamp_time"]) - origin) for frame in frames]
    if (not all(math.isfinite(value) for value in times)
            or any(b <= a for a, b in zip(times, times[1:]))):
        raise ValueError(f"Video timestamps are not strictly increasing: {path}")
    period = float(frames[-1].get("pkt_duration_time", 0) or 0)
    if period <= 0:
        period = times[-1] - times[-2] if len(times) > 1 else 1 / 25
    duration = max(times[-1] + period, float(stream.get("duration", 0) or 0))
    width, height = int(stream["width"]), int(stream["height"])
    if not math.isfinite(duration) or duration <= 0 or min(width, height) <= 0:
        raise ValueError(f"Invalid media metadata: {path}")
    return {"source_size": [width, height], "frame_count": len(times), "duration_sec": duration,
            "pts_sec": times, "source_pts_origin_sec": float(origin)}


def window_spans(duration: float, size: float, stride: float) -> list[tuple[float, float]]:
    if not (math.isfinite(size) and math.isfinite(stride) and 0 < stride <= size):
        raise ValueError("Require 0 < window_stride <= window_seconds")
    spans, start = [], 0.0
    while True:
        end = min(duration, start + size)
        spans.append((start, end))
        if end >= duration:
            return spans
        start += stride


def resized_shape(width: int, height: int, minimum: int, maximum: int) -> tuple[int, int]:
    """32-pixel grid and the same area budget used by the SFT input profile."""
    if not 0 < minimum <= maximum:
        raise ValueError("Invalid visual token budget")
    factor = 32
    new_w, new_h = max(factor, round(width / factor) * factor), max(factor, round(height / factor) * factor)
    if new_w * new_h > maximum * factor * factor:
        scale = math.sqrt(width * height / (maximum * factor * factor))
        new_w, new_h = max(factor, math.floor(width / scale / factor) * factor), max(factor, math.floor(height / scale / factor) * factor)
    elif new_w * new_h < minimum * factor * factor:
        scale = math.sqrt(minimum * factor * factor / (width * height))
        new_w, new_h = math.ceil(width * scale / factor) * factor, math.ceil(height * scale / factor) * factor
    return new_w, new_h


def prepare_inputs(args) -> Path:
    rows = index_rows(args.index)
    paths = [path for path in args.videos.iterdir()
             if path.is_file() and path.suffix.lower() == ".mp4" and not path.name.startswith("._")]
    actual = {path.stem: path for path in paths}
    if len(actual) != len(paths) or any(path.suffix != ".mp4" for path in paths):
        raise ValueError("Use unique filenames with lowercase .mp4 extensions")
    if set(actual) != {row["video_id"] for row in rows}:
        raise ValueError("Video IDs and supplied input configurations differ")
    sources = {row["video_id"]: digest(actual[row["video_id"]]) for row in rows}
    contract = {"index": rows, "source_sha256": sources, "fps": args.fps,
                "window_seconds": args.window_seconds, "window_stride": args.window_stride,
                "max_frames": args.max_frames, "min_tokens": args.video_min_token_num,
                "max_tokens": args.video_max_token_num, "prompt_sha256": digest(args.prompt),
                "preparer_sha256": digest(Path(__file__)), "ffmpeg": str(args.ffmpeg),
                "ffmpeg_version": subprocess.check_output([str(args.ffmpeg), "-version"], text=True, timeout=15).splitlines()[0]}
    output = args.pipeline_dir / "inputs" / identity(contract)[:16]
    with run_lock(output):
        manifest_path = output / "manifest.json"
        if manifest_path.is_file():
            manifest = read_json(manifest_path)
            if manifest.get("input_contract") != contract:
                raise ValueError("Prepared input identity changed")
            for name, info in manifest["prepared_files"].items():
                if digest(output / name) != info:
                    raise ValueError(f"Prepared input was modified: {name}")
            return output
        write_json(output / "PREPARING.json", contract)
        metadata, samples, requests, files = [], [], [], {}
        prompt = args.prompt.read_text(encoding="utf-8-sig").strip()
        for row in rows:
            video_id = row["video_id"]
            source = actual[video_id]
            meta_path = output / "media" / f"{video_id}.json"
            if meta_path.is_file():
                meta = read_json(meta_path)
                if meta.get("source_sha256") != sources[video_id]:
                    raise ValueError(f"Source changed during preparation: {video_id}")
            else:
                meta = {**probe_video(source, args.ffprobe), "video_id": video_id,
                        "source_sha256": sources[video_id], "target_ratio": row["targetRatioWH"]}
                write_json(meta_path, meta)
            metadata.append(meta)
            width, height = resized_shape(*meta["source_size"], args.video_min_token_num, args.video_max_token_num)
            for number, (start, end) in enumerate(window_spans(meta["duration_sec"], args.window_seconds, args.window_stride)):
                clip = output / "clips" / f"{video_id}_w{number:04d}.mp4"
                receipt = clip.with_suffix(".json")
                if clip.is_file() and receipt.is_file():
                    if read_json(receipt)["sha256"] != digest(clip):
                        raise ValueError(f"Prepared clip changed: {clip}")
                else:
                    clip.parent.mkdir(parents=True, exist_ok=True)
                    partial = clip.with_name(f".{clip.stem}.{uuid.uuid4().hex}.mp4")
                    subprocess.run([str(args.ffmpeg), "-hide_banner", "-loglevel", "error", "-n",
                                    "-threads", "2", "-ss", str(start), "-i", str(source),
                                    "-t", str(end - start), "-an", "-vf", f"scale={width}:{height}:flags=bicubic",
                                    "-vsync", "0", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                                    "-threads", "2", "-pix_fmt", "yuv420p", str(partial)], check=True, timeout=900)
                    clip_meta = probe_video(partial, args.ffprobe)
                    if abs(clip_meta["duration_sec"] - (end - start)) > max(0.1, 2 * (end - start) / clip_meta["frame_count"]):
                        raise ValueError(f"Materialized window duration mismatch: {clip}")
                    os.replace(partial, clip)
                    write_json(receipt, {"sha256": digest(clip), "start": start, "end": end})
                files[str(clip.relative_to(output))] = digest(clip)
                samples.append({"source_video_id": video_id, "source_video": source.name,
                                "source_duration_sec": meta["duration_sec"], "window_index": number,
                                "window_start_sec": start, "window_end_sec": end,
                                "window_duration_sec": end - start})
                requests.append({"videos": [str(clip.relative_to(output))], "messages": [
                    {"role": "user", "content": "<video>\n" + prompt}]})
        write_json(output / "source_metadata.json", metadata)
        write_json(output / "test_index.json", rows)
        request_file = output / "test.windows.jsonl"
        request_file.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in requests), encoding="utf-8")
        files.update({name: digest(output / name) for name in ("test.windows.jsonl", "source_metadata.json", "test_index.json")})
        write_json(manifest_path, {"schema": "video-ora-inference-windows/v1", "answer_contract": "highlights-only/v1",
                                  "source_counts": {"test": len(rows)}, "source_ids": [row["video_id"] for row in rows],
                                  "samples": {"test": samples}, "source_files": {"test_windows": {"rows": len(samples)}},
                                  "inference_only": True, "input_contract": contract, "prepared_files": files})
    return output


def prepared_metadata(materialized: Path) -> list[dict]:
    """Use current input media, never a release package's historical video table."""
    return read_json(materialized / "source_metadata.json")


def frame_spans(intervals, pts) -> list[list[int]]:
    result = []
    for start, end in intervals:
        first, stop = bisect_left(pts, start), bisect_left(pts, end)
        if first >= stop:
            continue
        if result and first <= result[-1][1] + 1:
            result[-1][1] = max(result[-1][1], stop - 1)
        else:
            result.append([first, stop - 1])
    return result
