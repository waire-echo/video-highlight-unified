#!/usr/bin/env python3
"""Run the ORA Stage-2 spatial pipeline from a Stage-1 handoff to submission.zip.

The runner accepts a temporal-handoff JSONL, a highlight-frames JSONL, or a ZIP
containing one of those files.  Stage-1 frame IDs are frozen: this stage may
only generate and refine crop boxes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2


HERE = Path(__file__).resolve().parent
CODE = HERE
CONFIG_PATH = Path(
    os.environ.get("HIGHLIGHT_PIPELINE_CONFIG", HERE.parents[2] / "configs/pipeline.local.json")
).expanduser().resolve()
if not CONFIG_PATH.is_file():
    raise FileNotFoundError(f"缺少本机配置: {CONFIG_PATH}")
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8-sig"))
PACKAGE = Path(CONFIG.get("run_root", CONFIG_PATH.parent / "run")).expanduser().resolve()


def configured_path(name: str) -> Path:
    value = CONFIG.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"config.local.json 缺少有效路径: {name}")
    return Path(value).expanduser().resolve()


def optional_path(name: str, default: Path) -> Path:
    value = CONFIG.get(name)
    return Path(value).expanduser().resolve() if isinstance(value, str) and value.strip() else default.resolve()


DROP_DIR = optional_path("input_dir", PACKAGE / "input")
LOG_DIR = optional_path("log_dir", PACKAGE / "logs")
OUTPUT_DIR = optional_path("output_dir", PACKAGE / "output")
WORK_DIR = configured_path("work_dir")
PYTHON_EXPECTED = optional_path("python_executable", Path(sys.executable))
VIDEOS = configured_path("video_dir")
YOLO_MODEL = configured_path("yolo_model")
POSE_MODEL = configured_path("pose_model")
INDEX = configured_path("test_index")
TRACKER = CODE / "botsort_reid.yaml"
STATUS_LOG = LOG_DIR / "heartbeat.txt"
STATUS_JSON = LOG_DIR / "status.json"
LIVE_LOG = LOG_DIR / "stage2-live.log"
FINAL_JSONL = OUTPUT_DIR / "predictions_spatial.jsonl"
FINAL_ZIP = OUTPUT_DIR / "submission.zip"
FINAL_REPORT = OUTPUT_DIR / "stage2_report.json"
INDEX_ROWS = json.loads(INDEX.read_text(encoding="utf-8-sig"))
EXPECTED_RATIOS = {str(row["video_id"]): row["targetRatioWH"] for row in INDEX_ROWS}
EXPECTED_IDS = set(EXPECTED_RATIOS)
EXPECTED_VIDEO_COUNT = int(CONFIG.get("expected_video_count", len(INDEX_ROWS)))
if len(EXPECTED_IDS) != len(INDEX_ROWS) or len(EXPECTED_IDS) != EXPECTED_VIDEO_COUNT:
    raise ValueError("Invalid test index count or duplicate video IDs")
if any(not video_id.isdecimal() for video_id in EXPECTED_IDS):
    raise ValueError("Video IDs must be decimal integers")
PIPELINE_VERSION = "video-highlight-pipeline/spatial-v1"
RUN_STARTED = time.monotonic()


class TeeText:
    def __init__(self, *streams: Any) -> None:
        self.streams = streams

    def write(self, value: str) -> int:
        for stream in self.streams:
            stream.write(value)
            stream.flush()
        return len(value)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def start_live_log() -> Any:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    handle = LIVE_LOG.open("a", encoding="utf-8", buffering=1, newline="\n")
    sys.stdout = TeeText(sys.__stdout__, handle)
    sys.stderr = TeeText(sys.__stderr__, handle)
    print(f"\n===== ORA_RUN_v2 {utc_now()} pid={os.getpid()} =====", flush=True)
    return handle


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def gpu_snapshot() -> dict[str, Any]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=5,
        )
        first = result.stdout.strip().splitlines()[0]
        name, used, total, utilization = [part.strip() for part in first.split(",", 3)]
        return {
            "name": name,
            "memory_used_mib": int(used),
            "memory_total_mib": int(total),
            "utilization_percent": int(utilization),
        }
    except Exception as error:
        return {"available": False, "error": str(error)}


def set_status(state: str, stage: str, detail: str) -> None:
    clean = " ".join(str(detail).split())
    timestamp = utc_now()
    atomic_text(STATUS_LOG, f"{state} | {stage} | {clean} | {timestamp}\n")
    payload = {
        "state": state,
        "stage": stage,
        "detail": clean,
        "updated_at": timestamp,
        "pid": os.getpid(),
        "elapsed_seconds": round(time.monotonic() - RUN_STARTED, 1),
        "gpu": gpu_snapshot(),
        "live_log": str(LIVE_LOG),
        "work_dir": str(WORK_DIR),
    }
    atomic_text(STATUS_JSON, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_dumps(value: Any, *, indent: int | None = None) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, indent=indent)


def load_jsonl_text(text: str, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{label}:{line_number} 不是有效 JSONL: {error}") from error
        if not isinstance(row, dict):
            raise ValueError(f"{label}:{line_number} 必须是 JSON 对象")
        rows.append(row)
    if not rows:
        raise ValueError(f"{label} 没有有效记录")
    return rows


def choose_zip_member(names: list[str]) -> str:
    candidates = [name for name in names if name.lower().endswith(".jsonl")]
    preferred = [
        name for name in candidates
        if "temporal_handoff" in Path(name).name.lower()
        and "window_score" not in Path(name).name.lower()
    ]
    if len(preferred) == 1:
        return preferred[0]
    frame_files = [name for name in candidates if "highlight_frames" in Path(name).name.lower()]
    if not preferred and len(frame_files) == 1:
        return frame_files[0]
    matches = preferred or frame_files
    if not matches:
        raise ValueError("ZIP 内没有 temporal_handoff*.jsonl 或 highlight_frames*.jsonl")
    raise ValueError("ZIP 内有多个可能的第一阶段结果，无法自动选择: " + ", ".join(matches))


def load_source(path: Path) -> tuple[list[dict[str, Any]], str, str]:
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            bad = archive.testzip()
            if bad:
                raise ValueError(f"ZIP CRC 校验失败: {bad}")
            member = choose_zip_member(archive.namelist())
            text = archive.read(member).decode("utf-8-sig")
        return load_jsonl_text(text, f"{path}!{member}"), member, sha256(path)
    if path.suffix.lower() != ".jsonl":
        raise ValueError("第一阶段输入只支持 .jsonl 或 .zip")
    return load_jsonl_text(path.read_text(encoding="utf-8-sig"), str(path)), path.name, sha256(path)


def discover_input(explicit: Path | None) -> Path:
    if explicit is not None:
        path = explicit.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    DROP_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(
        path for path in DROP_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in {".jsonl", ".zip"}
    )
    if len(files) != 1:
        raise ValueError(
            f"请在 {DROP_DIR} 只放一个第一阶段 .jsonl/.zip，当前找到 {len(files)} 个；"
            "也可以把文件直接拖到 run_stage2.cmd。"
        )
    return files[0].resolve()


def video_meta(video_id: str) -> tuple[int, int, int]:
    path = VIDEOS / f"{video_id}.mp4"
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"无法打开视频: {path}")
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    if count <= 0 or width <= 0 or height <= 0:
        raise RuntimeError(f"视频元数据无效: {path}")
    return count, width, height


def input_frames(row: dict[str, Any], actual_count: int) -> tuple[list[int], int]:
    if isinstance(row.get("segments_frame"), list):
        convention = row.get("frame_index_convention", "inclusive")
        if convention != "inclusive":
            raise ValueError(
                f"video_id={row.get('video_id')} 的 segments_frame 必须是 inclusive，当前为 {convention!r}"
            )
        frames: set[int] = set()
        segments = row["segments_frame"]
        for segment in segments:
            if not isinstance(segment, list) or len(segment) != 2:
                raise ValueError(f"video_id={row.get('video_id')} 的 segments_frame 格式无效")
            start, end = segment
            if type(start) is not int or type(end) is not int or start < 0 or end < start or end >= actual_count:
                raise ValueError(f"video_id={row.get('video_id')} 的帧段无效: {segment!r}")
            frames.update(range(start, end + 1))
        result = sorted(frames)
        if isinstance(row.get("highlight_frames"), list):
            explicit = sorted({int(value) for value in row["highlight_frames"]})
            if explicit != result:
                raise ValueError(f"video_id={row.get('video_id')} 的 highlight_frames 与 segments_frame 不一致")
        return result, len(segments)
    if isinstance(row.get("highlight_frames"), list):
        if any(type(value) is not int for value in row["highlight_frames"]):
            raise ValueError("highlight_frames must contain integer frame IDs")
        result = [int(value) for value in row["highlight_frames"]]
        if result != sorted(set(result)):
            raise ValueError(f"video_id={row.get('video_id')} 的 highlight_frames 必须递增且不重复")
        if result and (result[0] < 0 or result[-1] >= actual_count):
            raise ValueError("highlight_frames contains an out-of-range frame ID")
        segments = 0 if not result else 1 + sum(b != a + 1 for a, b in zip(result, result[1:]))
        return result, segments
    raise ValueError(f"video_id={row.get('video_id')} 缺少 segments_frame/highlight_frames")


def infer_main_subject(subject: str, action: str) -> str:
    text = f"{subject} {action}".lower()
    if any(word in text for word in ("dog", "cat", "animal", "horse", "bird", "puppy", "kitten")):
        return "animal"
    if any(word in text for word in ("person", "people", "man", "woman", "child", "kid", "human")):
        return "person"
    return "unknown"


def normalized_subject_hint(subject: str, action: str) -> str:
    value = subject.strip().lower()
    if value in {"person", "dog", "cat", "animal"}:
        return value
    inferred = infer_main_subject(subject, action)
    return inferred if inferred in {"person", "animal"} else "unknown"


def full_crop(width: int, height: int, ratio: list[int]) -> list[int]:
    ratio_w, ratio_h = ratio
    crop_width = max(1, min(width, math.floor(height * ratio_w / ratio_h)))
    crop_height = crop_width * ratio_h / ratio_w
    x = max(0, math.floor((width - crop_width) / 2))
    y = max(0, math.floor((height - crop_height) / 2))
    return [x, y, crop_width]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    atomic_text(path, "".join(json_dumps(row) + "\n" for row in rows))


def prepare_input(
    rows: list[dict[str, Any]], smoke_videos: list[str] | None
) -> tuple[list[dict[str, Any]], dict[str, list[int]], dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        video_id = str(row.get("video_id"))
        if video_id in by_id:
            raise ValueError(f"重复 video_id={video_id}")
        by_id[video_id] = row
    if set(by_id) != EXPECTED_IDS:
        missing = sorted(EXPECTED_IDS - set(by_id), key=int)
        extra = sorted(set(by_id) - EXPECTED_IDS)
        raise ValueError(f"第一阶段必须匹配测试索引的 {len(EXPECTED_IDS)} 条；缺少={missing[:10]}，额外={extra[:10]}")
    selected_ids = smoke_videos if smoke_videos is not None else sorted(EXPECTED_IDS, key=int)
    invalid_smoke = [video_id for video_id in selected_ids if video_id not in EXPECTED_IDS]
    if invalid_smoke:
        raise ValueError(f"--smoke-videos 必须在测试索引中，当前无效值: {invalid_smoke}")
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("--smoke-videos 不能包含重复 video_id")

    baseline: list[dict[str, Any]] = []
    frames_by_id: dict[str, list[int]] = {}
    hints_dir = WORK_DIR / "hints"
    hints_dir.mkdir(parents=True, exist_ok=True)
    total_frames = total_segments = unknown_subjects = empty_videos = 0
    for video_id in selected_ids:
        row = by_id[video_id]
        if row.get("valid") is False:
            raise ValueError(f"video_id={video_id} 的第一阶段 valid=false")
        ratio = row.get("targetRatioWH")
        if ratio not in ([16, 9], [9, 16]):
            raise ValueError(f"video_id={video_id} 的 targetRatioWH 无效: {ratio!r}")
        if ratio != EXPECTED_RATIOS[video_id]:
            raise ValueError(f"video_id={video_id} 的比例与原始测试索引不一致")
        actual_count, width, height = video_meta(video_id)
        declared_count = row.get("frame_count")
        if declared_count is not None and int(declared_count) != actual_count:
            raise ValueError(
                f"video_id={video_id} 第一阶段 frame_count={declared_count}，实际视频={actual_count}"
            )
        frames, segment_count = input_frames(row, actual_count)
        empty_videos += int(not frames)
        if frames and (frames[0] < 0 or frames[-1] >= actual_count):
            raise ValueError(f"video_id={video_id} 有越界高光帧")
        frames_by_id[video_id] = frames
        crop = full_crop(width, height, ratio)
        baseline.append(
            {
                "video_id": video_id,
                "targetRatioWH": ratio,
                "predictions": [{"frame": frame, "bboxes": crop} for frame in frames],
            }
        )
        subject = str(row.get("subject_hint", ""))
        action = str(row.get("action_description", ""))
        description = " ".join(part for part in (subject, action) if part).strip()
        subject_label = normalized_subject_hint(subject, action)
        unknown_subjects += int(subject_label == "unknown")
        hint = {
            "subject_hint": subject_label,
            "main_subject": infer_main_subject(subject, action),
            "action_description": action,
            "description": description,
            "segments": row.get("segments_sec", []),
            "source": "temporal_handoff",
        }
        atomic_text(hints_dir / f"{video_id}.json", json_dumps(hint, indent=2) + "\n")
        total_frames += len(frames)
        total_segments += segment_count
    write_jsonl(WORK_DIR / "01_center.jsonl", baseline)
    return baseline, frames_by_id, {
        "videos": len(selected_ids), "frames": total_frames, "segments": total_segments,
        "unknown_subjects": unknown_subjects,
        "empty_videos": empty_videos,
    }


def required_paths() -> dict[str, Path]:
    return {
        "Python 环境": PYTHON_EXPECTED,
        "视频目录": VIDEOS,
        "YOLO11m 权重": YOLO_MODEL,
        "YOLO11m-pose 权重": POSE_MODEL,
        "YOLO 脚本": CODE / "generate_highlight_yolo_center.py",
        "Pose 脚本": CODE / "generate_pose_head_anchor.py",
        "BoT-SORT 脚本": CODE / "generate_botsort_identity_lock.py",
        "校验脚本": CODE / "validate_submission.py",
        "BoT-SORT 配置": TRACKER,
        "原始测试索引": INDEX,
    }


def preflight(require_cuda: bool = True) -> dict[str, Any]:
    missing = [f"{label}: {path}" for label, path in required_paths().items() if not path.exists()]
    if Path(sys.executable).resolve() != PYTHON_EXPECTED:
        missing.append(f"必须使用配置中的 Python: {PYTHON_EXPECTED}；当前为 {sys.executable}")
    ids = {path.stem for path in VIDEOS.glob("*.mp4")} if VIDEOS.is_dir() else set()
    if ids != EXPECTED_IDS:
        missing.append(f"视频必须匹配测试索引的 {len(EXPECTED_IDS)} 个 ID，当前匹配 {len(ids)} 个")
    try:
        import torch
        import ultralytics
        import numpy
        import lap
        del numpy, lap
    except ImportError as error:
        missing.append(f"Python 依赖缺失: {error}")
        torch = None
        ultralytics = None
    if missing:
        raise FileNotFoundError("本机第二阶段依赖不完整:\n- " + "\n- ".join(missing))
    assert torch is not None and ultralytics is not None
    cuda_available = torch.cuda.is_available()
    if require_cuda and not cuda_available:
        raise RuntimeError("当前 Python 环境未检测到 CUDA GPU")
    return {
        "python": sys.executable,
        "opencv": cv2.__version__,
        "ultralytics": ultralytics.__version__,
        "torch": torch.__version__,
        "cuda_available": cuda_available,
        "gpu": torch.cuda.get_device_name(0) if cuda_available else None,
        "videos": len(ids),
    }


class Heartbeat:
    def __init__(self, stage: str) -> None:
        self.stage = stage
        self.detail = "starting"
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self.stop_event.wait(30):
            set_status("RUNNING", self.stage, self.detail)

    def __enter__(self) -> "Heartbeat":
        set_status("RUNNING", self.stage, self.detail)
        self.thread.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self.stop_event.set()
        self.thread.join(timeout=2)


def run_stage(stage: str, command: list[str]) -> None:
    print(f"\n[{stage}]", flush=True)
    with Heartbeat(stage) as heartbeat:
        process = subprocess.Popen(
            command,
            cwd=CODE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            line = line.rstrip()
            if line:
                heartbeat.detail = line
                print(line, flush=True)
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def code_fingerprint() -> dict[str, str]:
    return {
        path.name: sha256(path)
        for path in sorted(CODE.iterdir())
        if path.is_file() and path.suffix.lower() in {".py", ".yaml"}
    }


def ensure_work_state(input_sha: str, member: str, reset: bool) -> None:
    state_path = WORK_DIR / "state.json"
    expected = {
        "pipeline_version": PIPELINE_VERSION,
        "input_sha256": input_sha,
        "input_member": member,
        "test_index_sha256": sha256(INDEX),
        "code": code_fingerprint(),
        "yolo_model": {"path": str(YOLO_MODEL), "sha256": sha256(YOLO_MODEL)},
        "pose_model": {"path": str(POSE_MODEL), "sha256": sha256(POSE_MODEL)},
    }
    if reset and WORK_DIR.exists():
        if WORK_DIR == PACKAGE or not WORK_DIR.is_relative_to(PACKAGE):
            raise ValueError("--reset-work 只允许清理 run_root 内的独立 work 子目录")
        shutil.rmtree(WORK_DIR)
    if state_path.exists():
        current = json.loads(state_path.read_text(encoding="utf-8"))
        if current != expected:
            raise RuntimeError(
                f"{WORK_DIR} 属于不同输入或代码；请确认后加 --reset-work 清理本工具缓存。"
            )
    else:
        if WORK_DIR.exists() and any(WORK_DIR.iterdir()):
            raise RuntimeError(f"{WORK_DIR} 没有有效 state.json；请确认后加 --reset-work。")
        WORK_DIR.mkdir(parents=True, exist_ok=True)
        atomic_text(state_path, json_dumps(expected, indent=2) + "\n")


def compare_rows(before: list[dict[str, Any]], after: list[dict[str, Any]]) -> dict[str, int]:
    by_id = {str(row["video_id"]): row for row in after}
    if len(by_id) != len(after):
        raise ValueError("输出包含重复 video_id")
    changed = changed_videos = 0
    for row in before:
        video_id = str(row["video_id"])
        other = by_id.get(video_id)
        if other is None:
            raise ValueError(f"输出缺少 video_id={video_id}")
        if other.get("targetRatioWH") != row.get("targetRatioWH"):
            raise ValueError(f"video_id={video_id} 的比例被修改")
        first = row.get("predictions", [])
        second = other.get("predictions", [])
        if [x["frame"] for x in first] != [x["frame"] for x in second]:
            raise ValueError(f"video_id={video_id} 的第一阶段帧集合被修改")
        local = sum(a["bboxes"] != b["bboxes"] for a, b in zip(first, second))
        changed += local
        changed_videos += int(local > 0)
    if len(by_id) != len(before):
        raise ValueError("输出包含额外 video_id")
    return {"changed_frames": changed, "changed_videos": changed_videos}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return load_jsonl_text(path.read_text(encoding="utf-8-sig"), str(path))


def validate_output(
    baseline: list[dict[str, Any]], output_path: Path, expected_frames: dict[str, list[int]]
) -> dict[str, Any]:
    output = load_jsonl(output_path)
    comparison = compare_rows(baseline, output)
    invalid: list[str] = []
    for row in output:
        video_id = str(row["video_id"])
        count, width, height = video_meta(video_id)
        ratio_w, ratio_h = row["targetRatioWH"]
        predictions = row.get("predictions", [])
        if [item.get("frame") for item in predictions] != expected_frames[video_id]:
            invalid.append(f"{video_id}: frame set mismatch")
            continue
        for item in predictions:
            frame = item.get("frame")
            box = item.get("bboxes")
            if type(frame) is not int or not 0 <= frame < count:
                invalid.append(f"{video_id}: invalid frame={frame}")
                continue
            if not isinstance(box, list) or len(box) != 3 or any(type(x) is not int for x in box):
                invalid.append(f"{video_id}/{frame}: invalid bbox={box!r}")
                continue
            x, y, crop_width = box
            crop_height = crop_width * ratio_h / ratio_w
            if x < 0 or y < 0 or crop_width <= 0 or x + crop_width > width or y + crop_height > height + 1e-9:
                invalid.append(f"{video_id}/{frame}: out of bounds bbox={box!r}")
    if invalid:
        raise ValueError("最终输出校验失败: " + "; ".join(invalid[:10]))
    return {
        "videos": len(output),
        "frames": sum(len(row["predictions"]) for row in output),
        "invalid_count": 0,
        **comparison,
    }


def make_zip(source: Path, destination: Path) -> dict[str, Any]:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(source, arcname="predictions_spatial.jsonl")
    with zipfile.ZipFile(temporary) as archive:
        bad = archive.testzip()
        names = archive.namelist()
    if bad or names != ["predictions_spatial.jsonl"]:
        temporary.unlink(missing_ok=True)
        raise ValueError(f"最终 ZIP 校验失败: bad={bad}, members={names}")
    os.replace(temporary, destination)
    return {"path": str(destination), "sha256": sha256(destination), "members": names}


def commands() -> list[tuple[str, list[str]]]:
    python = sys.executable
    hints = WORK_DIR / "hints"
    center = WORK_DIR / "01_center.jsonl"
    yolo = WORK_DIR / "02_yolo.jsonl"
    pose = WORK_DIR / "03_pose.jsonl"
    final = WORK_DIR / "predictions_spatial.jsonl"
    result = [
        (
            "YOLO 主体定位",
            [python, "-B", str(CODE / "generate_highlight_yolo_center.py"),
             "--base-predictions", str(center), "--video-dir", str(VIDEOS),
             "--hints-dir", str(hints), "--output", str(yolo),
             "--cache", str(WORK_DIR / "02_yolo_cache.jsonl"), "--model", str(YOLO_MODEL),
             "--device", "0", "--imgsz", "640", "--batch-size", "8", "--stride", "3",
             "--conf", "0.15", "--iou", "0.50", "--smooth-window", "15",
             "--center-blend", "1.0", "--person-16x9-anchor-limit", "0.45"],
        ),
        (
            "Pose 人物构图",
            [python, "-B", str(CODE / "generate_pose_head_anchor.py"), "--input", str(yolo),
             "--video-dir", str(VIDEOS), "--hints-dir", str(hints), "--output", str(pose),
             "--cache", str(WORK_DIR / "03_pose_cache.jsonl"), "--model", str(POSE_MODEL),
             "--device", "0", "--imgsz", "640", "--batch-size", "4", "--stride", "3",
             "--conf", "0.15", "--iou", "0.50", "--smooth-window", "15",
             "--pose-blend", "1.0", "--target-ratio", "16x9", "--face-line-fraction", "0.30",
             "--max-shift-fraction", "0.35"],
        ),
        (
            "BoT-SORT 主体锁定",
            [python, "-B", str(CODE / "generate_botsort_identity_lock.py"), "--input", str(pose),
             "--video-dir", str(VIDEOS), "--hints-dir", str(hints), "--model", str(YOLO_MODEL),
             "--tracker", str(TRACKER), "--output", str(final),
             "--cache", str(WORK_DIR / "04_botsort_cache.jsonl"),
             "--report", str(WORK_DIR / "04_botsort_report.json"),
             "--zip-output", str(WORK_DIR / "04_botsort_preview.zip"),
             "--apply-mode", "confident", "--group-track-stride", "2",
             "--single-track-stride", "3", "--min-jump-pixels", "5",
             "--min-jump-fraction", "0.012", "--max-step-ratio", "0.70",
             "--min-score-margin", "0.06", "--min-coverage", "0.20",
             "--max-median-shift-fraction", "0.25", "--smooth-window", "15",
             "--center-blend", "1.0", "--imgsz", "640", "--conf", "0.15",
             "--iou", "0.50", "--device", "0", "--seed", "0"],
        ),
    ]
    from video_highlight.spatial_parameters import apply_parameters
    return apply_parameters(result, CONFIG.get('spatial_parameters', {}))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, help="第一阶段 .jsonl/.zip；省略时读取 input 文件夹")
    parser.add_argument("--check-only", action="store_true", help="只检查环境，不运行模型")
    parser.add_argument("--allow-cpu", action="store_true", help="仅用于无卡环境预检")
    parser.add_argument("--prepare-only", action="store_true", help="无卡校验全部输入并生成中心框，不运行模型")
    parser.add_argument("--smoke-videos", help="逗号分隔的 video_id；只做 smoke，不生成正式 submission.zip")
    parser.add_argument("--reset-work", action="store_true", help="清理本工具固定缓存后重跑")
    parser.add_argument("--replace-output", action="store_true", help="明确允许替换已有正式输出")
    return parser.parse_args()


def parse_smoke_videos(value: str | None) -> list[str] | None:
    if value is None:
        return None
    result = [part.strip() for part in value.split(",") if part.strip()]
    if not result:
        raise ValueError("--smoke-videos 至少需要一个 video_id")
    return result


def main() -> None:
    args = parse_args()
    if args.allow_cpu and not (args.check_only or args.prepare_only):
        raise ValueError("--allow-cpu 只允许环境预检或输入准备")
    smoke_videos = parse_smoke_videos(args.smoke_videos)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    environment = preflight(require_cuda=not (args.prepare_only or args.allow_cpu))
    if args.check_only:
        set_status("READY", "环境预检", f"Python, models and {len(EXPECTED_IDS)} videos are ready; CUDA={environment['cuda_available']}")
        print(json_dumps({"status": "READY", **environment}, indent=2))
        return
    if smoke_videos is None and not args.prepare_only and not args.replace_output:
        existing = [path for path in (FINAL_JSONL, FINAL_ZIP, FINAL_REPORT) if path.exists()]
        if existing:
            raise FileExistsError("正式结果已存在，不会覆盖: " + ", ".join(str(path) for path in existing))
    source = discover_input(args.input)
    rows, member, input_sha = load_source(source)
    ensure_work_state(input_sha, member, args.reset_work)
    baseline, frames_by_id, input_stats = prepare_input(rows, smoke_videos)
    print(json_dumps({"input": str(source), "member": member, **input_stats}, indent=2), flush=True)
    if args.prepare_only:
        validation = validate_output(baseline, WORK_DIR / "01_center.jsonl", frames_by_id)
        report = {
            "status": "CPU_PREPARATION_OK",
            "input": {"path": str(source), "sha256": input_sha, **input_stats},
            "environment": environment,
            "validation": validation,
            "gpu_inference_completed": False,
            "completed_at": utc_now(),
        }
        atomic_text(LOG_DIR / "preparation_report.json", json_dumps(report, indent=2) + "\n")
        set_status("CPU_PREPARATION_OK", "输入准备完成", f"videos={input_stats['videos']} frames={input_stats['frames']}; awaiting GPU")
        print(json_dumps(report, indent=2))
        return
    if input_stats["unknown_subjects"]:
        print(
            f"WARNING: {input_stats['unknown_subjects']} 个视频没有可识别的自动主体类别；"
            "建议使用包含 subject_hint/action_description 的 temporal_handoff。",
            flush=True,
        )
    for stage, command in commands():
        run_stage(stage, command)

    final_work = WORK_DIR / "predictions_spatial.jsonl"
    set_status("RUNNING", "严格校验", "checking frozen frames, ratios, bounds and ZIP")
    validation = validate_output(baseline, final_work, frames_by_id)
    stages = {
        "yolo": compare_rows(baseline, load_jsonl(WORK_DIR / "02_yolo.jsonl")),
        "pose_vs_yolo": compare_rows(load_jsonl(WORK_DIR / "02_yolo.jsonl"), load_jsonl(WORK_DIR / "03_pose.jsonl")),
        "botsort_vs_pose": compare_rows(load_jsonl(WORK_DIR / "03_pose.jsonl"), load_jsonl(final_work)),
    }

    if smoke_videos is not None:
        smoke_label = ",".join(smoke_videos)
        smoke_report = {
            "status": "SMOKE_OK", "video_ids": smoke_videos,
            "input": {"path": str(source), "member": member, "sha256": input_sha, **input_stats},
            "environment": environment, "validation": validation, "stages": stages,
            "completed_at": utc_now(),
        }
        atomic_text(LOG_DIR / "smoke_report.json", json_dumps(smoke_report, indent=2) + "\n")
        set_status("SMOKE_OK", "完成", f"video_ids={smoke_label}, frames={validation['frames']}")
        print(json_dumps(smoke_report, indent=2))
        return

    if not args.replace_output:
        existing = [path for path in (FINAL_JSONL, FINAL_ZIP, FINAL_REPORT) if path.exists()]
        if existing:
            raise FileExistsError(
                "正式结果已存在，不会覆盖: " + ", ".join(str(path) for path in existing)
                + "；确认替换时使用 --replace-output。"
            )
    shutil.copyfile(final_work, FINAL_JSONL)
    zip_report = make_zip(FINAL_JSONL, FINAL_ZIP)
    report = {
        "schema_version": "stage2-report/v1",
        "status": "COMPLETE_AND_VALID",
        "completed_at": utc_now(),
        "pipeline_version": PIPELINE_VERSION,
        "input": {"path": str(source), "member": member, "sha256": input_sha, **input_stats},
        "environment": environment,
        "models": {
            "yolo11m": str(YOLO_MODEL), "yolo11m_pose": str(POSE_MODEL)
        },
        "stages": stages,
        "validation": validation,
        "prediction": {"path": str(FINAL_JSONL), "sha256": sha256(FINAL_JSONL)},
        "submission_zip": zip_report,
        "temporal_invariant": "Stage-1 video IDs, target ratios and frame IDs are unchanged.",
    }
    atomic_text(FINAL_REPORT, json_dumps(report, indent=2) + "\n")
    set_status(
        "COMPLETE_AND_VALID", "完成",
        f"videos={validation['videos']} frames={validation['frames']} invalid=0 zip={FINAL_ZIP}",
    )
    print(json_dumps(report, indent=2))


if __name__ == "__main__":
    start_live_log()
    try:
        main()
    except Exception as error:
        set_status("FAILED", "异常退出", f"{type(error).__name__}: {error}")
        traceback.print_exc()
        raise SystemExit(1)
