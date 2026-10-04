#!/usr/bin/env python3
"""Create overview and selected-event contact sheets for temporal handoff JSONL."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np


Segment = Tuple[int, int]


def _load_records(path: Path) -> Dict[str, Mapping[str, object]]:
    records: Dict[str, Mapping[str, object]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL line {line_number}: {exc}") from exc
            video_id = str(record.get("video_id", "")).strip()
            if video_id:
                records[video_id] = record
    return records


def _segments(record: Mapping[str, object]) -> List[Segment]:
    result: List[Segment] = []
    for segment in record.get("segments_frame", []) or []:
        if isinstance(segment, (list, tuple)) and len(segment) == 2:
            start, end = int(segment[0]), int(segment[1])
            if end >= start:
                result.append((start, end))
    return result


def _segment_index(frame_id: int, segments: Sequence[Segment]) -> Optional[int]:
    for index, (start, end) in enumerate(segments):
        if start <= frame_id <= end:
            return index
    return None


def _sample_segment_frames(segments: Sequence[Segment], count: int) -> List[int]:
    sampled: List[int] = []
    for start, end in segments:
        if end <= start or count <= 1:
            sampled.append((start + end) // 2)
            continue
        sampled.extend(
            int(round(value))
            for value in np.linspace(start, end, num=count, endpoint=True)
        )
    return sampled


def _read_frame(capture: cv2.VideoCapture, frame_id: int) -> Optional[np.ndarray]:
    capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_id)))
    ok, frame = capture.read()
    return frame if ok and frame is not None else None


def _thumbnail(
    frame: np.ndarray,
    *,
    width: int,
    frame_id: int,
    fps: float,
    selected_index: Optional[int],
) -> np.ndarray:
    height = max(1, int(round(frame.shape[0] * width / max(1, frame.shape[1]))))
    image = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    border = (40, 190, 40) if selected_index is not None else (70, 70, 70)
    image = cv2.copyMakeBorder(image, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=border)
    time_sec = frame_id / fps if fps > 0 else 0.0
    label = f"{time_sec:.2f}s f={frame_id}"
    if selected_index is not None:
        label += f"  EVENT {selected_index + 1}"
    cv2.rectangle(image, (3, 3), (min(image.shape[1] - 3, 190), 27), (0, 0, 0), -1)
    cv2.putText(
        image,
        label,
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return image


def _tile(images: Sequence[np.ndarray], columns: int = 6) -> np.ndarray:
    if not images:
        raise ValueError("cannot tile an empty image list")
    columns = max(1, int(columns))
    rows = int(math.ceil(len(images) / columns))
    cell_height = max(image.shape[0] for image in images)
    cell_width = max(image.shape[1] for image in images)
    canvas = np.zeros((rows * cell_height, columns * cell_width, 3), dtype=np.uint8)
    for index, image in enumerate(images):
        row, column = divmod(index, columns)
        top = row * cell_height + (cell_height - image.shape[0]) // 2
        left = column * cell_width + (cell_width - image.shape[1]) // 2
        canvas[top : top + image.shape[0], left : left + image.shape[1]] = image
    return canvas


def _write_sheet(
    capture: cv2.VideoCapture,
    frame_ids: Iterable[int],
    segments: Sequence[Segment],
    fps: float,
    output_path: Path,
    tile_width: int,
) -> int:
    thumbnails = []
    for frame_id in frame_ids:
        frame = _read_frame(capture, int(frame_id))
        if frame is None:
            continue
        thumbnails.append(
            _thumbnail(
                frame,
                width=tile_width,
                frame_id=int(frame_id),
                fps=fps,
                selected_index=_segment_index(int(frame_id), segments),
            )
        )
    if not thumbnails:
        return 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    encoded_ok, encoded = cv2.imencode(".jpg", _tile(thumbnails))
    if not encoded_ok:
        raise RuntimeError(f"failed to encode preview: {output_path}")
    # OpenCV's Windows file APIs do not reliably accept non-ASCII paths.
    # NumPy's Unicode-aware file writer keeps previews on the requested E:
    # directory even when a parent folder contains Chinese characters.
    try:
        encoded.tofile(str(output_path))
    except OSError as exc:
        raise RuntimeError(f"failed to write preview: {output_path}") from exc
    return len(thumbnails)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff", required=True)
    parser.add_argument("--video-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--video-ids", default="")
    parser.add_argument("--overview-frames", type=int, default=30)
    parser.add_argument("--frames-per-segment", type=int, default=5)
    parser.add_argument("--tile-width", type=int, default=240)
    args = parser.parse_args()

    handoff = Path(args.handoff).expanduser().resolve()
    video_dir = Path(args.video_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    records = _load_records(handoff)
    requested = [value.strip() for value in args.video_ids.split(",") if value.strip()]
    video_ids = requested or sorted(records, key=lambda value: (not value.isdigit(), value))
    manifest = []
    for video_id in video_ids:
        if video_id not in records:
            raise KeyError(f"video {video_id} is missing from handoff")
        video_path = video_dir / f"{video_id}.mp4"
        capture = cv2.VideoCapture(str(video_path))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if not capture.isOpened() or frame_count <= 0 or fps <= 0:
            capture.release()
            raise RuntimeError(f"cannot open video: {video_path}")
        segments = _segments(records[video_id])
        overview_count = max(2, min(frame_count, int(args.overview_frames)))
        overview_ids = np.linspace(0, frame_count - 1, num=overview_count).round().astype(int)
        selected_ids = _sample_segment_frames(
            segments, max(1, int(args.frames_per_segment))
        )
        overview_path = out_dir / f"{video_id}_overview.jpg"
        selected_path = out_dir / f"{video_id}_selected.jpg"
        overview_written = _write_sheet(
            capture,
            overview_ids.tolist(),
            segments,
            fps,
            overview_path,
            max(120, int(args.tile_width)),
        )
        selected_written = _write_sheet(
            capture,
            selected_ids,
            segments,
            fps,
            selected_path,
            max(120, int(args.tile_width)),
        )
        capture.release()
        manifest.append(
            {
                "video_id": video_id,
                "duration_sec": frame_count / fps,
                "segments_frame": [list(item) for item in segments],
                "overview": str(overview_path),
                "selected": str(selected_path),
                "overview_frames": overview_written,
                "selected_frames": selected_written,
            }
        )
        print(f"preview video={video_id} segments={len(segments)}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"wrote previews={len(manifest)} -> {out_dir}")


if __name__ == "__main__":
    main()
