"""Opt-in I/O improvements; original frame numbering and resize arithmetic are retained."""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import time

from .contracts import write_json


class SequentialFrameReader:
    """Keep one decoder per current video; bound cached pixels and restart on cache misses."""
    def __init__(self, max_bytes=128 << 20):
        self.max_bytes = max_bytes
        self.cap = None
        self.video = None
        self.identity = None
        self.next_frame = 0
        self.cache = OrderedDict()
        self.bytes = 0
        self.metrics = {"decoder_opens": 0, "decoded_frames": 0, "cache_hits": 0,
                        "cache_peak_bytes": 0, "decode_seconds": 0.0}

    def close(self):
        if self.cap is not None:
            self.cap.release()
        self.cap = None

    def __call__(self, video, wanted, max_side=None):
        import cv2
        wanted = set(wanted)
        if not wanted:
            return {}
        if any(type(f) is not int or f < 0 for f in wanted):
            raise ValueError("Requested frames must be nonnegative original integer IDs")
        video = Path(video).resolve()
        stat = video.stat()
        identity = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        if self.video == video and self.identity != identity:
            raise ValueError("Video changed during decoding: " + str(video))
        if self.video != video:
            self.close()
            self.video, self.identity = video, identity
            self.cache.clear()
            self.bytes = 0
            self.next_frame = 0
        result = {}
        for frame in sorted(wanted):
            key = (frame, max_side)
            if key in self.cache:
                self.cache.move_to_end(key)
                result[frame] = self.cache[key].copy()
                self.metrics["cache_hits"] += 1
        missing = wanted - result.keys()
        if not missing:
            return result
        if self.cap is None or min(missing) < self.next_frame:
            self.close()
            self.cap = cv2.VideoCapture(str(video))
            if not self.cap.isOpened():
                self.close()
                raise RuntimeError("Cannot decode " + str(video))
            self.next_frame = 0
            self.metrics["decoder_opens"] += 1
        started = time.monotonic()
        last_frame = max(missing)
        try:
            while self.next_frame <= last_frame:
                frame = self.next_frame
                ok, bgr = self.cap.read()
                if not ok:
                    raise RuntimeError(f"Decode failed at original frame {frame}: {video}")
                self.next_frame += 1
                self.metrics["decoded_frames"] += 1
                if frame not in missing:
                    continue
                if max_side and max(bgr.shape[:2]) > max_side:
                    scale = max_side / max(bgr.shape[:2])
                    width = max(2, round(bgr.shape[1] * scale / 2) * 2)
                    height = max(2, round(bgr.shape[0] * scale / 2) * 2)
                    bgr = cv2.resize(bgr, (width, height))
                result[frame] = bgr
                if bgr.nbytes <= self.max_bytes:
                    while self.bytes + bgr.nbytes > self.max_bytes:
                        _, previous = self.cache.popitem(last=False)
                        self.bytes -= previous.nbytes
                    self.cache[(frame, max_side)] = bgr.copy()
                    self.bytes += bgr.nbytes
                    self.metrics["cache_peak_bytes"] = max(self.bytes, self.metrics["cache_peak_bytes"])
        finally:
            self.metrics["decode_seconds"] += time.monotonic() - started
        return result


class CheckpointExporter:
    """Validate each completed video; materialize the cumulative partial ZIP periodically."""
    def __init__(self, api, interval=25):
        if interval < 1:
            raise ValueError("checkpoint interval must be positive")
        self.api = api
        self.interval = interval
        self.completed = set()
        self.metrics = {"partial_full_exports": 0, "video_checkpoints": 0, "export_seconds": 0.0}

    def __call__(self, path, rows, baseline, manifest, preserve_width=True):
        started = time.monotonic()
        try:
            path = Path(path)
            if path.name == "partial":
                for row, original in zip(rows, baseline):
                    vid = str(row["video_id"])
                    if vid in self.completed:
                        continue
                    validation = self.api.validate([row], [original], manifest, preserve_width=preserve_width)
                    write_json(path.parent / "checkpoints" / (vid + ".json"),
                               {"prediction": row, "validation": validation})
                    self.completed.add(vid)
                    self.metrics["video_checkpoints"] += 1
                if len(rows) % self.interval:
                    return None
                self.metrics["partial_full_exports"] += 1
            return self.api.export(path, rows, baseline, manifest, preserve_width=preserve_width)
        finally:
            self.metrics["export_seconds"] += time.monotonic() - started
