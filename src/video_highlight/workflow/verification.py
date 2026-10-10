"""CPU evidence for optional I/O changes; no model responses or official predictions are edited."""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import time

from .contracts import read_json, write_json
from .handoff import common
from .io_optimized import SequentialFrameReader


def verify_io(source_root, data_root, video_dir, ids, output):
    import cv2
    import numpy as np
    source_root, data_root = Path(source_root), Path(data_root)
    spec = importlib.util.spec_from_file_location("scene_io_verification", source_root / "experiments/scene_general_v3/temporal.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cv2.setNumThreads(2)
    api = common(source_root)
    rows = [row for row in api.load_rows(data_root / "inputs/baseline.zip") if str(row["video_id"]) in set(ids)]
    if [str(row["video_id"]) for row in rows] != ids:
        raise ValueError("Verification IDs must be present in original baseline order")
    cfg = read_json(source_root / "experiments/scene_general_v3/config.json")
    basecfg = read_json(source_root / "experiments/scene_general_v1/config.json")
    reader = SequentialFrameReader()
    cases = []
    original_seconds = optimized_seconds = 0
    original_decoded = 0
    try:
        with np.load(data_root / "inputs/timestamps.npz") as timestamps:
            for row in rows:
                vid = str(row["video_id"])
                ts = timestamps[vid]
                shot = read_json(data_root / "output/shots" / (vid + ".json"))
                for number, frames in enumerate(api.groups([p["frame"] for p in row["predictions"]], shot["cuts"])):
                    sample = module.sample_frames(frames, ts, cfg["sample_interval_seconds"])
                    previous = [max(frames[0], int(np.searchsorted(ts, max(ts[frames[0]], ts[f] - basecfg["motion_lag_seconds"])))) for f in sample]
                    wanted = sorted(set(sample + previous))
                    video = Path(video_dir) / (vid + ".mp4")
                    started = time.monotonic()
                    old = module.read_frames(video, wanted, basecfg["image_max_side"])
                    original_seconds += time.monotonic() - started
                    original_decoded += max(wanted) + 1
                    started = time.monotonic()
                    new = reader(video, wanted, basecfg["image_max_side"])
                    optimized_seconds += time.monotonic() - started
                    matches = old.keys() == new.keys() and all(np.array_equal(old[f], new[f]) for f in wanted)
                    cases.append({"video_id": vid, "segment": number, "frames": len(wanted), "pixels_equal": matches,
                                  "image_sha256": {str(f): hashlib.sha256(new[f].tobytes()).hexdigest() for f in wanted}})
                    if not matches:
                        raise ValueError(f"Decoded pixels changed: {vid}/{number}")
    finally:
        reader.close()
    report = {"schema_version": "scene-io-verification/v1", "ids": ids, "segments": len(cases),
              "sampled_images": sum(case["frames"] for case in cases), "all_pixels_equal": True,
              "original_seconds": original_seconds, "optimized_seconds": optimized_seconds,
              "original_decoded_frames": original_decoded, "optimized_decoded_frames": reader.metrics["decoded_frames"],
              "reader": reader.metrics, "cases": cases, "gpu_inference_verified": False,
              "scope": "exact input image bytes for real baseline sample/motion frames; no GPU equivalence claim"}
    write_json(output, report)
    return report
