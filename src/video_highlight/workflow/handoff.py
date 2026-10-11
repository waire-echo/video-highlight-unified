"""Convert explicit original-frame handoff to spatial input without changing its mask."""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import zipfile

from .contracts import read_json, sha256, write_json


def common(source_root):
    spec = importlib.util.spec_from_file_location("scene_contracts", Path(source_root) / "src/common.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def jsonl(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("JSONL records must be objects")
    ids = [str(row["video_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate video IDs in " + str(path))
    return rows


def build_spatial_input(handoff, highlights, manifest_path, timestamps, output, ids, source_root):
    import numpy as np
    metadata = read_json(manifest_path)
    temporal = {str(row["video_id"]): row for row in jsonl(handoff)}
    masks = {str(row["video_id"]): row for row in jsonl(highlights)}
    if set(temporal) != set(ids) or set(masks) != set(ids):
        raise ValueError("Handoff and highlight frames must contain exactly the requested IDs")
    rows = []
    with np.load(timestamps) as pts:
        for vid in ids:
            row, mask, meta = temporal[vid], masks[vid], metadata["videos"][vid]
            if row.get("schema_version") != "temporal-handoff/v1" or row.get("valid") is not True:
                raise ValueError("Incomplete/invalid temporal handoff: " + vid)
            if row.get("timestamp_mapping") != "original_frame_pts" or row.get("frame_index_convention") != "inclusive":
                raise ValueError("Explicit original PTS and inclusive frame ranges required: " + vid)
            ratio = row.get("targetRatioWH")
            if (not isinstance(ratio, list) or len(ratio) != 2 or
                    any(type(x) is not int or x <= 0 for x in ratio) or ratio != mask.get("targetRatioWH")
                    or ("targetRatioWH" in meta and ratio != meta["targetRatioWH"])):
                raise ValueError("Invalid/mismatched target ratio: " + vid)
            duration = row.get("duration_sec")
            if type(duration) not in (float, int) or not math.isfinite(duration) or duration <= 0:
                raise ValueError("Invalid original duration: " + vid)
            if "duration" in meta and abs(duration - meta["duration"]) > 1e-6:
                raise ValueError("Original duration changed: " + vid)
            if mask.get("frame_index_base") != 0 or row.get("frame_count") != meta["frame_count"] or mask.get("frame_count") != meta["frame_count"]:
                raise ValueError("Frame count/index convention changed: " + vid)
            ts = pts[vid]
            if len(ts) != meta["frame_count"] or not np.all(np.isfinite(ts)) or np.any(np.diff(ts) < 0):
                raise ValueError("Invalid original timestamps: " + vid)
            frames = mask.get("highlight_frames")
            if (not isinstance(frames, list) or any(type(f) is not int or not 0 <= f < len(ts) for f in frames)
                    or frames != sorted(set(frames))):
                raise ValueError("Invalid original highlight frames: " + vid)
            ranges = row.get("segments_frame")
            seconds = row.get("segments_sec")
            if not isinstance(ranges, list) or not isinstance(seconds, list) or len(ranges) != len(seconds):
                raise ValueError("Invalid segment arrays: " + vid)
            expanded = []
            previous = -1
            for pair, span in zip(ranges, seconds):
                if (not isinstance(pair, list) or len(pair) != 2 or any(type(f) is not int for f in pair)
                        or not previous < pair[0] <= pair[1] < len(ts)):
                    raise ValueError("Invalid inclusive frame range: " + vid)
                if (not isinstance(span, list) or len(span) != 2 or
                        any(type(s) not in (int, float) or not math.isfinite(s) for s in span)):
                    raise ValueError("Invalid segment timestamps: " + vid)
                end = float(ts[pair[1] + 1]) if pair[1] + 1 < len(ts) else float(row["duration_sec"])
                if abs(span[0] - float(ts[pair[0]])) > 1e-6 or abs(span[1] - end) > 1e-6 or span[1] <= span[0]:
                    raise ValueError("Seconds do not match original PTS: " + vid)
                expanded.extend(range(pair[0], pair[1] + 1))
                previous = pair[1]
            if expanded != frames:
                raise ValueError("Temporal handoff and highlight frame mask differ: " + vid)
            rh = ratio[1] / ratio[0]
            width = min(meta["width"], math.floor(meta["height"] / rh))
            if width <= 0:
                raise ValueError("Target aspect ratio cannot fit source: " + vid)
            box = [math.floor((meta["width"] - width) / 2), math.floor((meta["height"] - width * rh) / 2), width]
            rows.append({"video_id": row["video_id"], "targetRatioWH": ratio,
                         "predictions": [{"frame": f, "bboxes": box.copy()} for f in frames]})
    output = Path(output)
    api = common(source_root)
    api.export(output / "baseline", rows, rows, metadata, preserve_width=False)
    metadata["baseline_sha256"] = sha256(output / "baseline.zip")
    write_json(output / "manifest.json", metadata)
    report = {"schema_version": "spatial-input/v1", "ids": ids,
              "handoff_sha256": sha256(handoff), "highlight_frames_sha256": sha256(highlights),
              "timestamps_sha256": sha256(timestamps), "baseline_sha256": metadata["baseline_sha256"],
              "frames": sum(len(row["predictions"]) for row in rows), "temporal_mask_unchanged": True,
              "rule": "exact original frame mask; centered boxes are explicit initial spatial input"}
    write_json(output / "adapter.json", report)
    return report


def validate_result(source_root, baseline, manifest, result, ids):
    api = common(source_root)
    original = [row for row in api.load_rows(baseline) if str(row["video_id"]) in set(ids)]
    rows = api.load_rows(result)
    report = api.validate(rows, original, read_json(manifest), preserve_width=False)
    archive = Path(result).with_suffix(".zip")
    if archive.is_file():
        if api.load_rows(archive) != rows:
            raise ValueError("Submission ZIP and JSONL differ")
        with zipfile.ZipFile(archive) as stream:
            if stream.read("predictions.jsonl") != Path(result).read_bytes():
                raise ValueError("Submission ZIP payload bytes differ from JSONL")
    else:
        raise ValueError("Missing submission ZIP")
    report.update(invalid_count=0, jsonl_sha256=sha256(result), submission_zip_sha256=sha256(archive))
    return report
