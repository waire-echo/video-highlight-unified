"""Independently check both temporal exports and package the handoff files."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import zipfile


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def valid_event_interval(a, b, duration):
    # core.merge_segments clamps to duration before rounding to five decimals.
    # Accept that exact rounded endpoint without changing exported predictions.
    return (
        math.isfinite(a) and math.isfinite(b)
        and 0 <= a < b
        and (b <= duration + 1e-6 or b == round(duration, 5))
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    index = read(args.index)
    expected = {str(row["video_id"]): row["targetRatioWH"] for row in index}
    require(len(expected) == len(index), "Duplicate input IDs")
    manifest = read(run / "manifest.json")
    require(set(manifest["input_ids"]) == set(expected), "Manifest IDs differ from input")
    validation = {"videos": len(expected), "signature": manifest["signature"], "variants": {},
                  "event_endpoint_policy": "Original duration or its exact five-decimal rounded value from core.merge_segments"}
    for variant in ("grounded", "recall"):
        folder = run / ("export_" + variant)
        handoff = rows(folder / "temporal_handoff.jsonl")
        dense_rows = rows(folder / "highlight_frames.jsonl")
        for label, output in (("handoff", handoff), ("dense", dense_rows)):
            ids = [str(row["video_id"]) for row in output]
            require(len(ids) == len(expected) and set(ids) == set(expected), variant + " " + label + " IDs are incomplete/duplicated")
        dense = {str(row["video_id"]): row for row in dense_rows}
        total_frames = empty = 0
        for row in handoff:
            vid = str(row["video_id"])
            prefix = variant + " " + vid + ": "
            frame_row = dense[vid]
            record = read(run / "records" / (vid + ".json"))
            require(record["status"] == "complete" and record["signature"] == manifest["signature"], prefix + "stale record")
            require(row["targetRatioWH"] == frame_row["targetRatioWH"] == expected[vid], prefix + "ratio differs")
            require(row["valid"] is True and not row["errors"], prefix + "invalid handoff")
            require(row["timestamp_mapping"] == "original_frame_pts" and row["frame_index_convention"] == "inclusive" and frame_row["frame_index_base"] == 0, prefix + "unexpected frame convention")
            cache = read(run / "timestamps" / (vid + ".json"))
            times = cache["times"]
            count = row["frame_count"]
            require(cache["source_sha256"] == record["source_sha256"], prefix + "stale timestamps")
            require(type(count) is int and count == frame_row["frame_count"] == len(times) and count > 0, prefix + "invalid frame count")
            require(all(math.isfinite(t) for t in times) and all(a <= b for a, b in zip(times, times[1:])), prefix + "invalid timestamps")
            duration = row["duration_sec"]
            require(math.isfinite(duration) and duration > 0 and abs(duration - record["duration"]) < 1e-8, prefix + "invalid duration")
            ranges = row["segments_frame"]
            seconds = row["segments_sec"]
            require(len(ranges) == len(seconds), prefix + "interval counts differ")
            rebuilt = []
            last = -2
            for (a, b), (start, end) in zip(ranges, seconds):
                require(type(a) is int and type(b) is int and 0 <= a <= b < count and a > last + 1, prefix + "invalid/overlapping frame range")
                require(all(math.isfinite(t) for t in (start, end)) and 0 <= start <= end <= duration + 1e-6, prefix + "invalid time interval")
                expected_end = times[b + 1] if b + 1 < count else duration
                require(abs(start - times[a]) < 1e-8 and abs(end - expected_end) < 1e-8, prefix + "PTS mismatch")
                rebuilt.extend(range(a, b + 1))
                last = b
            frames = frame_row["highlight_frames"]
            require(all(type(frame) is int for frame in frames) and frames == rebuilt, prefix + "dense frames differ from ranges")
            for a, b in row["event_segments_sec"]:
                require(valid_event_interval(a, b, duration), prefix + "invalid event interval")
            total_frames += len(frames)
            empty += not frames
        report = read(folder / "report.json")
        require(report["videos"] == len(expected) and report["prediction_frames"] == total_frames and report["empty_videos"] == empty, variant + " report differs")
        validation["variants"][variant] = {"valid": True, "videos": len(expected), "prediction_frames": total_frames, "empty_videos": empty}
    validation["ratios"] = dict(Counter(":".join(map(str, ratio)) for ratio in expected.values()))
    report_path = run / "independent_validation.json"
    report_path.write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(validation), flush=True)


if __name__ == "__main__":
    main()
