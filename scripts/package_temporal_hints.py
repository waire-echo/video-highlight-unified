#!/usr/bin/env python3
"""Package per-video temporal debug records into the Stage-1 ZIP format."""

import argparse
import json
import os
import re
import zipfile
from pathlib import Path


ANIMAL_WORDS = {
    "animal",
    "cat",
    "kitten",
    "dog",
    "puppy",
    "cow",
    "pig",
    "horse",
    "bird",
    "rabbit",
}
PERSON_WORDS = {
    "person",
    "man",
    "woman",
    "boy",
    "girl",
    "child",
    "athlete",
    "player",
    "runner",
    "hiker",
    "dancer",
    "singer",
}
VEHICLE_WORDS = {
    "vehicle",
    "car",
    "truck",
    "bus",
    "motorcycle",
    "motorbike",
    "bike",
    "bicycle",
}
EQUIPMENT_WORDS = {
    "racket",
    "bat",
    "skateboard",
    "ski",
    "skis",
    "snowboard",
    "surfboard",
}


def infer_main_subject(subject_hint: str) -> str:
    text = str(subject_hint or "").lower()
    words = set(re.findall(r"[a-z]+", text))
    if "sports ball" in text or "ball" in words:
        return "sports ball"
    if words & PERSON_WORDS:
        return "person"
    if words & ANIMAL_WORDS:
        return "animal"
    if words & VEHICLE_WORDS:
        return "vehicle"
    if words & EQUIPMENT_WORDS:
        return "sports equipment"
    return "other"


def _clean_segments(value) -> list[list[float]]:
    clean = []
    if not isinstance(value, list):
        return clean
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        try:
            start, end = float(item[0]), float(item[1])
        except (TypeError, ValueError):
            continue
        if 0.0 <= start < end:
            clean.append([start, end])
    return clean


def package_handoff_records(records, output_zip: Path) -> dict:
    """Write ``temporal-handoff/v1`` rows in the legacy per-video ZIP shape.

    Invalid handoff rows become explicit empty placeholders.  The ZIP is
    replaced atomically so an interrupted checkpoint never leaves a partially
    readable archive for the downstream team.
    """

    output_zip = Path(output_zip)
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(str(output_zip) + ".partial")
    valid_count = 0
    placeholder_ids = []
    try:
        with zipfile.ZipFile(
            partial, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as archive:
            for raw in records:
                if not isinstance(raw, dict):
                    continue
                video_id = str(raw.get("video_id", "")).strip()
                if not video_id:
                    continue
                is_valid = bool(
                    raw.get("schema_version") == "temporal-handoff/v1"
                    and raw.get("valid")
                )
                segments = _clean_segments(raw.get("segments_sec")) if is_valid else []
                subject_hint = str(raw.get("subject_hint") or "").strip() if is_valid else ""
                if is_valid and segments:
                    valid_count += 1
                else:
                    placeholder_ids.append(video_id)
                    segments = []
                    subject_hint = ""
                record = {
                    "segments": segments,
                    "subject_hint": subject_hint,
                    "main_subject": infer_main_subject(subject_hint),
                    "confidence": 0.0,
                }
                encoded = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
                archive.writestr(f"{video_id}.json", encoded.encode("utf-8"))
        os.replace(partial, output_zip)
    finally:
        if partial.exists():
            partial.unlink()
    return {
        "output": str(output_zip.resolve()),
        "entries": valid_count + len(placeholder_ids),
        "valid_entries": valid_count,
        "placeholder_count": len(placeholder_ids),
        "placeholder_video_ids": placeholder_ids,
        "confidence_note": "0.0 means not calibrated; it is not a model probability",
    }


def package_records(
    item_dir: Path,
    output_zip: Path,
    start_video_id: int = 0,
    end_video_id: int = 173,
) -> dict:
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    valid_count = 0
    placeholder_ids = []
    with zipfile.ZipFile(
        output_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for video_id in range(start_video_id, end_video_id + 1):
            raw_path = item_dir / f"{video_id}.jsonl.raw.jsonl"
            raw = None
            if raw_path.exists():
                try:
                    raw = json.loads(raw_path.read_text(encoding="utf-8-sig"))
                except (OSError, UnicodeError, json.JSONDecodeError):
                    raw = None

            is_valid = bool(
                isinstance(raw, dict)
                and raw.get("valid")
                and raw.get("source") in {"vlm", "vlm_retry"}
                and not raw.get("fallback_used")
            )
            segments = _clean_segments(raw.get("temporal_segments_sec")) if is_valid else []
            subject_hint = str(raw.get("subject_hint") or "").strip() if is_valid else ""
            if is_valid and segments:
                valid_count += 1
            else:
                placeholder_ids.append(str(video_id))
                segments = []
                subject_hint = ""

            record = {
                "segments": segments,
                "subject_hint": subject_hint,
                "main_subject": infer_main_subject(subject_hint),
                # Qwen confidence is not calibrated in this pipeline. Keep the
                # legacy field without inventing a probability.
                "confidence": 0.0,
            }
            encoded = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
            archive.writestr(f"{video_id}.json", encoded.encode("utf-8"))

    summary = {
        "output": str(output_zip.resolve()),
        "entries": end_video_id - start_video_id + 1,
        "valid_vlm_entries": valid_count,
        "placeholder_count": len(placeholder_ids),
        "placeholder_video_ids": placeholder_ids,
        "confidence_note": "0.0 means not calibrated; it is not a model probability",
    }
    summary_path = Path(str(output_zip) + ".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--item-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--start-video-id", type=int, default=0)
    parser.add_argument("--end-video-id", type=int, default=173)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = package_records(
        args.item_dir,
        args.out,
        start_video_id=args.start_video_id,
        end_video_id=args.end_video_id,
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
