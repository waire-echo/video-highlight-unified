#!/usr/bin/env python3
"""Repair strict highlight merge failures from recorded responses only."""

from __future__ import annotations

import argparse
from decimal import Decimal
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any

from video_highlight.highlight_output import (
    POLICY,
    lenient_normalized_answer,
)
from video_highlight.submission_recovery import recover_intervals, POLICY as RECOVERY_POLICY


EMPTY_THINK = re.compile(r"\A\s*<think>\s*</think>\s*", re.IGNORECASE)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def interval_union(intervals: list[list[float]]) -> list[list[float]]:
    ordered = sorted(
        (
            [float(start), float(end)]
            for start, end in intervals
            if Decimal(str(end)) > Decimal(str(start))
        ),
        key=lambda item: (item[0], item[1]),
    )
    merged: list[list[float]] = []
    for start, end in ordered:
        if not merged or Decimal(str(start)) > Decimal(str(merged[-1][1])):
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return merged


def response_content(record: dict[str, Any]) -> str | None:
    response = record.get("raw_response")
    if not isinstance(response, dict):
        return None
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    message = choices[0].get("message")
    if not isinstance(message, dict):
        return None
    content = message.get("content")
    return content if isinstance(content, str) and content.strip() else None


def map_to_source(
    spans: list[list[float]],
    *,
    window_start: float,
    source_duration: float,
    intersect: bool = False,
) -> list[list[float]]:
    mapped = []
    duration = Decimal(str(source_duration))
    window_origin = Decimal(str(window_start))
    for start, end in spans:
        source_start = window_origin + Decimal(str(start))
        source_end = window_origin + Decimal(str(end))
        if intersect:
            source_start = max(Decimal(0), source_start)
            source_end = min(duration, source_end)
            if source_start >= source_end:
                continue
        if source_end > duration and source_end - duration <= Decimal(
            "0.000000001"
        ):
            source_end = duration
        if source_start < 0 or source_end > duration:
            raise ValueError(
                "Mapped highlight outside source video: "
                f"{source_start}-{source_end}, duration={source_duration}"
            )
        mapped.append([float(source_start), float(source_end)])
    return mapped


def merge(args: argparse.Namespace) -> dict[str, Any]:
    intersection = getattr(args, "recovery_policy", "diagnostic") == "intersection"
    policy = RECOVERY_POLICY if intersection else POLICY
    strict_dir = args.strict_merged.resolve()
    output = args.output.resolve()
    if output == strict_dir:
        raise ValueError("Output must differ from the strict merge directory")

    highlight_rows = read_jsonl(strict_dir / "highlights.jsonl")
    window_rows = read_jsonl(strict_dir / "windows.jsonl")
    failures = read_jsonl(strict_dir / "failures.jsonl")
    windows = {str(row["window_id"]): row for row in window_rows}
    highlights = {str(row["video_id"]): row for row in highlight_rows}
    if len(windows) != len(window_rows):
        raise ValueError("Duplicate window IDs in strict windows.jsonl")
    if len(highlights) != len(highlight_rows):
        raise ValueError("Duplicate video IDs in strict highlights.jsonl")

    repairs = []
    unresolved = []
    repaired_windows = 0
    for failure in failures:
        window_id = str(failure["window_id"])
        window = windows.get(window_id)
        if window is None:
            raise ValueError(f"Missing window in strict merge: {window_id}")
        record_path = Path(str(failure["record_path"]))
        audit: dict[str, Any] = {
            "window_id": window_id,
            "source_video_id": str(failure["video_id"]),
            "policy": policy,
            "strict_error": failure.get("error"),
            "record_path": str(record_path),
            "accepted": False,
        }
        if not record_path.is_file():
            audit["error"] = "Record file is missing"
            repairs.append(audit)
            unresolved.append({**failure, "lenient_error": audit["error"]})
            continue

        record = read_json(record_path)
        audit["strict_record_identity"] = {
            "source_video_id": record.get("source_video_id"),
            "window_start_sec": record.get("window_start_sec"),
            "window_end_sec": record.get("window_end_sec"),
            "window_duration_sec": record.get("window_duration_sec"),
        }
        raw = response_content(record)
        if raw is None:
            audit["error"] = "No model response content; leniency cannot recover it"
            repairs.append(audit)
            unresolved.append({**failure, "lenient_error": audit["error"]})
            continue

        body = EMPTY_THINK.sub("", raw)
        try:
            if intersection:
                obj, spans, normalization = recover_intervals(body, record["window_duration_sec"])
            else:
                obj, spans, normalization = lenient_normalized_answer(
                    body, record["window_duration_sec"], lenient=True,
                    end_tolerance_sec=args.end_tolerance_sec,
                )
            if intersection and record.get("finish_reason") in ("length", "max_tokens", "error"):
                if not spans:
                    raise ValueError("Truncated/error generation cannot establish a no-highlights judgment")
                normalization.update(status="partial", finish_reason=record["finish_reason"])
            source_spans = map_to_source(
                spans,
                window_start=float(record["window_start_sec"]),
                source_duration=float(record["source_duration_sec"]),
                intersect=intersection,
            )
        except Exception as error:
            audit["error"] = f"{type(error).__name__}: {error}"
            repairs.append(audit)
            unresolved.append({**failure, "lenient_error": audit["error"]})
            continue

        source_id = str(record["source_video_id"])
        if source_id not in highlights:
            raise ValueError(f"Unknown source video ID: {source_id}")
        highlights[source_id]["highlights"].extend(
            {"start": start, "end": end} for start, end in source_spans
        )
        window.update(
            raw_json_valid=True,
            strict_raw_json_valid=False,
            accepted_by_lenient=True,
            lenient_policy=policy,
            lenient_window_highlights=spans,
            source_highlights=source_spans,
            recovery_status=normalization.get("status", "repaired"),
        )
        audit.update(
            accepted=True,
            raw_response=raw,
            lenient_object=obj,
            lenient_window_highlights=spans,
            source_highlights=source_spans,
            normalization=normalization,
        )
        repairs.append(audit)
        repaired_windows += 1

    final_highlights = []
    for row in highlight_rows:
        source_id = str(row["video_id"])
        intervals = interval_union(
            [
                [float(item["start"]), float(item["end"])]
                for item in highlights[source_id]["highlights"]
            ]
        )
        duration = Decimal(str(row["duration"]))
        for start, end in intervals:
            if (
                Decimal(str(start)) < 0
                or Decimal(str(start)) >= Decimal(str(end))
                or Decimal(str(end)) > duration
            ):
                raise ValueError(
                    f"Final interval outside source {source_id}: "
                    f"{start}-{end}, duration={duration}"
                )
        final_highlights.append(
            {
                "video_id": source_id,
                "duration": row["duration"],
                "highlights": [
                    {"start": start, "end": end} for start, end in intervals
                ],
            }
        )

    accepted_windows = len(window_rows) - len(unresolved)
    strict_valid_windows = len(window_rows) - len(failures)
    summary = {
        "schema": "video-ora-highlights-lenient-diagnostic/v1",
        "state": (
            "complete_with_failures" if unresolved else "complete"
        ),
        "policy": policy,
        "diagnostic_only": True,
        "strict_merged_input": str(strict_dir),
        "strict_inputs_sha256": {name: file_sha256(strict_dir / name) for name in
                                  ("summary.json", "windows.jsonl", "failures.jsonl", "highlights.jsonl")
                                  if (strict_dir / name).is_file()},
        "end_tolerance_sec": args.end_tolerance_sec,
        "source_videos": len(final_highlights),
        "total_windows": len(window_rows),
        "strict_valid_windows_preserved": strict_valid_windows,
        "strict_failed_windows_reprocessed": len(failures),
        "lenient_repaired_windows": repaired_windows,
        "valid_windows_after_lenient": accepted_windows,
        "unresolved_windows": len(unresolved),
        "source_videos_with_highlights": sum(
            bool(row["highlights"]) for row in final_highlights
        ),
        "total_merged_highlights": sum(
            len(row["highlights"]) for row in final_highlights
        ),
        "unresolved": unresolved,
    }

    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "highlights.jsonl", final_highlights)
    write_jsonl(output / "windows.jsonl", window_rows)
    write_jsonl(output / "failures.jsonl", unresolved)
    write_jsonl(output / "lenient_repairs.jsonl", repairs)
    summary["artifacts_sha256"] = {name: file_sha256(output / name) for name in
                                   ("highlights.jsonl", "windows.jsonl", "failures.jsonl", "lenient_repairs.jsonl")}
    write_json(output / "summary.json", summary)
    (output / "README.md").write_text(
        "# Video-ORA 高光结果：宽容诊断版\n\n"
        "本目录基于严格版已保存的模型回答做离线宽容修复，"
        "没有重新推理或训练。\n\n"
        f"- 策略：`{policy}`\n"
        f"- 片尾越界容差：`{args.end_tolerance_sec}` 秒\n"
        f"- 严格有效窗口：`{strict_valid_windows}`\n"
        f"- 本次修复窗口：`{repaired_windows}`\n"
        f"- 仍未解决窗口：`{len(unresolved)}`\n\n" +
        ("修复按显式策略取时间范围交集并隔离坏条目。" if intersection else
         "修复仅允许把片尾最多超出容差的结束时间裁到窗口末尾。") +
        "不会补写缺失回答，也不保证修复语义错误。每个窗口的原始回答、"
        "修复动作、局部时间和源视频时间均记录在 "
        "`lenient_repairs.jsonl`。\n",
        encoding="utf-8",
    )
    checksum_paths = (
        "README.md",
        "highlights.jsonl",
        "windows.jsonl",
        "failures.jsonl",
        "lenient_repairs.jsonl",
        "summary.json",
    )
    (output / "SHA256SUMS").write_text(
        "".join(
            f"{file_sha256(output / name)}  {name}\n"
            for name in checksum_paths
        ),
        encoding="ascii",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict-merged", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--recovery-policy", choices=("diagnostic", "intersection"), default="diagnostic")
    parser.add_argument(
        "--end-tolerance-sec",
        type=float,
        default=0.5,
    )
    args = parser.parse_args()
    if not 0 <= args.end_tolerance_sec <= 1:
        parser.error("--end-tolerance-sec must be between 0 and 1")
    print(json.dumps(merge(args), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
