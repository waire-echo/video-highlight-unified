#!/usr/bin/env python3
"""Compare base and highlights-only adapter on frozen validation windows.

Every raw model response is retained. Strict JSON validity and normalized
validity after an exact empty ``<think>...</think>`` prefix are reported
separately. Metrics are calculated only from a validated highlights object.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict
from decimal import Decimal
import json
from pathlib import Path
import re
import time
import traceback
from typing import Any


EMPTY_THINK = re.compile(r"\A\s*<think>\s*</think>\s*", re.IGNORECASE)


def interval_union(intervals: list[list[float]]) -> list[list[float]]:
    result: list[list[float]] = []
    for start, end in sorted(intervals):
        if result and start <= result[-1][1]:
            result[-1][1] = max(end, result[-1][1])
        else:
            result.append([start, end])
    return result


def temporal_metrics(
    predicted: list[dict[str, float]], reference: list[dict[str, float]]
) -> dict[str, Any]:
    pred = interval_union([[item["start"], item["end"]] for item in predicted])
    ref = interval_union([[item["start"], item["end"]] for item in reference])
    pred_duration = sum(end - start for start, end in pred)
    ref_duration = sum(end - start for start, end in ref)
    intersection = sum(
        max(0.0, min(a_end, b_end) - max(a_start, b_start))
        for a_start, a_end in pred
        for b_start, b_end in ref
    )
    union = pred_duration + ref_duration - intersection
    return {
        "temporal_iou": intersection / union if union else 1.0,
        "false_retain_duration_fraction": (
            (pred_duration - intersection) / pred_duration
            if pred_duration
            else None
        ),
        "recall": intersection / ref_duration if ref_duration else 1.0,
        "predicted_duration_sec": pred_duration,
        "reference_duration_sec": ref_duration,
        "intersection_duration_sec": intersection,
    }


def window_duration(metadata: dict[str, Any]) -> Decimal:
    end = Decimal(str(metadata["window_end_sec"]))
    start = Decimal(str(metadata["window_start_sec"]))
    if end <= start:
        raise ValueError("Invalid window endpoints")
    return end - start


def strip_exact_empty_think(content: str) -> tuple[str, bool]:
    match = EMPTY_THINK.match(content)
    if match is None:
        return content, False
    return content[match.end():], True


def validate_highlights(content: str, duration: Decimal) -> list[dict[str, float]]:
    parsed = json.loads(content)
    if not isinstance(parsed, dict) or set(parsed) != {"highlights"}:
        raise ValueError("Root object must contain only highlights")
    highlights = parsed["highlights"]
    if not isinstance(highlights, list):
        raise ValueError("highlights must be an array")
    intervals: list[list[float]] = []
    for index, item in enumerate(highlights):
        if not isinstance(item, dict) or set(item) != {"start", "end"}:
            raise ValueError(f"highlights[{index}] must contain only start/end")
        start = item["start"]
        end = item["end"]
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, (int, float))
            or not isinstance(end, (int, float))
        ):
            raise ValueError(f"highlights[{index}] start/end must be numbers")
        dec_start = Decimal(str(start))
        dec_end = Decimal(str(end))
        if not dec_start.is_finite() or not dec_end.is_finite():
            raise ValueError(f"highlights[{index}] must be finite")
        if dec_start < 0 or dec_start >= dec_end or dec_end > duration:
            raise ValueError(f"highlights[{index}] is outside the window")
        intervals.append([float(dec_start), float(dec_end)])
    return [
        {"start": start, "end": end}
        for start, end in interval_union(intervals)
    ]


def validate_generation(
    content: str, finish_reason: str, duration: Decimal
) -> dict[str, Any]:
    if finish_reason != "stop":
        raise ValueError(f"Generation did not finish normally: {finish_reason}")
    normalized, think_removed = strip_exact_empty_think(content)
    highlights = validate_highlights(normalized, duration)
    return {
        "highlights": highlights,
        "stripped_empty_think": think_removed,
        "normalized_json_valid": True,
    }


def reference_from_row(row: dict[str, Any]) -> list[dict[str, float]]:
    answer = json.loads(row["messages"][-1]["content"])
    return [
        {"start": float(item["start"]), "end": float(item["end"])}
        for item in answer["highlights"]
    ]


def source_intervals(
    record: dict[str, Any], *, prediction: bool
) -> list[list[float]]:
    metadata = record["window"]
    offset = float(metadata["window_start_sec"])
    if prediction:
        intervals = record.get("highlights") or []
    else:
        intervals = record.get("reference_highlights") or []
    return [
        [offset + float(item["start"]), offset + float(item["end"])]
        for item in intervals
    ]


def materialized_reference(metadata: dict[str, Any], source_video: str) -> str:
    """Resolve a training-relative media path through the manifest media table."""

    media = metadata.get("materialized_media")
    if isinstance(media, dict) and media.get("path"):
        return str(media["path"])

    source_name = Path(source_video).name
    candidates = []
    for item in (metadata.get("media") or {}).values():
        if not isinstance(item, dict) or not item.get("path"):
            continue
        candidates.append(item)
    exact = [
        item
        for item in candidates
        if Path(source_video) == Path(item["path"])
        or source_name == Path(item["path"]).name
    ]
    if len(exact) == 1:
        return str(exact[0]["path"])
    return str(source_video)


def row_window_metadata(row: dict[str, Any]) -> dict[str, Any]:
    """Return window metadata without inventing missing time information."""

    for key in ("window_metadata", "window"):
        metadata = row.get(key)
        if isinstance(metadata, dict):
            return metadata
    metadata = {
        key: row[key]
        for key in (
            "window_start_sec",
            "window_end_sec",
            "group",
            "source_video",
            "window_index",
            "clip_audit",
            "media",
        )
        if key in row
    }
    required = {"window_start_sec", "window_end_sec", "group"}
    if not required.issubset(metadata):
        missing = sorted(required - set(metadata))
        raise ValueError(
            f"Window metadata lacks required fields {missing}; "
            "refusing to guess timing"
        )
    return metadata


def attach_manifest_metadata(
    rows: list[dict[str, Any]], metadata_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Attach only row-aligned training metadata after exact consistency checks."""

    if len(rows) != len(metadata_rows):
        raise ValueError("Dataset and manifest row counts differ")
    result = []
    for index, (row, metadata) in enumerate(
        zip(rows, metadata_rows, strict=True), 1
    ):
        if not isinstance(metadata, dict):
            raise ValueError(f"Manifest row {index} is not an object")
        required = {
            "group",
            "window_start_sec",
            "window_end_sec",
            "window_highlights",
            "window_video",
        }
        if not required.issubset(metadata):
            raise ValueError(f"Manifest row {index} lacks timing metadata")
        row_video = Path(row["videos"][0]).name
        metadata_video = Path(metadata["window_video"]).name
        if row_video != metadata_video:
            raise ValueError(
                f"Manifest row {index} video names differ: "
                f"{row_video!r} != {metadata_video!r}"
            )
        expected_answer = json.dumps(
            {"highlights": metadata["window_highlights"]},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        if expected_answer != row["messages"][-1]["content"]:
            raise ValueError(
                f"Manifest row {index} answer and window metadata differ"
            )
        attached = deepcopy(row)
        attached["window_metadata"] = deepcopy(metadata)
        result.append(attached)
    return result


def evaluate(
    *,
    model_path: str,
    adapter: Path,
    base_adapter: Path | None,
    dataset: Path,
    output: Path,
    max_samples: int,
    expect_gpus: int,
    manifest: Path | None = None,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    predictions_path = output / "predictions.jsonl"
    import os

    os.environ.update(
        FPS="2",
        FPS_MAX_FRAMES="64",
        VIDEO_MAX_TOKEN_NUM="196",
        VIDEO_MIN_TOKEN_NUM="128",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
    )
    import torch
    from peft import PeftModel, get_peft_model_state_dict
    from safetensors.torch import load_file
    from swift.infer_engine import InferRequest, RequestConfig, TransformersEngine
    from swift.model import get_model_processor
    from swift.template import get_template

    if torch.cuda.device_count() != expect_gpus:
        raise ValueError(
            f"Expected {expect_gpus} visible GPUs, found {torch.cuda.device_count()}"
        )
    model, processor = get_model_processor(
        model_path,
        model_type="qwen3_5",
        download_model=False,
        torch_dtype=torch.bfloat16,
        device_map="balanced",
        attn_impl="sdpa",
        model_kwargs={"max_memory": {index: "8GiB" for index in range(expect_gpus)}},
    )
    model = PeftModel.from_pretrained(
        model,
        str(base_adapter or adapter),
        adapter_name="base",
        is_trainable=False,
    )
    if base_adapter is not None:
        model.load_adapter(
            str(adapter),
            adapter_name="candidate",
            is_trainable=False,
        )
        model.set_adapter("candidate")
    model.eval()
    adapter_reloads = {}
    for name in model.peft_config:
        saved = load_file(
            str((base_adapter if name == "base" else adapter) / "adapter_model.safetensors")
        )
        actual = get_peft_model_state_dict(model, adapter_name=name)
        if set(saved) != set(actual) or not all(
            torch.equal(actual[key].detach().cpu().to(saved[key].dtype), saved[key])
            for key in saved
        ):
            raise RuntimeError(f"Exact adapter reload verification failed: {name}")
        adapter_reloads[name] = len(saved)
    if base_adapter is not None and set(adapter_reloads) != {"base", "candidate"}:
        raise RuntimeError(f"Unexpected adapter reload set: {adapter_reloads}")

    engine = TransformersEngine(
        model,
        template=get_template(
            processor, template_type="qwen3_5", enable_thinking=False
        ),
    )
    rows = [
        json.loads(line)
        for line in dataset.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ][:max_samples]
    manifest_binding = None
    if manifest is not None:
        manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
        split = "val"
        metadata_rows = manifest_data["samples"][split]
        if max_samples != len(rows) or len(rows) != len(metadata_rows):
            raise ValueError(
                "Manifest mode requires all validation rows; refusing a prefix"
            )
        rows = attach_manifest_metadata(rows, metadata_rows)
        entry = manifest_data["source_files"]["val_windows"]
        dataset_sha256 = __import__("hashlib").sha256(
            dataset.read_bytes()
        ).hexdigest()
        if dataset_sha256 != entry.get("sha256"):
            raise ValueError("Validation dataset SHA256 does not match manifest")
        manifest_binding = {
            "path": str(manifest),
            "sha256": __import__("hashlib").sha256(
                manifest.read_bytes()
            ).hexdigest(),
            "dataset_sha256": dataset_sha256,
            "rows": len(rows),
        }
    records: list[dict[str, Any]] = []
    with predictions_path.open("x", encoding="utf-8", newline="\n") as handle:
        for row_index, item in enumerate(rows, 1):
            metadata = row_window_metadata(item)
            duration = window_duration(metadata)
            reference = reference_from_row(item)
            messages = [deepcopy(message) for message in item["messages"] if message["role"] != "assistant"]
            video_path = Path(
                materialized_reference(metadata, item["videos"][0])
            )
            videos = (
                [str(video_path)]
                if video_path.is_file()
                else [str(Path(metadata["source_video"]))]
            )
            if not Path(videos[0]).is_file():
                raise FileNotFoundError(
                    f"Row {row_index} has no materialized window or source video"
                )
            for variant in ("base", "adapter"):
                started = time.monotonic()
                record: dict[str, Any] = {
                    "row": row_index,
                    "variant": variant,
                    "window": metadata,
                    "reference_highlights": reference,
                    "teacher_answer_passed_to_model": False,
                    "raw_json_valid": False,
                    "normalized_json_valid": False,
                }
                try:
                    if variant == "base":
                        context = (
                            model.disable_adapter()
                            if base_adapter is None
                            else nullcontext(model.set_adapter("base"))
                        )
                    else:
                        context = (
                            nullcontext()
                            if base_adapter is None
                            else nullcontext(model.set_adapter("candidate"))
                        )
                    with context:
                        response = engine.infer(
                            [
                                InferRequest(
                                    messages=deepcopy(messages),
                                    videos=videos,
                                    chat_template_kwargs={"enable_thinking": False},
                                )
                            ],
                            RequestConfig(
                                max_tokens=512,
                                temperature=0,
                                seed=42,
                                stream=False,
                            ),
                        )[0]
                    record["raw_response"] = asdict(response)
                    choice = response.choices[0]
                    raw_content = choice.message.content
                    try:
                        validate_highlights(raw_content, duration)
                        record["raw_json_valid"] = True
                    except (ValueError, TypeError, json.JSONDecodeError) as exc:
                        record["raw_validation_error"] = str(exc)
                    try:
                        validated = validate_generation(
                            raw_content, choice.finish_reason, duration
                        )
                        record.update(validated)
                        record["reference_metrics"] = temporal_metrics(
                            validated["highlights"], reference
                        )
                    except (ValueError, TypeError, json.JSONDecodeError) as exc:
                        record["normalized_validation_error"] = str(exc)
                except Exception:
                    record["generation_error"] = traceback.format_exc()
                record["seconds"] = time.monotonic() - started
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                records.append(record)
                print(
                    json.dumps(
                        {
                            "row": row_index,
                            "variant": variant,
                            "raw_json_valid": record["raw_json_valid"],
                            "normalized_json_valid": record["normalized_json_valid"],
                        }
                    ),
                    flush=True,
                )
    summary: dict[str, Any] = {
        "evaluation_scope": (
            "frozen validation windows; proxy temporal metrics on supplied validation inputs"
        ),
        "selected_windows": len(rows),
        "exact_adapter_tensor_matches": adapter_reloads,
        "teacher_answer_passed_to_model": False,
        "base_generation": (
            "fresh base model with adapter disabled"
            if base_adapter is None
            else "first-round parent adapter fixed and made read-only"
        ),
        "raw_and_normalized_validity": "reported separately",
        "device_map": getattr(model, "hf_device_map", None),
        "manifest_binding": manifest_binding,
    }
    for variant in ("base", "adapter"):
        selected = [record for record in records if record["variant"] == variant]
        normalized_valid = [
            record for record in selected if record["normalized_json_valid"]
        ]
        metric_records = [
            record["reference_metrics"]
            for record in normalized_valid
        ]
        summary[variant] = {
            "requests": len(selected),
            "raw_json_valid": sum(record["raw_json_valid"] for record in selected),
            "normalized_json_valid": len(normalized_valid),
            "empty_think_prefix_stripped": sum(
                record.get("stripped_empty_think", False) for record in selected
            ),
            "generation_errors": sum(
                "generation_error" in record for record in selected
            ),
            "mean_temporal_iou_on_normalized_valid": (
                sum(
                    record["reference_metrics"]["temporal_iou"]
                    for record in normalized_valid
                )
                / len(normalized_valid)
                if normalized_valid
                else None
            ),
            "mean_recall_on_normalized_valid": (
                sum(
                    record["reference_metrics"]["recall"]
                    for record in normalized_valid
                )
                / len(normalized_valid)
                if normalized_valid
                else None
            ),
            "empty_highlight_answers": sum(
                not record.get("highlights") for record in normalized_valid
            ),
            "all_request_mean_temporal_iou_invalid_as_zero": (
                sum(metrics["temporal_iou"] for metrics in metric_records)
                / len(selected)
                if selected
                else None
            ),
            "all_request_mean_recall_invalid_as_zero": (
                sum(metrics["recall"] for metrics in metric_records)
                / len(selected)
                if selected
                else None
            ),
            "all_request_mean_false_retain_duration_fraction_invalid_as_zero": (
                sum(
                    metrics["false_retain_duration_fraction"]
                    if metrics["false_retain_duration_fraction"] is not None
                    else 0.0
                    for metrics in metric_records
                )
                / len(selected)
                if selected
                else None
            ),
        }
    summary["engineering_evaluation_completed"] = all(
        summary[variant]["generation_errors"] == 0
        for variant in ("base", "adapter")
    )
    summary["adapter_normalized_contract_all_selected_passed"] = (
        summary["adapter"]["normalized_json_valid"] == len(rows)
    )
    summary["adapter_raw_contract_all_selected_passed"] = (
        summary["adapter"]["raw_json_valid"] == len(rows)
    )
    metric_policy = (
        "all selected requests are in the denominator; invalid output, "
        "generation error, and empty prediction contribute zero to tIoU "
        "and recall rather than being dropped"
    )
    summary["metric_policy"] = metric_policy
    grouped_records: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for record in records:
        group = str(record["window"]["group"])
        grouped_records.setdefault(group, {}).setdefault(
            record["variant"], []
        ).append(record)
    grouped_summary = {}
    for group, variants in sorted(grouped_records.items()):
        grouped_summary[group] = {"windows": len(variants["base"])}
        for variant, selected in variants.items():
            metric_records = [
                record["reference_metrics"]
                for record in selected
                if record["normalized_json_valid"]
            ]
            grouped_summary[group][variant] = {
                "requests": len(selected),
                "normalized_json_valid": len(metric_records),
                "sum_temporal_iou_invalid_as_zero": sum(
                    metrics["temporal_iou"] for metrics in metric_records
                ),
                "sum_recall_invalid_as_zero": sum(
                    metrics["recall"] for metrics in metric_records
                ),
            }
    summary["grouped_source_summary"] = grouped_summary
    summary["source_group_mean_temporal_iou_invalid_as_zero"] = {}
    for variant in ("base", "adapter"):
        group_means = [
            value[variant]["sum_temporal_iou_invalid_as_zero"]
            / value["windows"]
            for value in grouped_summary.values()
        ]
        summary["source_group_mean_temporal_iou_invalid_as_zero"][variant] = (
            sum(group_means) / len(group_means) if group_means else None
        )
    merged_source_summary = {}
    for group, variants in sorted(grouped_records.items()):
        merged_source_summary[group] = {}
        for variant in ("base", "adapter"):
            selected = variants[variant]
            predicted = interval_union(
                [
                    interval
                    for record in selected
                    for interval in source_intervals(record, prediction=True)
                ]
            )
            reference = interval_union(
                [
                    interval
                    for record in selected
                    for interval in source_intervals(record, prediction=False)
                ]
            )
            metrics = temporal_metrics(
                [{"start": start, "end": end} for start, end in predicted],
                [{"start": start, "end": end} for start, end in reference],
            )
            merged_source_summary[group][variant] = metrics
    summary["merged_source_video_summary"] = merged_source_summary
    summary["source_video_mean_temporal_iou_invalid_as_zero"] = {}
    for variant in ("base", "adapter"):
        values = [
            metrics[variant]["temporal_iou"]
            for metrics in merged_source_summary.values()
        ]
        summary["source_video_mean_temporal_iou_invalid_as_zero"][variant] = (
            sum(values) / len(values) if values else None
        )
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-adapter", type=Path)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--expect-gpus", type=int, default=2)
    parser.add_argument(
        "--manifest",
        type=Path,
        help=(
            "Training manifest used to restore row-aligned window timing; "
            "requires the complete validation split"
        ),
    )
    args = parser.parse_args()
    summary = evaluate(
        model_path=args.model,
        adapter=args.adapter,
        base_adapter=args.base_adapter,
        dataset=args.dataset,
        output=args.output,
        max_samples=args.max_samples,
        expect_gpus=args.expect_gpus,
        manifest=args.manifest,
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
