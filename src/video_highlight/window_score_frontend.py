"""Experimental coarse-to-fine Qwen window-score temporal frontend.

The frontend is deliberately isolated from the existing TimeLens handoff and
from the YOLO/spatial backend.  It consumes an original local MP4, scores short
time ranges, adds inexpensive traditional signals, and returns the existing
``TemporalResult`` contract whose only spatially-relevant field is
``segments_sec``.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .io_utils import video_meta
from .score_curve import (
    build_score_curve,
    detect_tail_barrier,
    fallback_segments,
    overlay_score_curves,
    select_highlight_segments,
    select_multisignal_refinement_regions,
    smooth_score_curve,
)
from .temporal import TemporalResult, temporal_coverage_ratio
from .temporal_signals import (
    DEFAULT_SIGNAL_WEIGHTS,
    TRADITIONAL_SIGNAL_KEYS,
    TemporalSignalExtractor,
    apply_signal_calibration,
    fit_signal_calibration,
    fuse_window_scores,
)
from .window_scoring import (
    QWEN_SCORE_WEIGHTS,
    QwenWindowScorer,
    WindowScore,
    WindowScoreTemporalDetector,
)
from .windowing import WindowSpec, generate_windows


SegmentSeconds = Tuple[float, float]
SignalProvider = Callable[[str, WindowSpec], Dict[str, Any]]
YoloInteractionProvider = Callable[[str, WindowSpec], Optional[float]]


def _resource_error(errors: Iterable[str]) -> bool:
    text = " ".join(str(error).lower() for error in errors)
    return any(
        marker in text
        for marker in (
            "out of memory",
            "cuda oom",
            "model load failed",
            "vlm model is unavailable",
        )
    )


def _invalid_window_score(window: WindowSpec, reason: str) -> WindowScore:
    return WindowScore(
        window.window_id,
        window.start_sec,
        window.end_sec,
        window.sample_fps,
        valid=False,
        errors=[reason],
        attempts=0,
        stage=window.stage,
        parent_id=window.parent_id,
    )


class WindowScoreFrontend:
    """Orchestrate coarse Qwen scoring, fine rescoring, signals and selection."""

    def __init__(
        self,
        scorer: QwenWindowScorer,
        *,
        coarse_window_sec: float = 2.0,
        coarse_stride_sec: float = 1.0,
        coarse_fps: float = 2.0,
        fine_window_sec: float = 2.0,
        fine_stride_sec: float = 0.5,
        fine_fps: float = 4.0,
        max_fine_regions: int = 6,
        curve_step_sec: float = 0.5,
        smoothing_points: int = 3,
        relative_top_fraction: float = 0.30,
        absolute_threshold: float = 0.45,
        growth_threshold: float = 0.40,
        coarse_candidate_threshold: float = 0.45,
        merge_gap_sec: float = 1.0,
        boundary_expand_sec: float = 0.75,
        max_boundary_expand_sec: float = 1.5,
        max_segments: int = 12,
        max_coverage_ratio: float = 0.60,
        max_coverage_sec: float = 30.0,
        short_full_sec: float = 4.0,
        ffmpeg_bin: str = "ffmpeg",
        audio_sample_rate: int = 8000,
        signal_weights: Optional[Dict[str, float]] = None,
        signal_provider: Optional[SignalProvider] = None,
        yolo_interaction_provider: Optional[YoloInteractionProvider] = None,
    ):
        self.scorer = scorer
        self.coarse_window_sec = max(0.25, float(coarse_window_sec))
        self.coarse_stride_sec = max(0.1, float(coarse_stride_sec))
        self.coarse_fps = max(0.1, float(coarse_fps))
        self.fine_window_sec = max(0.25, float(fine_window_sec))
        self.fine_stride_sec = max(0.1, float(fine_stride_sec))
        self.fine_fps = max(0.1, float(fine_fps))
        self.max_fine_regions = max(0, int(max_fine_regions))
        self.curve_step_sec = max(0.1, float(curve_step_sec))
        self.smoothing_points = max(1, int(smoothing_points))
        self.relative_top_fraction = max(0.01, min(1.0, float(relative_top_fraction)))
        self.absolute_threshold = max(0.0, min(1.0, float(absolute_threshold)))
        self.growth_threshold = max(0.0, min(1.0, float(growth_threshold)))
        self.coarse_candidate_threshold = max(
            0.0, min(1.0, float(coarse_candidate_threshold))
        )
        self.merge_gap_sec = max(0.0, float(merge_gap_sec))
        self.boundary_expand_sec = max(0.0, float(boundary_expand_sec))
        self.max_boundary_expand_sec = max(0.0, float(max_boundary_expand_sec))
        self.max_segments = max(1, int(max_segments))
        self.max_coverage_ratio = max(0.0, min(1.0, float(max_coverage_ratio)))
        self.max_coverage_sec = max(0.0, float(max_coverage_sec))
        self.short_full_sec = max(0.0, float(short_full_sec))
        self.ffmpeg_bin = str(ffmpeg_bin or "ffmpeg")
        self.audio_sample_rate = max(1000, int(audio_sample_rate))
        self.signal_weights = dict(DEFAULT_SIGNAL_WEIGHTS)
        if signal_weights:
            self.signal_weights.update(signal_weights)
        self.signal_provider = signal_provider
        self.yolo_interaction_provider = yolo_interaction_provider

    def _score_windows(
        self, video_path: str, windows: Sequence[WindowSpec]
    ) -> Tuple[List[WindowScore], Optional[str]]:
        """Score a stage, using the scorer's GPU batch path when available."""

        batch_method = getattr(self.scorer, "score_windows", None)
        if callable(batch_method):
            try:
                batch_results = list(batch_method(video_path, windows))
            except Exception:
                batch_results = []
            if len(batch_results) == len(windows):
                stopped_reason = getattr(self.scorer, "fatal_disabled_reason", None)
                if not stopped_reason:
                    for result in batch_results:
                        if not result.valid and _resource_error(result.errors):
                            stopped_reason = (
                                "Qwen scoring disabled after resource/model failure: "
                                + "; ".join(result.errors)
                            )
                            break
                return batch_results, stopped_reason

        results: List[WindowScore] = []
        stopped_reason: Optional[str] = getattr(
            self.scorer, "fatal_disabled_reason", None
        )
        for index, window in enumerate(windows):
            if stopped_reason:
                results.append(_invalid_window_score(window, stopped_reason))
                continue
            # The scorer normally converts generation and parsing failures to
            # an invalid WindowScore itself.  Keep a second safety net here:
            # an injected provider, decoder, or future scorer implementation
            # may still raise unexpectedly, and one such window must not abort
            # the complete video.
            try:
                result = self.scorer.score_window(
                    video_path,
                    window.start_sec,
                    window.end_sec,
                    window.sample_fps,
                    window_id=window.window_id,
                    stage=window.stage,
                    parent_id=window.parent_id,
                )
            except Exception as exc:
                result = _invalid_window_score(
                    window,
                    f"window scorer exception: {type(exc).__name__}: {exc}",
                )
            results.append(result)
            if not result.valid and _resource_error(result.errors):
                stopped_reason = (
                    "Qwen scoring disabled for remaining windows after resource/model failure: "
                    + "; ".join(result.errors)
                )
        return results, stopped_reason

    def _signals_for_window(self, video_path: str, window: WindowSpec) -> Dict[str, Any]:
        if self.signal_provider is not None:
            try:
                payload = dict(self.signal_provider(video_path, window) or {})
            except Exception as exc:
                payload = {"signal_error": f"{type(exc).__name__}: {exc}"}
        else:
            payload = {}
        if self.yolo_interaction_provider is not None:
            try:
                payload["yolo_interaction_score"] = self.yolo_interaction_provider(
                    video_path, window
                )
            except Exception as exc:
                payload["yolo_interaction_score"] = None
                payload["yolo_interaction_error"] = f"{type(exc).__name__}: {exc}"
        else:
            payload.setdefault("yolo_interaction_score", None)
        return payload

    def _records_with_signals(
        self,
        video_path: str,
        windows: Sequence[WindowSpec],
        qwen_records: Sequence[WindowScore],
        signal_extractor: Optional[TemporalSignalExtractor] = None,
    ) -> List[Dict[str, Any]]:
        base_rows = [scored.to_dict() for scored in qwen_records]
        if self.signal_provider is None:
            extractor = signal_extractor or TemporalSignalExtractor(
                sample_fps=self.coarse_fps,
                ffmpeg_bin=self.ffmpeg_bin,
                audio_sample_rate=self.audio_sample_rate,
            )
            try:
                signal_rows = extractor.score_windows(video_path, windows)
            except Exception as exc:
                # A whole-stream decoder/FFmpeg failure is equivalent to all
                # traditional channels being missing.  Keep one diagnostic row
                # per window so Qwen-only fusion or the documented fallback can
                # still complete the video.
                signal_rows = [
                    {
                        "motion_score": None,
                        "audio_score": None,
                        "scene_score": None,
                        "yolo_interaction_score": None,
                        "signal_error": f"batch signal exception: {type(exc).__name__}: {exc}",
                    }
                    for _window in windows
                ]
            try:
                signal_rows = list(signal_rows or [])
            except Exception:
                signal_rows = []
            if len(signal_rows) != len(windows):
                signal_rows = list(signal_rows[: len(windows)]) + [
                    {
                        "motion_score": None,
                        "audio_score": None,
                        "scene_score": None,
                        "yolo_interaction_score": None,
                        "signal_error": "signal extractor returned an unexpected row count",
                    }
                    for _ in range(max(0, len(windows) - len(signal_rows)))
                ]
        else:
            signal_rows = [self._signals_for_window(video_path, window) for window in windows]

        records: List[Dict[str, Any]] = []
        for window, row, signals in zip(windows, base_rows, signal_rows):
            row.update(signals)
            if self.signal_provider is None and self.yolo_interaction_provider is not None:
                # The batch default extractor intentionally knows nothing
                # about YOLO, so attach the optional callback exactly once here.
                try:
                    row["yolo_interaction_score"] = self.yolo_interaction_provider(
                        video_path, window
                    )
                except Exception as exc:
                    row["yolo_interaction_score"] = None
                    row["yolo_interaction_error"] = f"{type(exc).__name__}: {exc}"
            # With a custom signal provider, ``_signals_for_window`` already
            # invoked the optional YOLO hook.  Avoid duplicate inference.
            row.setdefault("yolo_interaction_score", None)
            row["window_label"] = (
                ("C" if window.stage == "coarse" else "F")
                + f"{int(window.window_id) + 1:03d}"
            )
            records.append(row)
        return records

    def _fuse_records(
        self, records: Sequence[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        # Qwen is already divided by three into a fixed 0..1 rubric.  Only the
        # traditional channels need per-video scaling, and that scale is fitted
        # from coarse windows then reused unchanged for fine windows.
        rows = [dict(record) for record in records]
        qwen_values: List[float] = []
        for row in rows:
            if row.get("valid") is False or row.get("qwen_score") is None:
                continue
            try:
                value = float(row["qwen_score"])
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                qwen_values.append(max(0.0, min(1.0, value)))
        qwen_range = (
            max(qwen_values) - min(qwen_values) if qwen_values else None
        )
        qwen_degenerate = bool(
            qwen_values and qwen_range is not None and qwen_range <= 0.05 + 1e-12
        )

        coarse_reference = [row for row in rows if row.get("stage") == "coarse"] or rows
        calibration = fit_signal_calibration(
            coarse_reference,
            keys=TRADITIONAL_SIGNAL_KEYS,
            lower_percentile=10.0,
            upper_percentile=90.0,
        )
        normalized, normalization = apply_signal_calibration(
            rows,
            calibration,
            keys=TRADITIONAL_SIGNAL_KEYS,
        )
        disabled_constants = {
            key
            for key, detail in calibration.items()
            if isinstance(detail, dict) and detail.get("constant")
        }
        fused = fuse_window_scores(
            normalized,
            weights=self.signal_weights,
            normalize=False,
            disable_degenerate_qwen=False,
            disabled_signals=(
                disabled_constants | ({"qwen_score"} if qwen_degenerate else set())
            ),
        )
        for fused_row in fused:
            fused_row["qwen_disabled"] = qwen_degenerate
            diagnostics = dict(fused_row.get("score_diagnostics") or {})
            diagnostics.update(
                {
                    "qwen_degenerate": qwen_degenerate,
                    "qwen_range": qwen_range,
                    "traditional_calibration": calibration,
                }
            )
            fused_row["score_diagnostics"] = diagnostics

        stage_diagnostics: Dict[str, Any] = {}
        for stage in sorted({str(row.get("stage") or "coarse") for row in rows}):
            stage_rows = [row for row in rows if str(row.get("stage") or "coarse") == stage]
            qwen_present_count = sum(
                row.get("qwen_score") is not None and row.get("valid") is not False
                for row in stage_rows
            )
            stage_diagnostics[stage] = {
                "normalization": normalization,
                "qwen_present_count": qwen_present_count,
                "qwen_degenerate": qwen_degenerate,
                "disabled_constant_signals": sorted(disabled_constants),
            }

        return fused, {
            "qwen_normalization": {"method": "fixed_0_1", "range": qwen_range},
            "qwen_range": qwen_range,
            "traditional_calibration": calibration,
            "stages": stage_diagnostics,
            "qwen_degenerate": qwen_degenerate,
            "weights": dict(self.signal_weights),
        }

    def _fine_windows(
        self, regions: Sequence[SegmentSeconds], start_window_id: int
    ) -> List[WindowSpec]:
        windows: List[WindowSpec] = []
        next_id = int(start_window_id)
        for region_index, (region_start, region_end) in enumerate(regions):
            length = max(0.0, float(region_end) - float(region_start))
            local = generate_windows(
                length,
                self.fine_window_sec,
                self.fine_stride_sec,
                self.fine_fps,
                stage="fine",
                start_window_id=next_id,
                parent_id=region_index,
            )
            for item in local:
                windows.append(
                    WindowSpec(
                        item.window_id,
                        round(float(region_start) + item.start_sec, 6),
                        round(float(region_start) + item.end_sec, 6),
                        item.sample_fps,
                        "fine",
                        region_index,
                    )
                )
            if local:
                next_id = local[-1].window_id + 1
        return windows

    def _signal_curve(
        self,
        records: Sequence[Dict[str, Any]],
        duration_sec: float,
        key: str,
    ) -> List[Dict[str, Any]]:
        payload: List[Dict[str, Any]] = []
        for row in records:
            if key == "qwen_score" and row.get("valid") is False:
                continue
            try:
                value = float(row.get(key))
                start = float(row["start_sec"])
                end = float(row["end_sec"])
            except (KeyError, TypeError, ValueError):
                continue
            if not all(math.isfinite(item) for item in (value, start, end)):
                continue
            payload.append(
                {"start_sec": start, "end_sec": end, "fused_score": value}
            )
        return smooth_score_curve(
            build_score_curve(
                payload,
                duration_sec,
                grid_sec=self.curve_step_sec,
            ),
            window_points=self.smoothing_points,
        )

    @staticmethod
    def _canonical_subject_hint(subject: str, action: str) -> Optional[str]:
        """Map free Qwen wording to the exact aliases accepted by YOLO.

        The detailed subject/action remain in ``window_records``.  The handoff
        field itself must be a stable class hint; concatenating a sentence such
        as ``woman walking ...`` makes the existing selector treat it as an
        unsupported string.
        """

        def classify(text: str) -> Optional[str]:
            value = str(text or "").lower()
            tokens = set(re.findall(r"[a-z]+", value))
            if tokens & {"dog", "dogs", "puppy", "puppies", "canine"}:
                return "dog"
            if tokens & {"cat", "cats", "kitten", "kittens", "feline"}:
                return "cat"
            # A "basketball player" is a person, not a ball.  Prefer explicit
            # human nouns before sport/equipment words.
            if tokens & {
                "person", "people", "human", "man", "woman", "boy", "girl",
                "child", "children", "baby", "athlete", "player", "rider",
                "cyclist",
            }:
                return "person"
            if "sports ball" in value or tokens & {"ball", "football", "basketball", "soccer"}:
                return "ball"
            if tokens & {"bicycle", "bike", "cycling"}:
                return "bicycle"
            if tokens & {"motorcycle", "motorbike"}:
                return "motorcycle"
            if tokens & {"car", "automobile"}:
                return "car"
            if tokens & {"truck", "bus", "vehicle"}:
                return "vehicle"
            if tokens & {"animal", "horse", "bird", "pig"}:
                return "animal"
            return None

        return classify(subject) or classify(action)

    @staticmethod
    def _selected_semantics(
        records: Sequence[Dict[str, Any]], segments: Sequence[SegmentSeconds]
    ) -> Tuple[Optional[str], Optional[str]]:
        """Pick semantics only from the best valid selected-window overlap."""

        candidates: List[Tuple[float, float, str, str]] = []
        for row in records:
            if row.get("valid") is not True:
                continue
            try:
                start, end = float(row["start_sec"]), float(row["end_sec"])
            except (KeyError, TypeError, ValueError):
                continue
            if not any(
                min(end, right) > max(start, left) for left, right in segments
            ):
                continue
            try:
                qwen_score = float(row.get("qwen_score"))
            except (TypeError, ValueError):
                continue
            if not math.isfinite(qwen_score):
                continue
            try:
                fused_score = float(row.get("fused_score"))
            except (TypeError, ValueError):
                fused_score = -1.0
            if not math.isfinite(fused_score):
                fused_score = -1.0
            subject = str(row.get("main_subject") or "").strip()
            action = str(row.get("action_description") or "").strip()
            if subject or action:
                candidates.append((fused_score, qwen_score, subject, action))
        if not candidates:
            return None, None
        _fused, _qwen, subject, action = max(candidates, key=lambda item: (item[0], item[1]))
        return WindowScoreFrontend._canonical_subject_hint(subject, action), (action or None)

    def detect(self, video_path: str) -> TemporalResult:
        n_frames, fps, _width, _height = video_meta(video_path)
        if n_frames <= 0 or fps <= 0:
            return TemporalResult(
                [],
                raw="window-score",
                source="window_score",
                valid=False,
                errors=["unreadable video"],
                attempts=0,
            )
        duration = n_frames / float(fps)
        coarse_windows = generate_windows(
            duration,
            self.coarse_window_sec,
            self.coarse_stride_sec,
            self.coarse_fps,
            stage="coarse",
        )
        signal_extractor = (
            TemporalSignalExtractor(
                sample_fps=self.coarse_fps,
                ffmpeg_bin=self.ffmpeg_bin,
                audio_sample_rate=self.audio_sample_rate,
            )
            if self.signal_provider is None
            else None
        )
        coarse_qwen, coarse_stop = self._score_windows(video_path, coarse_windows)
        coarse_rows = self._records_with_signals(
            video_path,
            coarse_windows,
            coarse_qwen,
            signal_extractor,
        )
        coarse_fused_initial, coarse_diag = self._fuse_records(coarse_rows)
        coarse_curve_initial = smooth_score_curve(
            build_score_curve(
                coarse_fused_initial,
                duration,
                grid_sec=self.curve_step_sec,
            ),
            window_points=self.smoothing_points,
        )
        refinement_curves = {
            "fused": coarse_curve_initial,
            "qwen": self._signal_curve(
                coarse_fused_initial, duration, "qwen_score"
            ),
            "motion": self._signal_curve(
                coarse_fused_initial, duration, "normalized_motion_score"
            ),
            "audio_change": self._signal_curve(
                coarse_fused_initial, duration, "normalized_audio_score"
            ),
        }
        fine_regions = select_multisignal_refinement_regions(
            refinement_curves,
            duration,
            min_score=self.coarse_candidate_threshold,
            top_fraction=self.relative_top_fraction,
            min_prominence=0.10,
            padding_sec=1.0,
            merge_gap_sec=self.merge_gap_sec,
            max_regions=self.max_fine_regions,
        )
        fine_windows = self._fine_windows(fine_regions, len(coarse_windows))
        if coarse_stop:
            # Do not probe the same missing/OOM model again in the fine stage.
            # Fine traditional signals are still extracted below, so the
            # coarse-to-fine fallback remains useful without repeatedly
            # stressing the GPU or duplicating a known model-load failure.
            fine_qwen = [
                _invalid_window_score(window, coarse_stop) for window in fine_windows
            ]
            fine_stop = coarse_stop
        else:
            fine_qwen, fine_stop = self._score_windows(video_path, fine_windows)
        fine_rows = self._records_with_signals(
            video_path,
            fine_windows,
            fine_qwen,
            signal_extractor,
        )
        if signal_extractor is not None:
            signal_extractor.close()

        all_rows = coarse_rows + fine_rows
        all_fused, final_diag = self._fuse_records(all_rows)
        coarse_fused = [row for row in all_fused if row.get("stage") == "coarse"]
        fine_fused = [row for row in all_fused if row.get("stage") == "fine"]
        coarse_curve = build_score_curve(
            coarse_fused,
            duration,
            grid_sec=self.curve_step_sec,
        )
        if fine_fused:
            fine_curve = build_score_curve(
                fine_fused,
                duration,
                grid_sec=self.curve_step_sec,
                fill_missing=False,
            )
            fine_curve = [
                row for row in fine_curve if int(row.get("observation_count", 0) or 0) > 0
            ]
            combined_curve = overlay_score_curves(
                coarse_curve,
                fine_curve,
                max_refine_weight=0.35,
                full_evidence_windows=4,
                conflict_threshold=0.25,
                conflict_weight_cap=0.15,
            )
        else:
            fine_curve = []
            combined_curve = coarse_curve
        # The contract specifies one three-point moving average after coarse
        # and fine windows have been mapped/overlaid on the common grid.
        combined_curve = smooth_score_curve(
            combined_curve, window_points=self.smoothing_points
        )

        tail_diagnostics = detect_tail_barrier(
            all_fused,
            duration,
            grid_sec=self.curve_step_sec,
        )
        tail_barrier = tail_diagnostics.get("barrier_sec")

        selected = select_highlight_segments(
            combined_curve,
            duration,
            score_threshold=self.absolute_threshold,
            top_fraction=self.relative_top_fraction,
            min_segment_sec=1.0,
            merge_gap_sec=self.merge_gap_sec,
            padding_sec=self.boundary_expand_sec,
            max_padding_sec=self.max_boundary_expand_sec,
            max_segments=self.max_segments,
            max_coverage_sec=self.max_coverage_sec,
            max_coverage_ratio=self.max_coverage_ratio,
            growth_threshold=self.growth_threshold,
            use_local_peak_seeds=True,
            end_barrier_sec=tail_barrier,
        )
        selected_peaks = list(selected.get("selected_peaks") or [])
        fallback_used: Optional[str] = None
        qwen_used_in_fusion = any(
            "qwen_score" in (row.get("fusion_available_signals") or [])
            for row in all_fused
        )
        if selected.get("segments_sec"):
            if qwen_used_in_fusion:
                source = "window_score_curve"
            else:
                source = "window_score_traditional_only"
                fallback_used = "traditional_signals"
        else:
            selected = fallback_segments(
                all_fused,
                duration,
                grid_sec=self.curve_step_sec,
                short_video_sec=self.short_full_sec,
                middle_sec=4.0,
                min_segment_sec=1.0,
                weights=self.signal_weights,
            )
            fallback_used = str(selected.get("source") or "middle")
            source = f"window_score_{fallback_used}_fallback"
            selected.setdefault("score_curve", combined_curve)
            # Keep the peaks found before fallback.  They are useful for
            # diagnosing why the normal threshold rejected a curve even when
            # the final temporal interval came from a conservative fallback.
            if not selected.get("selected_peaks"):
                selected["selected_peaks"] = selected_peaks

        segments = [tuple(map(float, item)) for item in selected.get("segments_sec", [])]
        window_records = all_fused
        for row in window_records:
            try:
                row_start = float(row["start_sec"])
                row_end = float(row["end_sec"])
            except (KeyError, TypeError, ValueError):
                row_start = row_end = 0.0
            in_refinement_region = any(
                min(row_end, right) > max(row_start, left)
                for left, right in fine_regions
            )
            in_final_segment = any(
                min(row_end, right) > max(row_start, left)
                for left, right in segments
            )
            reasons: List[str] = []
            if row.get("stage") == "fine":
                reasons.append("fine_rescore")
            elif in_refinement_region:
                reasons.append("coarse_peak_region")
            if in_final_segment:
                reasons.append("overlaps_final_segment")
            if fallback_used:
                reasons.append(f"fallback:{fallback_used}")
            if not reasons:
                reasons.append("not_selected")
            row["qwen_retry_count"] = max(0, int(row.get("attempts", 0) or 0) - 1)
            row["selected_for_refinement"] = bool(in_refinement_region)
            row["selected_for_output"] = bool(in_final_segment)
            row["selection_reasons"] = reasons
        subject_hint, action_description = self._selected_semantics(
            window_records, segments
        )
        # A nearly constant Qwen curve was explicitly excluded from temporal
        # fusion, so its text must not leak back into the public handoff as if
        # it were trustworthy semantic evidence.  Apply this at the detector
        # contract itself, not only in one runner.
        if bool(final_diag.get("qwen_degenerate")):
            subject_hint = None
            action_description = None
        errors = [error for item in coarse_qwen + fine_qwen for error in item.errors]
        if coarse_stop:
            errors.append(coarse_stop)
        if fine_stop:
            errors.append(fine_stop)
        diagnostics = dict(selected.get("score_diagnostics") or {})
        diagnostics.update(
            {
                "coarse_window_count": len(coarse_windows),
                "fine_window_count": len(fine_windows),
                "coarse_valid_qwen_count": sum(item.valid for item in coarse_qwen),
                "fine_valid_qwen_count": sum(item.valid for item in fine_qwen),
                "fine_regions": [list(item) for item in fine_regions],
                "refinement_signals": sorted(refinement_curves),
                "coarse_fusion": coarse_diag,
                "final_fusion": final_diag,
                "tail_barrier": tail_diagnostics,
                "qwen_degenerate": bool(final_diag.get("qwen_degenerate")),
                "qwen_stopped_reason": coarse_stop or fine_stop,
                "fusion_weights": dict(self.signal_weights),
                "qwen_component_weights": dict(QWEN_SCORE_WEIGHTS),
            }
        )
        if diagnostics.get("qwen_degenerate"):
            warnings = list(diagnostics.get("warnings") or [])
            warnings.append("Qwen window scores have no useful variation; Qwen channel disabled")
            diagnostics["warnings"] = list(dict.fromkeys(warnings))
        if not any(item.valid for item in coarse_qwen + fine_qwen):
            warnings = list(diagnostics.get("warnings") or [])
            warnings.append("Qwen produced no valid window scores; traditional fallback used")
            diagnostics["warnings"] = list(dict.fromkeys(warnings))
        attempts = sum(item.attempts for item in coarse_qwen + fine_qwen)
        raw_responses: List[str] = []
        for item in coarse_qwen + fine_qwen:
            if item.raw_responses:
                raw_responses.extend(
                    str(raw) for raw in item.raw_responses if str(raw)
                )
            elif item.raw:
                raw_responses.append(str(item.raw))
        return TemporalResult(
            segments,
            subject_hint=subject_hint,
            raw=json.dumps(
                {
                    "mode": "window_score",
                    "coarse_windows": len(coarse_windows),
                    "fine_windows": len(fine_windows),
                },
                ensure_ascii=False,
            ),
            source=source,
            valid=bool(segments),
            errors=errors,
            attempts=attempts,
            fallback_used=fallback_used,
            duration_sec=duration,
            coverage_ratio=temporal_coverage_ratio(segments, duration),
            vlm_raw_responses=raw_responses,
            segments_before_normalization=list(segments),
            segments_after_normalization=list(segments),
            corrections=[
                f"coarse {self.coarse_fps:g} FPS window curve refined with "
                f"candidate-local {self.fine_fps:g} FPS windows"
            ]
            if fine_windows
            else [],
            window_records=window_records,
            score_curve=list(selected.get("score_curve") or combined_curve),
            selected_peaks=list(selected.get("selected_peaks") or selected_peaks),
            score_diagnostics=diagnostics,
            action_description=action_description,
        )

    def close(self) -> None:
        close = getattr(self.scorer, "close", None)
        if callable(close):
            close()


__all__ = ["WindowScoreFrontend", "WindowScoreTemporalDetector"]
