"""Query-free temporal highlight detection with Mr.HiSum PGL-SUM.

PGL-SUM is the only event-ranking signal in this frontend.  Motion, audio and
scene changes are evaluated after peak selection and may only move a selected
event's boundaries; they never create or rank events.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

from .io_utils import extract_frames, video_meta
from .pgl_sum_model import MRHISUM_PGL_INPUT_DIM, MrHiSumPGLScorer, file_sha256
from .temporal import MotionTemporalDetector, TemporalResult, temporal_coverage_ratio
from .temporal_signals import TemporalSignalExtractor


SegmentSeconds = Tuple[float, float]
ALLOWED_SUBJECT_HINTS = {"person", "dog", "cat", "animal"}
FEATURE_EXTRACTOR_VERSION = "youtube8m-inception-pca-quantized/v1"


def _require_e_path(path: str | os.PathLike[str], label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if os.name == "nt" and resolved.drive.upper() != "E:" and os.environ.get('VIDEO_HIGHLIGHT_PORTABLE_RUNTIME') != '1':
        raise ValueError(f"{label} must be on E: drive, got {resolved}")
    return resolved


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix=path.stem + ".",
        suffix=".tmp.npz",
        dir=str(path.parent),
        delete=False,
    )
    temp_path = Path(handle.name)
    handle.close()
    try:
        np.savez_compressed(str(temp_path), **arrays)
        os.replace(str(temp_path), str(path))
    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass


def _json_fingerprint(payload: Mapping[str, Any]) -> str:
    text = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class FeatureSequence:
    features: np.ndarray
    timestamps_sec: np.ndarray
    frame_ids: np.ndarray
    cache_path: str
    cache_hit: bool
    extractor_fingerprint: Dict[str, Any]


class YouTube8MFeatureExtractor:
    """Extract the exact 1-FPS 1024-D RGB representation used by Mr.HiSum.

    The released YouTube-8M frame records store PCA features after uint8
    quantisation.  Mr.HiSum dequantises those records before training.  This
    implementation deliberately performs the same quantise/dequantise roundtrip
    after Inception and PCA instead of feeding raw PCA floats to the checkpoint.
    """

    REQUIRED_FILES = (
        "classify_image_graph_def.pb",
        "mean.npy",
        "eigenvals.npy",
        "eigenvecs.npy",
    )

    def __init__(
        self,
        model_dir: str,
        cache_dir: str,
        *,
        sample_fps: float = 1.0,
        max_frames: int = 300,
        require_e_drive: bool = True,
    ) -> None:
        self.model_dir = (
            _require_e_path(model_dir, "YouTube-8M model directory")
            if require_e_drive
            else Path(model_dir).expanduser().resolve()
        )
        self.cache_dir = (
            _require_e_path(cache_dir, "YouTube-8M feature cache")
            if require_e_drive
            else Path(cache_dir).expanduser().resolve()
        )
        self.sample_fps = max(0.1, float(sample_fps))
        self.max_frames = max(1, min(300, int(max_frames)))
        missing = [name for name in self.REQUIRED_FILES if not (self.model_dir / name).is_file()]
        if missing:
            raise FileNotFoundError(
                f"YouTube-8M model directory is incomplete ({', '.join(missing)}): "
                f"{self.model_dir}"
            )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._backend_ready = False
        self._session = None
        self._graph = None
        self._decode_tensor = None
        self._pool_tensor = None
        self._pca_mean: Optional[np.ndarray] = None
        self._pca_eigenvals: Optional[np.ndarray] = None
        self._pca_eigenvecs: Optional[np.ndarray] = None
        self._fingerprint = {
            "extractor": FEATURE_EXTRACTOR_VERSION,
            "sample_fps": self.sample_fps,
            "max_frames": self.max_frames,
            "model_files": {
                name: file_sha256(self.model_dir / name) for name in self.REQUIRED_FILES
            },
        }
        self._fingerprint["fingerprint_sha256"] = _json_fingerprint(self._fingerprint)

    @property
    def fingerprint(self) -> Dict[str, Any]:
        return dict(self._fingerprint)

    def _ensure_backend(self) -> None:
        if self._backend_ready:
            return
        try:
            import tensorflow as tf
        except ImportError as exc:
            raise RuntimeError(
                "TensorFlow is required for exact YouTube-8M Inception features; "
                "run with E:\\AI\\envs\\mrhisum\\Scripts\\python.exe"
            ) from exc
        compat = tf.compat.v1
        graph_def = compat.GraphDef()
        graph_def.ParseFromString(
            (self.model_dir / "classify_image_graph_def.pb").read_bytes()
        )
        graph = tf.Graph()
        with graph.as_default():
            compat.import_graph_def(graph_def, name="")
        config = compat.ConfigProto(device_count={"GPU": 0})
        config.gpu_options.allow_growth = True
        self._session = compat.Session(graph=graph, config=config)
        self._graph = graph
        self._decode_tensor = graph.get_tensor_by_name("DecodeJpeg:0")
        self._pool_tensor = graph.get_tensor_by_name("pool_3/_reshape:0")
        self._pca_mean = np.load(self.model_dir / "mean.npy")[:, 0].astype(np.float32)
        self._pca_eigenvals = (
            np.load(self.model_dir / "eigenvals.npy")[:MRHISUM_PGL_INPUT_DIM, 0]
            .astype(np.float32)
        )
        self._pca_eigenvecs = (
            np.load(self.model_dir / "eigenvecs.npy")
            .T[:, :MRHISUM_PGL_INPUT_DIM]
            .astype(np.float32)
        )
        if self._pca_mean.shape != (2048,):
            raise ValueError(f"unexpected YouTube-8M PCA mean shape: {self._pca_mean.shape}")
        if self._pca_eigenvecs.shape != (2048, MRHISUM_PGL_INPUT_DIM):
            raise ValueError(
                f"unexpected YouTube-8M PCA matrix shape: {self._pca_eigenvecs.shape}"
            )
        self._backend_ready = True

    @staticmethod
    def _quantize_dequantize(features: np.ndarray) -> np.ndarray:
        clipped = np.clip(np.asarray(features, dtype=np.float32), -2.0, 2.0)
        quantized = np.rint((clipped + 2.0) * (255.0 / 4.0)).astype(np.uint8)
        return quantized.astype(np.float32) * (4.0 / 255.0) - 2.0

    def _frame_feature(self, bgr_frame: np.ndarray) -> np.ndarray:
        self._ensure_backend()
        if bgr_frame.ndim != 3 or bgr_frame.shape[2] != 3:
            raise ValueError(f"decoded frame is not BGR uint8: {bgr_frame.shape}")
        rgb = np.ascontiguousarray(bgr_frame[:, :, ::-1])
        pooled = self._session.run(  # type: ignore[union-attr]
            self._pool_tensor,
            feed_dict={self._decode_tensor: rgb},
        )[0].astype(np.float32)
        projected = (pooled - self._pca_mean).reshape(1, 2048).dot(
            self._pca_eigenvecs
        ).reshape(MRHISUM_PGL_INPUT_DIM)
        projected /= np.sqrt(self._pca_eigenvals + 1e-4)
        return self._quantize_dequantize(projected)

    def _video_cache_path(self, video_path: str) -> Path:
        path = Path(video_path).expanduser().resolve()
        stat = path.stat()
        payload = {
            "video_path": str(path),
            "video_size": int(stat.st_size),
            "video_mtime_ns": int(stat.st_mtime_ns),
            "extractor_fingerprint": self._fingerprint["fingerprint_sha256"],
        }
        return self.cache_dir / f"{_json_fingerprint(payload)}.npz"

    @staticmethod
    def _validate_cached(
        features: np.ndarray, timestamps: np.ndarray, frame_ids: np.ndarray
    ) -> None:
        if features.ndim != 2 or features.shape[1] != MRHISUM_PGL_INPUT_DIM:
            raise ValueError(f"cached features have invalid shape: {features.shape}")
        if features.shape[0] < 1 or timestamps.shape != (features.shape[0],):
            raise ValueError("cached feature timestamps are not aligned")
        if frame_ids.shape != (features.shape[0],):
            raise ValueError("cached feature frame IDs are not aligned")
        if not np.all(np.isfinite(features)) or not np.all(np.isfinite(timestamps)):
            raise ValueError("cached feature sequence contains non-finite values")

    def extract(self, video_path: str) -> FeatureSequence:
        video = Path(video_path).expanduser().resolve()
        if not video.is_file():
            raise FileNotFoundError(f"video not found: {video}")
        cache_path = self._video_cache_path(str(video))
        if cache_path.is_file():
            try:
                with np.load(cache_path, allow_pickle=False) as cached:
                    features = cached["features"].astype(np.float32)
                    timestamps = cached["timestamps_sec"].astype(np.float64)
                    frame_ids = cached["frame_ids"].astype(np.int64)
                self._validate_cached(features, timestamps, frame_ids)
                return FeatureSequence(
                    features,
                    timestamps,
                    frame_ids,
                    str(cache_path),
                    True,
                    self.fingerprint,
                )
            except (OSError, KeyError, ValueError):
                cache_path.unlink(missing_ok=True)

        frame_count, source_fps, _width, _height = video_meta(str(video))
        if frame_count <= 0 or source_fps <= 0:
            raise ValueError(f"cannot decode video metadata: {video}")
        duration = frame_count / source_fps
        count = min(
            self.max_frames,
            max(1, int(math.ceil(duration * self.sample_fps - 1e-9))),
        )
        frame_ids = np.asarray(
            [
                min(frame_count - 1, int(round(index * source_fps / self.sample_fps)))
                for index in range(count)
            ],
            dtype=np.int64,
        )
        frame_ids = np.unique(frame_ids)
        decoded = extract_frames(str(video), frame_ids.tolist())
        ordered_ids = [int(frame_id) for frame_id in frame_ids if int(frame_id) in decoded]
        if not ordered_ids:
            raise RuntimeError(f"no video frame could be decoded: {video}")
        features = np.stack(
            [self._frame_feature(decoded[frame_id]) for frame_id in ordered_ids], axis=0
        ).astype(np.float32)
        frame_ids = np.asarray(ordered_ids, dtype=np.int64)
        timestamps = frame_ids.astype(np.float64) / source_fps
        self._validate_cached(features, timestamps, frame_ids)
        _atomic_npz(
            cache_path,
            features=features,
            timestamps_sec=timestamps,
            frame_ids=frame_ids,
        )
        return FeatureSequence(
            features,
            timestamps,
            frame_ids,
            str(cache_path),
            False,
            self.fingerprint,
        )

    def close(self) -> None:
        session = self._session
        self._session = None
        self._graph = None
        self._decode_tensor = None
        self._pool_tensor = None
        self._backend_ready = False
        if session is not None:
            try:
                session.close()
            except Exception:
                pass


def _robust_normalize(values: Sequence[float]) -> Tuple[np.ndarray, Dict[str, Any]]:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return np.zeros_like(array), {"degenerate": True, "reason": "no_finite_values"}
    low = float(np.percentile(finite, 10.0))
    high = float(np.percentile(finite, 90.0))
    dynamic_range = high - low
    if dynamic_range <= 1e-6:
        normalized = np.zeros_like(array)
        degenerate = True
    else:
        normalized = np.clip((array - low) / dynamic_range, 0.0, 1.0)
        degenerate = bool(float(np.max(normalized) - np.min(normalized)) < 0.03)
    return normalized, {
        "method": "p10_p90",
        "low": low,
        "high": high,
        "range": dynamic_range,
        "degenerate": degenerate,
    }


def _smooth_three(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.size <= 1:
        return array.copy()
    padded = np.pad(array, (1, 1), mode="edge")
    return np.convolve(padded, np.asarray([0.25, 0.5, 0.25]), mode="valid")


def _fallback_prominence(values: np.ndarray, index: int) -> float:
    if values.size <= 1:
        return 0.0
    left_min = float(np.min(values[: index + 1]))
    right_min = float(np.min(values[index:]))
    return max(0.0, float(values[index]) - max(left_min, right_min))


def _find_peak_indices(
    scores: np.ndarray,
    *,
    duration_sec: float,
    sample_interval_sec: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if scores.size == 0:
        return [], {"peak_count": 0, "reason": "empty_curve"}
    iqr = float(np.percentile(scores, 75) - np.percentile(scores, 25))
    prominence_threshold = max(0.08, 0.35 * iqr)
    distance = max(1, int(math.ceil(2.0 / max(1e-6, sample_interval_sec))))
    endpoint_score_threshold = 0.80
    endpoint_prominence_threshold = max(prominence_threshold, 0.25)
    endpoint_horizon = max(
        2, int(math.ceil(4.0 / max(1e-6, sample_interval_sec))) + 1
    )
    try:
        from scipy.signal import find_peaks

        indices, properties = find_peaks(
            scores,
            distance=distance,
            prominence=prominence_threshold,
        )
        prominences = np.asarray(properties.get("prominences", []), dtype=float)
    except (ImportError, ValueError):
        raw = []
        for index in range(1, max(1, scores.size - 1)):
            value = scores[index]
            left = scores[index - 1]
            right = scores[index + 1]
            if value >= left and value >= right and (value > left or value > right):
                prominence = _fallback_prominence(scores, index)
                if prominence >= prominence_threshold:
                    raw.append((index, prominence))
        selected: List[Tuple[int, float]] = []
        for index, prominence in sorted(
            raw, key=lambda item: (item[1], scores[item[0]]), reverse=True
        ):
            if all(abs(index - existing[0]) >= distance for existing in selected):
                selected.append((index, prominence))
        indices = np.asarray([item[0] for item in selected], dtype=int)
        prominences = np.asarray([item[1] for item in selected], dtype=float)

    # scipy deliberately excludes endpoints.  The previous implementation
    # padded both sides with an artificial low value, which made an ordinary
    # first frame look globally prominent and recreated the project's old
    # "always choose the intro" failure.  An endpoint is now admitted only
    # when it is both absolutely strong and drops by a substantial amount
    # inside a real four-second, one-sided neighbourhood.  If a monotonic
    # short clip has no other peak, the global-maximum fallback below still
    # guarantees one legal event.
    endpoint_candidates: List[Tuple[int, float]] = []
    if scores.size > 1:
        horizon = min(scores.size, endpoint_horizon)
        endpoint_specs = (
            (0, 1, scores[1:horizon]),
            (scores.size - 1, scores.size - 2, scores[-horizon:-1]),
        )
        for index, neighbour, basin in endpoint_specs:
            if basin.size == 0:
                continue
            prominence = max(0.0, float(scores[index] - np.min(basin)))
            if (
                float(scores[index]) >= endpoint_score_threshold
                and float(scores[index]) > float(scores[neighbour])
                and prominence >= endpoint_prominence_threshold
            ):
                endpoint_candidates.append((int(index), prominence))
    if endpoint_candidates:
        indices = np.concatenate(
            (indices.astype(int, copy=False), np.asarray([item[0] for item in endpoint_candidates]))
        )
        prominences = np.concatenate(
            (prominences.astype(float, copy=False), np.asarray([item[1] for item in endpoint_candidates]))
        )

    if indices.size == 0:
        indices = np.asarray([int(np.argmax(scores))], dtype=int)
        prominences = np.asarray(
            [_fallback_prominence(scores, int(indices[0]))], dtype=float
        )
        fallback = True
    else:
        fallback = False
    event_budget = min(6, max(1, int(math.ceil(max(0.0, duration_sec) / 8.0))))
    ranked = sorted(
        zip(indices.tolist(), prominences.tolist()),
        key=lambda item: (item[1], float(scores[item[0]])),
        reverse=True,
    )[:event_budget]
    peaks = [
        {
            "index": int(index),
            "score": float(scores[index]),
            "prominence": float(prominence),
        }
        for index, prominence in sorted(ranked)
    ]
    rank_order = sorted(
        range(len(peaks)),
        key=lambda item: (peaks[item]["prominence"], peaks[item]["score"]),
        reverse=True,
    )
    for rank, peak_position in enumerate(rank_order, 1):
        peaks[peak_position]["rank"] = rank
    return peaks, {
        "prominence_threshold": prominence_threshold,
        "minimum_peak_distance_sec": 2.0,
        "endpoint_policy": {
            "artificial_padding": False,
            "minimum_normalized_score": endpoint_score_threshold,
            "minimum_one_sided_prominence": endpoint_prominence_threshold,
            "horizon_sec": 4.0,
        },
        "event_budget": event_budget,
        "used_global_max_fallback": fallback,
        "peak_count": len(peaks),
    }


def _expand_minimum(
    start: float,
    end: float,
    center: float,
    lower: float,
    upper: float,
    minimum: float = 1.0,
) -> SegmentSeconds:
    if end - start >= minimum - 1e-9:
        return max(lower, start), min(upper, end)
    start = max(lower, center - minimum / 2.0)
    end = min(upper, start + minimum)
    if end - start < minimum:
        start = max(lower, end - minimum)
    return start, end


def select_pgl_event_candidates(
    raw_scores: Sequence[float],
    timestamps_sec: Sequence[float],
    duration_sec: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Convert the full-video PGL curve into independent event candidates."""

    raw = np.asarray(raw_scores, dtype=np.float64)
    timestamps = np.asarray(timestamps_sec, dtype=np.float64)
    if raw.ndim != 1 or raw.size == 0 or timestamps.shape != raw.shape:
        raise ValueError("PGL score/timestamp arrays must be aligned one-dimensional arrays")
    normalized, normalization = _robust_normalize(raw)
    smoothed = _smooth_three(normalized)
    if timestamps.size > 1:
        sample_interval = float(np.median(np.diff(timestamps)))
    else:
        sample_interval = min(1.0, max(0.001, float(duration_sec)))
    peaks, peak_diagnostics = _find_peak_indices(
        smoothed,
        duration_sec=duration_sec,
        sample_interval_sec=sample_interval,
    )
    median = float(np.median(smoothed))
    peak_indices = [int(peak["index"]) for peak in peaks]
    candidates: List[Dict[str, Any]] = []
    for position, peak in enumerate(peaks):
        index = int(peak["index"])
        owner_left = 0
        owner_right = raw.size - 1
        if position > 0:
            previous = peak_indices[position - 1]
            owner_left = previous + int(np.argmin(smoothed[previous : index + 1]))
        if position + 1 < len(peaks):
            following = peak_indices[position + 1]
            owner_right = index + int(np.argmin(smoothed[index : following + 1]))
        peak_score = float(smoothed[index])
        growth_level = median + 0.35 * max(0.0, peak_score - median)
        left = index
        right = index
        while left > owner_left and smoothed[left - 1] >= growth_level:
            left -= 1
        while right < owner_right and smoothed[right + 1] >= growth_level:
            right += 1
        start = max(0.0, float(timestamps[left]) - sample_interval / 2.0)
        end = min(
            float(duration_sec),
            float(timestamps[right]) + sample_interval / 2.0,
        )
        peak_time = min(float(duration_sec), max(0.0, float(timestamps[index])))
        owner_start = max(
            0.0, float(timestamps[owner_left]) - sample_interval / 2.0
        )
        owner_end = min(
            float(duration_sec),
            float(timestamps[owner_right]) + sample_interval / 2.0,
        )
        start, end = _expand_minimum(
            start, end, peak_time, owner_start, owner_end, minimum=1.0
        )
        if end - start > 8.0:
            start = max(owner_start, peak_time - 4.0)
            end = min(owner_end, start + 8.0)
            if end - start < 8.0:
                start = max(owner_start, end - 8.0)
        candidates.append(
            {
                **peak,
                "time_sec": peak_time,
                "growth_level": float(growth_level),
                "ownership_start_sec": float(owner_start),
                "ownership_end_sec": float(owner_end),
                "initial_start_sec": float(start),
                "initial_end_sec": float(end),
            }
        )
    curve = []
    for index, (time_sec, raw_score, norm_score, smooth_score) in enumerate(
        zip(timestamps, raw, normalized, smoothed)
    ):
        start = max(0.0, float(time_sec) - sample_interval / 2.0)
        end = min(float(duration_sec), float(time_sec) + sample_interval / 2.0)
        curve.append(
            {
                "index": index,
                "time_sec": float(time_sec),
                "start_sec": start,
                "end_sec": end,
                "pgl_score": float(raw_score),
                "normalized_pgl_score": float(norm_score),
                "smoothed_score": float(smooth_score),
                # Existing preview/diagnostic tooling understands this generic
                # spelling.  It is still PGL-only, not a fused score.
                "fused_score": float(smooth_score),
            }
        )
    diagnostics = {
        "normalization": normalization,
        "peak_selection": peak_diagnostics,
        "curve_points": len(curve),
        "median_smoothed_score": median,
        "ranking_signal": "pgl_only",
    }
    return candidates, curve, diagnostics


def _normalise_optional(values: Sequence[Optional[float]]) -> List[Optional[float]]:
    finite = [float(value) for value in values if value is not None and math.isfinite(value)]
    if not finite:
        return [None for _ in values]
    low = float(np.percentile(finite, 10))
    high = float(np.percentile(finite, 90))
    if high - low <= 1e-6:
        return [0.0 if value is not None else None for value in values]
    return [
        max(0.0, min(1.0, (float(value) - low) / (high - low)))
        if value is not None and math.isfinite(value)
        else None
        for value in values
    ]


def build_boundary_records(
    video_path: str,
    duration_sec: float,
    signal_extractor: TemporalSignalExtractor,
    *,
    grid_sec: float = 0.25,
) -> List[Dict[str, Any]]:
    windows = []
    start = 0.0
    while start < duration_sec - 1e-9:
        end = min(duration_sec, start + grid_sec)
        if end > start:
            windows.append({"start_sec": start, "end_sec": end})
        start += grid_sec
    records = signal_extractor.score_windows(video_path, windows)
    motion = _normalise_optional([record.get("motion_score") for record in records])
    audio = _normalise_optional([record.get("audio_score") for record in records])
    scene = _normalise_optional([record.get("scene_score") for record in records])
    for index, record in enumerate(records):
        record["normalized_motion_score"] = motion[index]
        record["normalized_audio_score"] = audio[index]
        record["normalized_scene_score"] = scene[index]
        activity_values = [
            value for value in (motion[index], audio[index]) if value is not None
        ]
        record["boundary_activity"] = max(activity_values) if activity_values else None
        record["strong_scene_boundary"] = False
    # Histogram distance can stay high for several bins during a fast zoom or
    # heavy blur.  Treating every high bin as a cut would trap a peak inside a
    # 0.25-second interval.  A shot boundary must be a local transition peak,
    # not merely sustained visual change.
    raw_scene = [
        float(record.get("scene_score") or 0.0) for record in records
    ]
    for index, record in enumerate(records):
        value = raw_scene[index]
        left = raw_scene[index - 1] if index > 0 else 0.0
        right = raw_scene[index + 1] if index + 1 < len(raw_scene) else 0.0
        local_peak = value >= left and value >= right and (value > left or value > right)
        record["strong_scene_boundary"] = bool(
            local_peak
            and scene[index] is not None
            and scene[index] >= 0.85
            and value >= 0.08
        )
    return records


def _activity_groups(
    records: Sequence[Mapping[str, Any]],
    lower: float,
    upper: float,
    threshold: float = 0.30,
) -> List[SegmentSeconds]:
    active = []
    for record in records:
        start = float(record["start_sec"])
        end = float(record["end_sec"])
        value = record.get("boundary_activity")
        if end <= lower or start >= upper or value is None or float(value) < threshold:
            continue
        active.append((max(lower, start), min(upper, end)))
    if not active:
        return []
    groups = [[active[0][0], active[0][1]]]
    for start, end in active[1:]:
        if start <= groups[-1][1] + 0.250001:
            groups[-1][1] = max(groups[-1][1], end)
        else:
            groups.append([start, end])
    return [(float(start), float(end)) for start, end in groups]


def refine_event_boundaries(
    candidates: Sequence[Mapping[str, Any]],
    boundary_records: Sequence[Mapping[str, Any]],
    duration_sec: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    refined: List[Dict[str, Any]] = []
    scene_boundaries = sorted(
        float(record["end_sec"])
        for record in boundary_records
        if record.get("strong_scene_boundary")
    )
    for candidate in candidates:
        row = dict(candidate)
        peak = float(row["time_sec"])
        initial_start = float(row["initial_start_sec"])
        initial_end = float(row["initial_end_sec"])
        owner_start = float(row["ownership_start_sec"])
        owner_end = float(row["ownership_end_sec"])
        left_scene = max(
            [value for value in scene_boundaries if owner_start < value <= peak],
            default=owner_start,
        )
        right_scene = min(
            [value for value in scene_boundaries if peak < value < owner_end],
            default=owner_end,
        )
        # A one-second minimum event must remain feasible.  If two noisy cut
        # candidates squeeze the PGL peak more tightly, discard the closer
        # artificial barrier and retain the PGL ownership valley.
        if right_scene - left_scene < 1.0:
            left_space = peak - owner_start
            right_space = owner_end - peak
            if left_space >= right_space:
                left_scene = owner_start
            else:
                right_scene = owner_end
        if right_scene - left_scene < 1.0:
            left_scene, right_scene = owner_start, owner_end
        search_start = max(owner_start, left_scene, initial_start - 1.5)
        search_end = min(owner_end, right_scene, initial_end + 1.5)
        groups = _activity_groups(boundary_records, search_start, search_end)
        intersecting = [
            group
            for group in groups
            if min(group[1], initial_end) > max(group[0], initial_start)
        ]
        if intersecting:
            group = min(
                intersecting,
                key=lambda item: abs(((item[0] + item[1]) / 2.0) - peak),
            )
            start = max(search_start, min(initial_start + 0.75, group[0]))
            end = min(search_end, max(initial_end - 0.75, group[1]))
            boundary_reason = "activity_connected_to_pgl_event"
        else:
            start = max(search_start, initial_start)
            end = min(search_end, initial_end)
            boundary_reason = "pgl_valley_and_scene_only"
        start, end = _expand_minimum(
            start, end, peak, max(owner_start, left_scene), min(owner_end, right_scene), 1.0
        )
        if end - start > 12.0:
            start = max(owner_start, peak - 6.0)
            end = min(owner_end, start + 12.0)
        row.update(
            {
                "scene_left_barrier_sec": float(left_scene),
                "scene_right_barrier_sec": float(right_scene),
                "refined_start_sec": float(max(0.0, start)),
                "refined_end_sec": float(min(duration_sec, end)),
                "boundary_reason": boundary_reason,
                "activity_group_count": len(groups),
            }
        )
        if row["refined_end_sec"] > row["refined_start_sec"]:
            refined.append(row)
    return refined, {
        "grid_sec": 0.25,
        "scene_boundary_count": len(scene_boundaries),
        "scene_boundaries_sec": scene_boundaries,
        "motion_audio_role": "boundary_only",
        "ranking_signal": "pgl_only",
    }


def _limit_coverage(
    candidates: Sequence[Mapping[str, Any]], duration_sec: float
) -> List[Dict[str, Any]]:
    limit = min(30.0, max(0.0, duration_sec) * 0.75)
    total = sum(
        float(item["refined_end_sec"]) - float(item["refined_start_sec"])
        for item in candidates
    )
    if total <= limit + 1e-9:
        return [dict(item) for item in candidates]
    kept: List[Dict[str, Any]] = []
    used = 0.0
    for item in sorted(candidates, key=lambda row: int(row.get("rank", 999))):
        row = dict(item)
        start = float(row["refined_start_sec"])
        end = float(row["refined_end_sec"])
        available = limit - used
        if available < 1.0 - 1e-9:
            continue
        if end - start > available:
            peak = float(row["time_sec"])
            start = max(start, peak - available / 2.0)
            end = min(end, start + available)
            if end - start < available:
                start = max(float(row["refined_start_sec"]), end - available)
            row["refined_start_sec"] = start
            row["refined_end_sec"] = end
            row["coverage_limited"] = True
        used += end - start
        kept.append(row)
    return sorted(kept, key=lambda row: float(row["refined_start_sec"]))


def load_cached_semantic_hints(path: str) -> Dict[str, Dict[str, Optional[str]]]:
    if not path:
        return {}
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"cached semantic hints file not found: {source}")
    output: Dict[str, Dict[str, Optional[str]]] = {}
    with open(source, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid hints JSONL line {line_number}: {exc}") from exc
            video_id = str(record.get("video_id", "")).strip()
            if not video_id:
                continue
            subject = str(record.get("subject_hint") or "").strip().lower() or None
            if subject not in ALLOWED_SUBJECT_HINTS:
                subject = None
            action = str(record.get("action_description") or "").strip() or None
            output[video_id] = {"subject_hint": subject, "action_description": action}
    return output


class MrHiSumPGLTemporalDetector:
    """Complete-video PGL ranking followed by boundary-only signal repair."""

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        feature_model_dir: Optional[str] = None,
        feature_cache_dir: Optional[str] = None,
        *,
        device: str = "auto",
        sample_fps: float = 1.0,
        boundary_grid_sec: float = 0.25,
        ffmpeg_bin: str = "ffmpeg",
        audio_sample_rate: int = 8000,
        cached_hints: Optional[Mapping[str, Mapping[str, Optional[str]]]] = None,
        scorer: Optional[Any] = None,
        feature_extractor: Optional[Any] = None,
        signal_extractor: Optional[TemporalSignalExtractor] = None,
        fallback_detector: Optional[Any] = None,
        require_e_drive: bool = True,
    ) -> None:
        if scorer is None:
            if not checkpoint_path:
                raise ValueError("checkpoint_path is required")
            scorer = MrHiSumPGLScorer(checkpoint_path, device=device)
        if feature_extractor is None:
            if not feature_model_dir or not feature_cache_dir:
                raise ValueError("feature_model_dir and feature_cache_dir are required")
            feature_extractor = YouTube8MFeatureExtractor(
                feature_model_dir,
                feature_cache_dir,
                sample_fps=sample_fps,
                require_e_drive=require_e_drive,
            )
        self.scorer = scorer
        self.feature_extractor = feature_extractor
        self.boundary_grid_sec = max(0.1, float(boundary_grid_sec))
        self.signal_extractor = signal_extractor or TemporalSignalExtractor(
            sample_fps=max(4.0, 2.0 / self.boundary_grid_sec),
            ffmpeg_bin=ffmpeg_bin,
            audio_sample_rate=audio_sample_rate,
            include_audio=True,
        )
        self.fallback_detector = fallback_detector or MotionTemporalDetector(
            detect_fps=2.0,
            top_k=2,
            window_sec=4.0,
            min_segment_sec=1.0,
            padding_sec=0.25,
        )
        self.cached_hints = {
            str(video_id): dict(values) for video_id, values in (cached_hints or {}).items()
        }

    def _fallback(self, video_path: str, error: Exception) -> TemporalResult:
        result = self.fallback_detector.detect(video_path)
        result.errors = list(getattr(result, "errors", []) or []) + [
            f"PGL-SUM failed: {type(error).__name__}: {error}"
        ]
        original_source = result.source or "motion"
        result.source = f"pgl_sum_fallback:{original_source}"
        result.fallback_used = original_source
        result.score_diagnostics = {
            "model": "mrhisum/PGL_SUM",
            "ranking_signal": "fallback_motion",
            "pgl_error": f"{type(error).__name__}: {error}",
        }
        return result

    def detect(self, video_path: str) -> TemporalResult:
        frame_count, source_fps, _width, _height = video_meta(video_path)
        if frame_count <= 0 or source_fps <= 0:
            return self._fallback(video_path, ValueError("unreadable video metadata"))
        duration = frame_count / source_fps
        try:
            sequence = self.feature_extractor.extract(video_path)
            scores = self.scorer.score(sequence.features)
            candidates, curve, selection_diagnostics = select_pgl_event_candidates(
                scores, sequence.timestamps_sec, duration
            )
            boundary_records = build_boundary_records(
                video_path,
                duration,
                self.signal_extractor,
                grid_sec=self.boundary_grid_sec,
            )
            refined, boundary_diagnostics = refine_event_boundaries(
                candidates, boundary_records, duration
            )
            refined = _limit_coverage(refined, duration)
            segments = [
                (float(item["refined_start_sec"]), float(item["refined_end_sec"]))
                for item in refined
                if float(item["refined_end_sec"]) > float(item["refined_start_sec"])
            ]
            if not segments:
                raise RuntimeError("PGL-SUM peak selection produced no valid segment")
            video_id = Path(video_path).stem
            hints = self.cached_hints.get(video_id, {})
            subject = hints.get("subject_hint")
            if subject not in ALLOWED_SUBJECT_HINTS:
                subject = None
            action = hints.get("action_description") or None
            window_records = [
                {
                    "start_sec": row["start_sec"],
                    "end_sec": row["end_sec"],
                    "pgl_score": row["pgl_score"],
                    "normalized_pgl_score": row["normalized_pgl_score"],
                    "smoothed_score": row["smoothed_score"],
                    "valid": True,
                    "sample_fps": getattr(self.feature_extractor, "sample_fps", 1.0),
                }
                for row in curve
            ]
            model_fingerprint = getattr(self.scorer, "fingerprint", {})
            feature_fingerprint = getattr(sequence, "extractor_fingerprint", {})
            return TemporalResult(
                segments_sec=segments,
                subject_hint=subject,
                raw=None,
                source="mrhisum_pgl_sum",
                valid=True,
                errors=(
                    ["PGL score curve has low dynamic range"]
                    if selection_diagnostics["normalization"].get("degenerate")
                    else []
                ),
                attempts=1,
                fallback_used=None,
                duration_sec=duration,
                coverage_ratio=temporal_coverage_ratio(segments, duration),
                segments_before_normalization=list(segments),
                segments_after_normalization=list(segments),
                window_records=window_records,
                score_curve=curve,
                selected_peaks=refined,
                score_diagnostics={
                    "model": model_fingerprint,
                    "feature_extractor": feature_fingerprint,
                    "feature_cache_path": sequence.cache_path,
                    "feature_cache_hit": sequence.cache_hit,
                    "selection": selection_diagnostics,
                    "boundary": boundary_diagnostics,
                    "boundary_records": boundary_records,
                    "ranking_signal": "pgl_only",
                    "qwen_used_for_timing": False,
                    "motion_audio_scene_used_for_ranking": False,
                    "coverage_limit_ratio": 0.75,
                    "coverage_limit_sec": min(30.0, duration * 0.75),
                },
                action_description=action,
            )
        except Exception as exc:
            return self._fallback(video_path, exc)

    def close(self) -> None:
        for component in (self.signal_extractor, self.feature_extractor, self.scorer):
            close = getattr(component, "close", None)
            if callable(close):
                close()


__all__ = [
    "FeatureSequence",
    "MrHiSumPGLTemporalDetector",
    "YouTube8MFeatureExtractor",
    "build_boundary_records",
    "load_cached_semantic_hints",
    "refine_event_boundaries",
    "select_pgl_event_candidates",
]
