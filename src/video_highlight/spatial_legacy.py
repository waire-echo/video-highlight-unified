import math
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2


YOLO_HINT_ALIASES = {
    "person": {"person"},
    "people": {"person"},
    "human": {"person"},
    "athlete": {"person"},
    "player": {"person"},
    "cat": {"cat"},
    "dog": {"dog"},
    "animal": {"cat", "dog", "horse", "bird"},
    "ball": {"sports ball"},
    "sports ball": {"sports ball"},
    "car": {"car"},
    "vehicle": {"car", "truck", "bus", "motorcycle", "bicycle"},
    "bike": {"bicycle", "motorcycle"},
    "bicycle": {"bicycle"},
    "motorcycle": {"motorcycle"},
}

LEGACY_HINT_ALIASES = {
    "person": {"person"},
    "people": {"person"},
    "human": {"person"},
    "cat": {"cat"},
    "dog": {"dog"},
    "ball": {"sports ball"},
    "sports ball": {"sports ball"},
    "car": {"car", "truck", "bus", "motorcycle", "bicycle"},
    "vehicle": {"car", "truck", "bus", "motorcycle", "bicycle"},
    "bike": {"bicycle", "motorcycle"},
}

CLASS_GROUPS = {
    "person": "person",
    "sports ball": "ball",
    "car": "vehicle",
    "truck": "vehicle",
    "bus": "vehicle",
    "motorcycle": "vehicle",
    "bicycle": "vehicle",
    "cat": "animal",
    "dog": "animal",
    "horse": "animal",
    "bird": "animal",
}

SPORTS_EQUIPMENT = {
    "baseball bat",
    "baseball glove",
    "frisbee",
    "skis",
    "snowboard",
    "kite",
    "skateboard",
    "surfboard",
    "tennis racket",
}

NEUTRAL_HINTS = {"", "object", "stage", "text", "scene", "action", "unknown"}
GROUP_HINTS = {"vehicle": "vehicle", "animal": "animal"}


@dataclass(frozen=True)
class DetectionCandidate:
    index: int
    xyxy: Tuple[float, float, float, float]
    center: Tuple[float, float]
    width: float
    height: float
    area: float
    area_ratio: float
    class_id: int
    class_name: str
    class_group: str
    confidence: float
    touches_border: bool = False


@dataclass
class CandidateScore:
    candidate: DetectionCandidate
    total: float
    base_score: float
    confidence_score: float
    semantic_score: float
    area_score: float
    center_score: float
    prior_score: float
    relation_score: float
    continuity_score: float
    previous_iou: float
    center_distance_ratio: float


@dataclass
class SpatialTrackState:
    last_candidate: Optional[DetectionCandidate] = None
    last_frame_id: Optional[int] = None
    velocity_xy: Tuple[float, float] = (0.0, 0.0)
    last_focus_center: Optional[Tuple[float, float]] = None
    focus_velocity_xy: Tuple[float, float] = (0.0, 0.0)
    misses: int = 0
    age: int = 0
    track_id: int = 0


def _empty_spatial_summary() -> dict:
    return {
        "segments_started": 0,
        "state_resets": 0,
        "keyframes_seen": 0,
        "detection_frames": 0,
        "center_fallback_frames": 0,
        "motion_fallback_frames": 0,
        "track_hold_frames": 0,
        "candidates_total": 0,
        "multi_candidate_frames": 0,
        "hint_selected_frames": 0,
        "continuity_selected_frames": 0,
        "track_switches": 0,
        "track_reacquisitions": 0,
        "selected_conf_sum": 0.0,
        "selected_conf_count": 0,
        "selected_class_counts": Counter(),
        "scoring_counts": Counter(),
    }


class SpatialDetectorBase:
    def __init__(self):
        self._summary = _empty_spatial_summary()

    def begin_video(self, video_id, video_path, fps, width, height) -> None:
        del video_id, video_path, fps, width, height
        self._summary = _empty_spatial_summary()

    def begin_segment(self, segment_index, start_frame, end_frame, subject_hint=None) -> None:
        del segment_index, start_frame, end_frame, subject_hint
        self._summary["segments_started"] += 1

    def end_segment(self) -> None:
        return None

    def end_video(self) -> dict:
        out = dict(self._summary)
        out["selected_class_counts"] = dict(self._summary["selected_class_counts"])
        out["scoring_counts"] = dict(self._summary["scoring_counts"])
        count = int(out.get("selected_conf_count", 0))
        out["selected_conf_mean"] = (
            float(out.get("selected_conf_sum", 0.0)) / count if count else None
        )
        return out

    def _record(self, info: dict) -> None:
        summary = self._summary
        summary["keyframes_seen"] += 1
        detections = int(info.get("detections", 0) or 0)
        summary["candidates_total"] += detections
        if detections > 0:
            summary["detection_frames"] += 1
        if detections > 1:
            summary["multi_candidate_frames"] += 1
        fallback = str(info.get("fallback") or "")
        if fallback == "center":
            summary["center_fallback_frames"] += 1
        elif fallback == "motion-region":
            summary["motion_fallback_frames"] += 1
        elif fallback == "track-hold":
            summary["track_hold_frames"] += 1
        if info.get("hint_match"):
            summary["hint_selected_frames"] += 1
        if info.get("continuity_changed_selection"):
            summary["continuity_selected_frames"] += 1
        if info.get("track_switched"):
            summary["track_switches"] += 1
        if info.get("track_reacquired"):
            summary["track_reacquisitions"] += 1
        class_name = info.get("class")
        if class_name:
            summary["selected_class_counts"][str(class_name)] += 1
        confidence = info.get("conf")
        if isinstance(confidence, (int, float)) and math.isfinite(float(confidence)):
            summary["selected_conf_sum"] += float(confidence)
            summary["selected_conf_count"] += 1
        scoring = info.get("scoring")
        if scoring:
            summary["scoring_counts"][str(scoring)] += 1


def _motion_region_center(frame) -> Tuple[float, float, dict]:
    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 60, 160)
    contours, _hier = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return width / 2.0, height / 2.0, {"fallback": "center"}
    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)
    if w * h < 0.005 * width * height:
        return width / 2.0, height / 2.0, {"fallback": "center"}
    return x + w / 2.0, y + h / 2.0, {
        "fallback": "motion-region",
        "bbox": [x, y, x + w, y + h],
        "motion_bbox": [x, y, x + w, y + h],
    }


class CenterSpatialDetector(SpatialDetectorBase):
    def detect_center(
        self, frame, subject_hint: Optional[str] = None, *, frame_id: Optional[int] = None
    ) -> Tuple[float, float, dict]:
        del subject_hint, frame_id
        height, width = frame.shape[:2]
        info = {"backend": "center", "fallback": "center", "detections": 0}
        self._record(info)
        return width / 2.0, height / 2.0, info


class MotionRegionSpatialDetector(SpatialDetectorBase):
    """Single-frame fallback based on salient edges, not true tracking."""

    def detect_center(
        self, frame, subject_hint: Optional[str] = None, *, frame_id: Optional[int] = None
    ) -> Tuple[float, float, dict]:
        del subject_hint, frame_id
        cx, cy, extra = _motion_region_center(frame)
        if extra.get("fallback") == "motion-region":
            extra = dict(extra)
            extra.pop("fallback", None)
            extra["region_detected"] = True
        info = {"backend": "motion-region", "detections": 0, **extra}
        self._record(info)
        return cx, cy, info


def _class_group(name: str) -> str:
    if name in CLASS_GROUPS:
        return CLASS_GROUPS[name]
    if name in SPORTS_EQUIPMENT:
        return "equipment"
    return "other"


def _name_for_class(names, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, class_id)).lower()
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id]).lower()
    return str(class_id).lower()


def _scalar(value, default=0.0) -> float:
    if value is None:
        return float(default)
    try:
        if hasattr(value, "item"):
            return float(value.item())
        if isinstance(value, (list, tuple)):
            return _scalar(value[0], default=default) if value else float(default)
        return float(value)
    except (TypeError, ValueError, IndexError):
        return float(default)


def _iou(left: Sequence[float], right: Sequence[float]) -> float:
    x1 = max(float(left[0]), float(right[0]))
    y1 = max(float(left[1]), float(right[1]))
    x2 = min(float(left[2]), float(right[2]))
    y2 = min(float(left[3]), float(right[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, float(left[2]) - float(left[0])) * max(
        0.0, float(left[3]) - float(left[1])
    )
    right_area = max(0.0, float(right[2]) - float(right[0])) * max(
        0.0, float(right[3]) - float(right[1])
    )
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def _distance_ratio(left: Tuple[float, float], right: Tuple[float, float], width: int, height: int) -> float:
    diagonal = max(1.0, math.hypot(width, height))
    return min(1.0, math.hypot(left[0] - right[0], left[1] - right[1]) / diagonal)


def _area_score(candidate: DetectionCandidate) -> float:
    ideal = {
        "person": 0.10,
        "ball": 0.004,
        "vehicle": 0.18,
        "animal": 0.12,
        "equipment": 0.03,
        "other": 0.12,
    }.get(candidate.class_group, 0.12)
    ratio = max(1e-6, candidate.area_ratio)
    score = math.exp(-0.55 * abs(math.log(ratio / ideal)))
    if candidate.area_ratio > 0.80:
        score *= max(0.15, (1.0 - candidate.area_ratio) / 0.20)
    return max(0.0, min(1.0, score))


def _sports_prior(candidate: DetectionCandidate) -> float:
    return {
        "person": 1.0,
        "ball": 1.0,
        "vehicle": 0.90,
        "equipment": 0.80,
        "animal": 0.55,
        "other": 0.30,
    }.get(candidate.class_group, 0.30)


class YOLOSpatialDetector(SpatialDetectorBase):
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf: float = 0.25,
        *,
        ball_conf: float = 0.15,
        scoring: str = "enhanced",
        stateful: bool = True,
        continuity_weight: float = 0.30,
        max_missed_keyframes: int = 1,
        max_raw_candidates: int = 5,
        person_ball_weight: float = 0.20,
        model=None,
    ):
        super().__init__()
        if model is None:
            from ultralytics import YOLO

            model = YOLO(model_path)
        self.model = model
        self.conf = max(0.0, float(conf))
        self.ball_conf = max(0.0, min(self.conf, float(ball_conf)))
        self.names = getattr(self.model, "names", {})
        self.scoring = str(scoring or "enhanced").lower()
        self.stateful = bool(stateful) and self.scoring != "legacy"
        self.continuity_weight = max(0.0, min(0.75, float(continuity_weight)))
        self.max_missed_keyframes = max(0, int(max_missed_keyframes))
        self.max_raw_candidates = max(0, int(max_raw_candidates))
        self.person_ball_weight = max(0.0, min(0.5, float(person_ball_weight)))
        self.track = SpatialTrackState()
        self._geometric_reset_pending = False

    def begin_video(self, video_id, video_path, fps, width, height) -> None:
        super().begin_video(video_id, video_path, fps, width, height)
        self.track = SpatialTrackState()
        self._geometric_reset_pending = False

    def begin_segment(self, segment_index, start_frame, end_frame, subject_hint=None) -> None:
        super().begin_segment(segment_index, start_frame, end_frame, subject_hint)
        self.track = SpatialTrackState()
        self._geometric_reset_pending = False
        self._summary["state_resets"] += 1

    def end_segment(self) -> None:
        self.track = SpatialTrackState()
        self._geometric_reset_pending = False

    def end_video(self) -> dict:
        out = super().end_video()
        out["backend"] = "yolo"
        out["scoring"] = self.scoring
        out["stateful"] = self.stateful
        self.track = SpatialTrackState()
        return out

    def detect_candidates(self, frame) -> List[DetectionCandidate]:
        height, width = frame.shape[:2]
        inference_conf = self.conf if self.scoring == "legacy" else min(self.conf, self.ball_conf)
        result = self.model.predict(frame, conf=inference_conf, verbose=False)[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        candidates: List[DetectionCandidate] = []
        frame_area = max(1.0, float(width * height))
        for index, box in enumerate(boxes):
            try:
                coords = box.xyxy[0]
                if hasattr(coords, "tolist"):
                    coords = coords.tolist()
                x1, y1, x2, y2 = [float(value) for value in coords[:4]]
            except (TypeError, ValueError, IndexError, AttributeError):
                continue
            if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
                continue
            if x2 <= x1 or y2 <= y1:
                continue
            x1, x2 = max(0.0, min(x1, width)), max(0.0, min(x2, width))
            y1, y2 = max(0.0, min(y1, height)), max(0.0, min(y2, height))
            box_width = x2 - x1
            box_height = y2 - y1
            if box_width <= 1e-6 or box_height <= 1e-6:
                continue
            class_value = _scalar(getattr(box, "cls", None), default=-1)
            if not math.isfinite(class_value) or abs(class_value - round(class_value)) > 1e-6:
                continue
            class_id = int(round(class_value))
            class_name = _name_for_class(self.names, class_id)
            confidence = _scalar(getattr(box, "conf", None), default=0.0)
            if not math.isfinite(confidence):
                continue
            threshold = (
                self.ball_conf
                if self.scoring != "legacy" and class_name == "sports ball"
                else self.conf
            )
            if confidence < threshold:
                continue
            area = box_width * box_height
            candidates.append(
                DetectionCandidate(
                    index=index,
                    xyxy=(x1, y1, x2, y2),
                    center=((x1 + x2) / 2.0, (y1 + y2) / 2.0),
                    width=box_width,
                    height=box_height,
                    area=area,
                    area_ratio=area / frame_area,
                    class_id=class_id,
                    class_name=class_name,
                    class_group=_class_group(class_name),
                    confidence=max(0.0, min(1.0, confidence)),
                    touches_border=x1 <= 1 or y1 <= 1 or x2 >= width - 1 or y2 >= height - 1,
                )
            )
        return candidates

    def _hint_context(self, subject_hint: Optional[str]):
        normalized = str(subject_hint or "").lower().strip()
        if normalized in NEUTRAL_HINTS:
            return normalized, set(), None, False
        preferred = YOLO_HINT_ALIASES.get(normalized)
        if not preferred:
            return normalized, set(), None, False
        group = GROUP_HINTS.get(normalized)
        return normalized, set(preferred), group, True

    def _semantic_scores(
        self, candidates: Sequence[DetectionCandidate], subject_hint: Optional[str]
    ) -> Tuple[Dict[int, float], bool]:
        _hint, preferred, hint_group, supported = self._hint_context(subject_hint)
        if not supported:
            return {candidate.index: 0.5 for candidate in candidates}, False
        scores = {}
        preferred_exists = False
        for candidate in candidates:
            if candidate.class_name in preferred:
                score = 1.0
                preferred_exists = True
            elif hint_group and candidate.class_group == hint_group:
                score = 0.82
                preferred_exists = True
            else:
                score = 0.0
            scores[candidate.index] = score
        if preferred_exists:
            return scores, True
        for candidate in candidates:
            if hint_group == "ball" and candidate.class_group == "person":
                scores[candidate.index] = 0.70
            else:
                scores[candidate.index] = 0.5
        return scores, False

    def _relation_scores(
        self, candidates: Sequence[DetectionCandidate], width: int, height: int
    ) -> Dict[int, float]:
        people = [candidate for candidate in candidates if candidate.class_group == "person"]
        balls = [candidate for candidate in candidates if candidate.class_group == "ball"]
        out = {candidate.index: 0.0 for candidate in candidates}
        for candidate in candidates:
            related = balls if candidate.class_group == "person" else people if candidate.class_group == "ball" else []
            if not related:
                continue
            nearest = min(
                _distance_ratio(candidate.center, other.center, width, height) for other in related
            )
            out[candidate.index] = max(0.0, 1.0 - nearest / 0.60)
        return out

    def _continuity(
        self, candidate: DetectionCandidate, width: int, height: int, frame_id: Optional[int]
    ) -> Tuple[float, float, float]:
        previous = self.track.last_candidate if self.stateful else None
        if not self.stateful or previous is None:
            return 0.0, 0.0, 1.0
        gap = 1.0
        if frame_id is not None and self.track.last_frame_id is not None:
            gap = max(1.0, float(frame_id - self.track.last_frame_id))
        predicted = (
            previous.center[0] + self.track.velocity_xy[0] * gap,
            previous.center[1] + self.track.velocity_xy[1] * gap,
        )
        distance_ratio = _distance_ratio(candidate.center, predicted, width, height)
        center_similarity = max(0.0, 1.0 - distance_ratio / 0.65)
        overlap = _iou(candidate.xyxy, previous.xyxy)
        scale_similarity = math.exp(
            -abs(math.log(max(candidate.area, 1e-6) / max(previous.area, 1e-6)))
        )
        if candidate.class_name == previous.class_name:
            class_compatibility = 1.0
        elif candidate.class_group == previous.class_group:
            class_compatibility = 0.75
        else:
            class_compatibility = 0.0
        if candidate.class_group == "ball":
            continuity = (
                0.60 * center_similarity
                + 0.10 * overlap
                + 0.10 * scale_similarity
                + 0.20 * class_compatibility
            )
        else:
            continuity = (
                0.40 * center_similarity
                + 0.30 * overlap
                + 0.15 * scale_similarity
                + 0.15 * class_compatibility
            )
        return max(0.0, min(1.0, continuity)), overlap, distance_ratio

    def score_candidates(
        self,
        candidates: Sequence[DetectionCandidate],
        frame_shape,
        subject_hint: Optional[str] = None,
        frame_id: Optional[int] = None,
    ) -> List[CandidateScore]:
        height, width = frame_shape[:2]
        semantic_scores, preferred_exists = self._semantic_scores(candidates, subject_hint)
        _hint, preferred_names, _hint_group, hint_supported = self._hint_context(subject_hint)
        relation_scores = self._relation_scores(candidates, width, height)
        scored: List[CandidateScore] = []
        self._geometric_reset_pending = False

        if self.scoring == "legacy":
            normalized_hint = str(subject_hint or "").lower().strip()
            preferred = LEGACY_HINT_ALIASES.get(normalized_hint, {normalized_hint} if normalized_hint else set())
            for candidate in candidates:
                hint_bonus = 2.0 if candidate.class_name in preferred else 1.0
                total = hint_bonus * candidate.area * max(0.1, candidate.confidence)
                scored.append(
                    CandidateScore(
                        candidate=candidate,
                        total=total,
                        base_score=total,
                        confidence_score=candidate.confidence,
                        semantic_score=1.0 if hint_bonus > 1 else 0.5,
                        area_score=candidate.area_ratio,
                        center_score=0.0,
                        prior_score=0.0,
                        relation_score=0.0,
                        continuity_score=0.0,
                        previous_iou=0.0,
                        center_distance_ratio=1.0,
                    )
                )
            return sorted(scored, key=lambda item: item.total, reverse=True)

        for candidate in candidates:
            confidence_score = candidate.confidence ** 0.70
            semantic_score = semantic_scores.get(candidate.index, 0.5)
            area_score = _area_score(candidate)
            center_distance = _distance_ratio(
                candidate.center, (width / 2.0, height / 2.0), width, height
            )
            center_score = max(0.0, 1.0 - center_distance / 0.75)
            prior_score = _sports_prior(candidate)
            relation_score = relation_scores.get(candidate.index, 0.0)
            base_score = (
                0.30 * confidence_score
                + 0.30 * semantic_score
                + 0.15 * area_score
                + 0.08 * center_score
                + 0.12 * prior_score
                + 0.05 * relation_score
            )
            continuity, previous_iou, previous_distance = self._continuity(
                candidate, width, height, frame_id
            )
            total = base_score
            if self.stateful and self.track.last_candidate is not None:
                total = (
                    (1.0 - self.continuity_weight) * base_score
                    + self.continuity_weight * continuity
                )
            scored.append(
                CandidateScore(
                    candidate=candidate,
                    total=total,
                    base_score=base_score,
                    confidence_score=confidence_score,
                    semantic_score=semantic_score,
                    area_score=area_score,
                    center_score=center_score,
                    prior_score=prior_score,
                    relation_score=relation_score,
                    continuity_score=continuity,
                    previous_iou=previous_iou,
                    center_distance_ratio=previous_distance,
                )
            )

        if self.stateful and self.track.last_candidate is not None and scored:
            geometric_reset = (
                max(item.previous_iou for item in scored) < 0.02
                and min(item.center_distance_ratio for item in scored) > 0.55
            )
            if geometric_reset:
                self._geometric_reset_pending = True
                for item in scored:
                    item.total = item.base_score
        if hint_supported and preferred_exists:
            for item in scored:
                if item.candidate.class_name not in preferred_names:
                    item.total -= 2.0
        return sorted(scored, key=lambda item: item.total, reverse=True)

    def _related_candidate(
        self, selected: DetectionCandidate, candidates: Sequence[DetectionCandidate]
    ) -> Optional[DetectionCandidate]:
        target_group = "person" if selected.class_group == "ball" else "ball" if selected.class_group == "person" else None
        if not target_group:
            return None
        related = [candidate for candidate in candidates if candidate.class_group == target_group]
        if not related:
            return None
        return min(related, key=lambda item: math.hypot(item.center[0] - selected.center[0], item.center[1] - selected.center[1]))

    def _update_track(
        self,
        selected: DetectionCandidate,
        frame_id: Optional[int],
        focus_center: Tuple[float, float],
        width: int,
        height: int,
        force_new_track: bool = False,
    ) -> Tuple[bool, bool]:
        previous = self.track.last_candidate if self.stateful else None
        switched = bool(force_new_track and previous is not None)
        reacquired = self.track.misses > 0
        if previous is not None and not switched:
            center_jump_ratio = _distance_ratio(
                selected.center, previous.center, width, height
            )
            switched = (
                selected.class_group != previous.class_group
                or (
                    selected.class_group != "ball"
                    and _iou(selected.xyxy, previous.xyxy) < 0.02
                    and center_jump_ratio > 0.35
                )
            )
        if previous is not None and frame_id is not None and self.track.last_frame_id is not None:
            gap = max(1.0, float(frame_id - self.track.last_frame_id))
            if switched:
                velocity = (0.0, 0.0)
                focus_velocity = (0.0, 0.0)
            else:
                velocity = (
                    (selected.center[0] - previous.center[0]) / gap,
                    (selected.center[1] - previous.center[1]) / gap,
                )
                previous_focus = self.track.last_focus_center or previous.center
                focus_velocity = (
                    (focus_center[0] - previous_focus[0]) / gap,
                    (focus_center[1] - previous_focus[1]) / gap,
                )
        else:
            velocity = (0.0, 0.0)
            focus_velocity = (0.0, 0.0)
        track_id = self.track.track_id
        if previous is None or switched:
            track_id += 1
        self.track = SpatialTrackState(
            last_candidate=selected,
            last_frame_id=frame_id,
            velocity_xy=velocity,
            last_focus_center=focus_center,
            focus_velocity_xy=focus_velocity,
            misses=0,
            age=(self.track.age + 1 if previous is not None and not switched else 1),
            track_id=track_id,
        )
        return switched, reacquired

    def _candidate_log(self, item: CandidateScore, selected_index: int) -> dict:
        candidate = item.candidate
        return {
            "index": candidate.index,
            "bbox": [round(value) for value in candidate.xyxy],
            "class": candidate.class_name,
            "class_group": candidate.class_group,
            "conf": round(candidate.confidence, 4),
            "area_ratio": round(candidate.area_ratio, 6),
            "touches_border": candidate.touches_border,
            "base_score": round(item.base_score, 6),
            "semantic_score": round(item.semantic_score, 6),
            "area_score": round(item.area_score, 6),
            "center_score": round(item.center_score, 6),
            "prior_score": round(item.prior_score, 6),
            "relation_score": round(item.relation_score, 6),
            "continuity_score": round(item.continuity_score, 6),
            "previous_iou": round(item.previous_iou, 6),
            "center_distance_ratio": round(item.center_distance_ratio, 6),
            "final_score": round(item.total, 6),
            "selected": candidate.index == selected_index,
        }

    def _no_detection(
        self, frame, subject_hint: Optional[str], frame_id: Optional[int]
    ) -> Tuple[float, float, dict]:
        del subject_hint
        height, width = frame.shape[:2]
        if self.scoring == "legacy":
            info = {
                "backend": "yolo",
                "fallback": "center",
                "detections": 0,
                "scoring": "legacy",
                "stateful": False,
            }
            self._record(info)
            return width / 2.0, height / 2.0, info
        previous = self.track.last_candidate
        if self.stateful and previous is not None and self.track.misses < self.max_missed_keyframes:
            gap = 1.0
            if frame_id is not None and self.track.last_frame_id is not None:
                gap = max(1.0, float(frame_id - self.track.last_frame_id))
            previous_focus = self.track.last_focus_center or previous.center
            cx = max(0.0, min(width, previous_focus[0] + self.track.focus_velocity_xy[0] * gap))
            cy = max(0.0, min(height, previous_focus[1] + self.track.focus_velocity_xy[1] * gap))
            self.track.misses += 1
            info = {
                "backend": "yolo",
                "fallback": "track-hold",
                "detections": 0,
                "class": previous.class_name,
                "class_group": previous.class_group,
                "bbox": [round(value) for value in previous.xyxy],
                "track_id": self.track.track_id,
                "track_age": self.track.age,
                "track_misses": self.track.misses,
                "scoring": self.scoring,
                "stateful": self.stateful,
            }
            self._record(info)
            return cx, cy, info

        self.track = SpatialTrackState()
        cx, cy, extra = _motion_region_center(frame)
        info = {
            "backend": "yolo",
            "detections": 0,
            "scoring": self.scoring,
            "stateful": self.stateful,
            **extra,
        }
        self._record(info)
        return cx, cy, info

    def detect_center(
        self,
        frame,
        subject_hint: Optional[str] = None,
        *,
        frame_id: Optional[int] = None,
    ) -> Tuple[float, float, dict]:
        height, width = frame.shape[:2]
        candidates = self.detect_candidates(frame)
        if not candidates:
            return self._no_detection(frame, subject_hint, frame_id)

        scored = self.score_candidates(
            candidates, frame.shape, subject_hint=subject_hint, frame_id=frame_id
        )
        if not scored:
            return self._no_detection(frame, subject_hint, frame_id)
        selected_item = scored[0]
        selected = selected_item.candidate
        base_best = max(scored, key=lambda item: item.base_score)
        continuity_changed = base_best.candidate.index != selected.index
        previous = self.track.last_candidate if self.stateful else None

        cx, cy = selected.center
        related = self._related_candidate(selected, candidates)
        if (
            self.scoring != "legacy"
            and related is not None
            and selected_item.relation_score > 0
            and self.person_ball_weight > 0
        ):
            cx = (1.0 - self.person_ball_weight) * cx + self.person_ball_weight * related.center[0]
            cy = (1.0 - self.person_ball_weight) * cy + self.person_ball_weight * related.center[1]
        else:
            related = None
        if self.stateful:
            switched, reacquired = self._update_track(
                selected,
                frame_id,
                (cx, cy),
                width,
                height,
                force_new_track=self._geometric_reset_pending,
            )
        else:
            switched, reacquired = False, False
            self.track = SpatialTrackState()

        _hint, preferred, hint_group, hint_supported = self._hint_context(subject_hint)
        hint_match = selected.class_name in preferred or (
            hint_group is not None and selected.class_group == hint_group
        )
        reason_parts = []
        if hint_match:
            reason_parts.append("hint")
        if continuity_changed:
            reason_parts.append("continuity")
        if selected_item.relation_score > 0:
            reason_parts.append("person-ball")
        if not reason_parts:
            reason_parts.append("enhanced-score" if self.scoring != "legacy" else "legacy-score")

        candidates_log = [
            self._candidate_log(item, selected.index)
            for item in scored[: self.max_raw_candidates]
        ]
        info = {
            "backend": "yolo",
            "bbox": [round(value) for value in selected.xyxy],
            "class": selected.class_name,
            "class_group": selected.class_group,
            "conf": round(selected.confidence, 4),
            "detections": len(candidates),
            "candidates_total": len(candidates),
            "candidates_logged": len(candidates_log),
            "candidates": candidates_log,
            "selected_candidate_index": selected.index,
            "selected_score": round(selected_item.total, 6),
            "base_score": round(selected_item.base_score, 6),
            "semantic_score": round(selected_item.semantic_score, 6),
            "continuity_score": round(selected_item.continuity_score, 6),
            "hint_supported": hint_supported,
            "hint_match": hint_match,
            "selection_reason": "+".join(reason_parts),
            "continuity_changed_selection": continuity_changed,
            "track_id": self.track.track_id,
            "track_age": self.track.age,
            "track_misses": self.track.misses,
            "track_switched": switched,
            "track_reacquired": reacquired,
            "previous_bbox": [round(value) for value in previous.xyxy] if previous else None,
            "scoring": self.scoring,
            "stateful": self.stateful,
            "focus_adjusted": related is not None and self.person_ball_weight > 0,
            "related_class": related.class_name if related else None,
        }
        self._record(info)
        return cx, cy, info


def build_spatial_detector(args):
    if args.spatial_backend == "center":
        return CenterSpatialDetector()
    if args.spatial_backend == "motion":
        return MotionRegionSpatialDetector()
    if args.spatial_backend == "yolo":
        return YOLOSpatialDetector(
            args.yolo_model,
            conf=args.yolo_conf,
            ball_conf=getattr(args, "yolo_ball_conf", 0.15),
            scoring=getattr(args, "spatial_scoring", "enhanced"),
            stateful=getattr(args, "spatial_stateful", True),
            continuity_weight=getattr(args, "spatial_continuity_weight", 0.30),
            max_missed_keyframes=getattr(args, "spatial_max_missed_keyframes", 1),
            max_raw_candidates=getattr(args, "max_raw_spatial_candidates", 5),
            person_ball_weight=getattr(args, "spatial_person_ball_weight", 0.20),
        )
    raise ValueError(f"Unknown spatial backend: {args.spatial_backend}")
