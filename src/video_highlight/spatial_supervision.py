"""Parameterised reference alignment for the enclosing project's training workflow.

Candidate generation is completed before reference quality is computed. No data
location, sample identity, source label, or historical cache is bundled here.
"""
from bisect import bisect_left
import math

from .rgb_rank_core import candidates, iou, legal, xywh
from .spatial_source_roles import require_source_role


def align_spatial_references(*, video_id, group_id, split, source_sha256, source_roles,
                             source_size, ratio, segments, baseline_boxes,
                             tracked_by_segment, native_pts, feature_times,
                             references, max_reference_error=0.06,
                             max_detection_error=0.20):
    """Build training records using caller-provided, source-bound inputs.

    ``references`` supplies ``time`` and x/y/width/height ``box`` fields.
    ``tracked_by_segment`` maps (inclusive_first, inclusive_last) to selections.
    The enclosing project owns input admission, source validation, and splits.
    Only train and validation records are accepted by this material adapter.
    """
    if split not in {'train', 'validation'}:
        raise ValueError('Unsupported training role')
    source_sha256 = require_source_role(source_sha256, split, source_roles)
    if not native_pts or not feature_times:
        raise ValueError('Native and feature timestamps are required')
    for times in [native_pts, feature_times]:
        if not all(math.isfinite(float(t)) for t in times):
            raise ValueError('Timestamps must be finite')
        if not all(float(b) > float(a) for a, b in zip(times, times[1:])):
            raise ValueError('Timestamps must increase')
    frame_segments = {}
    selection = {}
    for first, last in segments:
        if not 0 <= first <= last < len(native_pts):
            raise ValueError('Invalid frame interval')
        for frame in range(first, last + 1):
            if frame in frame_segments:
                raise ValueError('Overlapping frame intervals')
            frame_segments[frame] = (first, last)
        tracked = tracked_by_segment.get((first, last), [])
        for row in tracked:
            frame = int(row['frame'])
            if not first <= frame <= last or frame in selection:
                raise ValueError('Invalid or duplicate detection sample')
            selection[frame] = row
    if set(baseline_boxes) != set(frame_segments):
        raise ValueError('Baseline frame coverage differs from selected intervals')
    if not all(legal(box, source_size, ratio) for box in baseline_boxes.values()):
        raise ValueError('Invalid baseline geometry')
    sample_frames = sorted(selection)
    records = []
    excluded = {'ineligible': 0, 'time_unmatched': 0, 'outside_selection': 0}
    for reference in references:
        if not reference.get('eligible', True):
            excluded['ineligible'] += 1
            continue
        at = float(reference['time'])
        if not math.isfinite(at):
            raise ValueError('Reference time must be finite')
        index = bisect_left(native_pts, at)
        possible = [max(0, min(len(native_pts)-1, index)), max(0, index-1)]
        frame = min(possible, key=lambda k: abs(native_pts[k]-at))
        if abs(native_pts[frame]-at) > max_reference_error:
            excluded['time_unmatched'] += 1
            continue
        if frame not in baseline_boxes:
            excluded['outside_selection'] += 1
            continue
        near = min(sample_frames, key=lambda f: abs(f-frame)) if sample_frames else None
        subject = selection[near] if near is not None else {}
        if (near is None or abs(native_pts[near]-native_pts[frame]) > max_detection_error
                or frame_segments.get(near) != frame_segments[frame]):
            subject = {}
        proposals = candidates(baseline_boxes[frame], source_size, ratio, subject.get('box'))
        # Labels affect targets only; they never affect the proposal locations.
        reference_box = reference['box']
        if (len(reference_box) != 4 or not all(math.isfinite(float(v)) for v in reference_box)
                or reference_box[2] <= 0 or reference_box[3] <= 0):
            raise ValueError('Reference uses finite x/y/width/height with positive size')
        quality = [iou(xywh(box, ratio), reference_box) for box in proposals]
        feature_index = min(range(len(feature_times)), key=lambda k: abs(feature_times[k]-native_pts[frame]))
        records.append(dict(video_id=str(video_id), group_id=str(group_id), split=split,
                            frame=frame, time=native_pts[frame], reference_time=at,
                            feature_index=feature_index, feature_time=feature_times[feature_index],
                            source_sha256=source_sha256, size=list(source_size), ratio=list(ratio),
                            segment=list(frame_segments[frame]), baseline=list(baseline_boxes[frame]),
                            candidates=proposals, reference_xywh=list(reference_box),
                            quality_iou=quality, subject=subject))
    return records, excluded
