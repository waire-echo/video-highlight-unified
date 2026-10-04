from typing import Dict, Iterable, List, Sequence, Tuple


Box4 = List[int]
Segment = Tuple[int, int]


def compute_crop_size(width: int, height: int, target_w: int, target_h: int) -> Tuple[int, int]:
    """Largest target-ratio crop rectangle inside the source frame."""
    if width <= 0 or height <= 0 or target_w <= 0 or target_h <= 0:
        return max(1, width), max(1, height)

    # The submission contains only width; the evaluator reconstructs height
    # with this exact rounding rule. Find the largest width whose reconstructed
    # height still fits instead of independently rounding both dimensions.
    low, high = 1, int(width)
    crop_w = 0
    while low <= high:
        candidate = (low + high) // 2
        candidate_h = int(round(candidate * float(target_h) / float(target_w)))
        if candidate_h <= height:
            crop_w = candidate
            low = candidate + 1
        else:
            high = candidate - 1

    # A positive-width crop can be mathematically impossible only for tiny,
    # degenerate frames. Keep the historical positive-size contract there.
    crop_w = max(1, crop_w)
    crop_h = int(round(crop_w * float(target_h) / float(target_w)))
    return crop_w, min(height, max(1, crop_h))


def center_to_box(cx: float, cy: float, width: int, height: int, crop_w: int, crop_h: int) -> Box4:
    """Normalized center in 0..1 to clamped [x, y, w, h]."""
    cx = max(0.0, min(1.0, float(cx)))
    cy = max(0.0, min(1.0, float(cy)))
    x = int(round(cx * width - crop_w / 2.0))
    y = int(round(cy * height - crop_h / 2.0))
    x = max(0, min(x, max(0, width - crop_w)))
    y = max(0, min(y, max(0, height - crop_h)))
    return [x, y, int(crop_w), int(crop_h)]


def absolute_center_to_box(cx: float, cy: float, width: int, height: int, crop_w: int, crop_h: int) -> Box4:
    return center_to_box(cx / float(width), cy / float(height), width, height, crop_w, crop_h)


def sec_segments_to_frames(segments_sec: Iterable[Tuple[float, float]], fps: float, n_frames: int) -> List[Segment]:
    out = []
    if n_frames <= 0:
        return out
    for start_sec, end_sec in segments_sec:
        a = int(round(min(start_sec, end_sec) * fps))
        b = int(round(max(start_sec, end_sec) * fps))
        a = max(0, min(a, n_frames - 1))
        b = max(0, min(b, n_frames - 1))
        if b >= a:
            out.append((a, b))
    return out


def merge_segments(segments: Iterable[Segment], n_frames: int) -> List[Segment]:
    clean = []
    for a, b in segments:
        a = max(0, min(int(a), max(0, n_frames - 1)))
        b = max(0, min(int(b), max(0, n_frames - 1)))
        if b < a:
            a, b = b, a
        clean.append((a, b))
    if not clean:
        return []
    clean.sort()
    merged = [[clean[0][0], clean[0][1]]]
    for a, b in clean[1:]:
        if a <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged]


def keyframes_for_segment(start: int, end: int, stride: int) -> List[int]:
    stride = max(1, int(stride))
    frames = list(range(int(start), int(end) + 1, stride))
    if not frames or frames[-1] != int(end):
        frames.append(int(end))
    return frames


def lerp_box(left: Sequence[int], right: Sequence[int], t: float) -> Box4:
    return [int(round(left[i] + (right[i] - left[i]) * t)) for i in range(4)]


def densify_boxes(start: int, end: int, key_frames: Sequence[int], key_boxes: Sequence[Sequence[int]]) -> Dict[int, Box4]:
    """Linearly interpolate key boxes to every frame in [start, end]."""
    if not key_frames:
        return {}
    pairs = sorted(zip([int(f) for f in key_frames], [list(b) for b in key_boxes]))
    kfs = [p[0] for p in pairs]
    kbs = [p[1] for p in pairs]
    out = {}
    cursor = 0
    for frame in range(int(start), int(end) + 1):
        while cursor + 1 < len(kfs) and kfs[cursor + 1] < frame:
            cursor += 1
        if frame <= kfs[0]:
            out[frame] = kbs[0]
        elif frame >= kfs[-1]:
            out[frame] = kbs[-1]
        else:
            f0, f1 = kfs[cursor], kfs[cursor + 1]
            t = (frame - f0) / float(f1 - f0) if f1 > f0 else 0.0
            out[frame] = lerp_box(kbs[cursor], kbs[cursor + 1], t)
    return out


def smooth_boxes(boxes_by_frame: Dict[int, Sequence[int]], window: int) -> Dict[int, Box4]:
    """Moving average smoothing over [x, y, w, h]."""
    window = int(window)
    if window <= 1 or not boxes_by_frame:
        return {int(f): list(b) for f, b in boxes_by_frame.items()}
    if window % 2 == 0:
        window += 1
    radius = window // 2
    frames = sorted(boxes_by_frame)
    out = {}
    for i, frame in enumerate(frames):
        lo = max(0, i - radius)
        hi = min(len(frames), i + radius + 1)
        vals = [boxes_by_frame[frames[j]] for j in range(lo, hi)]
        out[frame] = [int(round(sum(v[k] for v in vals) / float(len(vals)))) for k in range(4)]
    return out
