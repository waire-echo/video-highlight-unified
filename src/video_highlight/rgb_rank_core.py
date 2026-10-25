"""Translation candidates and content-score path search; no detector or teacher."""
import math


def legal(box, size, ratio):
    x, y, w = box
    h = w * ratio[1] / ratio[0]
    return (all(math.isfinite(v) for v in box) and w > 0 and x >= 0 and y >= 0
            and x + w <= size[0] + 1e-6 and y + h <= size[1] + 1e-6)


def candidates(baseline, size, ratio, subject=None, grid=9):
    """Preserve exact B box at index zero. Fixed size; no reference-dependent proposals."""
    assert legal(baseline, size, ratio), (baseline, size, ratio)
    assert grid >= 2
    x, y, w = baseline
    h = w * ratio[1] / ratio[0]
    mx, my = math.floor(size[0] - w + 1e-6), math.floor(size[1] - h + 1e-6)
    axis = 0 if mx >= my else 1
    bound, current = (mx, x) if axis == 0 else (my, y)
    values = [current] + [round(bound * k / (grid - 1)) for k in range(grid)]
    values += [round(current + d * (w if axis == 0 else h)) for d in [-.5, -.25, .25, .5]]
    if subject:
        lo, hi = subject[axis], subject[axis + 2]
        span = w if axis == 0 else h
        values += [round(lo), round(hi - span), round((lo + hi - span) / 2)]
    out = []
    for v in values:
        b = [int(x), int(y), int(w)]
        b[axis] = min(bound, max(0, int(v)))
        if b not in out:
            assert legal(b, size, ratio)
            out.append(b)
    assert out[0] == list(baseline)
    return out


def xywh(box, ratio):
    return [box[0], box[1], box[2], box[2] * ratio[1] / ratio[0]]


def iou(a, b):
    """Both arguments use x,y,width,height; reference boxes are not xyxy."""
    area = max(0, min(a[0]+a[2], b[0]+b[2])-max(a[0], b[0])) * max(0, min(a[1]+a[3], b[1]+b[3])-max(a[1], b[1]))
    return area / max(1e-9, a[2]*a[3] + b[2]*b[3] - area)


def path_search(boxes, scores, times, size, resets=None, movement=.05):
    """Exact first-order DP. Caller supplies content scores and scene/gap resets.

    Ties without any content discrimination return B exactly. Scene boundaries
    sever all transition costs. This module does not infer scene cuts itself.
    """
    n = len(boxes)
    assert len(scores) == len(times) == n and movement >= 0
    if not n:
        return []
    assert all(times[i] > times[i-1] for i in range(1, n))
    resets = set(resets or []) | {0, n}
    assert all(isinstance(i, int) and 0 <= i <= n for i in resets)
    for bs, ss in zip(boxes, scores):
        assert bs and len(bs) == len(ss) and all(math.isfinite(s) for s in ss)
    result = [0] * n
    cuts = sorted(resets)
    for start, end in zip(cuts, cuts[1:]):
        if all(max(scores[i])-min(scores[i]) <= 1e-9 for i in range(start, end)):
            continue
        vals, back = list(scores[start]), []
        for i in range(start + 1, end):
            dt = times[i] - times[i-1]
            nxt, links = [], []
            for k, b in enumerate(boxes[i]):
                choices = []
                for j, prev in enumerate(boxes[i-1]):
                    dist = abs(b[0]-prev[0])/size[0] + abs(b[1]-prev[1])/size[1]
                    choices.append(vals[j] - movement * dist / dt)
                best = max(range(len(choices)), key=lambda j: (choices[j], -j))
                nxt.append(scores[i][k] + choices[best]); links.append(best)
            vals = nxt; back.append(links)
        k = max(range(len(vals)), key=lambda j: (vals[j], -j))
        result[end-1] = k
        for i in range(end-2, start-1, -1):
            k = back[i-start][k]; result[i] = k
    return result
