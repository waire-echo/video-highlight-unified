"""Offline fixed/pan camera. No trainable weights and no cross-shot smoothing."""
from __future__ import annotations
import numpy as np
from scipy.ndimage import median_filter
from scipy.sparse import diags, eye
from scipy.sparse.linalg import spsolve


def smooth_camera(times, centers, reliability, frame_size, crop_size, cfg):
    times = np.asarray(times, dtype=float)
    centers = np.asarray(centers, dtype=float)
    reliability = np.clip(np.asarray(reliability, dtype=float), 0, 1)
    n = len(times)
    if n == 0:
        return centers, {'mode': 'empty'}
    if centers.shape != (n, 2) or not np.isfinite(centers).all():
        raise ValueError('Nonfinite camera centers')
    if reliability.shape != (n,) or not np.isfinite(reliability).all() or not np.isfinite(times).all():
        raise ValueError('Nonfinite/misaligned camera evidence')
    if n > 1 and np.any(np.diff(times) <= 0):
        raise ValueError('Camera timestamps must increase')
    # Fixed 10-Hz optimization grid avoids frame-rate dependent regularization.
    span = times[-1] - times[0]
    count = max(1, int(np.ceil(span * cfg['grid_hz'])) + 1)
    grid = np.linspace(times[0], times[-1], count)
    w = np.interp(grid, times, reliability)
    fallback = np.asarray(frame_size, float) / 2
    desired = np.column_stack([np.interp(grid, times, centers[:, j]) for j in range(2)])
    # Abrupt isolated excursions are unreliable; genuine sustained pans survive.
    med = median_filter(desired, size=(5, 1), mode='nearest')
    scale = np.maximum(np.asarray(crop_size, float), 1)
    residual = np.max(np.abs(desired - med) / scale, axis=1)
    w[residual > cfg['outlier_fraction']] = 0
    target = w[:, None] * desired + (1 - w[:, None]) * fallback
    # Median is only used for robust static-mode selection; motion uses original targets.
    clean = target[w >= cfg['reliable_threshold']]
    spread = np.ptp(np.percentile(clean, [10, 90], axis=0), axis=0) / scale if len(clean) > 1 else np.zeros(2)
    static = np.mean(w) >= cfg['static_min_reliability'] and np.max(spread) <= cfg['static_fraction']
    if static:
        trajectory = np.tile(np.median(clean, axis=0), (count, 1))
        mode = 'fixed'
    elif count < 3:
        trajectory = target
        mode = 'pan'
    else:
        D = diags([np.ones(count-2), -2*np.ones(count-2), np.ones(count-2)], [0, 1, 2], shape=(count-2, count))
        A = eye(count, format='csc') + cfg['acceleration_weight'] * (D.T @ D)
        trajectory = np.column_stack([spsolve(A, target[:, j]) for j in range(2)])
        dt = max(span / (count-1), 1e-6)
        step = scale * cfg['max_speed_crop_per_sec'] * dt
        for k in range(1, count):
            trajectory[k] = trajectory[k-1] + np.clip(trajectory[k]-trajectory[k-1], -step, step)
        for k in range(count-2, -1, -1):
            trajectory[k] = trajectory[k+1] + np.clip(trajectory[k]-trajectory[k+1], -step, step)
        mode = 'pan'
    output = np.column_stack([np.interp(times, grid, trajectory[:, j]) for j in range(2)])
    return output, {'mode': mode, 'mean_reliability': float(np.mean(w)),
                    'uncertain_grid_points': int(np.sum(w < cfg['reliable_threshold']))}
