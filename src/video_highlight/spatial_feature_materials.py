"""Feature preparation callable by the enclosing project; no launcher or data."""
from pathlib import Path
import hashlib
import torch

from .stage2_rgb_rank import encode_actual


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8388608), b''):
            digest.update(chunk)
    return digest.hexdigest()


@torch.no_grad()
def prepare_rgb_feature_rows(records, *, video_path, source_cache, vision, processor):
    """Encode one video's ordered native frames without seek-based frame drift.

    The caller passes a frozen vision runtime and source features. No language
    model training occurs. Supervision fields are excluded from encoder inputs.
    The returned list remains in caller record order, including repeated labels.
    """
    import cv2
    if not records:
        return []
    source_identity = source_cache['source_sha256']
    if _file_sha256(video_path) != source_identity:
        raise ValueError('Source file differs from feature cache')
    if any(row['source_sha256'] != source_identity for row in records):
        raise ValueError('Mixed source identities')
    if any(parameter.requires_grad for parameter in vision.parameters()):
        raise ValueError('Vision encoder must be frozen')
    vision.eval()
    if next(vision.parameters()).device.type != 'cuda':
        raise ValueError('Actual-crop encoder currently requires a CUDA runtime')
    groups = {}
    for index, row in enumerate(records):
        frame = int(row['frame'])
        if not 0 <= frame < len(source_cache['native_pts']):
            raise ValueError('Native frame outside source cache')
        if abs(float(source_cache['times'][row['feature_index']])-row['time']) > 0.251:
            raise ValueError('Feature timestamp differs from training reference alignment')
        groups.setdefault(frame, []).append((index, row))
    result = [None] * len(records)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError('Cannot decode source')
    try:
        for frame in range(max(groups)+1):
            if not capture.grab():
                raise ValueError('Sequential source decode failed')
            if frame not in groups:
                continue
            ok, image = capture.retrieve()
            if not ok:
                raise ValueError('Cannot retrieve native RGB frame')
            for index, row in groups[frame]:
                if [image.shape[1], image.shape[0]] != list(row['size']):
                    raise ValueError('Source geometry differs from record')
                encoder_record = {name: row[name] for name in
                                  ['size', 'ratio', 'feature_index', 'baseline', 'subject', 'candidates', 'segment']}
                visual, geometry = encode_actual(encoder_record, source_cache, image, vision, processor)
                target = torch.tensor(row['quality_iou'], dtype=torch.float32)
                if not (torch.isfinite(visual).all() and torch.isfinite(geometry).all()
                        and torch.isfinite(target).all()):
                    raise ValueError('Non-finite training inputs or targets')
                # encode_actual uses inference_mode internally. Clone here,
                # after that context exits, to create ordinary training inputs.
                result[index] = dict(record=dict(row), visual=visual.detach().clone().cpu(),
                                     geometry=geometry.detach().clone().cpu(), target=target)
    finally:
        capture.release()
    return result
