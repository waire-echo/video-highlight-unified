"""Raw selected video frames -> scored B's original detections/association/crops."""
import math
import torch
from pipeline.common import artifact, fingerprint, read, write
from vendor import yolo_center as gen
from vendor.subject_association import apply_control

_detector = None


def infer_group(ctx, video, frames, width, height, nms_iou):
    global _detector
    if _detector is None:
        import logging
        from ultralytics import YOLO
        from ultralytics.utils import LOGGER
        LOGGER.setLevel(logging.ERROR)
        _detector = YOLO(str(artifact(ctx['root'], ctx['config'], 'yolo')))
    previous, records = None, []
    allowed = gen.allowed_classes('')
    for ids, images in gen.read_frame_batches(video['path'], frames, 8):
        results = _detector.predict(source=images, imgsz=640, conf=.15, iou=nms_iou,
                                    device='0', half=True, verbose=False)
        assert len(results) == len(ids)
        for frame, result in zip(ids, results):
            picked = gen.choose_candidate(result, _detector.names, allowed, width, height, previous)
            eligible = []
            if result.boxes is not None:
                for box, confidence, cls in zip(result.boxes.xyxy.detach().cpu().tolist(),
                                               result.boxes.conf.detach().cpu().tolist(),
                                               result.boxes.cls.detach().cpu().tolist()):
                    name = str(_detector.names[int(cls)]).lower()
                    if name in allowed and box[2] > box[0] and box[3] > box[1]:
                        eligible.append(dict(target_id=len(eligible)+1, box=box,
                                             confidence=float(confidence), class_name=name, class_id=int(cls)))
            ident = None
            if picked is None:
                original = dict(frame=frame, detected=False)
            else:
                previous = tuple(picked['center'])
                original = dict(**picked, frame=frame, detected=True)
                ident = next(c['target_id'] for c in eligible if max(abs(x-y) for x, y in zip(c['box'], picked['box'])) < 1e-5)
            records.append(dict(frame=frame, eligible_candidates=eligible,
                                original_rule_choice_id=ident, original_rule_selection=original))
    return records


def run(ctx, video, stage1, source_size):
    width, height = source_size
    config = ctx['config']['stage2']
    profile = ctx.get('historical_detector_profile', {})
    crops, audit = {}, []
    for first, last in stage1['segments_frame']:
        group = list(range(first, last+1))
        sampled = gen.sampled_ids(group, config['stride'])
        profile_key = f'{video["sha256"]}:{first}:{last}'
        nms_iou = profile.get(profile_key, config['default_nms_iou'])
        key = fingerprint(dict(source_sha256=video['sha256'], ratio=video['ratio'],
                               first=first, last=last, stage2=config, nms_iou=nms_iou,
                               yolo=ctx['config']['assets']['yolo'], code=ctx['code_fingerprint']))
        path = ctx['video_dir'] / f'detections_{first}_{last}.json'
        if ctx['resume'] and path.exists() and read(path).get('fingerprint') == key:
            records = read(path)['records']
        else:
            records = infer_group(ctx, video, sampled, width, height, nms_iou)
            write(path, dict(fingerprint=key, nms_iou=nms_iou, records=records))
        assert [r['frame'] for r in records] == sampled
        selected, info = apply_control(records)
        ratio = stage1['targetRatioWH']
        cw = max(1, min(width, math.floor(height * ratio[0] / ratio[1])))
        center = [math.floor((width-cw)/2), math.floor((height-cw*ratio[1]/ratio[0])/2), cw]
        centers = gen.median_smooth(group, gen.interpolate_centers(group,
                     gen.adjust_person_16x9_centers(selected, ratio, cw, .45)), 15)
        for frame in group:
            crops[frame] = gen.place_fixed_crop(centers[frame], center, ratio, width, height, 1.) if frame in centers else center.copy()
        audit.append(dict(first=first, last=last, nms_iou=nms_iou, tracking=info))
    write(ctx['video_dir'] / 'stage2_audit.json', dict(groups=audit))
    return dict(video_id=video['video_id'], targetRatioWH=stage1['targetRatioWH'],
                predictions=[dict(frame=f, bboxes=box) for f, box in crops.items()])
