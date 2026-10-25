"""Parameterised training and evaluation material for the shared project.

This module deliberately has no command-line entry. The shared project's own
workflow supplies admitted feature rows, paths, device and publication policy.
"""
from pathlib import Path
import collections
import json
import math
import random
import statistics
import torch
from torch import nn
import torch.nn.functional as F

from .stage2_rgb_rank import RankHead
from .spatial_source_roles import normalise_source_roles, require_source_role


DEFAULTS = dict(learning_rate=0.0001, weight_decay=0.01, seed=23, gradient_clip=1.0,
                max_epochs=40, patience=6, min_improvement=0.002, videos_per_batch=4,
                points_per_video_max=6, ranking_iou_gap=0.1, calibration_mse_weight=0.5,
                score_advantage=0.03)


def forward_row(model, row):
    device = next(model.parameters()).device
    visual, geometry = row['visual'], row['geometry']
    # Frozen feature extraction may return inference tensors. Make ordinary
    # detached inputs before autograd saves them for trainable linear layers.
    if visual.is_inference():
        visual = visual.clone()
    if geometry.is_inference():
        geometry = geometry.clone()
    return model(visual.float().to(device), geometry.float().to(device))


def ranking_loss(model, row, *, ranking_iou_gap=0.1, calibration_mse_weight=0.5):
    logits = forward_row(model, row)
    target = row['target']
    if target.is_inference():
        target = target.clone()
    target = target.float().to(logits.device)
    gaps = target[:, None]-target[None, :]
    mask = gaps >= ranking_iou_gap
    differences = logits[:, None]-logits[None, :]
    ranking = F.softplus(-differences[mask]).mean() if mask.any() else logits.sum()*0
    return ranking + calibration_mse_weight*F.mse_loss(logits.sigmoid(), target)


def validate_feature_rows(rows, *, source_roles):
    reviewed_roles = normalise_source_roles(source_roles)
    groups = collections.defaultdict(set)
    identities = collections.defaultdict(set)
    source_identities = collections.defaultdict(set)
    count = collections.Counter()
    for row in rows:
        record = row['record']
        split = record['split']
        if split not in {'train', 'validation'}:
            raise ValueError('Unsupported feature-row role')
        digest = require_source_role(record.get('source_sha256'), split, reviewed_roles)
        source_identities[digest].add(split)
        groups[str(record['group_id'])].add(split)
        identities[str(record['video_id'])].add(split)
        if row['target'].ndim != 1:
            raise ValueError('Quality target must have exactly one dimension')
        n = len(row['target'])
        if row['visual'].shape != (n, 5, 2560) or row['geometry'].shape != (n, 16) or n < 1:
            raise ValueError('Feature shape differs from scoring architecture')
        if any(not torch.isfinite(row[key]).all() for key in ['visual', 'geometry', 'target']):
            raise ValueError('Non-finite feature rows')
        if not ((row['target'] >= 0).all() and (row['target'] <= 1).all()):
            raise ValueError('Quality targets must lie in [0, 1]')
        count[split] += 1
    if any(len(split) != 1 for split in list(groups.values())+list(identities.values())+list(source_identities.values())):
        raise ValueError('Source group or video crosses training roles')
    if not count['train'] or not count['validation']:
        raise ValueError('Both training and validation records are required')
    return dict(count)


@torch.inference_mode()
def evaluate_rank_head(model, rows, *, score_advantage=0.03):
    if not rows:
        raise ValueError('Evaluation rows are empty')
    model.eval()
    records = []
    for row in rows:
        target = row['target'].float().cpu()
        values = forward_row(model, row).sigmoid().cpu()
        raw = int(values.argmax())
        picked = raw if values[raw]-values[0] >= score_advantage else 0
        item = row['record']
        records.append(dict(video_id=str(item['video_id']), baseline=float(target[0]),
                            raw=float(target[raw]), chosen=float(target[picked]),
                            oracle=float(target.max()), chosen_index=picked, raw_index=raw))
    groups = collections.defaultdict(list)
    for row in records:
        groups[row['video_id']].append(row)
    summary = {name: statistics.mean(statistics.mean(x[name] for x in group) for group in groups.values())
               for name in ['baseline', 'raw', 'chosen', 'oracle']}
    summary.update(videos=len(groups), points=len(records), changed_points=sum(r['chosen_index'] != 0 for r in records))
    return dict(summary=summary, records=records)


def _write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def fit_rank_head(rows, output_dir, *, source_roles, settings=None, device='cuda', arm='visual'):
    """Retain video-balanced sampling, objective and early-stop semantics.

    Execution is left to the shared project. Output must be a new directory.
    If no checkpoint improves over the initial conservative choice, no trained
    checkpoint is silently promoted as a successful replacement.
    """
    if arm not in {'visual', 'geometry'}:
        raise ValueError('Unsupported scoring head')
    config = dict(DEFAULTS)
    if settings:
        unknown = set(settings)-set(config)
        if unknown:
            raise ValueError('Unknown training settings: '+','.join(sorted(unknown)))
        config.update(settings)
    integer_keys = ['seed', 'max_epochs', 'patience', 'videos_per_batch', 'points_per_video_max']
    if any(type(config[k]) is not int for k in integer_keys):
        raise ValueError('Integer training settings required')
    if any(config[k] < 1 for k in integer_keys if k != 'seed'):
        raise ValueError('Training budgets and batch sizes must be positive')
    if any(not math.isfinite(float(value)) for value in config.values()):
        raise ValueError('Training settings must be finite')
    if config['learning_rate'] <= 0 or any(config[k] < 0 for k in
           ['weight_decay', 'min_improvement', 'calibration_mse_weight', 'score_advantage']):
        raise ValueError('Invalid optimization settings')
    if config['gradient_clip'] <= 0 or config['ranking_iou_gap'] <= 0:
        raise ValueError('Gradient clip and ranking gap must be positive')
    validate_feature_rows(rows, source_roles=source_roles)
    target = torch.device(device)
    if target.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('Requested CUDA device unavailable')
    torch.set_num_threads(2)
    torch.manual_seed(config['seed'])
    random.seed(config['seed'])
    train = collections.defaultdict(list)
    validation = []
    for row in rows:
        if row['record']['split'] == 'train':
            train[str(row['record']['video_id'])].append(row)
        else:
            validation.append(row)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    model = RankHead(arm).to(target)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    initial = evaluate_rank_head(model, validation, score_advantage=config['score_advantage'])
    if abs(initial['summary']['chosen']-initial['summary']['baseline']) > 1e-8:
        raise ValueError('Zero-output initialization failed to retain baseline')
    _write_new(output/'CONFIG.json', dict(config, arm=arm, parameters=sum(p.numel() for p in model.parameters())))
    _write_new(output/'eval_0000.json', initial)
    best = initial['summary']['chosen']
    stale = steps = best_epoch = 0
    checkpoint = None
    for epoch in range(1, config['max_epochs']+1):
        model.train()
        ids = list(train)
        random.shuffle(ids)
        losses = []
        for start in range(0, len(ids), config['videos_per_batch']):
            optimizer.zero_grad(set_to_none=True)
            values = []
            for video_id in ids[start:start+config['videos_per_batch']]:
                selected = random.sample(train[video_id], min(config['points_per_video_max'], len(train[video_id])))
                values.append(torch.stack([ranking_loss(model, row,
                    ranking_iou_gap=config['ranking_iou_gap'],
                    calibration_mse_weight=config['calibration_mse_weight']) for row in selected]).mean())
            loss = torch.stack(values).mean()
            if not torch.isfinite(loss):
                raise ValueError('Non-finite optimization objective')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'])
            optimizer.step()
            steps += 1
            losses.append(float(loss.detach()))
        result = evaluate_rank_head(model, validation, score_advantage=config['score_advantage'])
        result.update(epoch=epoch, steps=steps, train_loss=statistics.mean(losses))
        _write_new(output/f'eval_{epoch:04d}.json', result)
        value = result['summary']['chosen']
        if value > best+config['min_improvement']:
            best, best_epoch, stale = value, epoch, 0
            checkpoint = output/f'checkpoint_{epoch:04d}.pt'
            torch.save(dict(state=model.state_dict(), arm=arm, epoch=epoch, steps=steps), checkpoint)
        else:
            stale += 1
        if stale >= config['patience']:
            break
    summary = dict(status='COMPLETE', epochs=epoch, steps=steps, best_epoch=best_epoch,
                   best_validation_quality=best, baseline_validation_quality=initial['summary']['baseline'],
                   stop='patience' if stale >= config['patience'] else 'budget',
                   selected_checkpoint=str(checkpoint) if checkpoint else None)
    _write_new(output/'SUMMARY.json', summary)
    return summary
