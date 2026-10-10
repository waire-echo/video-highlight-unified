"""Dense SAM2.1 propagation with repeated verified ORA prompts and explicit fallback."""
import argparse
import tempfile
from pathlib import Path
import numpy as np
import cv2
from common import read_json,write_json,load_rows,digest,sha,iou
from jobs import build_jobs
from ora_tools import read_frames
from fallback import fallback_focus


def mask_measure(mask,W,H):
    ys,xs=np.where(mask)
    if not len(xs):return None
    sx,sy=W/mask.shape[1],H/mask.shape[0]
    return {'box':[float(xs.min()*sx),float(ys.min()*sy),float((xs.max()+1)*sx),float((ys.max()+1)*sy)],
            'center':[float((xs.mean()+0.5)*sx),float((ys.mean()+0.5)*sy)],
            'area_fraction':len(xs)/mask.size}


def contain_center(center,box,w,h,frame_size,person=False):
    x1,y1,x2,y2=box
    cx,cy=center
    if x2-x1<=w:cx=float(np.clip(cx,x2-w/2,x1+w/2))
    if y2-y1<=h:
        cy=float(np.clip(cy,y2-h/2,y1+h/2))
    elif person:
        cy=y1+0.45*h
    return [cx,cy]


def has_verified_support(time,anchor_times,barrier_times,max_age):
    """An accepted anchor cannot support propagation across a rejected/absent review."""
    if not len(anchor_times):return False
    if len(barrier_times) and np.any(np.isclose(barrier_times,time,atol=1e-8,rtol=0)):return False
    left=barrier_times[barrier_times<time];right=barrier_times[barrier_times>time]
    same_interval=anchor_times[(anchor_times>(left.max() if len(left) else -np.inf)) &
                               (anchor_times<(right.min() if len(right) else np.inf))]
    return bool(len(same_interval) and np.min(np.abs(same_interval-time))<=max_age)


def run_job(predictor,job,ora,meta,ts,video,cfg,crop_width,ratio):
    import torch
    first,last=job['first'],job['last'];frames=list(range(first,last+1));W,H=meta['width'],meta['height']
    images=read_frames(video,frames,cfg['sam']['work_max_side'])
    centers=np.tile([W/2,H/2],(len(frames),1)).astype(float)
    quality=np.zeros(len(frames),float)
    boxes=np.zeros((len(frames),4),float)
    status=np.zeros(len(frames),np.int8) # 0 centered; 1 saliency; 2 valid SAM
    anchors=[a for a in ora['anchors'] if a.get('accepted')]
    measures={}
    if anchors:
        with tempfile.TemporaryDirectory(prefix='sam_clip_') as tmp:
            for j,f in enumerate(frames):
                cv2.imwrite(str(Path(tmp)/f'{j:06d}.jpg'),images[f],[cv2.IMWRITE_JPEG_QUALITY,95])
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                state=predictor.init_state(video_path=tmp,offload_video_to_cpu=True,offload_state_to_cpu=True,async_loading_frames=False)
                ih,iw=images[first].shape[:2]
                for anchor in anchors:
                    bx=np.asarray(anchor['box'],np.float32)*np.array([iw/W,ih/H,iw/W,ih/H],np.float32)
                    predictor.add_new_points_or_box(state,frame_idx=anchor['frame']-first,obj_id=1,box=bx)
                start=min(a['frame'] for a in anchors)-first
                for reverse in ([True,False] if start>0 else [False]):
                    for idx,ids,logits in predictor.propagate_in_video(state,start_frame_idx=start,reverse=reverse):
                        if not 0<=idx<len(frames):raise ValueError('SAM frame outside chunk')
                        mask=(logits[0,0]>0).detach().cpu().numpy()
                        measures[idx]=mask_measure(mask,W,H)
                predictor.reset_state(state)
                del state
            torch.cuda.empty_cache()
    acfg=cfg['sam'];anchor_map={a['frame']:a for a in anchors}
    anchor_times=np.array([ts[a['frame']] for a in anchors])
    barrier_times=np.array([ts[a['frame']] for a in
                           ora.get('abstentions',[])+ora.get('verification_barriers',[])])
    prior=None
    crop_height=crop_width*ratio[1]/ratio[0]
    for j,f in enumerate(frames):
        m=measures.get(j)
        valid=m is not None and acfg['min_mask_fraction']<=m['area_fraction']<=acfg['max_mask_fraction']
        if valid:
            valid=has_verified_support(ts[f],anchor_times,barrier_times,acfg['max_unverified_seconds'])
        if valid and f in anchor_map:
            valid=iou(m['box'],anchor_map[f]['box'])>=acfg['anchor_min_iou']
        if valid and prior is not None and f not in anchor_map:
            jump=np.linalg.norm((np.array(m['center'])-prior)/[W,H])
            valid=jump<=acfg['max_center_jump_fraction']
        if valid:
            centers[j]=contain_center(m['center'],m['box'],crop_width,crop_height,(W,H),meta['subject_hint']=='person')
            quality[j]=1;boxes[j]=m['box'];status[j]=2;prior=np.asarray(m['center'])
        else:
            prior=None
    # Sample fallback only where tracking is unreliable, then interpolate within this shot chunk.
    step=max(1,round(meta['fps']/2));sample=sorted(set(range(0,len(frames),step))|{len(frames)-1})
    fallback_points=[];fallback_quality=[]
    previous=None
    for j in sample:
        im=images[frames[j]];point,q=fallback_focus(im,previous);previous=im
        fallback_points.append([point[0]/im.shape[1]*W,point[1]/im.shape[0]*H]);fallback_quality.append(q)
    ft=ts[np.array(frames)[sample]]
    alltimes=ts[frames]
    fallback_points=np.asarray(fallback_points)
    for axis in range(2):
        values=np.interp(alltimes,ft,fallback_points[:,axis])
        centers[status!=2,axis]=values[status!=2]
    fq=np.interp(alltimes,ft,fallback_quality)
    quality[status!=2]=fq[status!=2]
    status[(status!=2)&(fq>0)]=1
    return {'frames':np.asarray(frames),'centers':centers,'quality':quality,'boxes':boxes,'status':status}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--video-dir',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--inputs',default='inputs');p.add_argument('--shots',default='output/shots')
    p.add_argument('--ora',default='output/ora');p.add_argument('--output',default='output/sam')
    p.add_argument('--config',default='config/default.json');p.add_argument('--ledger',default='output/parameter_ledger.json')
    p.add_argument('--scope',choices=['partial','all'],default='all');p.add_argument('--ids',default='')
    a=p.parse_args();cfg=read_json(a.config);ledger=read_json(a.ledger)
    if not ledger.get('pass') or ledger['total_upper_bound']>cfg['parameter_limit']:raise ValueError('Budget gate failed')
    if sha(a.checkpoint)!=ledger['sam_checkpoint_sha256']:raise ValueError('SAM checkpoint changed')
    import torch
    if not torch.cuda.is_available():raise RuntimeError('SAM inference requires GPU; preparation/tests do not')
    from sam2.build_sam import build_sam2_video_predictor
    predictor=build_sam2_video_predictor(cfg['sam']['config'],a.checkpoint,device='cuda')
    actual=sum(p.numel() for p in predictor.parameters())
    if actual>ledger['sam_tensor_elements']:raise ValueError('SAM runtime count exceeds audited upper bound')
    manifest=read_json(Path(a.inputs)/'manifest.json');ts_cache=np.load(Path(a.inputs)/'timestamps.npz')
    if sha(Path(a.inputs)/'baseline.zip')!=manifest['baseline_sha256'] or sha(Path(a.inputs)/'timestamps.npz')!=manifest['timestamps_sha256']:
        raise ValueError('Input cache changed')
    selection=set(a.ids.split(',')) if a.ids else None
    code=digest({f:sha(Path(__file__).parent/f) for f in ['sam_stage.py','fallback.py','jobs.py','ora_tools.py']})
    for row in load_rows(Path(a.inputs)/'baseline.zip'):
        vid=str(row['video_id']);m=manifest['videos'][vid]
        if selection is not None and vid not in selection:continue
        if a.scope=='partial' and m['subject_hint'] not in cfg['partial_hints']:continue
        if not row['predictions']:continue
        video=Path(a.video_dir)/(vid+'.mp4')
        if sha(video)!=m['source_sha256']:raise ValueError('Source mismatch')
        shot=read_json(Path(a.shots)/(vid+'.json'));ts=ts_cache[vid]
        if shot['source_sha256']!=m['source_sha256']:raise ValueError('Stale shot cache')
        for job in build_jobs(row,m,ts,shot['cuts'],cfg['ora']['max_chunk_seconds'],cfg['ora']['max_chunk_frames']):
            op=Path(a.ora)/vid/(job['id']+'.json');ora=read_json(op)
            if ora['source_sha256']!=m['source_sha256'] or ora['job']!=job:raise ValueError('Stale ORA job')
            ora_code=digest({f:sha(Path(__file__).parent/f) for f in ['ora_stage.py','ora_tools.py','ora_verify.py','jobs.py']})
            if (ora['baseline_sha256']!=manifest['baseline_sha256'] or ora['config_digest']!=digest(cfg)
                or ora['shots_signature']!=shot['signature'] or ora['code_digest']!=ora_code
                or ora['ledger_signature']!=ledger['signature']):raise ValueError('Re-run ORA with current inputs/code/config')
            out=Path(a.output)/vid/job['id'];meta_file=out.with_suffix('.json')
            sig=digest({'ora':sha(op),'sam':ledger['sam_checkpoint_sha256'],'config':cfg['sam'],'code':code,
                        'baseline':manifest['baseline_sha256']})
            if meta_file.exists() and out.with_suffix('.npz').exists():
                previous=read_json(meta_file)
                if previous.get('signature')==sig and previous.get('npz_sha256')==sha(out.with_suffix('.npz')):
                    continue
            data=run_job(predictor,job,ora,m,ts,video,cfg,row['predictions'][0]['bboxes'][2],row['targetRatioWH'])
            out.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(out.with_suffix('.npz'),**data)
            write_json(meta_file,{'signature':sig,'video_id':vid,'job':job,'ora_sha256':sha(op),
                'source_sha256':m['source_sha256'],'sam_frames':int(np.sum(data['status']==2)),
                'saliency_frames':int(np.sum(data['status']==1)),'center_frames':int(np.sum(data['status']==0)),
                'config_digest':digest(cfg),'npz_sha256':sha(out.with_suffix('.npz'))})
            print(f'{vid}/{job["id"]}: SAM={np.sum(data["status"]==2)}/{len(data["frames"])}',flush=True)


if __name__=='__main__':main()
