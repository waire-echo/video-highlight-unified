"""Variable-scale composition and conservative visibility states, without new weights.

States are evidence-based hypotheses, not calibrated labels of physical causes.
An unclear review is not proof of departure. A rejected graphic is never a seed.
"""
import cv2
import numpy as np
from scipy.ndimage import median_filter
from scipy.sparse import diags,eye
from scipy.sparse.linalg import spsolve
from common import center_of,box_from_center,iou


def valid_box(b,W,H):
    return len(b)==4 and np.isfinite(b).all() and 0<=b[0]<b[2]<=W and 0<=b[1]<b[3]<=H


def nearest_yolo(frames,ts,observations,cfg):
    result={}
    if observations is None or len(observations)==0:return result
    times=ts[observations[:,0].astype(int)]
    for f in frames:
        k=int(np.searchsorted(times,ts[f]));choices=[j for j in (k-1,k) if 0<=j<len(times)]
        j=min(choices,key=lambda j:abs(times[j]-ts[f]))
        if abs(times[j]-ts[f])<=cfg['yolo_max_age_seconds'] and observations[j,5]>=cfg['yolo_min_confidence']:
            result[f]=observations[j,1:5].copy()
    return result


def visual_evidence(video,frames,boxes):
    """ROI sharpness and simple appearance signatures; sequential original-frame decoding."""
    cap=cv2.VideoCapture(str(video));wanted=set(frames);result={}
    if not cap.isOpened():raise RuntimeError('Cannot decode '+str(video))
    prior=None
    try:
        for f in range(max(frames)+1):
            ok,im=cap.read()
            if not ok:raise RuntimeError(f'Decode failed {video}/{f}')
            if f not in wanted:continue
            h,w=im.shape[:2];b=boxes.get(f)
            if b is not None:prior=b
            roi=im
            if prior is not None:
                x1,y1,x2,y2=np.asarray(prior,int)
                x1,y1=max(0,x1-8),max(0,y1-8);x2,y2=min(w,x2+8),min(h,y2+8)
                if x2>x1 and y2>y1:roi=im[y1:y2,x1:x2]
            roi=cv2.resize(roi,(96,96))
            gray=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
            sharp=float(cv2.Laplacian(gray,cv2.CV_32F).var())
            hsv=cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
            hist=cv2.calcHist([hsv],[0,1],None,[12,4],[0,180,0,256]).ravel()
            hist/=hist.sum()+1e-9
            result[f]=(sharp,hist)
    finally:cap.release()
    return result


def edge_departure(previous,last,dt,frame_size,cfg):
    if previous is None or dt<=0:return False
    W,H=frame_size
    c0=(np.asarray(previous[:2])+previous[2:])/2
    c1=(np.asarray(last[:2])+last[2:])/2
    velocity=(c1-c0)/dt/[W,H]
    e=cfg['edge_fraction'];v=cfg['exit_velocity_fraction']
    return bool((last[0]<e*W and velocity[0]<-v) or
                (last[2]>(1-e)*W and velocity[0]>v) or
                (last[1]<e*H and velocity[1]<-v) or
                (last[3]>(1-e)*H and velocity[1]>v))


def subject_crop(box,frame_size,ratio,scene,cfg):
    W,H=frame_size;rh=ratio[1]/ratio[0];maximum=min(W,H/rh)
    bw,bh=box[2]-box[0],box[3]-box[1]
    floor=cfg['scene_min_width_fraction'] if scene else cfg['subject_min_width_fraction']
    width=min(maximum,max(floor*maximum,cfg['subject_context_factor']*max(bw,bh/rh)))
    return np.array([(box[0]+box[2])/2,(box[1]+box[3])/2,width],float)


def bridge_compatible(first,last,dt,size,hist0,hist1,cfg):
    if dt<=0:return False
    a=np.asarray(first);b=np.asarray(last)
    speed=np.linalg.norm(((a[:2]+a[2:]-b[:2]-b[2:])/2)/np.asarray(size))/dt
    area0=(a[2]-a[0])*(a[3]-a[1]);area1=(b[2]-b[0])*(b[3]-b[1])
    similarity=float(np.minimum(hist0,hist1).sum())
    return bool(speed<=cfg['bridge_max_speed_fraction'] and .4<=area1/max(area0,1e-6)<=2.5
                and similarity>=cfg['appearance_min_similarity'])


def compose_segment(frames,ts,observations,yolo,baseline,ratio,frame_size,scene,
                    routes,rejected,absent_frames,visual,cfg):
    """Return targets plus protected extents. Call separately for each shot/temporal group."""
    W,H=frame_size;maximum=min(W,H*ratio[0]/ratio[1]);frames=list(frames)
    candidates={};sources={}
    for f in frames:
        obs=observations[f]
        b=np.asarray(obs.get('box',[0,0,0,0]),float)
        if obs.get('status')==2 and valid_box(b,W,H):candidates[f]=b;sources[f]='visible_sam'
        elif f in yolo and not scene and valid_box(yolo[f],W,H):
            # Reject overlap with an identified overlay/wrong object, not all other detections.
            blocked=any((abs(ts[r['frame']]-ts[f])<=.12 or
                         (r['reason']=='graphic_or_text' and r.get('first',r['frame'])<=f<=r.get('last',r['frame']))) and
                        r['reason'] in ('graphic_or_text','wrong_object') and
                        iou(yolo[f],r['box'])>.2 for r in rejected)
            if not blocked:candidates[f]=yolo[f];sources[f]='visible_yolo'
    good=sorted(candidates);good_times=ts[good]
    reference_sharp=np.median([visual[f][0] for f in good]) if good else 0.
    targets=[];weights=[];protected=[];states=[]
    absent_frames=sorted(set(absent_frames)&set(frames))
    for f in frames:
        old=np.array([*center_of(baseline[f],ratio),baseline[f][2]],float)
        if f in candidates:
            b=candidates[f]
            target=subject_crop(b,frame_size,ratio,scene,cfg)
            if routes.get(f)=='multiple' and sources[f]=='visible_yolo':
                # A single detector cannot define the complete interacting group.
                target=old
            targets.append(target);weights.append(1.);protected.append(b);states.append(sources[f]);continue
        k=int(np.searchsorted(good_times,ts[f]));left=good[k-1] if k else None;right=good[k] if k<len(good) else None
        age=ts[f]-ts[left] if left is not None else np.inf
        blurry=reference_sharp>1 and visual[f][0]<cfg['blur_relative_threshold']*reference_sharp
        # Only actual absence observations count; unclear/graphic/bad_extent do not.
        negatives=[g for g in absent_frames if left is not None and ts[left]<ts[g]<=ts[f]]
        repeated_absent=len(negatives)>=2 and ts[negatives[-1]]-ts[negatives[0]]>=.2
        # Nearest-frame YOLO reuse can duplicate the final box several times.
        # Estimate departure from a distinct recent observation, not a zero-velocity duplicate.
        prior=None
        if left is not None:
            for g in reversed(good[:k-1]):
                dt=ts[left]-ts[g]
                if dt>.35:break
                if dt>=.05-1e-8 and np.linalg.norm(candidates[left]-candidates[g])>1:
                    prior=g;break
        departing=left is not None and edge_departure(candidates.get(prior),candidates[left],
                         ts[left]-ts[prior] if prior is not None else 0,frame_size,cfg)
        exited=not blurry and age>=cfg['exit_min_missing_seconds'] and (repeated_absent or departing)
        bridge=(left is not None and right is not None and not exited and not repeated_absent
                and ts[right]-ts[left]<=cfg['bridge_seconds']
                and bridge_compatible(candidates[left],candidates[right],ts[right]-ts[left],frame_size,
                                      visual[left][1],visual[right][1],cfg))
        if bridge:
            alpha=age/(ts[right]-ts[left]);b=candidates[left]*(1-alpha)+candidates[right]*alpha
            target=subject_crop(b,frame_size,ratio,scene,cfg)
            target[2]=min(maximum,target[2]*1.08)
            state='blur_bridge' if blurry else 'occlusion_candidate_bridge';weight=.7;protect=b
        elif not exited and left is not None and age<=cfg['hold_seconds']:
            # Short uncertain gap: hold position and gently widen; do not pan toward center.
            target=subject_crop(candidates[left],frame_size,ratio,scene,cfg)
            target[2]=min(maximum,target[2]*(1+.2*age/cfg['hold_seconds']))
            state='blur_hold' if blurry else 'uncertain_hold';weight=.5;protect=None
        else:
            # The broad previous crop is a conservative candidate, not proof of object presence.
            target=old;weight=.2;protect=None
            state='exit_likely' if exited else 'unresolved_baseline'
        targets.append(target);weights.append(weight);protected.append(protect);states.append(state)
    return np.asarray(targets),np.asarray(weights),protected,states


def smooth_crops(times,targets,weights,protected,frame_size,ratio,cfg):
    """Smooth center and log-width, then project into legal geometry and subject coverage."""
    W,H=frame_size;rh=ratio[1]/ratio[0];maximum=min(W,H/rh)
    times=np.asarray(times,float);targets=np.asarray(targets,float)
    if not np.isfinite(targets).all() or np.any(targets[:,2]<=0):raise ValueError('Bad adaptive target')
    n=max(1,int(np.ceil((times[-1]-times[0])*cfg['grid_hz']))+1)
    grid=np.linspace(times[0],times[-1],n)
    values=targets.copy();values[:,:2]/=[W,H];values[:,2]=np.log(values[:,2]/maximum)
    desired=np.column_stack([np.interp(grid,times,values[:,j]) for j in range(3)])
    w=np.maximum(.05,np.interp(grid,times,weights))
    desired[:,2]=median_filter(desired[:,2],size=min(5,n if n%2 else max(1,n-1)),mode='nearest')
    if n>=3:
        D=diags([np.ones(n-2),-2*np.ones(n-2),np.ones(n-2)],[0,1,2],shape=(n-2,n))
        A=diags(w)+cfg['smoothing_acceleration']*(D.T@D)+eye(n)*1e-9
        desired=np.column_stack([spsolve(A.tocsc(),w*desired[:,j]) for j in range(3)])
    if n>1:
        limit=cfg['max_log_zoom_per_second']*(grid[1]-grid[0])
        for i in range(1,n):desired[i,2]=np.clip(desired[i,2],desired[i-1,2]-limit,desired[i-1,2]+limit)
        for i in range(n-2,-1,-1):desired[i,2]=np.clip(desired[i,2],desired[i+1,2]-limit,desired[i+1,2]+limit)
    path=np.column_stack([np.interp(times,grid,desired[:,j]) for j in range(3)])
    path[:,:2]*=[W,H];path[:,2]=np.exp(path[:,2])*maximum
    out=[];containment_conflicts=0;expanded=0
    for i,(cx,cy,width) in enumerate(path):
        b=protected[i]
        if b is not None:
            bw,bh=b[2]-b[0],b[3]-b[1];pad=cfg['containment_margin']
            padded=np.array([max(0,b[0]-bw*pad),max(0,b[1]-bh*pad),min(W,b[2]+bw*pad),min(H,b[3]+bh*pad)])
            needed=max(padded[2]-padded[0]+2,(padded[3]-padded[1]+2)/rh)
            if needed>width:expanded+=1
            width=min(maximum,max(width,needed))
            if needed>maximum+1e-6:containment_conflicts+=1
        width=int(min(np.floor(maximum+1e-8),max(2,np.ceil(width))))
        if b is not None:
            h=width*rh
            if padded[2]-padded[0]<=width:cx=np.clip(cx,padded[2]-width/2,padded[0]+width/2)
            if padded[3]-padded[1]<=h:cy=np.clip(cy,padded[3]-h/2,padded[1]+h/2)
        out.append(box_from_center([cx,cy],width,ratio,W,H))
    return out,{'width_min':min(b[2] for b in out),'width_max':max(b[2] for b in out),
                'coverage_forced_expansions':expanded,'unfit_subject_frames':containment_conflicts,
                'note':'Zoom-rate target is soft when subject coverage requires a larger crop.'}
