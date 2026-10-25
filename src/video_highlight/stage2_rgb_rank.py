from pathlib import Path
import json,hashlib
import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image
from .rgb_rank_core import candidates,path_search,legal

class RankHead(nn.Module):
    def __init__(self,arm):
        super().__init__();self.arm=arm
        self.geometry=nn.Sequential(nn.LayerNorm(16),nn.Linear(16,32),nn.GELU())
        if arm=='visual':self.visual=nn.Sequential(nn.LayerNorm(2560),nn.Linear(2560,64),nn.GELU())
        self.score=nn.Sequential(nn.Linear(32+(320 if arm=='visual' else 0),64),nn.GELU(),nn.Linear(64,1))
        nn.init.zeros_(self.score[-1].weight);nn.init.zeros_(self.score[-1].bias)

    def forward(self,v,g):
        x=self.geometry(g)
        if self.arm=='visual':x=torch.cat([x,self.visual(v).flatten(1)],-1)
        return self.score(x).squeeze(-1)

def roi(grid, box, size):
    h,w,_=grid.shape; x,y,cw,ch=box
    xs=torch.arange(w+1,dtype=torch.float32,device=grid.device)*size[0]/w
    ys=torch.arange(h+1,dtype=torch.float32,device=grid.device)*size[1]/h
    wx=(torch.minimum(xs[1:],grid.new_tensor(x+cw))-torch.maximum(xs[:-1],grid.new_tensor(x))).clamp_min(0)
    wy=(torch.minimum(ys[1:],grid.new_tensor(y+ch))-torch.maximum(ys[:-1],grid.new_tensor(y))).clamp_min(0)
    weights=wy[:,None]*wx[None,:];assert weights.sum()>0
    return (grid*weights[:,:,None]).sum((0,1))/weights.sum()

def encode(rec, cache):
    size,ratio=rec['size'],rec['ratio'];k=rec['feature_index']
    frame_grid=cache['grids'][k].float();W,H=size
    first,last=rec['segment'];context=[]
    for j in [max(0,k-2),min(len(cache['times'])-1,k+2)]:
        f=int(cache['frames'][j])
        if first<=f<=last:
            g=cache['grids'][j].float()
            # Conservative feature discontinuity guard, not verified shot detection.
            sim=torch.nn.functional.cosine_similarity(frame_grid.mean((0,1)),g.mean((0,1)),dim=0)
            if sim>=.65:context.append(g)
    context_grid=torch.stack(context).mean(0) if context else frame_grid
    b=rec['baseline'];out=[];geometry=[]
    for x,y,w in rec['candidates']:
        h=w*ratio[1]/ratio[0];vertical=H-h>=W-w
        bins=[[x,y+h*i/3,w,h/3] if vertical else [x+w*i/3,y,w/3,h] for i in range(3)]
        out.append(torch.stack([frame_grid.mean((0,1))]+[roi(frame_grid,z,size) for z in bins]+[roi(context_grid,[x,y,w,h],size)]))
        sub=rec['subject'];box=sub.get('box');det=bool(sub.get('detected') and box)
        rel=[(box[0]-x)/w,(box[1]-y)/h,(box[2]-x)/w,(box[3]-y)/h] if det else [0]*4
        geometry.append([x/W,y/H,w/W,h/H,(x-b[0])/W,(y-b[1])/H,(W-w)/W,(H-h)/H]+rel+[float(sub.get('confidence',0)),float(det),float(sub.get('class_name')=='person'),float(det and sub.get('class_name')!='person')])
    return torch.stack(out).half(),torch.tensor(geometry,dtype=torch.float32)

@torch.inference_mode()
def encode_actual(rec,cache,image,vision,processor):
    v,g=encode(rec,cache);rgb=Image.fromarray(image[:,:,::-1]);ratio=rec['ratio'];crops=[]
    for x,y,w in rec['candidates']:
        h=round(w*ratio[1]/ratio[0]);crops.append(rgb.crop((x,y,x+w,y+h)))
    for offset in range(0,len(crops),4):
        group=crops[offset:offset+4];inputs=processor(images=group,return_tensors='pt',min_pixels=65536,max_pixels=196608);grids=inputs['image_grid_thw'].to('cuda');out=vision(inputs['pixel_values'].to('cuda',dtype=torch.bfloat16),grid_thw=grids).pooler_output.float();cursor=0
        for j,(t,hh,ww) in enumerate(grids.tolist()):
            merge=vision.config.spatial_merge_size;hh//=merge;ww//=merge;assert t==1;spatial=out[cursor:cursor+hh*ww].reshape(hh,ww,-1);cursor+=hh*ww
            index=offset+j;w,h=group[j].size;vertical=rec['size'][1]-h>=rec['size'][0]-w
            bins=[[0,h*k/3,w,h/3] if vertical else [w*k/3,0,w/3,h] for k in range(3)]
            v[index,1:4]=torch.stack([roi(spatial,z,[w,h]) for z in bins]).cpu().half()
        assert cursor==out.shape[0]
    return v,g

def asset_contract(folder):
    """Validate actual bytes, not only declared names; safe before loading a GPU model."""
    folder=Path(folder)
    manifest=json.loads((folder/'ASSETS.json').read_text())
    if manifest['version']!='B_RGBRank_DP_v1':raise ValueError('Unexpected RGB backend asset version')
    required={'visual.safetensors','config.json','processor_config.json','checkpoint_0001.pt'}
    if set(manifest['files'])!=required:raise ValueError('Incomplete RGB backend assets')
    for name,entry in manifest['files'].items():
        p=folder/name;h=hashlib.sha256()
        with p.open('rb') as f:
            for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
        if h.hexdigest()!=entry['sha256'] or p.stat().st_size!=entry['bytes']:raise ValueError('RGB backend asset changed: '+name)
    return manifest


_runtime=None
def load_runtime(folder):
    global _runtime
    key=str(Path(folder).resolve())
    if _runtime is not None:
        if _runtime[0]!=key:raise RuntimeError('Cannot mix RGB models in one worker')
        return _runtime[1:]
    contract=asset_contract(folder)
    from transformers import AutoConfig,AutoImageProcessor
    from .rgb_vision_compat import RGBTrainingVisionModel as Qwen3_5VisionModel
    from safetensors.torch import load_file
    config=AutoConfig.from_pretrained(folder,local_files_only=True).vision_config
    config._attn_implementation='sdpa'
    vision=Qwen3_5VisionModel(config)
    vision.load_state_dict(load_file(str(Path(folder)/'visual.safetensors')),strict=True)
    vision.requires_grad_(False).eval().to('cuda',dtype=torch.bfloat16)
    processor=AutoImageProcessor.from_pretrained(folder,local_files_only=True)
    ck=torch.load(Path(folder)/'checkpoint_0001.pt',map_location='cpu',weights_only=False)
    assert ck['epoch']==1
    head=RankHead('visual').cuda().eval();head.load_state_dict(ck['state'],strict=True);head.requires_grad_(False)
    _runtime=(key,vision,processor,head,contract)
    return _runtime[1:]


@torch.inference_mode()
def extract_source(video,vision,processor,asset_identity):
    """Same original-frame4Hz sampling and6x6 pooling as the trained feature view."""
    import cv2
    path=Path(video['path']);cap=cv2.VideoCapture(str(path))
    if not cap.isOpened():raise ValueError('Cannot decode source video')
    frames=[];times=[];pts=[];grids=[];next_time=0.;frame=-1
    try:
        while cap.grab():
            frame+=1;at=float(cap.get(cv2.CAP_PROP_POS_MSEC))/1000
            if pts and at<=pts[-1]:raise ValueError('Source PTS must increase')
            pts.append(at)
            if at+1e-6<next_time:continue
            ok,image=cap.retrieve()
            if not ok:raise ValueError('Source frame cannot decode')
            next_time=(int(at*4+1e-5)+1)/4
            inputs=processor(images=Image.fromarray(image[:,:,::-1]),return_tensors='pt',min_pixels=65536,max_pixels=196608)
            grid=inputs['image_grid_thw'].to('cuda');out=vision(inputs['pixel_values'].to('cuda',dtype=torch.bfloat16),grid_thw=grid).pooler_output
            t,h,w=map(int,grid[0].tolist());merge=vision.config.spatial_merge_size;assert t==1
            spatial=out.float().reshape(h//merge,w//merge,-1)
            small=F.adaptive_avg_pool2d(spatial.permute(2,0,1).unsqueeze(0),(6,6))[0].permute(1,2,0)
            grids.append(small.cpu().half());frames.append(frame);times.append(at)
    finally:cap.release()
    if not grids:raise ValueError('No visual source samples')
    return dict(source_sha256=video['sha256'],frames=torch.tensor(frames),times=torch.tensor(times,dtype=torch.float64),native_pts=torch.tensor(pts,dtype=torch.float64),grids=torch.stack(grids),visual_asset_identity=asset_identity)


def read_native_frame_batches(video_path, sample_ids, batch_size):
    """Decode original frame numbers from zero, without seeking or unbounded RGB storage."""
    import cv2
    if not sample_ids:
        return
    if batch_size <= 0 or sample_ids != sorted(set(sample_ids)) or sample_ids[0] < 0:
        raise ValueError('Native sample IDs must be sorted, unique and nonnegative')
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise ValueError('Cannot open RGB source video')
    wanted = 0
    ids, images = [], []
    try:
        for frame in range(sample_ids[-1] + 1):
            if not cap.grab():
                raise ValueError(f'Native RGB decode failed at frame {frame}')
            if frame != sample_ids[wanted]:
                continue
            ok, image = cap.retrieve()
            if not ok:
                raise ValueError(f'Native RGB retrieval failed at frame {frame}')
            ids.append(frame)
            images.append(image)
            wanted += 1
            if len(ids) >= batch_size:
                yield ids, images
                ids, images = [], []
            if wanted == len(sample_ids):
                break
        if wanted != len(sample_ids):
            raise ValueError('Native RGB source ended before requested frames')
        if ids:
            yield ids, images
    finally:
        cap.release()


@torch.inference_mode()
def refine(ctx,video,first,size,baseline,folder):
    from vendor.subject_association import apply_control
    from vendor.yolo_center import read_frame_batches
    missing_evidence = bool(ctx.get('rgb_missing_evidence', False))
    if not baseline['predictions']:return baseline
    vision,processor,head,manifest=load_runtime(folder)
    identity=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    work=ctx['video_dir'];cache_path=work/'rgb_source_features_v1.pt'
    if ctx.get('resume') and cache_path.exists():
        cache=torch.load(cache_path,map_location='cpu',weights_only=False)
        if cache.get('source_sha256')!=video['sha256'] or cache.get('visual_asset_identity')!=identity:raise ValueError('RGB feature cache identity mismatch')
    else:
        cache=extract_source(video,vision,processor,identity)
        temp=cache_path.with_suffix('.pt.writing');torch.save(cache,temp);temp.replace(cache_path)
    if len(cache['native_pts'])!=first['frame_count']:raise ValueError('RGB source frame count mismatch')
    ratio=first['targetRatioWH'];pts=cache['native_pts'].tolist();ft=cache['times'].tolist()
    original={p['frame']:p['bboxes'][:] for p in baseline['predictions']};crops={f:b[:] for f,b in original.items()};traces=[]
    for start,end in first['segments_frame']:
        det=json.loads((work/f'detections_{start}_{end}.json').read_text())['records'];tracked,info=apply_control(det);byframe={r['frame']:r for r in tracked};detframes=sorted(byframe)
        frames=sorted(set([int(f) for f in cache['frames'] if start<=int(f)<=end]+[start,end]));cslist=[];values=[];times=[];resets=[0];previous_grid=None;previous_missing=False
        records=[]
        for frame in frames:
            k=min(range(len(ft)),key=lambda j:abs(ft[j]-pts[frame]));near=min(detframes,key=lambda f:abs(f-frame));sub=byframe[near]
            if abs(pts[near]-pts[frame])>.20:sub={}
            cs=candidates(original[frame],size,ratio,sub.get('box'))
            records.append(dict(frame=frame,size=size,ratio=ratio,feature_index=k,baseline=original[frame],subject=sub,candidates=cs,segment=[start,end]))
        # Decode in bounded batches; do not retain all native RGB images of long clips.
        byid={r['frame']:r for r in records};scores={}
        needed=[r['frame'] for r in records if len(r['candidates'])>1 and (missing_evidence or r['subject'].get('detected'))]
        frame_reader=read_native_frame_batches if missing_evidence else read_frame_batches
        for ids,images in frame_reader(Path(video['path']),needed,4) if needed else []:
            for f,image in zip(ids,images):
                v,g=encode_actual(byid[f],cache,image,vision,processor);scores[f]=head(v.float().cuda(),g.cuda()).sigmoid().cpu().tolist()
        for rec in records:
            frame=rec['frame'];k=rec['feature_index'];sub=rec['subject'];cs=rec['candidates'];v=scores.get(frame,[.5]*len(cs));adv=[x-v[0] for x in v];missing=not sub.get('detected') and (not missing_evidence or max(adv)<.03)
            bounded=[0.]+[x if x>=.03 and not missing else -1e6 for x in adv[1:]]
            grid=cache['grids'][k].float().mean((0,1));cut=previous_grid is not None and float(F.cosine_similarity(grid,previous_grid,dim=0))<.65
            reset=bool(cut or missing or previous_missing or sub.get('origin')=='original_rule_reinitialization')
            if reset and cslist:resets.append(len(cslist))
            cslist.append(cs);values.append(bounded);times.append(pts[frame]);previous_grid=grid;previous_missing=missing
        picked=path_search(cslist,values,times,size,resets=resets,movement=.05);cuts=sorted(set(resets+[len(frames)]))
        for left,right in zip(cuts,cuts[1:]):
            if right-left<2:continue
            for i in range(left,right-1):
                fa,fb=frames[i:i+2];a=cslist[i][picked[i]];bb=cslist[i+1][picked[i+1]];da=[a[j]-original[fa][j] for j in [0,1]];db=[bb[j]-original[fb][j] for j in [0,1]]
                for f in range(fa,fb+1):
                    w=original[f][2];h=w*ratio[1]/ratio[0];fraction=(pts[f]-pts[fa])/(pts[fb]-pts[fa]);shift=[(1-fraction)*da[j]+fraction*db[j] for j in [0,1]]
                    crops[f]=[min(max(0,round(original[f][0]+shift[0])),int(size[0]-w)),min(max(0,round(original[f][1]+shift[1])),int(size[1]-h+1e-6)),w];assert legal(crops[f],size,ratio)
        traces.append(dict(first=start,last=end,anchor_frames=frames,selected_indices=picked,resets=resets,tracking=info))
    result=dict(video_id=video['video_id'],targetRatioWH=ratio,predictions=[dict(frame=f,bboxes=box) for f,box in crops.items()])
    (work/'rgb_rank_audit.json').write_text(json.dumps(dict(version='B_RGBRank_MissingEvidenceNative_v1' if missing_evidence else 'B_RGBRank_DP_v1',asset_identity=identity,changed_frames=sum(crops[f]!=original[f] for f in crops),frames=len(crops),groups=traces)))
    return result
