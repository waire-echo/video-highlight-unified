"""Consensus target hypotheses and hard coverage gates. No per-video rules."""
import argparse,copy,itertools,json,math,sys,datetime,zipfile,os
from pathlib import Path
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/scene_general_v1'))
from pipeline import Model,make_panel,crop_path,interpolate_crops,read_frames
sys.path.insert(0,str(ROOT/'src'))
from common import read_json,write_json,sha,digest,load_rows,export,groups,box_from_center,iou

def coverage(crop,box,ratio):
    x,y,w=crop;h=w*ratio[1]/ratio[0]
    area=(box[2]-box[0])*(box[3]-box[1])
    return max(0,min(x+w,box[2])-max(x,box[0]))*max(0,min(y+h,box[3])-max(y,box[1]))/max(area,1e-12)

def descriptor(image,box,size):
    W,H=size;b=np.array(box)*np.array([image.shape[1]/W,image.shape[0]/H]*2)
    x,y,r,t=np.round(b).astype(int);roi=image[max(0,y):max(y+1,t),max(0,x):max(x+1,r)]
    if not roi.size:return np.zeros(40)
    hsv=cv2.cvtColor(cv2.resize(roi,(48,48)),cv2.COLOR_BGR2HSV)
    hist=cv2.calcHist([hsv],[0,1],None,[8,4],[0,180,0,256]).ravel()
    gray=cv2.cvtColor(cv2.resize(roi,(48,48)),cv2.COLOR_BGR2GRAY)
    value=cv2.calcHist([gray],[0],None,[8],[0,256]).ravel()
    result=np.r_[hist/(hist.sum()+1e-9),value/(value.sum()+1e-9)]/2
    return result

def consensus(proposals,images,times,size,cfg):
    valid=[i for i,p in enumerate(proposals) if p.get('box') is not None]
    boxes={i:np.array(proposals[i]['box'],float) for i in valid};hist={i:descriptor(images[i],boxes[i],size) for i in valid}
    edges={};quality={};details=[];W,H=size
    for i,j in itertools.combinations(valid,2):
        a,b=boxes[i],boxes[j];area1=np.prod(a[2:]-a[:2]);area2=np.prod(b[2:]-b[:2]);scale=max(area1,area2)/max(min(area1,area2),1e-9)
        distance=float(np.linalg.norm(((a[:2]+a[2:]-b[:2]-b[2:])/2)/[W,H]))
        dt=abs(float(times[j]-times[i]));limit=min(cfg['position_max_fraction'],cfg['position_base_fraction']+cfg['position_fraction_per_second']*dt)
        appearance=float(np.minimum(hist[i],hist[j]).sum());overlap=iou(a,b)
        compatible=bool(scale<=cfg['maximum_area_ratio'] and distance<=limit and (appearance>=cfg['minimum_histogram_intersection'] or overlap>=cfg['minimum_pair_iou']))
        edges[i,j]=compatible;quality[i,j]=appearance+overlap-distance
        details.append({'indices':[i,j],'compatible':compatible,'area_ratio':float(scale),'position_distance':distance,'appearance_intersection':appearance,'iou':overlap})
    best=[];bestscore=-float('inf')
    for n in range(len(valid),1,-1):
        for subset in itertools.combinations(valid,n):
            if all(edges[i,j] for i,j in itertools.combinations(subset,2)):
                score=sum(quality[i,j] for i,j in itertools.combinations(subset,2))
                if score>bestscore:best=list(subset);bestscore=score
        if best:break
    required=max(2,math.ceil(len(proposals)*cfg['minimum_support_fraction']))
    trusted=len(best)>=required
    return {'trusted':trusted,'accepted_indices':best if trusted else [],'largest_consistent_indices':best,'required_support':required,'rejected_indices':[i for i in range(len(proposals)) if i not in best] if trusted else list(range(len(proposals))),'pair_checks':details,'interpretation':'Geometric/appearance consistency hypothesis, not verified semantic identity.'}

def predicted_targets(proposals,indices,times):
    bs=np.array([proposals[i]['box'] for i in indices]);t=np.array(times)[indices]
    return np.column_stack([np.interp(times,t,bs[:,k]) for k in range(4)])

def target_candidates(targets,times,size,ratio,basecfg,cfg):
    W,H=size;rh=ratio[1]/ratio[0];maximum=int(min(W,H/rh));out=[]
    widths=np.maximum(targets[:,2]-targets[:,0],(targets[:,3]-targets[:,1])/rh)
    centers=(targets[:,:2]+targets[:,2:])/2
    width=min(maximum,math.ceil(max(cfg['minimum_width_fraction']*maximum,cfg['target_context_factor']*max(widths)+2)))
    path,mode=crop_path(centers.copy(),width,times,size,ratio,basecfg)
    protected=[]
    for b,target in zip(path,targets):
        cx=b[0]+width/2;cy=b[1]+width*rh/2
        if target[2]-target[0]<=width:cx=np.clip(cx,target[2]-width/2,target[0]+width/2)
        if target[3]-target[1]<=width*rh:cy=np.clip(cy,target[3]-width*rh/2,target[1]+width*rh/2)
        protected.append(box_from_center([cx,cy],width,ratio,W,H))
    out.append({'source':'consensus_follow','width':width,'mode':mode,'boxes':protected,'ratio':ratio})
    union=np.r_[targets[:,:2].min(axis=0),targets[:,2:].max(axis=0)]
    width=min(maximum,math.ceil(max(cfg['minimum_width_fraction']*maximum,cfg['target_context_factor']*max(union[2]-union[0],(union[3]-union[1])/rh)+2)))
    b=box_from_center((union[:2]+union[2:])/2,width,ratio,W,H)
    out.append({'source':'consensus_fixed','width':width,'mode':'fixed','boxes':[b[:] for _ in targets],'ratio':ratio})
    return out

def evaluate(candidate,targets,accepted,times,size,basecfg,cfg):
    ratio=candidate['ratio'];bs=np.array(candidate['boxes']);maximum=min(size[0],size[1]*ratio[0]/ratio[1]);cov=np.array([coverage(b,t,ratio) for b,t in zip(bs,targets)])
    centers=np.c_[bs[:,0]+bs[:,2]/2,bs[:,1]+bs[:,2]*ratio[1]/ratio[0]/2];targetcenters=(targets[:,:2]+targets[:,2:])/2
    distances=np.linalg.norm(np.diff(centers,axis=0),axis=1);dts=np.diff(times)
    speed=distances/np.maximum(dts,1e-6)/candidate['width']
    speed_ok=not len(speed) or bool(np.all(distances<=dts*basecfg['max_pan_crop_per_second']*candidate['width']+3))
    cost=cfg['area_cost']*(candidate['width']/maximum)**2+cfg['centering_cost']*float(np.mean(np.linalg.norm((centers-targetcenters)/candidate['width'],axis=1)))+cfg['movement_cost']*(float(np.mean(speed)) if len(speed) else 0)
    return {'eligible':bool(np.min(cov)>=cfg['minimum_coverage'] and speed_ok),'coverage_by_sample':cov.tolist(),'minimum_observed_coverage':float(min(cov[accepted])),'minimum_interpolated_coverage':float(min(cov)),'speed_ok':speed_ok,'deterministic_cost':cost}

def arbitrate(choices,eligible):
    if len(choices)==2 and choices[0] is not None and choices[0]==choices[1] and choices[0] in eligible:
        return choices[0],'order_consistent_among_coverage_safe_candidates'
    return eligible[0],'conflict_rejected_deterministic_safe_selection'

def main():
    import fcntl
    p=argparse.ArgumentParser();p.add_argument('--previous',required=True);p.add_argument('--output',required=True);p.add_argument('--video-dir',required=True);a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);lock=open(out/'run.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    prev=Path(a.previous);original=read_json(prev/'frozen.json');basecfg=original['config'];cfg=read_json(Path(__file__).with_name('config.json'));model=Model(basecfg)
    assert sha(ROOT/'experiments/scene_general_v1/pipeline.py')==original['code_sha256']
    assert original['model_ledger']['total_upper_bound']<=9_000_000_000
    frozen={'config':cfg,'inherited':original,'code_sha256':sha(__file__),'previous_results_sha256':sha(prev/'SCENERY_257_291_general_v1.jsonl')};signature=digest(frozen)
    if (out/'frozen.json').exists():assert read_json(out/'frozen.json')==frozen
    else:write_json(out/'frozen.json',frozen)
    state={'status':'running','pid':os.getpid(),'signature':signature,'completed_ids':[],'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()};write_json(out/'status.json',state)
    cv2.setNumThreads(2);cv2.setRNGSeed(0)
    base=load_rows(ROOT/'inputs/baseline.zip');ids=original['requested_ids'];rows=[copy.deepcopy(r) for r in base if str(r['video_id']) in ids];subbase=[r for r in base if str(r['video_id']) in ids]
    assert sha(ROOT/'inputs/baseline.zip')==original['baseline_sha256']
    ts_all=np.load(ROOT/'inputs/timestamps.npz');manifest=read_json(ROOT/'inputs/manifest.json');report={}
    try:
        for row in rows:
            vid=str(row['video_id']);meta=manifest['videos'][vid];size=(meta['width'],meta['height']);ratio=row['targetRatioWH'];ts=ts_all[vid];video=Path(a.video_dir)/(vid+'.mp4')
            assert sha(video)==meta['source_sha256'];state['current_video']=vid;write_json(out/'status.json',state)
            segments=groups([p['frame'] for p in row['predictions']],read_json(ROOT/'output/shots'/(vid+'.json'))['cuts']);fd={p['frame']:p for p in row['predictions']};report[vid]=[]
            for k,fs in enumerate(segments):
                oldpath=prev/'records'/vid/(str(k)+'.json');old=read_json(oldpath);cache=out/'records'/vid/(str(k)+'.json');key=digest({'run':signature,'old_record':sha(oldpath),'source':meta['source_sha256']})
                if cache.exists():rec=read_json(cache);assert rec['signature']==key
                else:
                    sampled=old['sample_frames'];decoded=read_frames(video,sampled,basecfg['image_max_side']);images=[decoded[f] for f in sampled];times=ts[sampled]
                    proposals=dict(old['proposals']);audit=[]
                    for i,im in enumerate(images):
                        if i not in proposals:proposals[i]=model.proposal(im,size,audit)
                    proposals=[proposals[i] for i in range(len(images))];identity=consensus(proposals,images,times,size,cfg)
                    pool=copy.deepcopy(old['candidates']);checks=[];decision='insufficient_consistent_evidence_keep_baseline';selected=None;choices=[];shortlist=[]
                    if identity['trusted']:
                        targets=predicted_targets(proposals,identity['accepted_indices'],times)
                        pool+=target_candidates(targets,times,size,ratio,basecfg,cfg)
                        checks=[evaluate(c,targets,identity['accepted_indices'],times,size,basecfg,cfg) for c in pool]
                        eligible=sorted([i for i,c in enumerate(checks) if c['eligible']],key=lambda i:checks[i]['deterministic_cost'])
                        shortlist=eligible[:cfg['maximum_ranking_candidates']]
                        if 0 in eligible and 0 not in shortlist:shortlist[-1]=0
                        if len(shortlist)==1:selected=shortlist[0];decision='only_coverage_safe_candidate'
                        elif len(shortlist)>1:
                            order=shortlist.copy();np.random.default_rng(basecfg['seed']).shuffle(order)
                            for turn,ordering in enumerate([order,list(reversed(order))]):
                                panel=make_panel(images,pool,ordering,size,basecfg);pp=out/'panels'/vid/f'{k}_{turn}.jpg';pp.parent.mkdir(parents=True,exist_ok=True);cv2.imwrite(str(pp),panel)
                                answer=model.choose(panel,len(ordering),audit,f'guarded_choice_{turn}');choices.append(ordering[answer] if answer is not None else None)
                            selected,decision=arbitrate(choices,eligible)
                        else:decision='no_coverage_safe_candidate_keep_baseline'
                    rec={'signature':key,'sample_frames':sampled,'proposals':proposals,'identity':identity,'candidates':pool,'checks':checks,'shortlist':shortlist,'choices':choices,'selected':selected,'decision':decision,'audit':audit,'previous_selected':old['selected']}
                    write_json(cache,rec)
                if rec['selected'] is not None:
                    candidate=rec['candidates'][rec['selected']];assert rec['checks'][rec['selected']]['eligible']
                    for f,b in zip(fs,interpolate_crops(rec['sample_frames'],candidate['boxes'],fs,ts,size,ratio)):fd[f]['bboxes']=b
                detail={'segment':k,'decision':rec['decision'],'consistent_anchors':len(rec['identity']['accepted_indices']),'total_samples':len(rec['sample_frames']),'old_choice_rejected':bool(rec['checks'] and not rec['checks'][rec['previous_selected']]['eligible']),'selected':rec['selected']}
                if rec['selected'] is not None:detail.update(source=candidate['source'],width=candidate['width'],coverage=rec['checks'][rec['selected']]['minimum_observed_coverage'])
                report[vid].append(detail);print(vid,k,json.dumps(detail),flush=True)
            state['completed_ids'].append(vid);write_json(out/'status.json',state)
            done=[r for r in rows if str(r['video_id']) in state['completed_ids']];oldrows=[r for r in subbase if str(r['video_id']) in state['completed_ids']]
            export(out/'partial',done,oldrows,manifest,preserve_width=False);write_json(out/'report.partial.json',report)
        validation=export(out/'SCENERY_257_291_guarded_v2',rows,subbase,manifest,preserve_width=False)
        write_json(out/'report.json',{'validation':validation,'per_video':report,'no_video_specific_rules':True,'official_score':None,'limitations':'Consistency is a heuristic hypothesis, not a semantic guarantee. Coverage is checked at sampled frames and against interpolated target hypotheses, not ground truth for every frame. No new SAM propagation.'})
        bundle=out/'results.zip'
        with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED) as z:
            for name in ['SCENERY_257_291_guarded_v2.jsonl','SCENERY_257_291_guarded_v2.zip','SCENERY_257_291_guarded_v2.validation.json','report.json','frozen.json']:z.write(out/name,name)
            for f in (out/'records').glob('*/*.json'):z.write(f,f.relative_to(out).as_posix())
        state.update(status='complete',bundle=str(bundle),bundle_sha256=sha(bundle),jsonl_sha256=sha(out/'SCENERY_257_291_guarded_v2.jsonl'),finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());write_json(out/'status.json',state)
    except BaseException as exc:state.update(status='failed',error=repr(exc));write_json(out/'status.json',state);raise
if __name__=='__main__':main()
