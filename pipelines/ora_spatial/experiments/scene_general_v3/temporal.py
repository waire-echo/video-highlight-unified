"""Uniform reference-conditioned attention, local decisions and protected variable crops.

No video identifiers, scene names or hand-chosen coordinates in algorithm decisions.
All ORA observations are new in each new run directory. Shot/source metadata only
are reused. This stage does not perform SAM or Pose inference.
"""
import argparse, copy, datetime, json, math, os, re, sys, time, zipfile
from pathlib import Path
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/scene_general_v1'))
from pipeline import Model, candidates, make_panel, read_frames
sys.path.insert(0,str(ROOT/'src'))
from common import read_json,write_json,sha,digest,load_rows,export,groups,box_from_center
from ora_tools import parse_answer,convert_box,image_data


def stamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def coverage(crop,target,ratio):
    if target is None:return 1.0
    x,y,w=crop;h=w*ratio[1]/ratio[0];a,b,c,d=target
    return max(0,min(x+w,c)-max(x,a))*max(0,min(y+h,d)-max(y,b))/max((c-a)*(d-b),1e-9)


def marked(image,box,size):
    im=image.copy()
    if box is not None:
        b=np.array(box)*np.array([im.shape[1]/size[0],im.shape[0]/size[1]]*2)
        x,y,r,t=np.round(b).astype(int);cv2.rectangle(im,(x,y),(r,t),(0,255,0),2)
    return im


class TemporalModel(Model):
    def multi(self,images,prompt,audit,tag,max_tokens=256,schema=None):
        # Reuse the trained model, without adding any other learned network.
        import requests
        content=[{'type':'image_url','image_url':{'url':image_data(im)}} for im in images]
        content.append({'type':'text','text':prompt})
        payload={'model':self.cfg['model'],'messages':[{'role':'user','content':content}],
                 'temperature':0,'seed':self.cfg['seed'],'max_tokens':max_tokens,
                 'chat_template_kwargs':{'enable_thinking':False},
                 'mm_processor_kwargs':{'size':{'shortest_edge':65536,'longest_edge':786432}}}
        if schema is not None:
            payload['response_format']={'type':'json_schema','json_schema':{'name':tag,'strict':True,'schema':schema}}
        for attempt in range(3):
            try:
                response=self.session.post(self.cfg['endpoint']+'/chat/completions',json=payload,timeout=240)
                response.raise_for_status();body=response.json();raw=body['choices'][0]['message']['content']
                audit.append({'tag':tag,'prompt':prompt,'response':raw,'finish_reason':body['choices'][0].get('finish_reason'),
                              'image_hashes':[__import__('hashlib').sha256(im.tobytes()).hexdigest() for im in images]})
                return raw
            except requests.RequestException:
                if attempt==2:raise
                time.sleep(2**attempt)

    def identity(self,refs,current,box,size,cfg,audit):
        images=[marked(im,b,size) for im,b in refs]+[marked(current,box,size)]
        states=['SAME','EVOLVING','DIFFERENT','ABSENT','UNCERTAIN']
        schema={'type':'object','properties':{'verdict':{'type':'string','enum':states}},'required':['verdict'],'additionalProperties':False}
        raw=self.multi(images,cfg['identity_prompt']+' Output ONLY a JSON object with the key verdict containing that word. No explanation or answer tags.',audit,'identity',64,schema)
        try:
            verdict=parse_answer(raw)['verdict']
            return verdict if verdict in states else 'PARSE_FAILURE'
        except (ValueError,TypeError,KeyError):return 'PARSE_FAILURE'

    def relocate(self,refs,current,size,cfg,audit):
        schema={'type':'object','properties':{
            'state':{'type':'string','enum':['visible','partial','occluded','out_of_view','uncertain']},
            'bbox_2d':{'anyOf':[{'type':'null'},{'type':'array','items':{'type':'number'},'minItems':4,'maxItems':4}]}},
            'required':['state','bbox_2d'],'additionalProperties':False}
        raw=self.multi([marked(im,b,size) for im,b in refs]+[current],cfg['relocalize_prompt']+' Output ONLY the JSON object, without answer tags.',audit,'relocalize',256,schema)
        try:
            obj=parse_answer(raw);state=obj['state'];box=obj.get('bbox_2d')
            if state in ('visible','partial'):
                return {'state':state,'box':convert_box(box,'norm1000',size,size),'reason':'reference_conditioned'}
            if state not in ('occluded','out_of_view','uncertain') or box is not None:raise ValueError('state/box mismatch')
            return {'state':state,'box':None,'reason':'reference_conditioned'}
        except (ValueError,TypeError,KeyError):return {'state':'uncertain','box':None,'reason':'parse_failure'}


def camera_transform(previous,current,size):
    """Feature-based global camera compensation; never a semantic identity vote."""
    gray1=cv2.cvtColor(previous,cv2.COLOR_BGR2GRAY);gray2=cv2.cvtColor(current,cv2.COLOR_BGR2GRAY)
    detector=cv2.SIFT_create(nfeatures=600)
    k1,d1=detector.detectAndCompute(gray1,None);k2,d2=detector.detectAndCompute(gray2,None)
    info={'valid':False,'inliers':0,'matrix':None}
    if d1 is None or d2 is None or len(d2)<2:return info
    pairs=cv2.BFMatcher().knnMatch(d1,d2,k=2)
    good=[a for pair in pairs if len(pair)==2 for a,b in [pair] if a.distance<.7*b.distance]
    if len(good)<10:return info
    a=np.float32([k1[m.queryIdx].pt for m in good]);b=np.float32([k2[m.trainIdx].pt for m in good])
    matrix,mask=cv2.estimateAffinePartial2D(a,b,method=cv2.RANSAC,ransacReprojThreshold=3)
    if matrix is None or mask is None:return info
    count=int(mask.sum());scale=float(np.linalg.norm(matrix[:,0]))
    if count<10 or count/len(good)<.5 or not .65<scale<1.55:return info
    sx,sy=size[0]/current.shape[1],size[1]/current.shape[0]
    full=np.eye(3);full[:2]=matrix;D=np.diag([sx,sy,1]);full=D@full@np.linalg.inv(D)
    info.update(valid=True,inliers=count,matrix=full[:2].tolist());return info


def warp_box(box,matrix,size):
    x,y,r,b=box;points=np.array([[x,y,1],[r,y,1],[r,b,1],[x,b,1]])@np.array(matrix).T
    lo=points.min(axis=0);hi=points.max(axis=0);W,H=size
    result=[max(0.,lo[0]),max(0.,lo[1]),min(float(W),hi[0]),min(float(H),hi[1])]
    return result if result[2]>result[0] and result[3]>result[1] else None


def trusted_sequence(observations):
    """Trust confirmed adjacent links; an invalid interval cannot erase earlier links."""
    targets=[None for _ in observations]
    for i,obs in enumerate(observations):
        if obs.get('confirmed') and obs.get('box') is not None:targets[i]=obs['box']
        parent=obs.get('confirms_seed')
        if parent is not None and obs.get('confirmed'):targets[parent]=observations[parent]['box']
    return targets


def resolve_exit(obs):
    # A failed repair must not downgrade explicit exit into a holdable occlusion.
    # A positively verified re-localization can correct an earlier erroneous ABSENT.
    if obs.get('verdict')=='ABSENT' and not obs.get('confirmed'):
        obs.update(state='out_of_view',box=None,reason='explicit_exit_no_extrapolation')
    return obs


def fill_short_gaps(targets,observations,times,transforms,size,cfg):
    """Only brief uncertainty/occlusion may use camera-warped last target. Never exit."""
    result=copy.deepcopy(targets);kinds=['observed' if b is not None else 'wide_fallback' for b in targets]
    previous=None;last_time=None
    for i,box in enumerate(targets):
        if box is not None:previous=box;last_time=float(times[i]);continue
        state=observations[i]['state']
        if state in ('out_of_view','different'):previous=None;last_time=None;continue
        if previous is not None and last_time is not None and float(times[i])-last_time<=cfg['hold_seconds']:
            motion=transforms[i]
            if motion['valid']:
                previous=warp_box(previous,motion['matrix'],size)
                if previous is not None:result[i]=previous;kinds[i]='brief_camera_compensated_hold'
            else:previous=None
        else:previous=None
    return result,kinds


def protect(crop,target,size,ratio,cfg):
    """Project crop onto the feasible region retaining visible focus, with context."""
    W,H=size;rh=ratio[1]/ratio[0];maximum=int(min(W,H/rh));x,y,w=crop
    center=np.array([x+w/2,y+w*rh/2]);width=int(min(maximum,max(1,round(w))))
    if target is not None:
        a,b,r,t=target;needed=max(r-a,(t-b)/rh)
        width=min(maximum,max(width,int(math.ceil(needed*cfg['context_factor']+2))))
        if r-a<=width:center[0]=np.clip(center[0],r-width/2,a+width/2)
        if t-b<=width*rh:center[1]=np.clip(center[1],t-width*rh/2,b+width*rh/2)
    return box_from_center(center,width,ratio,W,H)


def focus_candidates(targets,times,size,ratio,cfg):
    W,H=size;rh=ratio[1]/ratio[0];maximum=int(min(W,H/rh));follow=[]
    for target in targets:
        if target is None:follow.append(box_from_center([W/2,H/2],maximum,ratio,W,H));continue
        center=(np.array(target[:2])+target[2:])/2
        w=min(maximum,max(int(maximum*cfg['minimum_width_fraction']),math.ceil(cfg['context_factor']*max(target[2]-target[0],(target[3]-target[1])/rh)+2)))
        follow.append(protect(box_from_center(center,w,ratio,W,H),target,size,ratio,cfg))
    out=[{'source':'reference_follow','boxes':follow,'width':max(b[2] for b in follow),'ratio':ratio,'mode':'variable'}]
    known=[b for b in targets if b is not None]
    if known and len(known)==len(targets):
        a=np.array(known);union=np.r_[a[:,:2].min(axis=0),a[:,2:].max(axis=0)]
        center=(union[:2]+union[2:])/2;w=min(maximum,max(int(maximum*cfg['minimum_width_fraction']),math.ceil(cfg['context_factor']*max(union[2]-union[0],(union[3]-union[1])/rh)+2)))
        box=protect(box_from_center(center,w,ratio,W,H),union,size,ratio,cfg)
        out.append({'source':'reference_fixed','boxes':[box[:] for _ in targets],'width':w,'ratio':ratio,'mode':'fixed'})
    return out


def evaluate(candidate,targets,times,size,ratio,cfg):
    bs=np.array(candidate['boxes'],float);rh=ratio[1]/ratio[0];maximum=min(size[0],size[1]/rh)
    cov=[coverage(b,t,ratio) for b,t in zip(bs,targets)];known=[i for i,t in enumerate(targets) if t is not None]
    centers=np.c_[bs[:,0]+bs[:,2]/2,bs[:,1]+bs[:,2]*rh/2]
    movement=float(np.mean(np.linalg.norm(np.diff(centers,axis=0),axis=1)/np.maximum(np.diff(times),1e-4)/maximum)) if len(bs)>1 else 0
    centering=float(np.mean([np.linalg.norm((centers[i]-(np.array(targets[i][:2])+targets[i][2:])/2)/bs[i,2]) for i in known])) if known else 0
    # With no evidence, avoid aggressive automatic zoom. The model can still judge alternatives.
    area=float(np.mean((bs[:,2]/maximum)**2));cost=.55*area+.3*centering+.15*movement if known else 1-area+.15*movement
    unknown_safe=all(t is not None or b[2]>=maximum*.90 for b,t in zip(bs,targets))
    return {'eligible':bool(min(cov)>=cfg['minimum_coverage'] and unknown_safe),'coverage':cov,'cost':cost,'unknown_safe':unknown_safe}


def orders_without_fixed_positions(indices,seed):
    order=list(indices);np.random.default_rng(seed).shuffle(order)
    # Reversing an odd list leaves its middle letter unchanged. A cyclic derangement does not.
    return [order,order[1:]+order[:1]]


def interpolate_variable(sample_frames,boxes,frames,ts,size,ratio):
    bs=np.array(boxes,float);rh=ratio[1]/ratio[0];st=ts[sample_frames];ft=ts[frames]
    centers=np.c_[bs[:,0]+bs[:,2]/2,bs[:,1]+bs[:,2]*rh/2]
    cs=np.column_stack([np.interp(ft,st,centers[:,j]) for j in range(2)])
    ws=np.interp(ft,st,bs[:,2]);maximum=int(min(size[0],size[1]/rh))
    return [box_from_center(c,min(maximum,max(1,int(math.ceil(w)))),ratio,*size) for c,w in zip(cs,ws)]


def smooth_protected(boxes,targets,size,ratio,cfg,times=None):
    from scipy.optimize import lsq_linear
    bs=np.array(boxes,float);rh=ratio[1]/ratio[0];cs=np.c_[bs[:,0]+bs[:,2]/2,bs[:,1]+bs[:,2]*rh/2]
    n=len(bs);times=np.arange(n,dtype=float) if times is None else np.array(times,float)
    maximum=int(min(size[0],size[1]/rh))
    required=np.array([protect(b,t,size,ratio,cfg)[2] for b,t in zip(bs,targets)],float)
    # Log-scale envelope limits zoom changes while anticipating known later coverage needs.
    distance=np.abs(times[:,None]-times[None,:])
    widths=np.minimum(maximum,np.ceil(np.max(required[None,:]*np.exp(-cfg['maximum_log_zoom_per_second']*distance),axis=1)))
    lower=np.c_[widths/2,widths*rh/2];upper=np.array(size)-lower
    for i,target in enumerate(targets):
        if target is None:continue
        target=np.array(target);extent=target[2:]-target[:2];half=np.array([widths[i]/2,widths[i]*rh/2])
        for dim in range(2):
            if extent[dim]<=2*half[dim]:
                lower[i,dim]=max(lower[i,dim],target[dim+2]-half[dim])
                upper[i,dim]=min(upper[i,dim],target[dim]+half[dim])
    # Bounded least squares smooths center position/velocity without moving outside target-preserving intervals.
    blocks=[np.eye(n)];rhs=[cs]
    if n>1:
        D=np.diff(np.eye(n),axis=0)/np.maximum(np.diff(times)[:,None],.001)
        blocks.append(.8*D);rhs.append(np.zeros((n-1,2)))
    if n>2:
        blocks.append(.6*np.diff(D,axis=0));rhs.append(np.zeros((n-2,2)))
    A=np.vstack(blocks);Y=np.vstack(rhs);solved=cs.copy()
    for dim in range(2):
        fixed=upper[:,dim]-lower[:,dim]<1e-6;free=~fixed
        solved[fixed,dim]=(lower[fixed,dim]+upper[fixed,dim])/2
        if np.any(free):
            solution=lsq_linear(A[:,free],Y[:,dim]-A[:,fixed]@solved[fixed,dim],bounds=(lower[free,dim],upper[free,dim]),tol=1e-8,max_iter=200)
            if not solution.success:raise RuntimeError('Constrained camera optimization failed: '+solution.message)
            solved[free,dim]=solution.x
    out=[]
    for i,target in enumerate(targets):
        initial=box_from_center(solved[i],int(widths[i]),ratio,*size)
        out.append(protect(initial,target,size,ratio,cfg))
    return out


def sample_frames(fs,ts,interval):
    wanted=np.arange(float(ts[fs[0]]),float(ts[fs[-1]])+1e-7,interval)
    indices=np.searchsorted(ts,wanted);return sorted(set([fs[0],fs[-1]]+[min(fs[-1],max(fs[0],int(f))) for f in indices]))


def infer_segment(model,video,fs,ts,size,ratio,basecfg,cfg,out,signature,progress):
    samples=sample_frames(fs,ts,cfg['sample_interval_seconds']);times=ts[samples]
    prev=[max(fs[0],int(np.searchsorted(ts,max(ts[fs[0]],ts[f]-basecfg['motion_lag_seconds'])))) for f in samples]
    images_by_id=read_frames(video,set(samples+prev),basecfg['image_max_side'])
    images=[images_by_id[f] for f in samples];observations=[];transforms=[];seed=None;anchor=None;last_good=None
    for i,(f,im) in enumerate(zip(samples,images)):
        progress(stage='grounding',sample=i,total_samples=len(samples),original_frame=f)
        motion=camera_transform(images[i-1],im,size) if i else {'valid':False,'inliers':0,'matrix':None}
        transforms.append(motion);cache=out/'observations'/f'{i}.json'
        key=digest({'signature':signature,'frame':f,'history':observations})
        if cache.exists():obs=read_json(cache);assert obs['signature']==key
        else:
            audit=[];proposal=model.proposal(im,size,audit)
            obs={'signature':key,'frame':f,'proposal':proposal,'box':proposal['box'],'state':'unconfirmed',
                 'confirmed':False,'audit':audit,'verdict':None,'confirms_seed':None}
            ref=anchor if anchor is not None else seed
            if ref is not None:
                reference_indices=list(dict.fromkeys([ref,last_good if last_good is not None else ref]))
                refs=[(images[j],observations[j]['box']) for j in reference_indices]
                verdict=model.identity(refs,im,proposal['box'],size,cfg,audit) if proposal['box'] is not None else 'UNCERTAIN'
                obs['verdict']=verdict
                if verdict in ('SAME','EVOLVING'):
                    obs.update(state='visible' if verdict=='SAME' else 'partial',confirmed=True)
                elif anchor is not None:
                    located=model.relocate(refs,im,size,cfg,audit);obs.update(located)
                    if located['box'] is not None:
                        confirm=model.identity(refs,im,located['box'],size,cfg,audit)
                        obs['repair_verdict']=confirm
                        if confirm in ('SAME','EVOLVING'):obs['confirmed']=True
                        else:obs.update(box=None,state='uncertain')
                else:obs.update(state='different' if verdict=='DIFFERENT' else 'uncertain',box=proposal['box'])
                if obs['confirmed'] and anchor is None:obs['confirms_seed']=ref
            resolve_exit(obs)
            write_json(cache,obs)
        observations.append(obs)
        if obs['confirmed']:
            if anchor is None:anchor=obs['confirms_seed']
            last_good=i
        elif anchor is None:seed=i if obs['proposal']['box'] is not None else None
    rawtargets=trusted_sequence(observations)
    targets,kinds=fill_short_gaps(rawtargets,observations,times,transforms,size,cfg)
    sample_crops=[[] for _ in samples];windows=[]
    starts=list(range(0,max(1,len(samples)-1),cfg['window_intervals']))
    for k,start in enumerate(starts):
        stop=min(len(samples),start+cfg['window_intervals']+1);ix=list(range(start,stop))
        progress(stage='candidates',window=k,total_windows=len(starts))
        ims=[images[j] for j in ix];tt=[targets[j] for j in ix];tm=times[ix]
        previous=[images_by_id[prev[j]] if prev[j]<samples[j] else None for j in ix]
        props=[(q,{'box':targets[j]}) for q,j in enumerate(ix)]
        pool,mapdiag=candidates(ims,previous,props,tm,size,ratio,basecfg)
        for c in pool:c['ratio']=ratio
        pool+=focus_candidates(tt,tm,size,ratio,cfg)
        checks=[evaluate(c,tt,tm,size,ratio,cfg) for c in pool]
        eligible=sorted([j for j,c in enumerate(checks) if c['eligible']],key=lambda j:checks[j]['cost'])
        shortlist=eligible[:cfg['maximum_ranking_candidates']];choices=[];audit=[];reason='only_eligible'
        if not eligible:
            # Aspect-ratio constraints can make complete coverage impossible. Keep the best attainable crop,
            # record failure explicitly, and do not call it coverage-safe.
            selected=max(range(len(pool)),key=lambda j:(min(checks[j]['coverage']),-checks[j]['cost']))
            reason='coverage_infeasible_best_effort'
        elif len(shortlist)==1:selected=shortlist[0]
        else:
            orders=orders_without_fixed_positions(shortlist,basecfg['seed'])
            for turn,order in enumerate(orders):
                panel=make_panel(ims,pool,order,size,basecfg);p=out/'panels'/f'{k}_{turn}.jpg';p.parent.mkdir(parents=True,exist_ok=True);cv2.imwrite(str(p),panel)
                answer=model.choose(panel,len(order),audit,f'local_choice_{turn}');choices.append(order[answer] if answer is not None else None)
            if choices[0] is not None and choices[0]==choices[1]:selected=choices[0];reason='agreement_after_derangement'
            else:selected=eligible[0];reason='conflict_rejected_safe_cost'
        for j,b in zip(ix,pool[selected]['boxes']):sample_crops[j].append(b)
        rec={'sample_indices':ix,'candidates':pool,'checks':checks,'shortlist':shortlist,'choices':choices,'selected':selected,'reason':reason,'audit':audit,'map_diagnostics':mapdiag}
        write_json(out/'windows'/f'{k}.json',rec);windows.append(rec)
    merged=[np.mean(bs,axis=0).tolist() for bs in sample_crops]
    final=smooth_protected(merged,targets,size,ratio,cfg,times)
    cov=[coverage(b,t,ratio) for b,t in zip(final,targets)]
    record={'signature':signature,'sample_frames':samples,'observations':observations,'camera_transforms':transforms,'targets':targets,
            'target_kinds':kinds,'sample_boxes':final,'coverage_by_sample':cov,'windows':windows,
            'coverage_failures':[int(samples[j]) for j,c in enumerate(cov) if c<cfg['minimum_coverage']],
            'limitations':'Semantic identity is model-estimated; between-sample paths are interpolated. No new SAM propagation.'}
    write_json(out/'segment.json',record);return record


def main():
    import fcntl
    parser=argparse.ArgumentParser();parser.add_argument('--ids',required=True);parser.add_argument('--video-dir',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    lock=open(out/'run.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cv2.setNumThreads(2);cv2.setRNGSeed(0)
    cfg=read_json(Path(__file__).with_name('config.json'));basecfg=read_json(ROOT/'experiments/scene_general_v1/config.json')
    ids=args.ids.split(',');assert len(ids)==len(set(ids));baseline=load_rows(ROOT/'inputs/baseline.zip')
    old=[r for r in baseline if str(r['video_id']) in ids];assert len(old)==len(ids);rows=copy.deepcopy(old)
    manifest=read_json(ROOT/'inputs/manifest.json');ts_all=np.load(ROOT/'inputs/timestamps.npz');ledger=read_json(ROOT/'output/parameter_ledger.json')
    assert sha(ROOT/'inputs/baseline.zip')==manifest['baseline_sha256'];assert ledger['pass'] and ledger['total_upper_bound']<=9_000_000_000
    sources=[Path(__file__),Path(__file__).with_name('config.json'),ROOT/'experiments/scene_general_v1/pipeline.py',ROOT/'experiments/scene_general_v1/config.json',ROOT/'src/common.py',ROOT/'src/ora_tools.py']
    frozen={'config':cfg,'base_config':basecfg,'files':{p.relative_to(ROOT).as_posix():sha(p) for p in sources},'requested_ids':ids,
            'baseline_sha256':sha(ROOT/'inputs/baseline.zip'),'model_ledger':ledger,'reuse_prior_inference':False}
    signature=digest(frozen)
    if (out/'frozen.json').exists():assert read_json(out/'frozen.json')==frozen
    else:write_json(out/'frozen.json',frozen)
    status={'status':'running','pid':os.getpid(),'started_utc':stamp(),'signature':signature,'requested_ids':ids,'completed_ids':[]}
    def progress(**kw):status.update(kw,updated_utc=stamp());write_json(out/'status.json',status)
    model=TemporalModel(basecfg);report={};progress(stage='preflight')
    try:
        for row in rows:
            for rel,expected in frozen['files'].items():assert sha(ROOT/rel)==expected,'Source changed during run'
            vid=str(row['video_id']);meta=manifest['videos'][vid];video=Path(args.video_dir)/(vid+'.mp4')
            assert sha(video)==meta['source_sha256'];shot=read_json(ROOT/'output/shots'/f'{vid}.json');assert shot['source_sha256']==meta['source_sha256']
            ts=ts_all[vid];size=(meta['width'],meta['height']);ratio=row['targetRatioWH'];lookup={p['frame']:p for p in row['predictions']}
            segments=groups(list(lookup),shot['cuts']);report[vid]=[];progress(current_video=vid,stage='video',total_segments=len(segments))
            for k,fs in enumerate(segments):
                progress(segment=k);where=out/'records'/vid/str(k);key=digest({'run':signature,'source':meta['source_sha256'],'frames':fs,'cuts':shot['cuts']})
                if (where/'segment.json').exists():rec=read_json(where/'segment.json');assert rec['signature']==key
                else:rec=infer_segment(model,video,fs,ts,size,ratio,basecfg,cfg,where,key,progress)
                boxes=interpolate_variable(rec['sample_frames'],rec['sample_boxes'],fs,ts,size,ratio)
                for f,b in zip(fs,boxes):lookup[f]['bboxes']=b
                summary={'segment':k,'samples':len(rec['sample_frames']),'confirmed':sum(o['confirmed'] for o in rec['observations']),
                         'state_counts':{s:sum(o['state']==s for o in rec['observations']) for s in sorted(set(o['state'] for o in rec['observations']))},
                         'coverage_failures':rec['coverage_failures'],'window_decisions':[w['reason'] for w in rec['windows']]}
                report[vid].append(summary);print(vid,k,json.dumps(summary),flush=True)
            status['completed_ids'].append(vid)
            done=[r for r in rows if str(r['video_id']) in status['completed_ids']];subold=[r for r in old if str(r['video_id']) in status['completed_ids']]
            export(out/'partial',done,subold,manifest,preserve_width=False);write_json(out/'report.partial.json',report);progress(stage='video_complete')
            print('COMPLETED',len(done),'/',len(rows),vid,flush=True)
        name='SPATIAL_temporal_v3';validation=export(out/name,rows,old,manifest,preserve_width=False)
        write_json(out/'report.json',{'validation':validation,'per_video':report,'no_video_specific_rules':True,'fresh_inference':True,'official_score':None,
                                    'limitations':'ORA identity and localization remain fallible. Coverage verified at sampled observations; intervening crops are interpolated. Existing shot cache retained. No new SAM or Pose inference.'})
        bundle=out/'results.zip'
        with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED) as archive:
            for fn in [name+'.jsonl',name+'.zip',name+'.validation.json','report.json','frozen.json']:archive.write(out/fn,fn)
            for f in (out/'records').rglob('*.json'):archive.write(f,f.relative_to(out).as_posix())
        progress(status='complete',stage='complete',bundle=str(bundle),bundle_sha256=sha(bundle),jsonl_sha256=sha(out/(name+'.jsonl')),finished_utc=stamp())
    except BaseException as exc:progress(status='failed',error=repr(exc));raise


if __name__=='__main__':main()
