"""Uniform candidate-based scenery reframing. Video IDs are used only for I/O.

No scene-specific nouns, per-video parameters, or previous manually refined anchors.
"""
import argparse,copy,datetime,hashlib,json,os,re,sys,time,zipfile
from pathlib import Path
import cv2,numpy as np,requests
from scipy.ndimage import gaussian_filter1d
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from common import read_json,write_json,load_rows,export,validate,sha,digest,groups,box_from_center
from ora_tools import read_frames,image_data,parse_answer,parse_spatial

def normalize_map(a):
    a=np.maximum(np.nan_to_num(a),0).astype(np.float32)
    return a/(float(a.sum())+1e-12)

def evidence_maps(image,previous,cfg):
    size=tuple(cfg['map_size']);small=cv2.resize(image,size)
    gray=cv2.cvtColor(small,cv2.COLOR_BGR2GRAY)
    ft=np.fft.fft2(gray.astype(float)+1);amp=np.log(np.abs(ft)+1e-8)
    sr=np.abs(np.fft.ifft2(np.exp(amp-cv2.blur(amp,(3,3))+1j*np.angle(ft))))**2
    sr=cv2.GaussianBlur(sr.astype(np.float32),(9,9),2)
    # Border suppression is identical for every image; not a scene-specific mask.
    edge=np.ones_like(sr);edge[:2]=0;edge[-2:]=0;edge[:,:2]=0;edge[:,-2:]=0
    sal=normalize_map(sr*edge);motion=np.zeros_like(sal);diag={'motion_usable':False,'affine_inliers':0}
    if previous is not None:
        old=cv2.cvtColor(cv2.resize(previous,size),cv2.COLOR_BGR2GRAY)
        pts=cv2.goodFeaturesToTrack(old,150,.01,5)
        matrix=None
        if pts is not None and len(pts)>=8:
            nxt,status,_=cv2.calcOpticalFlowPyrLK(old,gray,pts,None)
            if nxt is not None:
                good=status.ravel().astype(bool);p=pts[good].reshape(-1,2);q=nxt[good].reshape(-1,2)
                if len(p)>=8:
                    matrix,inliers=cv2.estimateAffinePartial2D(p,q,method=cv2.RANSAC,ransacReprojThreshold=2)
                    diag['affine_inliers']=int(inliers.sum()) if inliers is not None else 0
        flow=cv2.calcOpticalFlowFarneback(old,gray,None,.5,3,15,3,5,1.2,0)
        yy,xx=np.indices(gray.shape,dtype=np.float32)
        if matrix is not None and diag['affine_inliers']>=8:
            camera=np.stack([matrix[0,0]*xx+matrix[0,1]*yy+matrix[0,2]-xx,matrix[1,0]*xx+matrix[1,1]*yy+matrix[1,2]-yy],axis=-1)
        else:camera=np.median(flow.reshape(-1,2),axis=0)
        residual=np.linalg.norm(flow-camera,axis=2)
        residual=np.maximum(0,residual-np.percentile(residual,70))*edge
        if float(residual.max())>.10:
            motion=normalize_map(cv2.GaussianBlur(residual,(9,9),2));diag['motion_usable']=True
    return sal,motion,diag

def crop_path(centers,width,times,size,ratio,cfg):
    W,H=size;centers=np.asarray(centers,float)
    if len(centers)>1 and np.max(np.ptp(centers,axis=0))<cfg['static_range_crop_fraction']*width:
        centers[:]=np.median(centers,axis=0);mode='fixed'
    else:
        centers=gaussian_filter1d(centers,.65,axis=0,mode='nearest') if len(centers)>1 else centers
        for i in range(1,len(centers)):
            delta=centers[i]-centers[i-1];dist=float(np.linalg.norm(delta));limit=cfg['max_pan_crop_per_second']*width*max(0,float(times[i]-times[i-1]))
            if dist>limit and dist>0:centers[i]=centers[i-1]+delta*limit/dist
        mode='slow_pan'
    return [box_from_center(c,width,ratio,W,H) for c in centers],mode

def map_centers(maps,width,size,ratio,cfg):
    """DP over window saliency, penalizing center jumps between sampled frames."""
    W,H=size;mw,mh=cfg['map_size'];rh=ratio[1]/ratio[0]
    bw=max(1,min(mw,round(width/W*mw)));bh=max(1,min(mh,round(width*rh/H*mh)))
    xs=np.linspace(0,mw-bw,9).round().astype(int);ys=np.linspace(0,mh-bh,7).round().astype(int)
    points=np.array([(x+bw/2,y+bh/2) for y in ys for x in xs]);scores=[]
    for mp in maps:
        ii=cv2.integral(mp);s=np.array([ii[y+bh,x+bw]-ii[y,x+bw]-ii[y+bh,x]+ii[y,x] for y in ys for x in xs])
        scores.append(s)
    scores=np.asarray(scores);cost=((points[:,None]-points[None,:])/np.array([bw,bh]))**2
    penalty=cfg['path_jump_penalty']*cost.sum(axis=2)
    values=scores[0];back=[]
    for s in scores[1:]:
        v=values[:,None]-penalty;best=v.argmax(axis=0);back.append(best);values=v[best,np.arange(len(points))]+s
    j=int(values.argmax());indices=[j]
    for b in reversed(back):j=int(b[j]);indices.append(j)
    selected=points[list(reversed(indices))]*np.array([W/mw,H/mh])
    return selected

def candidates(images,prevs,proposal_records,sample_times,size,ratio,cfg):
    W,H=size;maxwidth=int(min(W,H*ratio[0]/ratio[1]));maps=[];motions=[];mapdiag=[]
    for im,prev in zip(images,prevs):
        s,m,d=evidence_maps(im,prev,cfg);maps.append(s);motions.append(m);mapdiag.append(d)
    result=[]
    def add(source,width,centers):
        path,mode=crop_path(centers,width,sample_times,size,ratio,cfg)
        if any(np.mean(np.abs(np.array(path)-np.array(r['boxes']))/np.array([W,H,W]))<.015 for r in result):return
        result.append({'source':source,'width':int(width),'mode':mode,'boxes':path})
    add('wide_context',maxwidth,np.tile([W/2,H/2],(len(images),1)))
    valid=[(i,r['box']) for i,r in proposal_records if r.get('box') is not None]
    if valid:
        indices=np.array([x[0] for x in valid]);bs=np.array([x[1] for x in valid]);cs=(bs[:,:2]+bs[:,2:])/2
        ora_centers=np.column_stack([np.interp(np.arange(len(images)),indices,cs[:,i]) for i in range(2)])
        for scale in cfg['scales']:add('ora_focus',int(maxwidth*scale),ora_centers.copy())
    for scale in cfg['scales']:
        width=int(maxwidth*scale);add('spectral_saliency',width,map_centers(maps,width,size,ratio,cfg))
    if any(d['motion_usable'] for d in mapdiag):
        fused=[normalize_map(.4*s+.6*m) if d['motion_usable'] else s for s,m,d in zip(maps,motions,mapdiag)]
        for scale in cfg['scales']:
            width=int(maxwidth*scale);add('motion_saliency',width,map_centers(fused,width,size,ratio,cfg))
    return result[:cfg['candidate_limit']],mapdiag

class Model:
    def __init__(self,cfg):self.cfg=cfg;self.session=requests.Session()
    def ask(self,im,prompt,max_tokens,audit,tag):
        payload={'model':self.cfg['model'],'messages':[{'role':'user','content':[{'type':'image_url','image_url':{'url':image_data(im)}},{'type':'text','text':prompt}]}],'temperature':0,'seed':self.cfg['seed'],'max_tokens':max_tokens,'chat_template_kwargs':{'enable_thinking':False},'mm_processor_kwargs':{'size':{'shortest_edge':65536,'longest_edge':self.cfg['ranking_max_pixels']}}}
        for attempt in range(3):
            try:
                res=self.session.post(self.cfg['endpoint']+'/chat/completions',json=payload,timeout=240);res.raise_for_status();body=res.json();raw=body['choices'][0]['message']['content']
                audit.append({'tag':tag,'prompt':prompt,'response':raw,'finish_reason':body['choices'][0].get('finish_reason'),'attempt':attempt,'image_sha256':hashlib.sha256(im.tobytes()).hexdigest()})
                return raw
            except requests.RequestException:
                if attempt==2:raise
                time.sleep(2**attempt)
    def proposal(self,im,size,audit):
        raw=self.ask(im,self.cfg['proposal_prompt'],self.cfg['max_proposal_tokens'],audit,'proposal')
        try:
            if parse_answer(raw)==[]:return {'box':None,'reason':'no_localized_focus'}
            return {'box':parse_spatial(raw,'norm1000',(im.shape[1],im.shape[0]),size),'reason':'parsed_unverified_proposal'}
        except (ValueError,TypeError,KeyError) as exc:return {'box':None,'reason':'parse_failure:'+str(exc)}
    def choose(self,panel,count,audit,tag):
        letters='ABCDEFG'[:count]
        prompt=self.cfg['ranking_prompt']+'\nOptions: '+', '.join(letters)
        for attempt in range(2):
            raw=self.ask(panel,prompt,self.cfg['max_choice_tokens'],audit,tag)
            m=re.fullmatch(r'\s*(?:<answer>\s*)?([A-G])(?:\s*</answer>)?\s*',raw)
            if m and m.group(1) in letters:return letters.index(m.group(1))
        return None

def make_panel(images,pool,order,size,cfg):
    W,H=size;tw=cfg['rank_tile_width'];th=round(tw*H/W);label=26
    sheet=np.zeros(((len(order)+1)*(th+label),len(images)*tw,3),np.uint8)
    for row,idx in enumerate([None]+order):
        title='ORIGINAL (context)' if idx is None else chr(64+row)
        y=row*(th+label);cv2.putText(sheet,title,(6,y+20),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),2)
        for col,im in enumerate(images):
            tile=im
            if idx is not None:
                x,top,w=pool[idx]['boxes'][col];h=w*pool[idx]['ratio'][1]/pool[idx]['ratio'][0]
                sx,sy=im.shape[1]/W,im.shape[0]/H
                l,t=int(x*sx),int(top*sy);r,b=max(int((x+w)*sx),l+1),max(int((top+h)*sy),t+1)
                tile=im[t:b,l:r]
            if not tile.size:raise ValueError('Empty candidate image')
            # Letterboxing avoids distorting portrait or non-source-aspect candidates.
            scale=min(tw/tile.shape[1],th/tile.shape[0]);rw,rh=max(1,round(tile.shape[1]*scale)),max(1,round(tile.shape[0]*scale))
            tile=cv2.resize(tile,(rw,rh));ox=col*tw+(tw-rw)//2;oy=y+label+(th-rh)//2
            sheet[oy:oy+rh,ox:ox+rw]=tile
    return sheet

def interpolate_crops(sample_frames,sample_boxes,frames,ts,size,ratio):
    W,H=size;bs=np.asarray(sample_boxes,float);centers=np.c_[bs[:,0]+bs[:,2]/2,bs[:,1]+bs[:,2]*ratio[1]/ratio[0]/2]
    times=ts[sample_frames];width=int(bs[0,2]);assert np.all(bs[:,2]==width)
    cs=np.column_stack([np.interp(ts[frames],times,centers[:,i]) for i in range(2)])
    return [box_from_center(c,width,ratio,W,H) for c in cs]

def main():
    import fcntl
    p=argparse.ArgumentParser();p.add_argument('--ids',required=True);p.add_argument('--video-dir',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    cv2.setNumThreads(2);cv2.setRNGSeed(0)
    cfg=read_json(Path(__file__).with_name('config.json'));out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    lock=open(out/'run.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    model=Model(cfg);baseline=load_rows(ROOT/'inputs/baseline.zip');manifest=read_json(ROOT/'inputs/manifest.json');ts_all=np.load(ROOT/'inputs/timestamps.npz')
    ledger=read_json(ROOT/'output/parameter_ledger.json');assert ledger['pass'] and ledger['total_upper_bound']<=9_000_000_000
    assert sha(ROOT/'inputs/baseline.zip')==manifest['baseline_sha256']
    requested=a.ids.split(',');assert len(set(requested))==len(requested)
    rows=[copy.deepcopy(r) for r in baseline if str(r['video_id']) in requested];subbase=[r for r in baseline if str(r['video_id']) in requested]
    assert len(rows)==len(requested)
    frozen={'config':cfg,'code_sha256':sha(__file__),'common_sha256':sha(ROOT/'src/common.py'),'ora_tools_sha256':sha(ROOT/'src/ora_tools.py'),'baseline_sha256':sha(ROOT/'inputs/baseline.zip'),'model_ledger':ledger,'requested_ids':requested}
    signature=digest(frozen)
    if (out/'frozen.json').exists():assert read_json(out/'frozen.json')==frozen,'Frozen experiment changed'
    else:write_json(out/'frozen.json',frozen)
    state={'status':'running','pid':os.getpid(),'signature':signature,'completed_ids':[],'requested_ids':requested,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()};write_json(out/'status.json',state)
    report={}
    try:
        for row in rows:
            vid=str(row['video_id']);meta=manifest['videos'][vid];video=Path(a.video_dir)/(vid+'.mp4');assert sha(video)==meta['source_sha256'];ts=ts_all[vid];size=(meta['width'],meta['height']);ratio=row['targetRatioWH']
            state.update(current_video=vid);write_json(out/'status.json',state)
            shot=read_json(ROOT/'output/shots'/(vid+'.json'));assert shot['source_sha256']==meta['source_sha256']
            segments=groups([p['frame'] for p in row['predictions']],shot['cuts']);report[vid]=[];lookup={p['frame']:p for p in row['predictions']}
            for k,fs in enumerate(segments):
                cache=out/'records'/vid/(str(k)+'.json');key=digest({'run':signature,'source':meta['source_sha256'],'frames':fs,'cuts':shot['cuts']})
                if cache.exists():
                    rec=read_json(cache);assert rec['signature']==key
                else:
                    sampled=sorted(set(fs[i] for i in np.linspace(0,len(fs)-1,min(len(fs),cfg['sample_count'])).round().astype(int)))
                    prev=[max(fs[0],int(np.searchsorted(ts,max(ts[fs[0]],ts[f]-cfg['motion_lag_seconds'])))) for f in sampled]
                    decoded=read_frames(video,set(sampled+prev),cfg['image_max_side']);images=[decoded[f] for f in sampled];previous=[decoded[q] if q<f else None for q,f in zip(prev,sampled)]
                    proposal_indices=sorted(set(np.linspace(0,len(sampled)-1,min(len(sampled),cfg['ora_proposal_count'])).round().astype(int)))
                    audit=[];proposals=[(int(i),model.proposal(images[i],size,audit)) for i in proposal_indices]
                    pool,mapdiag=candidates(images,previous,proposals,ts[sampled],size,ratio,cfg)
                    for c in pool:c['ratio']=ratio
                    # Same deterministic order for all clips; reversed second pass checks position bias.
                    order=list(range(len(pool)));np.random.default_rng(cfg['seed']).shuffle(order)
                    orders=[order,list(reversed(order))];chosen=[]
                    for turn,ordering in enumerate(orders):
                        panel=make_panel(images,pool,ordering,size,cfg);panelpath=out/'panels'/vid/f'{k}_{turn}.jpg';panelpath.parent.mkdir(parents=True,exist_ok=True);cv2.imwrite(str(panelpath),panel)
                        answer=model.choose(panel,len(ordering),audit,f'choice_{turn}');chosen.append(ordering[answer] if answer is not None else None)
                    if chosen[0] is not None and chosen[0]==chosen[1]:selected=chosen[0];reason='agreement_under_reversed_order'
                    elif all(c is not None for c in chosen):
                        tie=list(dict.fromkeys(chosen));panel=make_panel(images,pool,tie,size,cfg);answer=model.choose(panel,len(tie),audit,'tie_break')
                        if answer is not None:selected=tie[answer];reason='pairwise_tie_break'
                        else:selected=max(tie,key=lambda i:pool[i]['width']);reason='parse_failure_wider_of_disagreement'
                    else:selected=0;reason='unresolved_choice_wide_context'
                    rec={'signature':key,'sample_frames':sampled,'proposals':proposals,'map_diagnostics':mapdiag,'candidates':pool,'first_two_choices':chosen,'selected':selected,'selection_reason':reason,'audit':audit}
                    write_json(cache,rec)
                selected=rec['candidates'][rec['selected']];boxes=interpolate_crops(rec['sample_frames'],selected['boxes'],fs,ts,size,ratio)
                for f,b in zip(fs,boxes):lookup[f]['bboxes']=b
                report[vid].append({'segment':k,'first':fs[0],'last':fs[-1],'source':selected['source'],'width':selected['width'],'mode':selected['mode'],'selection_reason':rec['selection_reason'],'proposal_failures':sum(r['reason'].startswith('parse_failure') for _,r in rec['proposals'])})
                print(vid,k,selected['source'],selected['width'],rec['selection_reason'],flush=True)
            state['completed_ids'].append(vid);write_json(out/'status.json',state)
            done=[r for r in rows if str(r['video_id']) in state['completed_ids']];old=[r for r in subbase if str(r['video_id']) in state['completed_ids']]
            export(out/'partial',done,old,manifest,preserve_width=False);write_json(out/'report.partial.json',report)
            print('COMPLETED',len(state['completed_ids']),'/',len(rows),vid,flush=True)
        validation=export(out/'SCENERY_257_291_general_v1',rows,subbase,manifest,preserve_width=False)
        write_json(out/'report.json',{'validation':validation,'per_video':report,'method':'Uniform candidates from ORA/spectral saliency/camera-compensated flow, two ordering comparisons, stable crops. No new SAM propagation.','no_video_specific_rules':True,'official_score':None,'empty_input_preserved':[r['video_id'] for r in rows if not r['predictions']]})
        bundle=out/'results.zip'
        with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED) as z:
            for name in ['SCENERY_257_291_general_v1.jsonl','SCENERY_257_291_general_v1.zip','SCENERY_257_291_general_v1.validation.json','report.json','frozen.json']:z.write(out/name,name)
            for f in (out/'records').glob('*/*.json'):z.write(f,f.relative_to(out).as_posix())
        state.update(status='complete',finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),bundle=str(bundle),bundle_sha256=sha(bundle),jsonl_sha256=sha(out/'SCENERY_257_291_general_v1.jsonl'));write_json(out/'status.json',state)
    except BaseException as exc:
        state.update(status='failed',error=type(exc).__name__+': '+str(exc));write_json(out/'status.json',state);raise
if __name__=='__main__':main()
