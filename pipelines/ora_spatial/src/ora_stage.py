"""Sparse ORA anchors. Exact native templates, explicit units; no temporal edits."""
import argparse
import base64
import re
from pathlib import Path
import sys
import numpy as np
from common import load_rows,read_json,write_json,digest,sha,iou
from jobs import build_jobs
from ora_tools import OraClient,read_frames,image_data,make_video,parse_answer,parse_spatial,parse_tracking,nearest_frame
from ora_verify import verify_result
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'vendor'))
from eval_prompt import build_spatial_grounding_prompt


ROUTE_PROMPT = '''Determine the visual focus of this short video, using the event description only as a fallible hint.
Event description: {description}
Return JSON only: {{"mode":"single|multiple|scene|unknown","target":"short visible referring phrase"}}.
single: one clearly identifiable focal object/person. multiple: interacting or equally important focal objects.
scene: overall environment/composition is the focus. unknown: insufficient evidence or ambiguity.
Do not classify failed detection as scenery. Do not invent an object from the description.
For multiple, target must describe the important objects together. For scene/unknown, target may be empty.
The target must be an object description such as "the small white dog", never numeric coordinates.
Return exactly one valid JSON object, for example {{"mode":"single","target":"the small white dog"}}.'''


def parse_route(raw):
    # An explicit scene/unknown abstention is distinct from malformed numeric output.
    tag=re.fullmatch(r'\s*<mode>(scene|unknown)</mode>\s*',raw)
    route={'mode':tag[1],'target':''} if tag else parse_answer(raw)
    if not isinstance(route,dict) or route.get('mode') not in ('single','multiple','scene','unknown'):
        raise ValueError('Invalid route')
    if not isinstance(route.get('target',''),str):raise ValueError('Invalid target')
    if route['mode'] in ('single','multiple') and not re.search(r'[A-Za-z\u4e00-\u9fff]',route.get('target','')):
        raise ValueError('Target must name a visible object, not coordinates')
    return route


def infer_job(client,job,meta,ts,video,cfg,work):
    ocfg=cfg['ora'];a,b=job['first'],job['last'];W,H=meta['width'],meta['height']
    dur=ts[b]-ts[a]+1/meta['fps']
    sample_times=ts[a]+np.arange(max(1,int(np.ceil(dur*ocfg['input_fps']))))/ocfg['input_fps']
    sample_ids=[nearest_frame(ts,min(t,ts[b]),a,b) for t in sample_times]
    review_times=list(np.arange(ts[a],ts[b]+1e-8,ocfg['review_interval_seconds']))+[ts[b]]
    review_ids=sorted({a,b,*[nearest_frame(ts,t,a,b) for t in review_times]})
    images=read_frames(video,set(sample_ids+review_ids),ocfg['image_max_side'])
    clip=work/'proxy.mp4';make_video(clip,[images[f] for f in sample_ids],ocfg['input_fps'])
    video_content={'type':'video_url','video_url':{'url':'data:video/mp4;base64,'+base64.b64encode(clip.read_bytes()).decode()}}
    audit=[];errors=[]
    route_prompt=ROUTE_PROMPT.format(description=job['action_description'][:1600])
    for attempt in range(2):
        raw=client.ask(video_content,route_prompt,audit,'route' if attempt==0 else 'route_retry')
        try:
            route=parse_route(raw);break
        except (ValueError,TypeError) as e:
            route={'mode':'unknown','target':''}
            if attempt==1:errors.append('route_parse:'+str(e))
            route_prompt+='\nYour previous answer did not follow the schema. Return valid JSON with mode and a noun-phrase target; do not output coordinates.'
    result={'job':job,'route':route,'anchors':[],'tracking':[],'audit':audit,'errors':errors,'abstentions':[],
            'coordinate_contract':{'spatial':ocfg['spatial_coordinates'],'tracking':'norm1000',
                                   'tracking_key_base':ocfg['tracking_key_base'],
                                   'tracking_first_key_seconds':ocfg['tracking_first_key_seconds']}}
    if route['mode'] in ('scene','unknown') or not route.get('target','').strip():
        return result
    target=route['target'].strip()[:400]
    spatial=[]
    for f in review_ids:
        im=images[f];h,w=im.shape[:2]
        prompt=build_spatial_grounding_prompt(target)
        if ocfg['spatial_coordinates']=='norm1000':
            prompt+='\nAll four coordinates must be normalized to 0..1000: x=1000*pixel_x/image_width, y=1000*pixel_y/image_height.'
        else:
            prompt+=f'\nCoordinates are pixels of the supplied {w} by {h} image.'
        prompt+=' If the target is not visible in this image, return <answer>[]</answer> instead of inventing a box.'
        raw=client.ask({'type':'image_url','image_url':{'url':image_data(im)}},prompt,audit,f'spatial_{f}')
        try:
            if parse_answer(raw)==[]:
                result['abstentions'].append({'frame':f,'reason':'target_not_visible'});continue
            box=parse_spatial(raw,ocfg['spatial_coordinates'],(w,h),(W,H),route['mode']=='multiple')
            spatial.append({'frame':f,'box':box,'kind':'spatial'})
        except (ValueError,KeyError,TypeError) as e:
            errors.append(f'spatial_{f}:{e}')
    initial=next((s for s in spatial if s['frame']==a),None)
    if spatial:
        question=f'Track {target}. Event hint: {job["action_description"][:800]}'
        if initial:
            ib=[round(initial['box'][i]/(W if i%2==0 else H)*1000) for i in range(4)]
            question+=f' Its bounding box in the first frame is {ib}.'
        else:
            question+=' The target is absent in the first frame. Track it only when it becomes visible.'
        final_second=int(np.floor(ts[b]-ts[a]+1e-6))
        prompt=question+f'\nProvide one bounding box at each integer second 0 through {final_second} of this supplied clip. Keys are actual elapsed seconds: "0" is its first frame, "1" is one second later. All boxes are normalized to 0..1000 coordinates. Keep the same target identity. Omit seconds where it is absent. Return only <answer>{{"boxes":{{"0":[100,200,300,400]}}}}</answer>, with one key for each visible integer second; the example coordinates are only a format example.'
        raw=client.ask(video_content,prompt,audit,'tracking')
        try:
            result['tracking']=parse_tracking(raw,ts,a,b,(W,H),ocfg['tracking_first_key_seconds'],ocfg['tracking_key_base'])
        except (ValueError,KeyError,TypeError) as e:
            errors.append('tracking_parse:'+str(e))
    # Agreement is an engineering check, not a calibrated semantic confidence.
    tracking=result['tracking']
    for s in spatial:
        near=min(tracking,key=lambda t:abs(ts[t['frame']]-ts[s['frame']])) if tracking else None
        score=iou(s['box'],near['box']) if near and abs(ts[near['frame']]-ts[s['frame']])<=0.55 else None
        s['agreement_iou']=score
        s['accepted']=score is not None and score>=ocfg['anchor_min_iou']
        if route['mode']=='multiple':
            # Single-object tracker can disagree with group extent; require agreement too.
            s['accepted']=bool(s['accepted'])
        result['anchors'].append(s)
    # For sub-second clips there may be only one tracking anchor; seed/frame equality still checked.
    return verify_result(client,result,images,(W,H))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--video-dir',required=True)
    p.add_argument('--inputs',default='inputs');p.add_argument('--shots',default='output/shots')
    p.add_argument('--config',default='config/default.json');p.add_argument('--output',default='output/ora')
    p.add_argument('--scope',choices=['partial','all'],default='all')
    p.add_argument('--url',default='http://127.0.0.1:8008/v1');p.add_argument('--model',default='Video-ORA-4B')
    p.add_argument('--ids',default='');p.add_argument('--ledger',default='output/parameter_ledger.json')
    a=p.parse_args();cfg=read_json(a.config);manifest=read_json(Path(a.inputs)/'manifest.json')
    ledger=read_json(a.ledger)
    if not ledger.get('pass') or ledger['total_upper_bound']>cfg['parameter_limit']:
        raise ValueError('Parameter gate failed')
    baseline=Path(a.inputs)/'baseline.zip'
    if sha(baseline)!=manifest['baseline_sha256']:raise ValueError('Baseline mismatch')
    if sha(Path(a.inputs)/'timestamps.npz')!=manifest['timestamps_sha256']:raise ValueError('Timestamps changed')
    ts_cache=np.load(Path(a.inputs)/'timestamps.npz')
    client=OraClient(a.url,a.model,cfg['ora'])
    selection=set(a.ids.split(',')) if a.ids else None
    codehash=digest({f:sha(Path(__file__).parent/f) for f in ['ora_stage.py','ora_tools.py','ora_verify.py','jobs.py']})
    prompt_hash=sha(Path(__file__).resolve().parents[1]/'vendor/eval_prompt.py')
    for row in load_rows(baseline):
        vid=str(row['video_id']);m=manifest['videos'][vid]
        if selection is not None and vid not in selection:continue
        if a.scope=='partial' and m['subject_hint'] not in cfg['partial_hints']:continue
        if not row['predictions']:continue
        video=Path(a.video_dir)/(vid+'.mp4')
        if sha(video)!=m['source_sha256']:raise ValueError(f'Video changed: {vid}')
        shot=read_json(Path(a.shots)/(vid+'.json'))
        if shot['source_sha256']!=m['source_sha256']:raise ValueError('Stale shots')
        ts=ts_cache[vid]
        for job in build_jobs(row,m,ts,shot['cuts'],cfg['ora']['max_chunk_seconds'],cfg['ora']['max_chunk_frames']):
            out=Path(a.output)/vid/(job['id']+'.json')
            signature=digest({'job':job,'source':m['source_sha256'],'cfg':cfg['ora'],
                              'code':codehash,'prompts':prompt_hash,'ledger':ledger['signature'],
                              'baseline':manifest['baseline_sha256'],'shots':shot['signature']})
            if out.exists() and read_json(out).get('signature')==signature:continue
            import tempfile
            with tempfile.TemporaryDirectory(prefix='ora_clip_') as tmp:
                result=infer_job(client,job,m,ts,video,cfg,Path(tmp))
            result.update(signature=signature,source_sha256=m['source_sha256'],video_id=vid,
                          baseline_sha256=manifest['baseline_sha256'],config_digest=digest(cfg),
                          code_digest=codehash,shots_signature=shot['signature'],
                          prompt_sha256=prompt_hash,ledger_signature=ledger['signature'])
            write_json(out,result)
            print(f'{vid}/{job["id"]}: {result["route"]["mode"]}, accepted={sum(x["accepted"] for x in result["anchors"])}, errors={len(result["errors"])}',flush=True)


if __name__=='__main__':main()
