"""Evidence-based temporal editing. Default = deterministic pilot; full run is explicit."""
import argparse,datetime,json,math,os,sys
from pathlib import Path
import numpy as np
from engine import Client,digest,sha,read,save,span,windows,union,frame_ids,visible_text,check_times
import prompts as P
from settings import data_arguments,git_commit

HERE=Path(__file__).resolve().parent

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()

def pilot_ids(manifest,count):
    # Stratify by existing coarse subject hints, ratio and duration, never reference answers.
    rows=manifest['videos']; ordered=sorted(rows,key=lambda i:(rows[i]['duration'],int(i)))
    selected=[]
    def category(i):
        s=rows[i].get('subject_hint','unknown')
        return 'person' if s=='person' else ('animal' if s in ('dog','cat','animal') else 'scene_object_unknown')
    strata=[]
    for subject in ('person','animal','scene_object_unknown'):
        for ratio in ([16,9],[9,16]):
            group=[i for i in ordered if rows[i]['targetRatioWH']==ratio and category(i)==subject]
            if group:strata.append(group)
    for group in strata:
        n=min(len(group),max(1,count//len(strata)))
        for j in np.linspace(0,len(group)-1,n).round().astype(int):
            if group[j] not in selected and len(selected)<count:selected.append(group[j])
    for i in ordered:
        if len(selected)>=count:break
        if i not in selected:selected.append(i)
    return sorted(selected,key=int)

def validate_discovery(obj,duration):
    if not obj['candidates']:visible_text(obj['empty_reason'])
    for c in obj['candidates']:
        span(c['start'],c['end'],duration)
        for k in ('event','evidence','distinct_change'):visible_text(c[k])

def validate_core(obj,duration):
    visible_text(obj['observed_event'])
    check_times(obj['evidence_times'],duration)
    for k in ('reason','evidence_before','evidence_core','evidence_after'):visible_text(obj[k])
    if obj['decision']=='keep':
        a,b=span(obj['start'],obj['end'],duration)
        if not any(a<=t<=b for t in obj['evidence_times']):raise ValueError('No supporting evidence inside retained core')
    elif obj['start'] is not None or obj['end'] is not None:raise ValueError('Unaccepted core must not contain times')

def validate_boundary(obj,duration,proposed):
    visible_text(obj['reason']);check_times([obj['evidence_time']],duration)
    if obj['decision']=='uncertain':
        if obj['time'] is not None:raise ValueError('Uncertain boundary must be null')
    else:
        check_times([obj['time']],duration)
        if obj['decision']=='unchanged' and abs(obj['time']-proposed)>1e-5:raise ValueError('Unchanged boundary moved')

def validate_verify(obj,duration,a,b):
    visible_text(obj['reason']);check_times(obj['evidence_times'],duration)
    if obj['decision']=='keep' and not any(a<=t<=b for t in obj['evidence_times']):
        raise ValueError('No evidence inside proposed interval')

def global_schema(ids):
    return P.obj({'plans':{'type':'array','minItems':len(ids),'maxItems':len(ids),'items':P.obj({
        'candidate_id':P.enum(*ids),'possible_duplicate_of':{'type':['string','null'],'enum':ids+[None]},
        'review_question':P.text()})}})

def validate_duplicate(obj):
    visible_text(obj['reason'])
    allowed={'duplicate':('first','second'),'distinct':('both',),'uncertain':('uncertain',)}
    if obj['representative'] not in allowed[obj['decision']]:raise ValueError('Inconsistent duplicate comparison')

def resolve_reference(ref,accepted,decisions):
    prior={r['candidate_id']:r for r in decisions};visited=set()
    while ref is not None and ref not in accepted:
        if ref in visited:raise ValueError('Cyclic duplicate replacement')
        visited.add(ref);record=prior.get(ref,{})
        if record.get('decision')=='replaced_by_better_visible_instance':ref=record['replacement_id']
        elif record.get('decision')=='drop_duplicate':ref=record['duplicate_review']['reference_id']
        else:return None
    return ref

def validate_plan(obj,ids):
    if sorted(x['candidate_id'] for x in obj['plans'])!=sorted(ids):raise ValueError('Global plan must cover each candidate exactly once')
    rank={cid:i for i,cid in enumerate(ids)}
    for p in obj['plans']:
        visible_text(p['review_question']);ref=p['possible_duplicate_of']
        if ref is not None and rank[ref]>=rank[p['candidate_id']]:raise ValueError('Duplicate reference must point earlier; no cycles')

class Editor:
    def __init__(self,client,cfg,progress):self.client,self.cfg,self.progress=client,cfg,progress
    def process(self,vid,source,meta):
        C=self.client;cfg=self.cfg;duration=meta['duration'];fingerprint=meta['source_sha256']
        audit=[];candidates=[];observations=[]
        C.video_id=vid
        def ask(tag,prompt,schema,clips,validator,tokens=1800):
            self.progress(stage=tag,current_video=vid,candidates=len(candidates),model_calls=C.calls,cache_hits=C.hits)
            answer,key=C.ask(tag,prompt,schema,clips,validator,tokens);audit.append({'stage':tag,'response_key':key});return answer
        def clip(a,b,fps):return C.clip(source,fingerprint,a,b,fps)
        def discover(a,b,depth=0):
            media=clip(a,b,cfg['discovery_fps'])
            obj=ask('discovery',P.DISCOVER,P.DISCOVERY_SCHEMA,[media],lambda o:validate_discovery(o,b-a),2400)
            observations.append({'range':[a,b],'result':obj})
            if obj['overflow']:
                if depth>=4 or b-a<2:raise RuntimeError('Discovery capacity exceeded after subdivision; no candidates silently omitted')
                mid=(a+b)/2;discover(a,mid,depth+1);discover(mid,b,depth+1);return
            for c in obj['candidates']:
                candidate={**c,'start':a+c['start'],'end':a+c['end'],'discovery_window':[a,b]}
                candidates.append(candidate)
                if len(candidates)>cfg['max_candidates']:raise RuntimeError('Candidate capacity exceeded; explicit review required')
        for a,b in windows(duration,cfg['window_seconds'],cfg['stride_seconds']):discover(a,b)
        candidates.sort(key=lambda c:(c['start'],c['end']))
        for i,c in enumerate(candidates):c['candidate_id']=f'c{i:03d}'
        if not candidates:
            # Empty output is genuine only after independent second-pass coverage of every window.
            for a,b in windows(duration,cfg['window_seconds'],cfg['stride_seconds']):
                media=clip(a,b,cfg['core_fps'])
                check=ask('empty_recheck',P.DISCOVER+'\nIndependently recheck whether any supported highlight was overlooked. Do not force a nonempty answer.',P.DISCOVERY_SCHEMA,[media],lambda o:validate_discovery(o,b-a),2400)
                if check['candidates'] or check['overflow']:raise RuntimeError('Empty discovery contradicted by dense recheck; needs investigation')
            return {'candidates':[],'decisions':[],'segments':[],'unresolved':[],'audit':audit,'observations':observations,'quality_flags':['visually_rechecked_empty']}
        ids=[c['candidate_id'] for c in candidates]
        catalog=[{k:c[k] for k in ('candidate_id','start','end','event','distinct_change')} for c in candidates]
        planning_text=P.GLOBAL+'\nCATALOG:'+json.dumps(catalog,ensure_ascii=False)
        # Conservatively bound the text budget rather than silently truncating global context.
        if len(planning_text)>23000:raise RuntimeError('Global catalog too large for audited context budget')
        plan=ask('global_plan',planning_text,global_schema(ids),[],lambda o:validate_plan(o,ids),min(6000,512+100*len(ids)))
        plans={p['candidate_id']:p for p in plan['plans']};decisions=[];accepted={};unresolved=[]
        for c in candidates:
            cid=c['candidate_id'];hypothesis={'candidate':c,'review_plan':plans[cid]}
            a=max(0,c['start']-cfg['context_seconds']);b=min(duration,c['end']+cfg['context_seconds'])
            media=clip(a,b,cfg['core_fps'])
            blind=None
            if cfg.get('blind_observation'):
                def validate_observe(o):
                    for k in ('subject','observed_event','uncertainty'):visible_text(o[k])
                    check_times(o['evidence_times'],b-a)
                blind=ask('blind_observation',P.OBSERVE,P.OBSERVE_SCHEMA,[media],validate_observe,900)
                hypothesis['blind_observation']=blind
            core=ask('core_review',P.CORE+'\nHYPOTHESIS:'+json.dumps(hypothesis,ensure_ascii=False),P.CORE_SCHEMA,[media],lambda o:validate_core(o,b-a))
            record={'candidate_id':cid,'candidate':c,'plan':plans[cid],'core_review':core,'blind_observation':blind,
                    'viewed_range':[a,b],'sampling':{'core_fps':media['effective_fps']}}
            if core['decision']=='review':
                denser=clip(a,b,cfg['boundary_fps'])
                core=ask('core_recheck',P.CORE+'\nIndependent second review. Prior result was uncertain. HYPOTHESIS:'+json.dumps(hypothesis,ensure_ascii=False),P.CORE_SCHEMA,[denser],lambda o:validate_core(o,b-a))
                record['core_recheck']=core
                record['sampling']['recheck_fps']=denser['effective_fps']
                record['sampling']['recheck_added_density']=denser['effective_fps']>media['effective_fps']+1e-9
            if core['decision']!='keep':
                record['decision']=core['decision'];decisions.append(record)
                if core['decision']=='review':unresolved.append(cid+':core')
                continue
            start,end=a+core['start'],a+core['end'];refined=[start,end];edges=[];edge_uncertain=False
            for index,t in enumerate((start,end)):
                ea=max(0,t-cfg['boundary_radius_seconds']);eb=min(duration,t+cfg['boundary_radius_seconds'])
                em=clip(ea,eb,cfg['boundary_fps']);local=t-ea
                prompt=P.BOUNDARY+'\n'+json.dumps({'boundary':'start' if index==0 else 'end','proposed_time':local,'event':core['observed_event'],'core_evidence':core['evidence_core']},ensure_ascii=False)
                edge=ask('boundary_'+('start' if index==0 else 'end'),prompt,P.BOUNDARY_SCHEMA,[em],lambda o:validate_boundary(o,eb-ea,local),700)
                edges.append({'viewed_range':[ea,eb],'result':edge})
                if edge['decision']=='uncertain':edge_uncertain=True
                else:refined[index]=ea+edge['time']
            record['boundary_reviews']=edges
            if edge_uncertain or refined[1]<=refined[0]:
                record['decision']='review';unresolved.append(cid+':boundary');decisions.append(record);continue
            start,end=refined;va=max(0,start-cfg['context_seconds']);vb=min(duration,end+cfg['context_seconds'])
            vm=clip(va,vb,cfg['core_fps'])
            verify=ask('final_verify',P.VERIFY+'\nPROPOSAL:'+json.dumps({'event':core['observed_event'],'start':start-va,'end':end-va},ensure_ascii=False),P.VERIFY_SCHEMA,[vm],lambda o:validate_verify(o,vb-va,start-va,end-va),800)
            record['verification']=verify;record['proposed_interval']=[start,end];record['decision']=verify['decision']
            if verify['decision']=='review':unresolved.append(cid+':final')
            if verify['decision']=='keep':
                ref=resolve_reference(plans[cid]['possible_duplicate_of'],accepted,decisions)
                # Only an already accepted, visually supported event can replace this candidate.
                if ref in accepted:
                    ra,rb=accepted[ref];rm=clip(ra,rb,cfg['discovery_fps']);cm=clip(start,end,cfg['discovery_fps'])
                    duplicate=ask('visual_duplicate',P.DUPLICATE,P.DUPLICATE_SCHEMA,[rm,cm],validate_duplicate,700)
                    record['duplicate_review']={'reference_id':ref,'result':duplicate}
                    if duplicate['decision']=='duplicate':
                        if duplicate['representative']=='first':record['decision']='drop_duplicate'
                        else:
                            del accepted[ref]
                            earlier=next(r for r in decisions if r['candidate_id']==ref)
                            earlier['decision']='replaced_by_better_visible_instance';earlier['replacement_id']=cid
                    elif duplicate['decision']=='uncertain':record['decision']='review';unresolved.append(cid+':duplicate')
                if record['decision']=='keep':accepted[cid]=[start,end]
            decisions.append(record)
        segments=union(list(accepted.values()));flags=[]
        accepted_count=sum(r['decision']=='keep' for r in decisions)
        if len(candidates)>=5 and accepted_count/len(candidates)>.95:flags.append('near_all_candidates_accepted_review_required')
        if sum(b-a for a,b in segments)/duration>.95:flags.append('near_full_video_retention_review_required')
        return {'candidates':candidates,'decisions':decisions,'segments':segments,'unresolved':unresolved,'audit':audit,'observations':observations,'quality_flags':flags}

def export_temporal(records,manifest,timestamps,out):
    # No fabricated geometry for newly selected frames. This handoff feeds the independent spatial stage.
    rows=[]
    for r in records:
        vid=r['video_id'];fs=frame_ids(timestamps[vid],r['segments'])
        rows.append({'video_id':vid,'targetRatioWH':manifest['videos'][vid]['targetRatioWH'],
                     'frame_index_base':0,'frame_count':len(timestamps[vid]),'highlight_frames':fs,
                     'segments_sec':r['segments'],'evidence_resolved':not r['unresolved'],
                     'requires_quality_review':bool([f for f in r.get('quality_flags',[]) if f!='visually_rechecked_empty'])})
    dest=Path(out)/'temporal_handoff.jsonl';tmp=dest.with_suffix('.jsonl.tmp')
    tmp.write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in rows),encoding='utf-8');os.replace(tmp,dest)
    return {'videos':len(rows),'frames':sum(len(r['highlight_frames']) for r in rows),'unresolved_videos':[r['video_id'] for r in rows if not r['evidence_resolved']],'sha256':sha(dest)}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    data_arguments(parser)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--ids',help='Explicit comma-separated original IDs; defaults to metadata-stratified pilot')
    parser.add_argument('--all',action='store_true',help='Explicit full dataset run; never triggered after pilot automatically')
    parser.add_argument('--preflight',action='store_true',help='Read-only CPU checks plus source fingerprints; no inference')
    args=parser.parse_args()
    if args.ids and args.all:parser.error('Choose ids or all, not both')
    ROOT=args.data_root.resolve()
    cfg=read(args.config);manifest=read(ROOT/'inputs/manifest.json');provenance=read(ROOT/'inputs/ora_model_provenance.json')
    if cfg['prompt_version']!='ora/temporal-editor-test-v3':raise ValueError('Prompt version does not match this test implementation')
    ids=[i.strip() for i in args.ids.split(',')] if args.ids else (sorted(manifest['videos'],key=int) if args.all else cfg.get('pilot_ids',pilot_ids(manifest,cfg['pilot_count'])))
    if len(set(ids))!=len(ids) or any(i not in manifest['videos'] for i in ids):raise ValueError('Invalid or duplicate video IDs')
    ts_path=ROOT/'inputs/timestamps.npz'
    if sha(ts_path)!=manifest['timestamps_sha256']:raise ValueError('Original PTS fingerprint changed')
    timestamps=np.load(ts_path,allow_pickle=False)
    for vid in ids:
        m=manifest['videos'][vid];path=args.video_dir/(vid+'.mp4')
        if sha(path)!=m['source_sha256']:raise ValueError('Original video fingerprint changed: '+vid)
        if len(timestamps[vid])!=m['frame_count']:raise ValueError('Frame count mismatch')
        frame_ids(timestamps[vid],[])
    ledger=read(ROOT/'output/parameter_ledger.json')
    if ledger['total_upper_bound']>=9_000_000_000:raise ValueError('Model budget exceeded')
    # No added checkpoint: all stages reuse the same ORA weights; spatial models remain inherited.
    files={p.name:sha(p) for p in HERE.iterdir() if p.suffix in ('.py','.json','.sh')}
    files.update({'prompts/'+p.name:sha(p) for p in P.PROMPT_DIR.glob('*.txt')})
    frozen={'config':cfg,'ids':ids,'files':files,'model':provenance,'parameter_upper_bound':ledger['total_upper_bound'],
            'git_commit':git_commit(),'config_sha256':sha(args.config),'experiment_id':args.output.name,
            'manifest_sha256':sha(ROOT/'inputs/manifest.json'),'timestamps_sha256':sha(ts_path)}
    signature=digest(frozen)
    if args.preflight:
        print(json.dumps({'preflight':'passed','ids':ids,'signature':signature,'source_files_verified':len(ids),'parameter_upper_bound':ledger['total_upper_bound'],'gpu_inference_tested':False}));return
    args.output.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (args.output/'run.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        freeze=args.output/'frozen.json'
        if freeze.exists() and read(freeze)!=frozen:raise ValueError('Existing output uses another source/config/model; choose a new directory')
        save(freeze,frozen);status={'status':'running','started_utc':now(),'pid':os.getpid(),'signature':signature,'requested_ids':ids,'completed_ids':[]}
        def progress(**kwargs):status.update(kwargs,updated_utc=now());save(args.output/'status.json',status)
        client=Client(cfg,args.output,signature);records=[];failed={}
        try:
            # Ensure the endpoint serves the expected checkpoint name before any paid inference.
            models=client.session.get(cfg['endpoint']+'/models',timeout=10);models.raise_for_status()
            if cfg['model'] not in [m['id'] for m in models.json()['data']]:raise RuntimeError('Model endpoint mismatch')
            editor=Editor(client,cfg,progress)
            for vid in ids:
                progress(current_video=vid)
                target=args.output/'records'/(vid+'.json')
                if target.exists():
                    rec=read(target)
                    if rec.get('signature')!=signature:raise ValueError('Stale record')
                else:
                    try:
                        rec={'video_id':vid,'signature':signature,'source_sha256':manifest['videos'][vid]['source_sha256'],
                             **editor.process(vid,args.video_dir/(vid+'.mp4'),manifest['videos'][vid])}
                        save(target,rec)
                    except Exception as error:
                        failed[vid]={'video_id':vid,'signature':signature,'source_sha256':manifest['videos'][vid]['source_sha256'],
                                     'stage':status.get('stage'),'error':f'{type(error).__name__}: {error}'}
                        save(args.output/'errors'/(vid+'.json'),failed[vid])
                        progress(stage='video_failed',failed_ids=list(failed))
                        print('FAILED',vid,failed[vid]['error'],flush=True)
                        continue
                records.append(rec);status['completed_ids'].append(vid)
                export_temporal(records,manifest,timestamps,args.output)
                progress(stage='video_complete');print('COMPLETE',vid,len(status['completed_ids']),len(ids),flush=True)
            report=export_temporal(records,manifest,timestamps,args.output)
            report.update(model_calls=client.calls,cache_hits=client.hits,quality_flags={r['video_id']:r['quality_flags'] for r in records if r['quality_flags']},
                          failed_videos=failed,requested_videos=len(ids),git_commit=frozen['git_commit'],
                          official_score_unknown=True,contest_submission_created=False,pilot=not args.all and not args.ids)
            save(args.output/'report.json',report)
            if failed:raise RuntimeError(f'{len(failed)} videos failed; all remaining IDs were attempted. See errors/ and report.json')
            progress(status='needs_review' if report['unresolved_videos'] or report['quality_flags'] else 'complete',stage='finished',report=report)
        except Exception as error:
            progress(status='failed',error=f'{type(error).__name__}: {error}');raise

if __name__=='__main__':main()
