"""E2/E3 share the same ORA/SAM cache; out-of-scope E2 rows equal E1 exactly."""
import argparse
import copy
from pathlib import Path
import numpy as np
from common import load_rows,read_json,write_json,sha,digest,groups,export,validate,box_from_center
from jobs import build_jobs
from camera import smooth_camera
from adaptive_camera import nearest_yolo,visual_evidence,compose_segment,smooth_crops
from collections import Counter


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--inputs',default='inputs');p.add_argument('--e1',default='output/E1_camera.jsonl')
    p.add_argument('--shots',default='output/shots');p.add_argument('--sam',default='output/sam')
    p.add_argument('--ora',default='output/ora');p.add_argument('--scope',choices=['partial','all'],required=True)
    p.add_argument('--config',default='config/default.json');p.add_argument('--output',required=True)
    p.add_argument('--ids',default='',help='Diagnostic subset: other videos stay exactly E1')
    p.add_argument('--fixed-width',action='store_true',help='Explicit legacy ablation; adaptive size is the default')
    p.add_argument('--video-dir',help='Original videos, required for adaptive visibility evidence')
    p.add_argument('--reframe-config',default='config/reframe.json')
    a=p.parse_args();cfg=read_json(a.config);manifest=read_json(Path(a.inputs)/'manifest.json')
    if sha(Path(a.inputs)/'baseline.zip')!=manifest['baseline_sha256'] or sha(Path(a.inputs)/'timestamps.npz')!=manifest['timestamps_sha256']:
        raise ValueError('Input fingerprint mismatch')
    base=load_rows(Path(a.inputs)/'baseline.zip');e1=load_rows(a.e1);validate(e1,base,manifest)
    e1_report=read_json(Path(a.e1).with_suffix('.report.json'))
    if e1_report['baseline_sha256']!=manifest['baseline_sha256'] or e1_report['config']!=cfg:
        raise ValueError('E1 configuration mismatch')
    if e1_report['output_jsonl_sha256']!=sha(a.e1):raise ValueError('E1 file changed')
    adaptive=None;geometry=None
    if not a.fixed_width:
        if not a.video_dir:raise ValueError('Adaptive composition requires --video-dir')
        adaptive=read_json(a.reframe_config)
        provenance=read_json(Path(a.inputs)/'yolo_geometry.provenance.json')
        if (provenance['baseline_sha256']!=manifest['baseline_sha256'] or
            provenance['source_sha256']!=manifest['yolo_cache_source_sha256'] or
            provenance['npz_sha256']!=sha(Path(a.inputs)/'yolo_geometry.npz')):
            raise ValueError('Stale YOLO geometry')
        geometry=np.load(Path(a.inputs)/'yolo_geometry.npz')
    selection=set(a.ids.split(',')) if a.ids else None
    if selection is not None and not selection.issubset({str(r['video_id']) for r in base}):
        raise ValueError('Unknown sample video IDs')
    rows=copy.deepcopy(e1);ts_cache=np.load(Path(a.inputs)/'timestamps.npz')
    report={'scope':a.scope,'selected_video_ids':[],'groups':{},'e1_sha256':sha(a.e1),
            'baseline_sha256':manifest['baseline_sha256'],'config':cfg,
            'diagnostic_subset':sorted(selection) if selection is not None else None,
            'composition':'legacy_fixed_width' if a.fixed_width else 'adaptive_center_scale_visibility',
            'reframe_config':adaptive,'reframe_code_sha256':sha(Path(__file__).parent/'adaptive_camera.py')}
    for row in rows:
        vid=str(row['video_id']);m=manifest['videos'][vid]
        if selection is not None and vid not in selection:continue
        if a.scope=='partial' and m['subject_hint'] not in cfg['partial_hints']:continue
        report['selected_video_ids'].append(vid)
        if not row['predictions']:continue
        shot=read_json(Path(a.shots)/(vid+'.json'));ts=ts_cache[vid];observations={}
        routes={};rejected=[];absent=[]
        for job in build_jobs(row,m,ts,shot['cuts'],cfg['ora']['max_chunk_seconds'],cfg['ora']['max_chunk_frames']):
            stem=Path(a.sam)/vid/job['id'];cache=read_json(stem.with_suffix('.json'))
            ora=Path(a.ora)/vid/(job['id']+'.json')
            if cache['source_sha256']!=m['source_sha256'] or cache['config_digest']!=digest(cfg) or cache['ora_sha256']!=sha(ora):
                raise ValueError(f'Stale SAM cache: {vid}/{job["id"]}')
            if cache['npz_sha256']!=sha(stem.with_suffix('.npz')):raise ValueError('SAM cache damaged')
            data=np.load(stem.with_suffix('.npz'))
            expected=list(range(job['first'],job['last']+1))
            if data['frames'].tolist()!=expected:raise ValueError('Missing/reordered dense frames')
            if data['centers'].shape!=(len(expected),2) or data['quality'].shape!=(len(expected),):raise ValueError('Invalid cache dimensions')
            if not np.isfinite(data['centers']).all() or not np.isfinite(data['quality']).all():raise ValueError('Nonfinite dense cache')
            record=read_json(ora)
            if not a.fixed_width:
                if 'boxes' not in data or 'status' not in data:raise ValueError('Adaptive mode needs SAM extents and validity')
                if data['boxes'].shape!=(len(expected),4) or data['status'].shape!=(len(expected),):raise ValueError('Invalid dense extents')
                if not np.isfinite(data['boxes']).all():raise ValueError('Nonfinite SAM box')
                for anchor in record['anchors']:
                    if 'verification' in anchor and not anchor['verification']['passed']:
                        reason=anchor['verification']['status']
                        rejected.append({'frame':anchor['frame'],'box':anchor['box'],'reason':reason,
                                         'first':job['first'],'last':job['last']})
                        if reason=='target_absent':absent.append(anchor['frame'])
                absent.extend(x['frame'] for x in record.get('abstentions',[]))
            for j,(f,c,q) in enumerate(zip(data['frames'],data['centers'],data['quality'])):
                observations[int(f)]={'center':c,'quality':q,'box':data['boxes'][j] if 'boxes' in data else None,
                                      'status':int(data['status'][j]) if 'status' in data else 0}
                routes[int(f)]=record.get('route',{}).get('mode','unknown')
        frame_map={p['frame']:p for p in row['predictions']};diags=[]
        old_map={p['frame']:p['bboxes'] for p in next(r for r in base if str(r['video_id'])==vid)['predictions']}
        yolo={};visual={}
        if not a.fixed_width:
            video=Path(a.video_dir)/(vid+'.mp4')
            if sha(video)!=m['source_sha256']:raise ValueError('Original video changed')
            yolo=nearest_yolo(list(frame_map),ts,geometry[vid] if vid in geometry else None,adaptive)
            hint_boxes={f:obs['box'] for f,obs in observations.items() if obs['status']==2}
            for f,b in yolo.items():hint_boxes.setdefault(f,b)
            visual=visual_evidence(video,list(frame_map),hint_boxes)
        states_by_frame=[]
        for fs in groups(list(frame_map),shot['cuts']):
            if a.fixed_width:
                points=[observations[f]['center'] for f in fs];q=[observations[f]['quality'] for f in fs]
                w=frame_map[fs[0]]['bboxes'][2];h=w*row['targetRatioWH'][1]/row['targetRatioWH'][0]
                path,diag=smooth_camera(ts[fs],points,q,(m['width'],m['height']),(w,h),cfg['camera'])
                for f,c in zip(fs,path):frame_map[f]['bboxes']=box_from_center(c,w,row['targetRatioWH'],m['width'],m['height'])
            else:
                targets,q,protected,states=compose_segment(fs,ts,observations,yolo,old_map,row['targetRatioWH'],
                    (m['width'],m['height']),m['subject_hint']=='scene',routes,rejected,absent,visual,adaptive)
                boxes,diag=smooth_crops(ts[fs],targets,q,protected,(m['width'],m['height']),row['targetRatioWH'],adaptive)
                for f,b in zip(fs,boxes):frame_map[f]['bboxes']=b
                diag['states']=dict(Counter(states));states_by_frame.extend([{'frame':f,'state':s} for f,s in zip(fs,states)])
            diags.append({'first':fs[0],'last':fs[-1],**diag})
        report['groups'][vid]=diags
        if not a.fixed_width:write_json(Path(a.output).parent/(Path(a.output).stem+'_states')/(vid+'.json'),states_by_frame)
    report['validation']=export(a.output,rows,base,manifest,preserve_width=a.fixed_width)
    report['changes_vs_e1']=validate(rows,e1,manifest,preserve_width=a.fixed_width)
    if a.scope=='partial' or selection is not None:
        for row,old in zip(rows,e1):
            if str(row['video_id']) not in report['selected_video_ids'] and row!=old:raise AssertionError('Out-of-scope mutation')
    write_json(Path(a.output).with_suffix('.report.json'),report)
    print({k:v for k,v in report['validation'].items() if k!='changes_by_video'})


if __name__=='__main__':main()
