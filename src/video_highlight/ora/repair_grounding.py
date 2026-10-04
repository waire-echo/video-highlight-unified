#!/usr/bin/env python3
"""Retry failed temporal grounding with OraRL's exact native task prompt.

Run only after full temporal inference, with the local model server still alive.
Original records and automatic fallback intervals are retained for comparison.
This is format recovery, not a claim that revised boundaries are more accurate.
"""
import argparse
from pathlib import Path
import time

from core import Engine, file_hash, merge_segments, overlap, parse_grounding, read_json, write_json
from video_highlight.prompts import load_text


NATIVE_PROMPT = load_text('ora/v5', 'repair.txt').rstrip('\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--videos',type=Path,required=True)
    p.add_argument('--base-url',default='http://127.0.0.1:8000/v1')
    args=p.parse_args()
    root=args.run.resolve()
    manifest=read_json(root/'manifest.json')
    cfg=manifest['configuration']
    records=[]
    for vid in manifest['input_ids']:
        path=root/'records'/f'{vid}.json'
        row=read_json(path)
        if row.get('status')!='complete' or row.get('signature')!=manifest['signature']:
            raise RuntimeError(f'Incomplete or stale temporal record: {vid}')
        if file_hash(args.videos/f'{vid}.mp4')!=row['source_sha256']:
            raise RuntimeError(f'Source file changed: {vid}')
        records.append((path,row))
    client_args=argparse.Namespace(output=root/'native_repair',model=cfg['model'],model_revision=cfg['model_revision'],
        side=cfg['side'],max_frames=cfg['max_frames'],no_structured=False,
        base_url=args.base_url,timeout=300,ffmpeg='ffmpeg')
    client=Engine(client_args)
    clips=Engine(argparse.Namespace(output=root,side=cfg['side'],ffmpeg='ffmpeg'))
    spec={'parent_signature':manifest['signature'],
        'repair_code_sha256':file_hash(Path(__file__)),
        'prompt':NATIVE_PROMPT,'policy':'Retry only discovery_fallback events; keep fallback if invalid',
        'model_revision':cfg['model_revision'],'pad_seconds':cfg['pad']}
    spec_path=root/'native_repair_manifest.json'
    if spec_path.exists() and read_json(spec_path)!=spec:
        raise RuntimeError('Existing native repair uses different code/configuration')
    write_json(spec_path,spec)
    started=time.monotonic()
    attempts=recovered=unchanged=0
    for path,row in records:
        failures=[e for e in row['events'] if e.get('grounding_status')=='discovery_fallback']
        if not failures:
            continue
        original=root/'records_before_native_repair'/path.name
        if not original.exists():
            write_json(original,row)
        for event in failures:
            if event.get('native_repair_attempted'):
                continue
            attempts+=1
            a,b=event['grounding_context']
            clip=clips.clip(args.videos/f"{row['video_id']}.mp4",row,a,b,cfg['refine_fps'])
            answer=''
            previous={k:event.get(k) for k in ('grounding_status','grounding_answer','grounding_error','grounded_segment')}
            try:
                answer=client.infer(NATIVE_PROMPT.format(event['query']),clip,cfg['refine_fps'],tokens=128)
                local=parse_grounding(answer,b-a)
                segment=[a+local[0],a+local[1]]
                if overlap(segment,[event['start'],event['end']])<=0:
                    raise ValueError('Native interval does not overlap the discovered event')
                event['grounding_status']='native_repair_ok'
                event['grounding_answer']=answer
                event['grounded_segment']=segment
                event['touches_context_edge']=segment[0]<=a+.3 or segment[1]>=b-.3
                event.pop('grounding_error',None)
                recovered+=1
            except ValueError as error:
                event['native_repair_error']=str(error)
                unchanged+=1
            event['native_repair_attempted']=True
            event['native_repair_answer']=answer
            event['before_native_repair']=previous
            print(f"{row['video_id']} {event['event_id']}: {event['grounding_status']}",flush=True)
        row.setdefault('segments_grounded_before_native_repair',row['segments_grounded'])
        spans=[]
        for e in row['events']:
            if e['selection']!='discard':
                a,b=e['grounded_segment']
                spans.append([max(0,a-cfg['pad']),min(row['duration'],b+cfg['pad'])])
        row['segments_grounded']=merge_segments(spans,row['duration'])
        row['native_repair_code_sha256']=spec['repair_code_sha256']
        write_json(path,row)
    report={'attempted_this_run':attempts,'recovered_this_run':recovered,
        'kept_fallback_this_run':unchanged,'elapsed_seconds':round(time.monotonic()-started,2),
        'note':'Valid syntax and overlap do not establish temporal accuracy; original records are preserved.'}
    write_json(root/'native_repair_report.json',report)
    print(report,flush=True)


if __name__=='__main__':main()
