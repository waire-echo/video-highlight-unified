"""Offline temporal comparison. Local AI references are editorial references, not ground truth."""
import argparse,bisect,csv,hashlib,html,json,math,zipfile
from pathlib import Path


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(8<<20),b''):h.update(part)
    return h.hexdigest()


def rows(path):
    path=Path(path)
    if path.suffix=='.zip':
        with zipfile.ZipFile(path) as z:
            names=[n for n in z.namelist() if n.endswith('.jsonl')]
            if len(names)!=1:raise ValueError('Reference ZIP must contain exactly one JSONL')
            text=z.read(names[0]).decode('utf-8-sig')
    else:text=path.read_text(encoding='utf-8-sig')
    result={}
    for line in text.splitlines():
        if not line.strip():continue
        row=json.loads(line);vid=str(row['video_id'])
        if vid in result:raise ValueError('Duplicate video ID: '+vid)
        result[vid]=row
    return result


def mask(row,count):
    if row.get('frame_index_base',0)!=0:raise ValueError('Expected zero-based original frame IDs')
    values=row['highlight_frames'] if 'highlight_frames' in row else [p['frame'] for p in row['predictions']]
    if any(type(i) is not int or not 0<=i<count for i in values):raise ValueError('Frame index outside original source')
    if len(set(values))!=len(values):raise ValueError('Duplicate frame index')
    if row.get('frame_count',count)!=count:raise ValueError('Source frame count mismatch')
    return set(values)


def compare(a,b,count):
    shared=len(a&b)
    return {'ora_frames':len(a),'reference_frames':len(b),'shared_frames':shared,
            'ora_only_frames':len(a-b),'reference_only_frames':len(b-a),
            'ora_retention':len(a)/count,'reference_retention':len(b)/count,
            'agreement_iou':shared/len(a|b) if a|b else 1.0,
            'reference_coverage':shared/len(b) if b else None}


def interval_frames(times,segments):
    result=set()
    for a,b in segments:
        if not (math.isfinite(a+b) and 0<=a<b):raise ValueError('Invalid temporal interval')
        result.update(range(bisect.bisect_left(times,a),bisect.bisect_left(times,b)))
    return len(result)


def audit(args):
    ref=rows(args.reference);old=rows(args.ora_frames)
    ids=[i.strip() for i in args.ids.split(',')] if args.ids else sorted(ref,key=int)
    if len(set(ids))!=len(ids):raise ValueError('Duplicate selected IDs')
    new=rows(args.new_handoff) if args.new_handoff else {}
    result=[];missing_new=[]
    for vid in ids:
        if vid not in old or vid not in ref:raise ValueError('Video missing from paired inputs: '+vid)
        meta=read(args.reference_root/'metadata'/f'{vid}.json')
        annotation=read(args.reference_root/'annotations'/f'{vid}.json')
        pts=read(args.reference_root/'timestamps'/f'{vid}.json')
        record=read(args.ora_records/f'{vid}.json')
        if record['source_sha256']!=meta['source_sha256'] or pts['source_sha256']!=meta['source_sha256']:
            raise ValueError('Videos/PTS from different source bytes: '+vid)
        count=meta['frame_count'];times=pts['times']
        if len(times)!=count or any(not math.isfinite(t) for t in times) or any(a>=b for a,b in zip(times,times[1:])):
            raise ValueError('Invalid original PTS: '+vid)
        # Format duration may end at/slightly before the last decoded PTS. Preserve PTS.
        if times[0]<0:raise ValueError('Negative original PTS: '+vid)
        if old[vid]['targetRatioWH']!=ref[vid]['targetRatioWH'] or ref[vid]['targetRatioWH']!=meta['targetRatioWH']:
            raise ValueError('Paired target ratios differ: '+vid)
        am=mask(old[vid],count);bm=mask(ref[vid],count)
        events=record['events'];kept=[e for e in events if e.get('selection')!='discard']
        stages={'all_discovered_candidates':interval_frames(times,[[e['start'],e['end']] for e in events]),
                'discovery_not_discard':interval_frames(times,[[e['start'],e['end']] for e in events if e.get('value')!='discard']),
                'after_review':interval_frames(times,[[e['start'],e['end']] for e in kept]),
                'grounded_without_padding':interval_frames(times,[e.get('grounded_segment') or [e['start'],e['end']] for e in kept]),
                'exported':len(am)}
        item={'video_id':vid,'source_sha256':meta['source_sha256'],'source_frames':count,'duration':meta['duration'],
              'last_pts_minus_format_duration':times[-1]-meta['duration'],
              **compare(am,bm,count),'ora_segment_seconds':sum(b-a for a,b in record['segments_grounded']),
              'reference_segment_seconds':sum(s['end']-s['start'] for s in annotation['segments']),
              'ora_stages_frames':stages,'ora_segments':record['segments_grounded'],
              'reference_segments':[{k:s[k] for k in ('start','end','event','why_keep')} for s in annotation['segments']],
              'reference_excluded':annotation.get('excluded',[]),
              'ora_events':[{k:e.get(k) for k in ('query','start','end','selection','grounded_segment','grounding_status')} for e in events]}
        if args.new_handoff:
            if vid not in new:missing_new.append(vid)
            else:
                nr=read(args.new_records/f'{vid}.json')
                if nr['source_sha256']!=meta['source_sha256']:raise ValueError('New run used another source: '+vid)
                if new[vid]['targetRatioWH']!=meta['targetRatioWH']:raise ValueError('New ratio differs: '+vid)
                nm=mask(new[vid],count)
                item['new_run']={**compare(nm,bm,count),'segments':nr['segments'],
                                 'unresolved':nr['unresolved'],'quality_flags':nr.get('quality_flags',[]),
                                 'candidates':nr['candidates'],'decisions':nr['decisions'],'audit':nr['audit']}
        result.append(item)
    total=sum(r['source_frames'] for r in result);seconds=sum(r['duration'] for r in result)
    aggregate={k:sum(r[k] for r in result) for k in ('ora_frames','reference_frames','shared_frames','ora_only_frames','reference_only_frames')}
    aggregate.update(videos=len(result),source_frames=total,source_seconds=seconds,
                     ora_retention_frames=aggregate['ora_frames']/total,reference_retention_frames=aggregate['reference_frames']/total,
                     ora_retention_seconds=sum(r['ora_segment_seconds'] for r in result)/seconds,
                     reference_retention_seconds=sum(r['reference_segment_seconds'] for r in result)/seconds,
                     ora_stages_frames={k:sum(r['ora_stages_frames'][k] for r in result) for k in result[0]['ora_stages_frames']})
    report={'interpretation':'Reference agreement and retention; not official accuracy or leaderboard score.',
            'reference_sha256':sha(args.reference),'ora_frames_sha256':sha(args.ora_frames),'selected_ids':ids,
            'source_hashes_paired':True,'aggregate':aggregate,'missing_new_ids':missing_new,'videos':result}
    if args.new_handoff:report['new_handoff_sha256']=sha(args.new_handoff)
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    fields=['video_id','source_frames','duration','ora_frames','reference_frames','shared_frames','ora_only_frames','reference_only_frames','ora_retention','reference_retention','agreement_iou','reference_coverage']
    with (args.output/'comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fields,extrasaction='ignore');w.writeheader();w.writerows(result)
    panels=[]
    def timeline(segments,duration,color):
        pieces=''.join(f'<i style="left:{100*a/duration:.4f}%;width:{100*(b-a)/duration:.4f}%;background:{color}"></i>' for a,b in segments)
        return '<div class="timeline">'+pieces+'</div>'
    for r in result:
        panels.append('<section><h2>Video '+html.escape(r['video_id'])+'</h2>ORA '+timeline(r['ora_segments'],r['duration'],'#d97706')+
                      'GPT reference '+timeline([[s['start'],s['end']] for s in r['reference_segments']],r['duration'],'#2563eb')+
                      ('New test '+timeline(r['new_run']['segments'],r['duration'],'#16a34a') if 'new_run' in r else '')+
                      '<details><summary>Events, evidence and decisions</summary><pre>'+html.escape(json.dumps(r,ensure_ascii=False,indent=2))+'</pre></details></section>')
    page='<!doctype html><meta charset="utf-8"><title>Temporal comparison</title><style>body{font:16px system-ui;max-width:1100px;margin:30px auto;padding:0 15px}section{margin:25px 0;padding:18px;border:1px solid #ddd}.timeline{height:20px;position:relative;background:#eee;margin:6px 0 12px}.timeline i{position:absolute;height:100%}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style><h1>Temporal comparison</h1><p>Offline editorial reference agreement; no official score. Timeline = entire original video.</p>'+''.join(panels)
    (args.output/'index.html').write_text(page,encoding='utf-8')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,required=True);p.add_argument('--reference-root',type=Path,required=True)
    p.add_argument('--ora-frames',type=Path,required=True);p.add_argument('--ora-records',type=Path,required=True)
    p.add_argument('--ids');p.add_argument('--new-handoff',type=Path);p.add_argument('--new-records',type=Path)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if bool(a.new_handoff)!=bool(a.new_records):p.error('Provide both new-handoff and new-records')
    report=audit(a);print(json.dumps(report['aggregate'],ensure_ascii=False,indent=2))


if __name__=='__main__':main()
