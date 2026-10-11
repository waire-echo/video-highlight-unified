import numpy as np
from common import groups


def build_jobs(row,meta,ts,cuts,max_seconds,max_frames=240):
    jobs=[]
    frames=[p['frame'] for p in row['predictions']]
    for group in groups(frames,cuts):
        start=0
        while start<len(group):
            end=start+1
            while end<len(group) and end-start<max_frames and ts[group[end]]-ts[group[start]] < max_seconds:
                end+=1
            first,last=group[start],group[end-1]
            # Event matching uses the original timeline, not chunk-local seconds.
            events=[]
            for e in meta['events']:
                a,b=e['grounded_segment'] if e.get('grounded_segment') else [e['start'],e['end']]
                overlap=max(0,min(b,ts[last]+1/meta['fps'])-max(a,ts[first]))
                if overlap>0:
                    events.append((overlap,e.get('query','')))
            description=max(events,key=lambda v:v[0])[1] if events else ''
            jobs.append({'id':f'{first:08d}_{last:08d}','first':first,'last':last,
                         'action_description':description,'shot_group_first':group[0]})
            start=end
    return jobs
