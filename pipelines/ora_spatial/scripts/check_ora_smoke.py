"""Fail early on malformed model outputs, never hide a failed server as center crops."""
import argparse
import json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--ids',default='0,32,269');a=p.parse_args()
files=[p for vid in a.ids.split(',') for p in (Path('output/ora')/vid).glob('*.json')]
if not files:raise RuntimeError('ORA smoke produced no cache')
data=[json.loads(p.read_text()) for p in files]
errors=[(d['video_id'],d['job']['id'],e) for d in data for e in d['errors']]
subjects=[d for d in data if d['route']['mode'] in ('single','multiple')]
anchors=sum(sum(a.get('accepted',False) for a in d['anchors']) for d in subjects)
summary={'chunks':len(data),'subject_chunks':len(subjects),'accepted_anchors':anchors,'errors':errors,
         'note':'Protocol smoke only; not an accuracy evaluation.'}
Path('output/ora_smoke.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
if errors or not subjects or anchors==0:
    raise RuntimeError('Inspect output/ora_smoke.json and raw responses before full run; check units/prompt/video input')
