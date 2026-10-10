"""Fail before the long job if sources, shots, baseline or endpoint are unavailable."""
import argparse,json,sys
from pathlib import Path
from temporal import ROOT,read_json,sha,load_rows
import requests,numpy as np

p=argparse.ArgumentParser();p.add_argument('--ids',required=True);p.add_argument('--video-dir',required=True);a=p.parse_args()
ids=a.ids.split(',');manifest=read_json(ROOT/'inputs/manifest.json');rows=load_rows(ROOT/'inputs/baseline.zip');ts=np.load(ROOT/'inputs/timestamps.npz')
assert sha(ROOT/'inputs/baseline.zip')==manifest['baseline_sha256']
subset=[r for r in rows if str(r['video_id']) in ids];assert len(subset)==len(ids)==len(set(ids))
for r in subset:
    vid=str(r['video_id']);m=manifest['videos'][vid];assert sha(Path(a.video_dir)/(vid+'.mp4'))==m['source_sha256']
    assert read_json(ROOT/'output/shots'/(vid+'.json'))['source_sha256']==m['source_sha256']
    assert len(ts[vid])==m['frame_count']
response=requests.get('http://127.0.0.1:8008/v1/models',timeout=10);response.raise_for_status()
assert any(x['id']=='Video-ORA-4B' for x in response.json()['data'])
print(json.dumps({'videos':len(subset),'frames':sum(len(r['predictions']) for r in subset),'empty_ids':[r['video_id'] for r in subset if not r['predictions']],'sources_verified':True,'ready':True}))
