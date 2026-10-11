"""Reuse exactly the baseline detector cache; no extra model or new predictions."""
import argparse
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from common import read_json,load_rows,sha,write_json
p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--inputs',default='inputs')
a=p.parse_args();root=Path(a.inputs);m=read_json(root/'manifest.json')
if sha(a.cache)!=m['yolo_cache_source_sha256']:raise ValueError('Not the matching baseline YOLO cache')
arrays={}
for row in load_rows(a.cache):
    vid=str(row['video_id']);obs={}
    for group in row['groups']:
        for d in group['detections']:
            if d.get('detected') and d.get('box') is not None:
                obs[int(d['frame'])]=[int(d['frame']),*d['box'],float(d['confidence'])]
    arrays[vid]=np.asarray([obs[k] for k in sorted(obs)],float).reshape(-1,6)
np.savez_compressed(root/'yolo_geometry.npz',**arrays)
write_json(root/'yolo_geometry.provenance.json',{'source_sha256':sha(a.cache),
    'baseline_sha256':m['baseline_sha256'],'npz_sha256':sha(root/'yolo_geometry.npz'),
    'columns':['frame','x1','y1','x2','y2','confidence'],'videos':len(arrays)})
print('Prepared matching YOLO geometry for',len(arrays),'videos')
