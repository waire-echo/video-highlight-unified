"""Paired experiment: retain the old candidate boxes and only change acceptance."""
import argparse
import copy
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from common import read_json,write_json,sha
from ora_tools import OraClient,read_frames
from ora_verify import verify_result

p=argparse.ArgumentParser()
p.add_argument('--video-dir',required=True)
p.add_argument('--input',default='output/ora')
p.add_argument('--output',default='output/ora_rechecked')
p.add_argument('--ids',default='0,32,269')
a=p.parse_args()
cfg=read_json('config/default.json');manifest=read_json('inputs/manifest.json')
client=OraClient('http://127.0.0.1:8008/v1','Video-ORA-4B',cfg['ora'])
summary=[]
for vid in a.ids.split(','):
    for path in sorted((Path(a.input)/vid).glob('*.json')):
        record=copy.deepcopy(read_json(path));meta=manifest['videos'][vid]
        candidates=[x for x in record['anchors'] if x.get('accepted')]
        images=read_frames(Path(a.video_dir)/(vid+'.mp4'),[x['frame'] for x in candidates],cfg['ora']['image_max_side'])
        verify_result(client,record,images,(meta['width'],meta['height']))
        for anchor in candidates:
            entry={'video_id':vid,'frame':anchor['frame'],'target':record['route']['target'],**anchor['verification']}
            summary.append(entry);print(entry,flush=True)
        # Diagnostic cache deliberately cannot pass production's source/config/code signature gate.
        record['diagnostic_source_sha256']=sha(path)
        record['signature']='diagnostic_recheck_only'
        record['code_digest']='diagnostic_recheck_only'
        write_json(Path(a.output)/vid/path.name,record)
write_json('output/anchor_verification_review.json',summary)
print(f'Before={len(summary)}, after={sum(x["passed"] for x in summary)}',flush=True)
