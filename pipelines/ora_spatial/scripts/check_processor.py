"""Exercise installed ORA image/video processors without loading the language model."""
import argparse
import sys
from pathlib import Path
import numpy as np
import torch
from transformers import AutoProcessor
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from common import read_json,write_json
p=argparse.ArgumentParser();p.add_argument('--ora-dir',required=True);a=p.parse_args()
torch.set_num_threads(1)
cfg=read_json('config/default.json')['ora']
processor=AutoProcessor.from_pretrained(a.ora_dir,local_files_only=True)
video=np.zeros((32,540,960,3),dtype=np.uint8)
v=processor.video_processor(videos=[video],do_sample_frames=False,
    size={'shortest_edge':65536,'longest_edge':cfg['video_total_pixels']},return_tensors='pt')
i=processor.image_processor(images=video[0],
    size={'shortest_edge':65536,'longest_edge':cfg['image_max_pixels']},return_tensors='pt')
vt=int(torch.prod(v['video_grid_thw'][0]).item())//4
it=int(torch.prod(i['image_grid_thw'][0]).item())//4
if not 1024<vt<=8192:raise ValueError(f'Unexpected video token budget: {vt}')
if not 1<=it<=1024:raise ValueError(f'Unexpected image token budget: {it}')
result={'image_grid':i['image_grid_thw'].tolist(),'video_grid':v['video_grid_thw'].tolist(),
        'image_tokens':it,'video_tokens':vt,'context_limit':16384,'output_limit':cfg['max_new_tokens'],
        'actual_processor_tested':True,'language_model_inference_tested':False}
write_json('output/processor_preflight.json',result)
print(result)
