"""Automatic QA sample: largest median shifts plus evenly sampled changed videos."""
import argparse
import json
import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from common import read_json,load_rows,write_json
from ora_tools import read_frames
p=argparse.ArgumentParser();p.add_argument('--video-dir',required=True);a=p.parse_args()
base=load_rows('inputs/baseline.zip');new=load_rows('output/E1_camera.zip');manifest=read_json('inputs/manifest.json')
stats=[]
for b,n in zip(base,new):
    if not b['predictions']:continue
    ds=np.array([np.linalg.norm(np.array(x['bboxes'][:2])-y['bboxes'][:2]) for x,y in zip(n['predictions'],b['predictions'])])
    if np.max(ds)<2:continue
    k=int(np.argmin(np.abs(ds-np.median(ds))))
    stats.append({'video_id':str(b['video_id']),'median_shift':float(np.median(ds)),
                  'max_shift':float(np.max(ds)),'prediction_index':k,'frame':b['predictions'][k]['frame']})
top=sorted(stats,key=lambda s:s['median_shift'],reverse=True)[:8]
rest=[s for s in stats if s not in top]
sample=top+[rest[i] for i in np.linspace(0,len(rest)-1,min(4,len(rest)),dtype=int)]
lookup={str(b['video_id']):(b,n) for b,n in zip(base,new)}
canvas=Image.new('RGB',(1280,80+240*len(sample)),(15,18,23));d=ImageDraw.Draw(canvas)
d.text((12,12),'E1 automatic QA | left: baseline (red) | right: E1 (green)',fill='white')
d.text((12,35),'Original selected frame, crop size unchanged. No ground truth; visual comparison is not a score.',fill='white')
for j,s in enumerate(sample):
    b,n=lookup[s['video_id']];f=s['frame'];im=read_frames(Path(a.video_dir)/(s['video_id']+'.mp4'),[f])[f]
    for side,row,color in [(0,b,(255,80,80)),(1,n,(70,240,90))]:
        pil=Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB));scale=min(620/pil.width,205/pil.height)
        pil=pil.resize((round(pil.width*scale),round(pil.height*scale)));draw=ImageDraw.Draw(pil)
        x,y,w=row['predictions'][s['prediction_index']]['bboxes'];h=w*row['targetRatioWH'][1]/row['targetRatioWH'][0]
        draw.rectangle([x*scale,y*scale,(x+w)*scale,(y+h)*scale],outline=color,width=2)
        ox=side*640+(640-pil.width)//2;oy=80+j*240
        canvas.paste(pil,(ox,oy))
    d.text((12,80+j*240+208),f"ID {s['video_id']} | frame {f} | median shift {s['median_shift']:.1f}px | hint {manifest['videos'][s['video_id']]['subject_hint']}",fill='white')
canvas.save('output/E1_comparison.jpg',quality=92)
write_json('output/E1_visual_samples.json',{'selection':'8 largest median shifts + 4 evenly sampled remaining changed videos','samples':sample,'all_shift_stats':stats})
print('Saved output/E1_comparison.jpg')
