"""Inspect model-produced subject boxes on uncropped original frames."""
import argparse
import sys
import textwrap
from pathlib import Path
from PIL import Image,ImageDraw
import cv2
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from common import read_json,write_json
from ora_tools import read_frames
p=argparse.ArgumentParser();p.add_argument('--video-dir',required=True);p.add_argument('--ids',default='0,32,269')
p.add_argument('--input',default='output/ora');p.add_argument('--output',default='output/ORA_smoke_subject_boxes.jpg')
args=p.parse_args()
panels=[]
for vid in args.ids.split(','):
    paths=sorted((Path(args.input)/vid).glob('*.json'))
    for path in paths:
        record=read_json(path)
        anchors=record['anchors']
        shown=[anchors[0],anchors[-1]] if len(anchors)>1 else anchors or [{'frame':record['job']['first'],'box':None,'accepted':False}]
        for anchor in shown:
            panels.append({'video_id':vid,'record':record,'anchor':anchor})
cols=3;cw,ch=460,350
canvas=Image.new('RGB',(cols*cw,70+((len(panels)+cols-1)//cols)*ch),(15,18,23));draw=ImageDraw.Draw(canvas)
draw.text((12,10),'ORA raw spatial output on ORIGINAL full frames (these are subject boxes, not final crops)',fill='white')
draw.text((12,32),'Green: accepted by this pipeline. Orange: rejected/unverified. Acceptance is not ground truth.',fill='white')
for i,panel in enumerate(panels):
    vid=panel['video_id'];r=panel['record'];anchor=panel['anchor'];f=anchor['frame']
    im=read_frames(Path(args.video_dir)/(vid+'.mp4'),[f])[f]
    pil=Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB));scale=min(440/pil.width,215/pil.height)
    pil=pil.resize((round(pil.width*scale),round(pil.height*scale)))
    if anchor['box']:
        color=(60,245,80) if anchor['accepted'] else (255,180,40)
        ImageDraw.Draw(pil).rectangle([v*scale for v in anchor['box']],outline=color,width=3)
    ox=(i%cols)*cw;oy=70+(i//cols)*ch
    canvas.paste(pil,(ox+(cw-pil.width)//2,oy))
    lines=[f"ID {vid} frame {f}: {r['route']['mode']} | accepted={anchor['accepted']}",
           'Target: '+r['route'].get('target',''),
           'Event hint: '+r['job']['action_description']]
    y=oy+220
    for line in lines:
        for s in textwrap.wrap(line,68)[:2]:draw.text((ox+8,y),s,fill='white');y+=15
canvas.save(args.output,quality=94)
write_json(Path(args.output).with_suffix('.json'),[{'video_id':p['video_id'],'frame':p['anchor']['frame'],'mode':p['record']['route']['mode'],'target':p['record']['route'].get('target',''),'accepted':p['anchor']['accepted']} for p in panels])
print('Saved '+args.output)
