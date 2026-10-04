"""Competition contracts and a PDF-derived reference metric (not the organizer script)."""
import json
import math
import re
from pathlib import Path

def number(x):
    return type(x) in (int, float) and math.isfinite(x)

def temporal_answer(text, duration):
    if '<think>' in text.lower():
        raise ValueError('Unexpected reasoning output')
    blocks = re.findall(r'<answer>\s*(.*?)\s*</answer>', text, re.S | re.I)
    if len(blocks) != 1:
        raise ValueError('Require one complete answer block')
    body = blocks[0].strip()
    if body.startswith('[') and body.endswith(']'):
        body = body[1:-1].strip()
    m = re.fullmatch(r'(\d+(?:\.\d+)?)\s*(?:to|-)\s*(\d+(?:\.\d+)?)(?:\s*(?:seconds?|s))?\s*',body,re.I)
    if not m:
        raise ValueError('Ambiguous time interval')
    a,b = map(float,m.groups())
    if not 0 <= a < b <= duration + .51:
        raise ValueError('Out-of-range time interval')
    if a >= duration:
        raise ValueError('Empty clipped interval')
    return [a,min(b,duration)]

def answer_letter(text, allowed='ABC'):
    m = re.fullmatch(r'\s*(?:<answer>\s*)?([A-Z])(?:\s*</answer>)?\s*',text)
    if not m or m[1] not in allowed:
        raise ValueError('Invalid option answer')
    return m[1]

def spatial_answer(text, width, height, target=''):
    """Training-aligned Qwen3.5 output is explicitly treated as norm1000."""
    blocks = re.findall(r'<answer>\s*(.*?)\s*</answer>',text,re.S|re.I)
    body = blocks[0] if len(blocks)==1 else text.strip()
    if len(blocks)>1 or '<think>' in text.lower():
        raise ValueError('Ambiguous spatial answer')
    body=re.sub(r'^```(?:json)?\s*|\s*```$','',body.strip())
    obj=json.loads(body)
    if isinstance(obj,dict): obj=[obj]
    if not isinstance(obj,list) or not 1<=len(obj)<=10 or not all(isinstance(item,dict) for item in obj):
        raise ValueError('Invalid spatial box list')
    boxes=[]
    for item in obj:
        b=item.get('bbox_2d')
        if not isinstance(b,list) or len(b)!=4 or not all(number(x) for x in b):
            raise ValueError('Invalid spatial box')
        x1,y1,x2,y2=b
        if not (0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000):
            raise ValueError('Invalid norm1000 box')
        boxes.append(b)
    if len(boxes)>1:
        labels=[item.get('label') for item in obj]
        if not all(isinstance(label,str) and label.strip() for label in labels):
            raise ValueError('Ambiguous multiple spatial targets')
        names={label.strip().casefold() for label in labels}
        if len(names)!=1 and not all(re.search(r'(?<!\w)'+re.escape(name)+r'(?!\w)',target.casefold()) for name in names):
            raise ValueError('Ambiguous multiple spatial targets')
    x1=min(b[0] for b in boxes);y1=min(b[1] for b in boxes)
    x2=max(b[2] for b in boxes);y2=max(b[3] for b in boxes)
    return [x1*width/1000,y1*height/1000,x2*width/1000,y2*height/1000]

def crop_box(focus, width, height, ratio, previous=None, alpha=.35, margin=1.15):
    """Fit aspect ratio, smooth, then constrain to include the target when feasible."""
    rw,rh=ratio; aspect=rw/rh
    maximum=min(width,height*aspect)
    if focus is None:
        cx,cy,cw=width/2,height/2,maximum
    else:
        x1,y1,x2,y2=focus
        cx,cy=(x1+x2)/2,(y1+y2)/2
        cw=min(maximum,max(x2-x1,(y2-y1)*aspect)*margin)
    if previous is not None:
        px,py,pw=previous
        cx=alpha*cx+(1-alpha)*(px+pw/2)
        cy=alpha*cy+(1-alpha)*(py+pw/aspect/2)
        cw=alpha*cw+(1-alpha)*pw
    if focus is not None:
        cw=max(cw,min(maximum,max(focus[2]-focus[0],(focus[3]-focus[1])*aspect)))
    cw=min(maximum,max(1,cw)); ch=cw/aspect
    x=min(width-cw,max(0,cx-cw/2)); y=min(height-ch,max(0,cy-ch/2))
    if focus is not None:
        lo,hi=max(0,focus[2]-cw),min(width-cw,focus[0])
        if lo<=hi:x=min(hi,max(lo,x))
        lo,hi=max(0,focus[3]-ch),min(height-ch,focus[1])
        if lo<=hi:y=min(hi,max(lo,y))
    x,y,cw=(float(round(float(v),6)) for v in (x,y,cw))
    cw=min(cw,width-x,(height-y)*aspect)
    return [x,y,cw]

def valid_crop(box,width,height,ratio):
    if not isinstance(box,list) or len(box)!=3 or not all(number(v) for v in box):return False
    x,y,w=box;h=w*ratio[1]/ratio[0]
    return x>=0 and y>=0 and w>0 and x+w<=width+1e-5 and y+h<=height+1e-5

def validate_row(item,row,meta):
    if row.get('video_id')!=item['video_id'] or row.get('targetRatioWH')!=item['targetRatioWH']:
        raise ValueError('Index fields changed')
    for k,v in item.items():
        if k!='predictions' and row.get(k)!=v:raise ValueError('Index metadata changed: '+k)
    if not isinstance(row.get('predictions'),list):raise ValueError('Missing predictions')
    last=-1
    for pred in row['predictions']:
        f=pred.get('frame')
        if type(f) is not int or not last<f<meta['frame_count']:raise ValueError('Invalid/duplicate/unsorted frame')
        if not valid_crop(pred.get('bboxes'),meta['width'],meta['height'],item['targetRatioWH']):raise ValueError('Invalid crop')
        last=f

def iou(a,b):
    inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/union if union>0 else 0.

def weighted_f1(predictions,ground_truth,width,height,ratio):
    """Count every submitted attempt, including duplicates/invalid boxes, as Npred."""
    truth={x['frame']:x['bboxes'] for x in ground_truth};used=set();total=0.
    if len(truth)!=len(ground_truth):raise ValueError('Duplicate ground truth')
    def xyxy(b):return [b[0],b[1],b[0]+b[2],b[1]+b[2]*ratio[1]/ratio[0]]
    for p in predictions:
        f=p.get('frame');b=p.get('bboxes')
        if type(f) is not int or f<0 or not valid_crop(b,width,height,ratio) or f in used:continue
        used.add(f)
        if f in truth: total+=iou(xyxy(b),xyxy(truth[f]))
    npred,ngt=len(predictions),len(ground_truth)
    return {'n_pred':npred,'n_gt':ngt,'sum_iou':total,'f1':2*total/(npred+ngt) if npred+ngt else 1.}

def resolve_video(item,video_dir):
    for key in ('video_path','video_file','video_filename','filename','file_name','video_name','video'):
        if isinstance(item.get(key),str) and item[key]:
            p=Path(item[key]);return p if p.is_absolute() else video_dir/p
    return video_dir/(str(item['video_id'])+'.mp4')
