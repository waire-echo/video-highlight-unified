"""Silent local previews; exact selected frames and submitted crop geometry."""
import argparse
import sys
from pathlib import Path
import cv2
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from common import load_rows,read_json,write_json


def fit(image,w,h):
    out=np.full((h,w,3),18,np.uint8)
    scale=min(w/image.shape[1],h/image.shape[0])
    small=cv2.resize(image,(max(1,round(image.shape[1]*scale)),max(1,round(image.shape[0]*scale))))
    y=(h-small.shape[0])//2;x=(w-small.shape[1])//2
    out[y:y+small.shape[0],x:x+small.shape[1]]=small
    return out


def crop(image,box,ratio):
    x,y,w=box;h=w*ratio[1]/ratio[0]
    # Fractional h is permitted by the submission format. Sampling retains the exact ratio.
    scale=min(390/w,430/h)
    ow=max(2,round(w*scale));oh=max(2,round(h*scale))
    xx=x+(np.arange(ow,dtype=np.float32)+.5)*w/ow-.5
    yy=y+(np.arange(oh,dtype=np.float32)+.5)*h/oh-.5
    mx,my=np.meshgrid(xx,yy)
    return cv2.remap(image,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)


def main():
    p=argparse.ArgumentParser();p.add_argument('--video-dir',required=True)
    p.add_argument('--result',default='output/ORA_SAM_sample3_on_E1.jsonl')
    p.add_argument('--ids',default='0,32,269');p.add_argument('--output',default='output/video_preview')
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    versions=[{str(r['video_id']):r for r in load_rows(path)} for path in
              ['inputs/baseline.zip','output/E1_camera.jsonl',a.result]]
    titles=['Original / submitted boxes','YOLO-only baseline','E1 smoothing','ORA + verify + SAM']
    colors=[(255,255,255),(255,170,40),(40,210,255),(80,235,80)]
    summary=[]
    for vid in a.ids.split(','):
        rows=[v[vid] for v in versions];ratio=rows[0]['targetRatioWH']
        maps=[{p['frame']:p['bboxes'] for p in r['predictions']} for r in rows]
        ids=list(maps[0]);assert all(list(m)==ids for m in maps)
        if not ids:continue
        cap=cv2.VideoCapture(str(Path(a.video_dir)/(vid+'.mp4')))
        if not cap.isOpened():raise RuntimeError('Cannot open '+vid)
        fps=cap.get(cv2.CAP_PROP_FPS)
        path=out/(vid+'_comparison.mp4')
        writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),fps,(1600,530))
        if not writer.isOpened():raise RuntimeError('Cannot create video')
        count=0
        try:
            for f in range(ids[-1]+1):
                ok,im=cap.read()
                if not ok:raise RuntimeError(f'Decode failed {vid}/{f}')
                if f not in maps[0]:continue
                boxes=[m[f] for m in maps]
                annotated=im.copy()
                for box,color in zip(boxes,colors[1:]):
                    x,y,w=box;h=w*ratio[1]/ratio[0]
                    cv2.rectangle(annotated,(round(x),round(y)),(round(x+w-1),round(y+h-1)),color,3)
                panels=[annotated]+[crop(im,b,ratio) for b in boxes]
                canvas=np.full((530,1600,3),18,np.uint8)
                for i,(panel,title,color) in enumerate(zip(panels,titles,colors)):
                    canvas[45:475,i*400:(i+1)*400]=fit(panel,400,430)
                    cv2.putText(canvas,title,(i*400+10,28),cv2.FONT_HERSHEY_SIMPLEX,.62,color,1,cv2.LINE_AA)
                cv2.putText(canvas,f'ID {vid} | original frame {f} | {fps:.3f} fps | selected frames only | silent preview',
                            (12,513),cv2.FONT_HERSHEY_SIMPLEX,.65,(225,225,225),1,cv2.LINE_AA)
                writer.write(canvas)
                if count==0:cv2.imwrite(str(out/(vid+'_comparison.jpg')),canvas)
                count+=1
        finally:
            cap.release();writer.release()
        # Decode the finished artifact, not merely trust the writer's return value.
        check=cv2.VideoCapture(str(path));decoded=0
        while True:
            ok,_=check.read()
            if not ok:break
            decoded+=1
        check.release()
        if decoded!=count:raise RuntimeError(f'Incomplete output {vid}: {decoded}/{count}')
        entry={'video_id':vid,'frames':count,'fps':fps,'file':str(path),'silent':True,'decode_verified':True}
        summary.append(entry);print(entry,flush=True)
    write_json(out/'manifest.json',summary)


if __name__=='__main__':main()
