"""Periodic language-grounded anchors + dense optical flow; optional local YOLO correction."""
import math
import cv2
import numpy as np
from .contracts import spatial_answer,crop_box,iou

NATIVE_SPATIAL=('Locate "{}" in the image. Output its bounding box in JSON format within <answer>...</answer> tags. '
 'Example: <answer>[{{"bbox_2d": [123, 30, 404, 846]}}]</answer>')

def cut_score(previous,current):
    if previous is None:return 0.
    # HSV histogram avoids treating ordinary translations as scene cuts.
    def hist(im):
        hsv=cv2.cvtColor(cv2.resize(im,(128,72)),cv2.COLOR_BGR2HSV)
        h=cv2.calcHist([hsv],[0,1],None,[24,16],[0,180,0,256]);cv2.normalize(h,h)
        return h
    return float(cv2.compareHist(hist(previous),hist(current),cv2.HISTCMP_BHATTACHARYYA))

class FlowTracker:
    def __init__(self):self.gray=None;self.box=None;self.scale=1.
    def update(self,frame,anchor=None):
        h,w=frame.shape[:2];scale=min(1.,640/max(h,w))
        gray=cv2.cvtColor(cv2.resize(frame,(round(w*scale),round(h*scale))),cv2.COLOR_BGR2GRAY)
        result=anchor
        if anchor is None and self.gray is not None and self.box is not None:
            mask=np.zeros_like(self.gray);x1,y1,x2,y2=[int(v*scale) for v in self.box]
            mask[max(0,y1):min(mask.shape[0],y2+1),max(0,x1):min(mask.shape[1],x2+1)]=255
            pts=cv2.goodFeaturesToTrack(self.gray,100,.01,5,mask=mask)
            if pts is not None and len(pts)>=5:
                nxt,status,_=cv2.calcOpticalFlowPyrLK(self.gray,gray,pts,None)
                back,bst,_=cv2.calcOpticalFlowPyrLK(gray,self.gray,nxt,None)
                valid=(status.ravel()==1)&(bst.ravel()==1)&(np.linalg.norm(back-pts,axis=2).ravel()<1.5)
                if valid.sum()>=5:
                    old=pts[valid].reshape(-1,2);new=nxt[valid].reshape(-1,2)
                    shift=np.median(new-old,axis=0)/scale
                    x1,y1,x2,y2=self.box
                    result=[max(0,x1+shift[0]),max(0,y1+shift[1]),min(w,x2+shift[0]),min(h,y2+shift[1])]
                    if result[2]-result[0]<1 or result[3]-result[1]<1:result=None
        self.gray=gray;self.box=result;self.scale=scale
        return result

class Cropper:
    def __init__(self,client,cfg):
        self.client=client;self.cfg=cfg;self.yolo=None
        if cfg['tracker']=='yolo':
            from pathlib import Path
            weights=Path(cfg['yolo_weights'] or '')
            if not weights.is_file():raise ValueError('YOLO requires explicit existing local weights; no auto-download')
            from ultralytics import YOLO
            self.yolo=YOLO(str(weights))

    def locate(self,frame,event,frame_index,video_id):
        if event['focus_mode']=='scene':return None
        target=event['subject_description']
        if target.count(',')>=4:target=event['query']
        dest=self.client.root/'keyframes'/str(video_id)/(str(frame_index)+'.jpg')
        dest.parent.mkdir(parents=True,exist_ok=True)
        h,w=frame.shape[:2];factor=min(1.,1024/max(h,w))
        small=cv2.resize(frame,(round(w*factor),round(h*factor)))
        if not cv2.imwrite(str(dest),small,[cv2.IMWRITE_JPEG_QUALITY,95]):raise OSError('Cannot write keyframe')
        prompt=NATIVE_SPATIAL.format(target.rstrip('.')+'.')
        try:
            raw=self.client.ask(prompt,dest,kind='image',tokens=256)
            return spatial_answer(raw,w,h,target)
        except (ValueError,KeyError):
            raw=self.client.ask(prompt+'\nUse one bbox_2d with coordinates normalized to 0..1000. Output no explanation.',dest,kind='image',tokens=256)
            return spatial_answer(raw,w,h,target)

    def process(self,source,video_id,meta,segments,ratio):
        cap=cv2.VideoCapture(str(source))
        if hasattr(cv2,'CAP_PROP_ORIENTATION_AUTO'):cap.set(cv2.CAP_PROP_ORIENTATION_AUTO,0)
        if not cap.isOpened():raise ValueError('Cannot decode source video')
        active_id=None;previous_frame=None;previous_crop=None;tracker=FlowTracker();last_anchor=-math.inf
        predictions=[];anchors=[];warnings=[];index=0;current=None
        try:
            while True:
                ok,frame=cap.read()
                if not ok:break
                if index>=len(meta['times']):raise ValueError('Decoder has more frames than PTS inventory')
                if frame.shape[1]!=meta['width'] or frame.shape[0]!=meta['height']:raise ValueError('Raw-frame dimension mismatch')
                t=meta['times'][index]
                choices=[e for e in segments if e['segment_sec'][0]<=t<e['segment_sec'][1]]
                # Keep the current event identity while overlapping candidates remain active.
                event=next((e for e in choices if e['segment_id']==active_id),choices[0] if choices else None)
                if event is None:
                    active_id=None;previous_crop=None;previous_frame=None;tracker=FlowTracker();index+=1;continue
                reset=event['segment_id']!=active_id or cut_score(previous_frame,frame)>=self.cfg['scene_cut_threshold']
                if reset:tracker=FlowTracker();previous_crop=None;last_anchor=-math.inf
                scene=event['focus_mode']=='scene'
                due=reset or t-last_anchor>=self.cfg['anchor_seconds']
                box=None if scene else tracker.update(frame)
                if not scene and (due or box is None):
                    try:box=self.locate(frame,event,index,video_id)
                    except (ValueError,KeyError) as exc:
                        if self.cfg['spatial_failure']=='error':raise
                        # Explicit opt-in fallback, recorded; never silently represented as grounding.
                        box=None;warnings.append({'frame':index,'error':str(exc),'fallback':'wide_center'})
                    tracker.update(frame,anchor=box);last_anchor=t
                    anchors.append({'frame':index,'segment_id':event['segment_id'],'subject_description':event['subject_description'],
                        'bbox_xyxy':box,'coordinate_space':'original_pixels','scene_reset':reset})
                if self.yolo is not None and not scene and box is not None and index%self.cfg['yolo_interval']==0:
                    result=self.yolo.predict(frame,device=self.cfg['yolo_device'],verbose=False)[0]
                    candidates=result.boxes.xyxy.cpu().tolist() if result.boxes is not None else []
                    matched=max(candidates,key=lambda b:iou(b,box),default=None)
                    if matched is not None and iou(matched,box)>=.3:
                        box=matched;tracker.update(frame,anchor=box)
                crop=crop_box(box,meta['width'],meta['height'],ratio,previous_crop,
                    self.cfg['smooth_alpha'],self.cfg['crop_margin'])
                predictions.append({'frame':index,'bboxes':crop})
                previous_crop=crop;previous_frame=frame;active_id=event['segment_id'];index+=1
            if index!=meta['frame_count']:raise ValueError(f'Decode/PTS frame count mismatch: {index} != {meta["frame_count"]}')
        finally:cap.release()
        return predictions,{'anchors':anchors,'warnings':warnings}
