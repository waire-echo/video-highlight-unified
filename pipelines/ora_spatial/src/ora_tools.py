"""Explicit native coordinate contracts and original-PTS mapping."""
import base64
import json
import math
import re
import time
from pathlib import Path
import numpy as np
import cv2
import requests


def parse_answer(text):
    match = re.search(r'<answer>\s*(.*?)\s*</answer>', text, re.S | re.I)
    value = match.group(1) if match else text.strip()
    value = re.sub(r'^```(?:json)?\s*|\s*```$', '', value.strip())
    # No eval, no heuristic number scraping from explanations.
    return json.loads(value)


def convert_box(box, system, input_size, original_size):
    if system not in ('pixel','norm1000'):
        raise ValueError('Unknown coordinate system')
    if not isinstance(box, list) or len(box) != 4:
        raise ValueError('Expected xyxy box')
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in box):
        raise ValueError('Nonfinite box')
    sx, sy = (1000, 1000) if system == 'norm1000' else input_size
    x1, y1, x2, y2 = box
    if not (0 <= x1 < x2 <= sx and 0 <= y1 < y2 <= sy):
        raise ValueError(f'Invalid {system} coordinates')
    W, H = original_size
    return [x1/sx*W, y1/sy*H, x2/sx*W, y2/sy*H]


def parse_spatial(text, system, input_size, original_size, multiple=False):
    value = parse_answer(text)
    candidates = value if isinstance(value, list) else [value]
    boxes = [convert_box(v['bbox_2d'], system, input_size, original_size)
             for v in candidates if isinstance(v, dict) and 'bbox_2d' in v]
    if not boxes:
        raise ValueError('No native spatial box')
    if multiple:
        return [min(b[0] for b in boxes), min(b[1] for b in boxes),
                max(b[2] for b in boxes), max(b[3] for b in boxes)]
    return boxes[0]


def nearest_frame(ts, seconds, first, last):
    k = int(np.searchsorted(ts, seconds))
    choices = [min(last,max(first,j)) for j in (k-1,k)]
    return min(choices,key=lambda j:abs(ts[j]-seconds))


def parse_tracking(text, ts, first, last, original_size, first_key_seconds=0, key_base=0):
    value = parse_answer(text)
    if not isinstance(value,dict) or not isinstance(value.get('boxes'),dict):
        raise ValueError('Missing tracking boxes')
    observations = {}
    for key, b in value['boxes'].items():
        if not re.fullmatch(r'0|[1-9]\d*',str(key)) or not key_base <= int(key) <= key_base+31:
            continue
        sec = ts[first] + int(key)-key_base + first_key_seconds
        if sec > ts[last]+1e-6 or sec < ts[first]:
            continue
        try:
            box = convert_box(b,'norm1000',original_size,original_size)
        except ValueError:
            continue
        f = nearest_frame(ts,sec,first,last)
        observations[f] = {'frame':f,'box':box,'kind':'tracking'}
    return [observations[k] for k in sorted(observations)]


def read_frames(video, wanted, max_side=None):
    wanted = set(wanted)
    if not wanted:
        return {}
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f'Cannot decode {video}')
    result = {}
    try:
        # Sequential decode from zero avoids inaccurate keyframe seeks.
        for i in range(max(wanted)+1):
            ok, bgr = cap.read()
            if not ok:
                raise RuntimeError(f'Decode failed at original frame {i}: {video}')
            if i in wanted:
                if max_side and max(bgr.shape[:2]) > max_side:
                    scale = max_side/max(bgr.shape[:2])
                    w,h = max(2,round(bgr.shape[1]*scale/2)*2), max(2,round(bgr.shape[0]*scale/2)*2)
                    bgr = cv2.resize(bgr,(w,h))
                result[i] = bgr
    finally:
        cap.release()
    return result


def image_data(bgr):
    ok, data = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY,92])
    if not ok:
        raise RuntimeError('JPEG encoding failed')
    return 'data:image/jpeg;base64,'+base64.b64encode(data.tobytes()).decode()


def make_video(path, sampled_frames, fps):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    h,w = sampled_frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h))
    if not writer.isOpened():
        raise RuntimeError('MP4 encoder unavailable')
    try:
        for frame in sampled_frames:
            writer.write(frame)
    finally:
        writer.release()


class OraClient:
    def __init__(self, url, model, cfg):
        from urllib.parse import urlparse
        if urlparse(url).hostname not in ('localhost','127.0.0.1','::1'):
            raise ValueError('Use the self-hosted loopback ORA endpoint, not an external model API')
        self.url,self.model,self.cfg = url.rstrip('/'),model,cfg
        self.session = requests.Session()

    def ask(self, content, prompt, audit, tag):
        payload = {'model':self.model,'messages':[{'role':'user','content':[
            content,{'type':'text','text':prompt}]}], 'temperature':0,
            'max_tokens':self.cfg['max_new_tokens'],
            'chat_template_kwargs':{'enable_thinking':False}}
        if tag.startswith('route'):
            # Constrain routing syntax at decode time; short clips can otherwise
            # produce an object followed by repeated prose until the token limit.
            payload['max_tokens']=256
            payload['response_format']={'type':'json_schema','json_schema':{
                'name':'visible_focus','strict':True,'schema':{
                    'type':'object','properties':{
                        'mode':{'type':'string','enum':['single','multiple','scene','unknown']},
                        'target':{'type':'string','maxLength':200}},
                    'required':['mode','target'],'additionalProperties':False}}}
        elif tag.startswith(('evidence_','verify_')):
            evidence=tag.startswith('evidence_')
            key='kind' if evidence else 'status'
            description='description' if evidence else 'reason'
            choices=(['physical_subject','environment','graphic_or_text','unclear'] if evidence else
                     ['valid','target_absent','wrong_object','bad_extent','uncertain'])
            payload['max_tokens']=256
            payload['response_format']={'type':'json_schema','json_schema':{
                'name':'visual_check','strict':True,'schema':{
                    'type':'object','properties':{key:{'type':'string','enum':choices},
                    description:{'type':'string','maxLength':500}},
                    'required':[key,description],'additionalProperties':False}}}
        elif tag.startswith('spatial_'):
            payload['max_tokens']=256
        if content['type']=='video_url':
            payload['mm_processor_kwargs']={'fps':self.cfg['input_fps'],'do_sample_frames':True,
                'size':{'shortest_edge':65536,'longest_edge':self.cfg['video_total_pixels']}}
        else:
            payload['mm_processor_kwargs']={'size':{'shortest_edge':65536,'longest_edge':self.cfg['image_max_pixels']}}
        for attempt in range(self.cfg['request_retries']+1):
            try:
                res = self.session.post(self.url+'/chat/completions',json=payload,timeout=self.cfg['request_timeout'])
                res.raise_for_status()
                body=res.json()
                text = body['choices'][0]['message']['content']
                audit.append({'tag':tag,'prompt':prompt,'response':text,'attempt':attempt,
                              'finish_reason':body['choices'][0].get('finish_reason'),
                              'usage':body.get('usage')})
                return text
            except (requests.RequestException,KeyError,TypeError) as e:
                if attempt==self.cfg['request_retries']:
                    # Infrastructure failure must never silently become hundreds of centered crops.
                    raise RuntimeError(f'ORA service failure at {tag}: {e}') from e
                time.sleep(2**attempt)
