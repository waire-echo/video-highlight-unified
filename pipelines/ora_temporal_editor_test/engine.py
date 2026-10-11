"""Audited self-hosted ORA calls and deterministic geometry-free temporal utilities."""
import base64,hashlib,json,math,os,re,subprocess,time
from pathlib import Path
from urllib.parse import urlparse
import numpy as np
import requests
import jsonschema

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    os.replace(tmp,path)

def span(a,b,duration):
    if type(a) not in (int,float) or type(b) not in (int,float) or not all(map(math.isfinite,[a,b])):
        raise ValueError('Non-numeric or nonfinite interval')
    if not 0<=a<b<=duration+1e-6:raise ValueError(f'Interval outside viewed clip: {a}, {b}, duration={duration}')
    return [float(a),min(float(b),duration)]

def windows(duration,size,stride):
    if not 0<stride<=size or duration<=0:raise ValueError('Invalid window settings')
    a=0.
    while True:
        b=min(duration,a+size);yield [a,b]
        if b>=duration:break
        a+=stride

def union(spans):
    result=[]
    for a,b in sorted(spans):
        if result and a<=result[-1][1]+1e-9:result[-1][1]=max(result[-1][1],b)
        else:result.append([a,b])
    return result

def frame_ids(timestamps,spans):
    ts=np.asarray(timestamps)
    if not np.isfinite(ts).all() or np.any(np.diff(ts)<=0):raise ValueError('Invalid original PTS')
    mask=np.zeros(len(ts),bool)
    for a,b in spans:
        if not math.isfinite(a+b) or not 0<=a<b:raise ValueError('Invalid export interval')
        mask[np.searchsorted(ts,a,side='left'):np.searchsorted(ts,b,side='left')]=True
    return np.flatnonzero(mask).tolist()

def visible_text(s):
    if not isinstance(s,str) or sum(c.isalpha() for c in s)<4:
        raise ValueError(f'Descriptive evidence must use words, not numeric placeholders or timestamps; received {s!r}')

def check_times(values,duration):
    for v in values:
        if type(v) not in (int,float) or not math.isfinite(v) or not 0<=v<=duration+1e-6:
            raise ValueError('Evidence timestamp outside viewed clip')

def sampling_rate(fps,frame_budget,duration):
    if fps<=0 or frame_budget<3 or duration<=0:raise ValueError('Invalid sampling budget')
    return min(fps,(frame_budget-2)/duration)

def parse(raw):
    s=raw.strip();m=re.fullmatch(r'<answer>\s*(.*?)\s*</answer>',s,re.S)
    if m:s=m.group(1)
    if s.startswith('```'):
        m=re.fullmatch(r'```(?:json)?\s*(.*?)\s*```',s,re.S)
        if not m:raise ValueError('Malformed JSON fence')
        s=m.group(1)
    return json.loads(s,parse_constant=lambda x:(_ for _ in ()).throw(ValueError('Nonfinite JSON')))

class Client:
    def __init__(self,cfg,output,signature):
        if urlparse(cfg['endpoint']).hostname not in ('127.0.0.1','localhost','::1'):
            raise ValueError('Only local self-hosted model endpoint allowed')
        self.cfg,self.root,self.signature=cfg,Path(output),signature
        self.calls=self.hits=0
        self.video_id=None
        self.session=requests.Session()

    def clip(self,source,source_hash,a,b,fps):
        if not a<b:raise ValueError('Empty clip')
        effective=sampling_rate(fps,self.cfg['max_frames'],b-a)
        key=digest([source_hash,a,b,effective,self.cfg['side']])
        dest=self.root/'clips'/(key+'.mp4');meta=dest.with_suffix('.json')
        if dest.exists() and meta.exists():
            record=read(meta)
            if sha(dest)==record['sha256']:return record
        dest.parent.mkdir(parents=True,exist_ok=True)
        tmp=dest.with_name(dest.stem+'.tmp.mp4');side=self.cfg['side']
        vf=f'fps={effective:.9f},scale={side}:{side}:force_original_aspect_ratio=decrease:force_divisible_by=2'
        # Decode before seeking: exact clip-relative timestamps rather than keyframe seek.
        cmd=['ffmpeg','-v','error','-y','-threads','2','-filter_threads','1','-i',str(source),'-ss',f'{a:.9f}','-t',f'{b-a:.9f}',
             '-map','0:v:0','-an','-vf',vf,'-c:v','libx264','-preset','ultrafast','-crf','20',str(tmp)]
        # This container stores BLAS/LAPACK in subdirectories absent from ld.so's cache.
        # Scope the library path fix to the FFmpeg subprocess, not Python/CUDA or system files.
        env=dict(os.environ)
        media_libs=[s for s in ('/usr/lib/x86_64-linux-gnu/blas','/usr/lib/x86_64-linux-gnu/lapack') if Path(s).is_dir()]
        if media_libs:env['LD_LIBRARY_PATH']=':'.join(media_libs+[env.get('LD_LIBRARY_PATH','')])
        result=subprocess.run(cmd,capture_output=True,timeout=240,env=env)
        compatibility=None
        error=result.stderr.decode('utf-8',errors='replace')
        if result.returncode and 'Unsupported input' in error and 'trc:log316' in error:
            compatibility={'original_stderr':error,'action':'Treat unsupported transfer metadata as unknown for scaling only; original file unchanged'}
            cmd[cmd.index('-vf')+1]='setparams=color_trc=unknown,'+vf
            result=subprocess.run(cmd,capture_output=True,timeout=240,env=env)
            save(self.root/'media_compatibility'/(key+'.json'),{**compatibility,'retry_returncode':result.returncode})
        if result.returncode:raise RuntimeError('FFmpeg failed: '+result.stderr.decode('utf-8',errors='replace')[-1600:])
        import cv2
        cap=cv2.VideoCapture(str(tmp));n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));rate=cap.get(cv2.CAP_PROP_FPS);cap.release()
        if n<1 or n>self.cfg['max_frames'] or not rate>0:raise RuntimeError('Decoded clip outside frame budget')
        os.replace(tmp,dest)
        record={'path':str(dest.resolve()),'sha256':sha(dest),'start':a,'end':b,
                'source_path':str(Path(source).resolve()),'source_sha256':source_hash,'side':side,
                'requested_fps':fps,'encoded_fps':rate,'effective_fps':effective,'frames':n,'compatibility':compatibility}
        save(meta,record);return record

    def ask(self,tag,prompt,schema,media,validator,tokens=1800):
        # Use the slowest encoded rate when two evidence clips share one request.
        fps=min((m['effective_fps'] for m in media),default=self.cfg['discovery_fps'])
        frame_cap=self.cfg['max_frames'] if len(media)<2 else min(32,self.cfg['max_frames'])
        content=[{'type':'video_url','video_url':{'url':Path(m['path']).as_uri()}} for m in media]
        media_note=[{k:m[k] for k in ('start','end','requested_fps','effective_fps','frames')} for m in media]
        prompt+='\nCLIP METADATA (seconds in original video; answers use local seconds): '+json.dumps(media_note)
        payload={'model':self.cfg['model'],'messages':[{'role':'user','content':content+[{'type':'text','text':prompt}]}],
                 'temperature':0,'seed':self.cfg['seed'],'max_tokens':tokens,
                 'chat_template_kwargs':{'enable_thinking':False},
                 'structured_outputs':{'json':schema,'disable_any_whitespace':True}}
        if media:payload['mm_processor_kwargs']={'fps':fps,'max_frames':frame_cap,'do_sample_frames':True,
                 'size':{'shortest_edge':65536,'longest_edge':self.cfg['side']**2*frame_cap}}
        key=digest([self.signature,payload,[m['sha256'] for m in media]])
        cache=self.root/'responses'/(key+'.json')
        if cache.exists():
            saved=read(cache);obj=saved['parsed'];jsonschema.validate(obj,schema);validator(obj)
            self.hits+=1;return obj,key
        last_error=None
        for attempt in range(2):
            request=json.loads(json.dumps(payload))
            if attempt:request['messages'][0]['content'][-1]['text']+='\nPrevious response violated the output contract: '+str(last_error)+'\nUse real descriptive words in event/evidence/distinct_change/uncertainty/reason fields, never numeric placeholders. Return concise valid JSON with supported local times.'
            response=self.session.post(self.cfg['endpoint']+'/chat/completions',json=request,timeout=self.cfg['timeout_seconds'])
            response.raise_for_status();body=response.json();self.calls+=1
            entry={'video_id':self.video_id,'tag':tag,'request':request,'media':media,'response':body,'attempt':attempt,
                   'processor_fps':fps,'processor_max_frames':frame_cap}
            rawfile=self.root/'raw'/(key+f'.{attempt}.json')
            try:
                choice=body['choices'][0]
                if choice.get('finish_reason')!='stop':raise ValueError('Generation did not finish normally')
                obj=parse(choice['message']['content']);jsonschema.validate(obj,schema);validator(obj)
            except (ValueError,KeyError,TypeError,jsonschema.ValidationError) as error:
                last_error=str(error)
                entry['validation_error']=str(error);save(rawfile,entry)
                if attempt:raise RuntimeError(f'{tag}: invalid model answer; see {rawfile}') from error
            else:
                save(rawfile,entry);save(cache,{'parsed':obj,'raw':str(rawfile),'tag':tag});return obj,key
        raise AssertionError('Unreachable')
