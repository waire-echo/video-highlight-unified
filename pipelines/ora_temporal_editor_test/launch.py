"""Launch the isolated front-half pilot and its own self-hosted ORA service."""
import argparse,json,os,signal,socket,subprocess,time
from pathlib import Path
from urllib.parse import urlparse
import requests
import torch
from engine import sha,read,save
from settings import data_arguments,git_commit


def main():
    p=argparse.ArgumentParser(description=__doc__);data_arguments(p)
    p.add_argument('--runtime-root',type=Path,required=True,help='Existing ORA runtime with env/bin/vllm and python311/lib')
    p.add_argument('--model-path',type=Path,required=True)
    p.add_argument('--ids');p.add_argument('--all',action='store_true')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--check-only',action='store_true')
    a=p.parse_args()
    if a.ids and a.all:p.error('Choose ids or all, not both')
    here=Path(__file__).resolve().parent;root=a.data_root.resolve();cfg=read(a.config)
    if urlparse(cfg['endpoint']).hostname not in ('127.0.0.1','localhost','::1'):raise ValueError('Only local endpoint allowed')
    provenance=read(root/'inputs/ora_model_provenance.json')
    for f in provenance['weight_files']:
        if sha(a.model_path/f['filename'])!=f['sha256']:raise RuntimeError('Checkpoint fingerprint mismatch: '+f['filename'])
    common=['--data-root',str(root),'--video-dir',str(a.video_dir.resolve()),'--config',str(a.config.resolve())]
    python=str(root/'env/bin/python')
    cmd=[python,str(here/'run.py'),'--output',str(a.output.resolve())]+common
    if a.all:cmd+=['--all']
    if a.ids:cmd+=['--ids',a.ids]
    subprocess.run(cmd+['--preflight'],check=True,cwd=root)
    subprocess.run([python,'-m','unittest','discover','-s',str(here),'-p','test_editor.py','-v'],check=True,cwd=root)
    if a.check_only:
        print(json.dumps({'cpu_checks':'passed','git_commit':git_commit(),'cuda_available':torch.cuda.is_available(),'gpu_inference_tested':False}));return
    if not torch.cuda.is_available():raise RuntimeError('No GPU attached. CPU checks passed; no inference process started.')
    a.output.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (a.output/'launch.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        statuspath=a.output/'status.json'
        if statuspath.exists():
            pid=read(statuspath).get('pid');proc=Path('/proc')/str(pid)/'cmdline'
            if proc.exists() and str(here/'run.py').encode() in proc.read_bytes():raise RuntimeError('This job is already running')
        port=urlparse(cfg['endpoint']).port or 80
        with socket.socket() as s:
            if s.connect_ex(('127.0.0.1',port))==0:raise RuntimeError(f'Port {port} occupied; inspect before reuse')
        service=[str(a.runtime_root/'env/bin/vllm'),'serve',str(a.model_path.resolve()),
                 '--served-model-name',cfg['model'],'--host','127.0.0.1','--port',str(port),
                 '--dtype','bfloat16','--tensor-parallel-size','1','--max-model-len','16384',
                 '--max-num-seqs','1','--max-num-batched-tokens','8192','--enforce-eager',
                 '--gpu-memory-utilization','0.85','--mm-processor-cache-gb','0',
                 '--structured-outputs-config',json.dumps({'backend':'xgrammar','disable_any_whitespace':True}),
                 '--allowed-local-media-path',str(a.output.resolve()),
                 '--media-io-kwargs',json.dumps({'video':{'num_frames':-1}}),
                 '--limit-mm-per-prompt',json.dumps({'image':0,'video':2}),
                 '--mm-processor-kwargs',json.dumps({'max_frames':cfg['max_frames'],
                       'size':{'shortest_edge':65536,'longest_edge':cfg['side']**2*cfg['max_frames']}})]
        env=dict(os.environ);env['LD_LIBRARY_PATH']=str(a.runtime_root/'python311/lib')+':'+env.get('LD_LIBRARY_PATH','')
        env.update(VLLM_NO_USAGE_STATS='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',VLLM_VIDEO_LOADER_BACKEND='opencv')
        with (a.output/'server.log').open('ab') as logfile:
            server=subprocess.Popen(service,cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=logfile,
                                    stderr=subprocess.STDOUT,start_new_session=True)
            save(a.output/'server_receipt.json',{'pid':server.pid,'git_commit':git_commit(),'model_provenance':provenance,'command':service})
            try:
                for _ in range(180):
                    if server.poll() is not None:raise RuntimeError('ORA service exited; inspect server.log')
                    try:
                        r=requests.get(cfg['endpoint']+'/models',timeout=2)
                        if r.ok and cfg['model'] in [m['id'] for m in r.json()['data']]:break
                    except requests.RequestException:pass
                    time.sleep(2)
                else:raise RuntimeError('ORA service readiness timeout')
                subprocess.run([python,str(here/'protocol_smoke.py'),'--output',str(a.output.resolve()/'protocol')]+common,check=True,cwd=root)
                subprocess.run(cmd,check=True,cwd=root)
            finally:
                if server.poll() is None:
                    os.killpg(server.pid,signal.SIGTERM)
                    try:server.wait(timeout=30)
                    except subprocess.TimeoutExpired:os.killpg(server.pid,signal.SIGKILL);server.wait()


if __name__=='__main__':main()
