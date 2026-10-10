"""No-GPU deployment verification; all video bytes checked against source manifest."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from common import load_rows,read_json,sha,validate,write_json

p=argparse.ArgumentParser();p.add_argument('--video-dir',required=True);a=p.parse_args()
manifest=read_json('inputs/manifest.json')
for key,file in [('baseline_sha256','baseline.zip'),('timestamps_sha256','timestamps.npz'),('yolo_evidence_sha256','yolo_evidence.npz')]:
    if sha(Path('inputs')/file)!=manifest[key]:raise ValueError('Changed input: '+file)
base=load_rows('inputs/baseline.zip');validation=validate(base,base,manifest)
for vid,meta in manifest['videos'].items():
    if sha(Path(a.video_dir)/(vid+'.mp4'))!=meta['source_sha256']:raise ValueError('Wrong video: '+vid)
subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-v'],check=True)
import torch,cv2,scipy,requests
torch.set_num_threads(1)
from sam2.build_sam import build_sam2_video_predictor
from hydra import initialize_config_module,compose
# Load real checkpoint on CPU, checking package/config/architecture compatibility.
predictor=build_sam2_video_predictor('configs/sam2.1/sam2.1_hiera_l.yaml','models/sam2.1_hiera_large.pt',device='cpu')
actual=sum(p.numel() for p in predictor.parameters())
ledger=read_json('output/parameter_ledger.json')
if actual>ledger['sam_tensor_elements'] or not ledger['pass']:raise ValueError('Parameter gate failed')
report={'video_sha256_verified':len(manifest['videos']),'validation':validation,
        'sam_cpu_load_ok':True,'sam_runtime_parameters':actual,'torch':torch.__version__,
        'cuda_available':torch.cuda.is_available(),'gpu_inference_tested':False,
        'total_parameter_upper_bound':ledger['total_upper_bound']}
write_json('output/preflight.json',report)
print(json.dumps({k:v for k,v in report.items() if k!='validation'},indent=2))
