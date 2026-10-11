"""Pinned community conversion, loaded with weights_only; upstream architecture."""
import hashlib
from pathlib import Path
import requests

root=Path(__file__).resolve().parents[1]
target=root/'models/transnetv2-pytorch-weights.pth'
target.parent.mkdir(exist_ok=True)
expected='eed5336d5d6a013c67f5863505a26e7e835053e64a9ce413d6b089ccba07bb53'
if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest()==expected:
    print('TransNet weights already verified');raise SystemExit(0)
for host in ['https://hf-mirror.com','https://huggingface.co']:
    url=host+'/magnusdtd/TransNetV2/resolve/1c0e86a1a6333ff57c4458a44fe4b5b4a0faae2a/transnetv2-pytorch-weights.pth'
    try:
        h=hashlib.sha256();temp=target.with_suffix('.download')
        with requests.get(url,stream=True,timeout=(20,60)) as r:
            r.raise_for_status()
            with temp.open('wb') as f:
                for chunk in r.iter_content(1<<20):
                    f.write(chunk);h.update(chunk)
        if h.hexdigest()!=expected:raise ValueError('TransNet SHA256 mismatch')
        temp.replace(target);print('Downloaded verified TransNet weights',flush=True);break
    except Exception as e:
        print(type(e).__name__,str(e),flush=True)
else:
    raise RuntimeError('No verified TransNet weights downloaded')
