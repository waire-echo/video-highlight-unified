"""Reviewable snapshot without environments, secrets, large SAM/ORA weights or temp files."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--with-transnet',action='store_true');a=p.parse_args()
root=Path(__file__).resolve().parents[1];dest=Path(a.output).resolve()
files=[]
for item in root.rglob('*'):
    if not item.is_file():continue
    rel=item.relative_to(root)
    if any(v in rel.parts for v in ['__pycache__','.git','env']):continue
    if rel.suffix in ['.tmp','.download','.pyc']:continue
    if rel.parts[0]=='vendor' and len(rel.parts)>1 and rel.parts[1]=='sam2':continue
    if rel.parts[0]=='models' and not (a.with_transnet and rel.name=='transnetv2-pytorch-weights.pth'):continue
    if rel.parts[0]=='output' and len(rel.parts)>1 and rel.parts[1] in ['ora','sam','logs']:continue
    if item.resolve()==dest:continue
    files.append((item,rel.as_posix()))
manifest={rel:hashlib.sha256(p.read_bytes()).hexdigest() for p,rel in files}
with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED) as z:
    for item,rel in files:z.write(item,rel)
    z.writestr('PACKAGE_SHA256.json',json.dumps(manifest,indent=2))
print(json.dumps({'zip':str(dest),'files':len(files),'bytes':dest.stat().st_size,'sha256':hashlib.sha256(dest.read_bytes()).hexdigest()}))
