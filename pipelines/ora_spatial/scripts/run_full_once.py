"""One auditable background run with a durable completion marker and result bundle."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from common import read_json,write_json,sha,load_rows,validate

root=Path(__file__).resolve().parents[1];os.chdir(root)
run_id=sys.argv[1]
if not run_id.replace('_','').isalnum():raise ValueError('Invalid run name')
status_path=root/'output'/f'{run_id}.status.json'
status={'run_id':run_id,'pid':os.getpid(),'status':'running',
        'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'scope':'all','baseline':'User-selected YOLO-only; not R1','videos':426,
        'source_sha256':{p.as_posix():sha(p) for folder in ['src','scripts','config']
                         for p in Path(folder).glob('*') if p.is_file()},
        'baseline_sha256':sha('inputs/baseline.zip')}
write_json(status_path,status)
try:
    # Existing run_gpu holds output/run.lock, and resumes only matching signed caches.
    subprocess.run(['bash','scripts/run_gpu.sh','all'],check=True)
    manifest=read_json('inputs/manifest.json');baseline=load_rows('inputs/baseline.zip')
    artifacts=[];summaries={}
    for name in ['E2_partial','E3_all']:
        stem=Path('output')/name;report=read_json(stem.with_suffix('.report.json'))
        if report['composition']!='adaptive_center_scale_visibility':raise ValueError('Wrong composition mode')
        rows=load_rows(stem.with_suffix('.jsonl'))
        summaries[name]=validate(rows,baseline,manifest,preserve_width=False)
        if load_rows(stem.with_suffix('.zip'))!=rows:raise ValueError('ZIP/JSONL mismatch')
        artifacts.extend(stem.with_suffix(ext) for ext in ['.jsonl','.zip','.report.json','.validation.json'])
    # Confirm no source/config mutation took place during the long run.
    if any(sha(p)!=v for p,v in status['source_sha256'].items()):raise ValueError('Code changed during run')
    artifacts.append(Path('output/parameter_ledger.json'))
    bundle=Path('output')/(run_id+'_results.zip')
    with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED) as z:
        for p in artifacts:z.write(p,p.name)
    with zipfile.ZipFile(bundle) as z:
        if z.testzip() is not None:raise ValueError('Result archive CRC failed')
    status.update(status='complete',finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  results={p.as_posix():sha(p) for p in artifacts},bundle=bundle.as_posix(),
                  bundle_sha256=sha(bundle),validation=summaries,official_score=None)
    print('FULL_RUN_COMPLETE',run_id,flush=True)
except BaseException as e:
    status.update(status='failed',finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  error=type(e).__name__+': '+str(e))
    raise
finally:
    write_json(status_path,status)
