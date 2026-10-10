"""Validate the downloaded sample run and export a clearly labelled three-row preview."""
import json
import sys
from pathlib import Path
import zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from common import read_json,write_json,load_rows,validate,sha

root=Path(__file__).resolve().parents[1]
with zipfile.ZipFile(root/'output/sample_pipeline_artifacts.zip') as z:
    if z.testzip() is not None:raise ValueError('Corrupt downloaded archive')
    for name in z.namelist():
        target=(root/name).resolve()
        if not target.is_relative_to(root/'output'):raise ValueError('Archive path outside output')
    z.extractall(root)
manifest=read_json(root/'inputs/manifest.json')
base=load_rows(root/'inputs/baseline.zip');e1=load_rows(root/'output/E1_camera.jsonl')
path=root/'output/ORA_SAM_sample3_on_E1.jsonl';rows=load_rows(path)
report=validate(rows,base,manifest)
selection={'0','32','269'}
for new,old in zip(rows,e1):
    if str(new['video_id']) not in selection and new!=old:raise ValueError('Out-of-sample row changed')
subset=[r for r in rows if str(r['video_id']) in selection]
small=root/'output/ORA_SAM_only3_preview.jsonl'
small.write_text(''.join(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n' for r in subset),encoding='utf-8')
status=read_json(root/'output/release_status.json')
status.update(ora_sam_gpu_inference_tested=True,sam_gpu_inference_tested=True,
              sam_gpu_video_ids=sorted(selection),e2_e3_generated=False,
              sample_pipeline_jsonl=str(path),sample_pipeline_sha256=sha(path))
write_json(root/'output/release_status.json',status)
summaries=[read_json(p) for vid in selection for p in (root/'output/sam'/vid).glob('*.json')]
stats={v:{key:sum(d[key] for d in summaries if str(d['video_id'])==v)
          for key in ['sam_frames','saliency_frames','center_frames']} for v in sorted(selection)}
changes=validate(rows,e1,manifest)
write_json(root/'output/sample_delivery_summary.json',{
    'full_file':path.name,'subset_file':small.name,'inferred_ids':sorted(selection),
    'other_rows':'Exactly E1; not all-video ORA/SAM output',
    'validation':report,'changes_vs_e1':changes,'tracking_stats':stats})
print(json.dumps({'valid_rows':len(rows),'frames':report['frames'],
                  'changed_vs_e1':{v:changes['changes_by_video'][v] for v in selection},
                  'tracking_stats':stats},indent=2))
