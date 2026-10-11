"""Synthetic integration only; no real-video score or accuracy assertion."""
import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from common import read_json,write_json,export,load_rows,sha,digest


class ExportIntegration(unittest.TestCase):
    def test_e1_partial_all_preserve_masks_and_partial_scope(self):
        cfg=read_json(ROOT/'config/default.json')
        rows=[{'video_id':vid,'targetRatioWH':[1,2],
               'predictions':[{'frame':f,'bboxes':[20,0,100]} for f in [0,1,2]]}
              for vid in ['0','1']]
        manifest={'videos':{vid:{'width':400,'height':200,'fps':10,'frame_count':3,
                    'source_sha256':'synthetic','subject_hint':hint,'events':[]}
                   for vid,hint in [('0','object'),('1','person')]}}
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'inputs').mkdir();(root/'config').mkdir()
            export(root/'inputs/baseline',rows,rows,manifest)
            np.savez_compressed(root/'inputs/timestamps.npz',**{v:np.array([0,.1,.2]) for v in ['0','1']})
            np.savez_compressed(root/'inputs/yolo_evidence.npz',**{'0':np.array([[0,0],[1,0],[2,0]])})
            for key,name in [('baseline_sha256','baseline.zip'),('timestamps_sha256','timestamps.npz'),('yolo_evidence_sha256','yolo_evidence.npz')]:
                manifest[key]=sha(root/'inputs'/name)
            write_json(root/'inputs/manifest.json',manifest);write_json(root/'config/default.json',cfg)
            for vid in ['0','1']:
                write_json(root/f'output/shots/{vid}.json',{'cuts':[0],'source_sha256':'synthetic','signature':'synthetic'})
            def run(script,*args,success=True):
                if script=='export_spatial.py':args=(*args,'--fixed-width')
                result=subprocess.run([sys.executable,str(ROOT/'src'/script),*args],cwd=root,capture_output=True,text=True)
                if success:self.assertEqual(result.returncode,0,result.stderr)
                else:self.assertNotEqual(result.returncode,0)
            run('stage1.py')
            e1=load_rows(root/'output/E1_camera.zip')
            self.assertEqual(e1[0]['predictions'][0]['bboxes'],[150,0,100])
            self.assertEqual(e1[1]['predictions'][0]['bboxes'],[20,0,100])
            # A vague/scene-like subject label must not override confident detections.
            np.savez_compressed(root/'inputs/yolo_evidence.npz',**{'0':np.array([[0,.9],[1,.9],[2,.9]])})
            manifest['yolo_evidence_sha256']=sha(root/'inputs/yolo_evidence.npz')
            write_json(root/'inputs/manifest.json',manifest)
            run('stage1.py')
            e1=load_rows(root/'output/E1_camera.zip')
            self.assertEqual(e1[0]['predictions'][0]['bboxes'],[20,0,100])
            for vid in ['0','1']:
                jobid='00000000_00000002';op=root/f'output/ora/{vid}/{jobid}.json'
                write_json(op,{'fixture':'synthetic'})
                stem=root/f'output/sam/{vid}/{jobid}';stem.parent.mkdir(parents=True)
                np.savez_compressed(stem.with_suffix('.npz'),frames=np.arange(3),centers=np.array([[320,100]]*3),quality=np.ones(3))
                write_json(stem.with_suffix('.json'),{'source_sha256':'synthetic','config_digest':digest(cfg),
                           'ora_sha256':sha(op),'npz_sha256':sha(stem.with_suffix('.npz'))})
            run('export_spatial.py','--scope','partial','--output','output/E2_partial')
            e2=load_rows(root/'output/E2_partial.zip')
            self.assertEqual(e2[1],e1[1]);self.assertNotEqual(e2[0],e1[0])
            run('export_spatial.py','--scope','all','--output','output/E3_all')
            e3=load_rows(root/'output/E3_all.zip')
            self.assertNotEqual(e3[1],e1[1]);self.assertEqual(e3[0],e2[0])
            run('export_spatial.py','--scope','all','--ids','1','--output','output/sample')
            sample=load_rows(root/'output/sample.zip')
            self.assertEqual(sample[0],e1[0]);self.assertEqual(sample[1],e3[1])
            for new,old in zip(e3,rows):
                self.assertEqual([p['frame'] for p in new['predictions']],[p['frame'] for p in old['predictions']])
                self.assertEqual([p['bboxes'][2] for p in new['predictions']],[100]*3)
            # Corrupted/incomplete inference must fail rather than silently keep old boxes.
            (root/'output/sam/1/00000000_00000002.npz').write_bytes(b'broken')
            run('export_spatial.py','--scope','all','--output','output/bad',success=False)


if __name__=='__main__':unittest.main()
