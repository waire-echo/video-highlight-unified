import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from common import read_json
from ora_stage import infer_job,parse_route


class FakeClient:
    def __init__(self,absent=False):self.calls=[];self.absent=absent
    def ask(self,content,prompt,audit,tag):
        self.calls.append((content['type'],tag,prompt))
        if tag=='route':return '{"mode":"single","target":"the visible dog"}'
        if tag=='spatial_0' and self.absent:return '<answer>[]</answer>'
        if tag.startswith('spatial_'):return '<answer>[{"bbox_2d":[50,100,400,900]}]</answer>'
        if tag=='tracking':return '<answer>{"boxes":{"0":[50,100,400,900],"1":[50,100,400,900]}}</answer>'
        if tag.startswith('evidence_'):return '{"kind":"physical_subject","description":"A small white dog"}'
        if tag.startswith('verify_'):return '{"status":"valid","reason":"The dog is visible and the box fits"}'
        raise AssertionError(tag)


class OraProtocol(unittest.TestCase):
    def test_route_rejects_coordinates_but_accepts_explicit_abstention(self):
        self.assertEqual(parse_route('<mode>unknown</mode>'),{'mode':'unknown','target':''})
        with self.assertRaises(ValueError):parse_route('{"mode":"single","target":"655, 508"}')

    def run_fixture(self,absent):
        cfg=read_json(ROOT/'config/default.json');client=FakeClient(absent)
        images={i:np.zeros((100,200,3),np.uint8) for i in range(3)}
        job={'first':0,'last':2,'action_description':'A dog runs.'}
        with tempfile.TemporaryDirectory() as d,patch('ora_stage.read_frames',return_value=images),patch('ora_stage.make_video',side_effect=lambda p,fs,fps:Path(p).write_bytes(b'synthetic')):
            result=infer_job(client,job,{'width':400,'height':200,'fps':2},np.array([0,.5,1]),'unused',cfg,Path(d))
        return client,result

    def test_spatial_and_video_tracking_are_both_called(self):
        client,result=self.run_fixture(False)
        self.assertEqual([(t,k) for t,k,_ in client.calls[:4]],[('video_url','route'),('image_url','spatial_0'),('image_url','spatial_2'),('video_url','tracking')])
        self.assertEqual([k for _,k,_ in client.calls[4:]],['evidence_0','verify_0','evidence_2','verify_2'])
        self.assertEqual(len(result['anchors']),2)
        self.assertTrue(all(a['accepted'] for a in result['anchors']))
        self.assertEqual(result['anchors'][0]['box'],[20,20,160,180])

    def test_first_frame_absence_does_not_disable_later_localization(self):
        client,result=self.run_fixture(True)
        self.assertEqual(result['errors'],[])
        self.assertEqual(result['abstentions'][0]['frame'],0)
        self.assertTrue(result['anchors'][0]['accepted'])
        self.assertEqual(result['anchors'][0]['frame'],2)
        self.assertIn('absent in the first frame',next(prompt for _,tag,prompt in client.calls if tag=='tracking'))


if __name__=='__main__':unittest.main()
