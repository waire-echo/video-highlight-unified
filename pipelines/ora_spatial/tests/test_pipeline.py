import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from common import box_from_center, export, load_rows, validate, groups, read_json
from camera import smooth_camera
from ora_tools import parse_spatial, parse_tracking, convert_box
from jobs import build_jobs
from stage1 import cache_reliability
from sam_stage import mask_measure, contain_center
from shots import boundaries


class CoordinateTests(unittest.TestCase):
    def test_pixels_and_normalized_are_not_interchanged(self):
        text = '<answer>[{"bbox_2d":[100,50,300,200]}]</answer>'
        self.assertEqual(parse_spatial(text, 'pixel', (400,200), (800,400)), [200,100,600,400])
        self.assertEqual(convert_box([100,50,300,200], 'norm1000', (400,200), (800,400)), [80,20,240,80])

    def test_reject_bad_coordinates_and_no_number_scraping(self):
        for box in ([0,0,float('nan'),100], [100,0,0,100], [-1,0,20,30], [0,0,1200,30]):
            with self.assertRaises(ValueError):
                convert_box(box, 'norm1000', (800,400), (800,400))
        with self.assertRaises(ValueError):
            parse_spatial('I think 0 0 100 200', 'pixel', (800,400), (800,400))

    def test_tracking_uses_original_vfr_timestamps(self):
        ts = np.array([0,.04,.1,5,5.2,5.9,6.04,6.3,7.02])
        value = json.dumps({'boxes': {'0':[0,0,500,500], '1':[500,0,1000,500], '2':[0,0,1000,1000], '3':[0,0,1000,1000]}})
        found = parse_tracking(value, ts, 3, 8, (800,400))
        self.assertEqual([x['frame'] for x in found], [3,6,8])

    def test_multiple_union(self):
        text = '[{"bbox_2d":[0,0,30,40]},{"bbox_2d":[60,50,90,90]}]'
        self.assertEqual(parse_spatial(text, 'pixel', (100,100), (100,100), True), [0,0,90,90])


class CameraTests(unittest.TestCase):
    def setUp(self):
        self.cfg = read_json(ROOT/'config/default.json')['camera']

    def test_uncertain_is_center_and_full_width_stays_valid(self):
        path,_ = smooth_camera(np.arange(30)/30, np.tile([20,50],(30,1)), np.zeros(30), (800,400), (200,400), self.cfg)
        np.testing.assert_allclose(path, np.tile([400,200],(30,1)), atol=1e-8)
        self.assertEqual(box_from_center([800,400],800,[2,1],800,400), [0,0,800])

    def test_constant_target_does_not_drift(self):
        path,diag = smooth_camera([0,.1,.2], [[220,200]]*3, [1]*3, (800,400),(200,400),self.cfg)
        np.testing.assert_allclose(path, [[220,200]]*3)
        self.assertEqual(diag['mode'],'fixed')

    def test_pan_speed_bounded_and_one_frame_supported(self):
        t=np.linspace(0,4,121);c=np.column_stack([np.where(t<2,100,700),np.full(len(t),200)])
        path,_=smooth_camera(t,c,np.ones(len(t)),(800,400),(200,400),self.cfg)
        self.assertLessEqual(np.max(np.abs(np.diff(path[:,0])/np.diff(t))),120.00001)
        path,_=smooth_camera([4],[[100,200]],[0],(800,400),(200,400),self.cfg)
        np.testing.assert_allclose(path,[[400,200]])

    def test_absent_evidence_not_failed_detection(self):
        cfg=read_json(ROOT/'config/default.json')['stage1'];ts=np.arange(100)/10
        q=cache_reliability([0,1,99],ts,np.array([[0,0]]),cfg)
        np.testing.assert_equal(q,[0,0,1])


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.rows=[{'video_id':'0','targetRatioWH':[1,2],'predictions':[{'frame':f,'bboxes':[100,0,200]} for f in [2,3,8]]},
                   {'video_id':'1','targetRatioWH':[1,2],'predictions':[]}]
        self.manifest={'videos':{v:{'width':800,'height':400,'frame_count':30} for v in ['0','1']}}

    def test_export_preserves_temporal_width_empty_and_zip_format(self):
        rows=copy.deepcopy(self.rows);rows[0]['predictions'][0]['bboxes'][0]=300
        with tempfile.TemporaryDirectory() as td:
            report=export(Path(td)/'test',rows,self.rows,self.manifest)
            self.assertEqual(load_rows(Path(td)/'test.zip'),rows)
            self.assertEqual(report['empty_video_ids'],['1'])
        for mutation in ['frame','width','empty','bounds']:
            bad=copy.deepcopy(rows)
            if mutation=='frame':bad[0]['predictions'][0]['frame']=1
            if mutation=='width':bad[0]['predictions'][0]['bboxes'][2]=190
            if mutation=='empty':bad[1]['predictions']=copy.deepcopy(bad[0]['predictions'])
            if mutation=='bounds':bad[0]['predictions'][0]['bboxes'][0]=799
            with self.assertRaises(ValueError):validate(bad,self.rows,self.manifest)

    def test_cuts_and_temporal_gaps_never_join(self):
        self.assertEqual(groups([1,2,3,4,8,9],[3]),[[1,2],[3,4],[8,9]])
        row={'predictions':[{'frame':f} for f in list(range(40))+[50,51]]}
        meta={'fps':2,'events':[{'start':0,'end':50,'query':'visible action'}]}
        ts=np.arange(60)/2
        jobs=build_jobs(row,meta,ts,[20],16)
        self.assertEqual([(j['first'],j['last']) for j in jobs],[(0,19),(20,39),(50,51)])
        self.assertTrue(all(ts[j['last']]-ts[j['first']]<16 for j in jobs))
        jobs=build_jobs({'predictions':[{'frame':f} for f in range(1000)]},meta,np.arange(1000)/120,[],16,240)
        self.assertTrue(all(j['last']-j['first']+1<=240 for j in jobs))
        self.assertEqual(sum(j['last']-j['first']+1 for j in jobs),1000)

    def test_mask_original_coordinate_and_containment(self):
        mask=np.zeros((10,20),bool);mask[2:8,5:10]=True
        m=mask_measure(mask,200,100)
        self.assertEqual(m['box'],[50,20,100,80])
        self.assertEqual(contain_center([50,50],m['box'],60,100,(200,100)),[70,50])

    def test_fade_does_not_connect_before_and_after(self):
        self.assertEqual(boundaries(np.array([0,.8,.9,.7,.8,0]),.5),[0,2,5])


if __name__=='__main__':unittest.main()
