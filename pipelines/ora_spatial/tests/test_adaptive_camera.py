import copy
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from common import read_json,export,load_rows,validate
from adaptive_camera import compose_segment,smooth_crops,subject_crop


class AdaptiveCameraTests(unittest.TestCase):
    def setUp(self):self.cfg=read_json(ROOT/'config/reframe.json')

    def fixture(self,candidates,blur=(),absent=(),yolo=None,routes=None,rejected=None):
        frames=list(range(15));ts=np.arange(15)/10
        obs={f:{'status':2 if f in candidates else 0,'box':candidates.get(f,[0,0,0,0])} for f in frames}
        hist=np.ones(48)/48
        visual={f:(1 if f in blur else 100,hist) for f in frames}
        old={f:[0,0,100] for f in frames}
        return compose_segment(frames,ts,obs,yolo or {},old,[1,1],(100,100),False,
                               routes or {},rejected or [],absent,visual,self.cfg)

    def test_single_missed_frame_is_not_departure(self):
        targets,q,protect,states=self.fixture({0:[30,30,50,60]})
        self.assertEqual(states[1],'uncertain_hold')
        self.assertNotIn('exit_likely',states)
        self.assertEqual(states[-1],'unresolved_baseline')

    def test_blur_and_occlusion_bridges_have_different_evidence(self):
        boxes={0:[30,30,50,60],5:[34,30,54,60]}
        _,_,_,blur=self.fixture(boxes,blur=range(1,5))
        _,_,_,clear=self.fixture(boxes)
        self.assertEqual(blur[2],'blur_bridge')
        self.assertEqual(clear[2],'occlusion_candidate_bridge')

    def test_exit_requires_direction_and_sustained_loss(self):
        _,_,_,states=self.fixture({0:[80,30,95,60],1:[85,30,100,60]})
        self.assertEqual(states[2],'uncertain_hold')
        self.assertEqual(states[6],'exit_likely')
        _,_,_,blurry=self.fixture({0:[80,30,95,60],1:[85,30,100,60]},blur=range(2,15))
        self.assertNotIn('exit_likely',blurry)

    def test_repeated_absence_is_different_from_one_negative(self):
        _,_,_,single=self.fixture({0:[30,30,50,60]},absent=[2])
        _,_,_,repeat=self.fixture({0:[30,30,50,60]},absent=[2,5])
        self.assertNotIn('exit_likely',single)
        self.assertEqual(repeat[6],'exit_likely')

    def test_reliable_baseline_detection_prevents_road_center_fallback(self):
        targets,_,_,states=self.fixture({},yolo={0:np.array([20,0,30,10])},routes={0:'multiple'})
        self.assertEqual(states[0],'visible_yolo')
        np.testing.assert_equal(targets[0],[50,50,100])

    def test_baseline_detector_cannot_resurrect_rejected_overlay_between_reviews(self):
        logo=np.array([90,0,100,10])
        _,_,_,states=self.fixture({},yolo={7:logo},rejected=[
            {'frame':0,'first':0,'last':14,'box':logo,'reason':'graphic_or_text'}])
        self.assertNotEqual(states[7],'visible_yolo')

    def test_scale_not_locked_and_visible_subject_stays_in_crop(self):
        t=np.arange(20)/10;b=np.array([1250,450,1320,520])
        desired=subject_crop(b,(1920,1080),[16,9],True,self.cfg)
        crops,diag=smooth_crops(t,np.tile(desired,(20,1)),np.ones(20),[b]*20,(1920,1080),[16,9],self.cfg)
        self.assertLess(diag['width_max'],1920)
        self.assertGreater(diag['width_min'],b[2]-b[0])
        for x,y,w in crops:
            self.assertTrue(x<=b[0] and x+w>=b[2] and y<=b[1] and y+w*9/16>=b[3])
            self.assertTrue(x>=0 and y>=0 and x+w<=1920 and y+w*9/16<=1080)

    def test_variable_width_export_still_checks_bounds_time_and_zip(self):
        old=[{'video_id':'0','targetRatioWH':[16,9],'predictions':[{'frame':0,'bboxes':[0,0,1920]}]}]
        rows=copy.deepcopy(old);rows[0]['predictions'][0]['bboxes']=[500,200,1000]
        m={'videos':{'0':{'width':1920,'height':1080,'frame_count':1}}}
        with tempfile.TemporaryDirectory() as tmp:
            report=export(Path(tmp)/'adaptive',rows,old,m,preserve_width=False)
            self.assertEqual(report['changed_widths'],1)
            self.assertFalse(report['widths_unchanged'])
            self.assertEqual(load_rows(Path(tmp)/'adaptive.zip'),rows)
        with self.assertRaises(ValueError):validate(rows,old,m)
        rows[0]['predictions'][0]['bboxes']=[1800,0,1000]
        with self.assertRaises(ValueError):validate(rows,old,m,preserve_width=False)


if __name__=='__main__':unittest.main()
