import unittest
import numpy as np
import cv2
from temporal import *


class TemporalRegression(unittest.TestCase):
    def setUp(self):
        self.cfg=read_json(Path(__file__).with_name('config.json'))
        self.size=(1920,1080);self.ratio=[16,9]

    def test_no_fixed_letter_in_either_even_or_odd_candidate_counts(self):
        for n in range(2,8):
            a,b=orders_without_fixed_positions(range(n),18)
            self.assertEqual(set(a),set(b));self.assertTrue(all(x!=y for x,y in zip(a,b)))

    def test_progressive_visibility_does_not_require_all_pairs_area_agreement(self):
        observations=[{'box':[900,800,906,803],'confirmed':False},
                      {'box':[900,793,916,803],'confirmed':True,'confirms_seed':0},
                      {'box':[900,777,929,806],'confirmed':True}]
        targets=trusted_sequence(observations)
        self.assertTrue(all(b is not None for b in targets))
        self.assertGreater(np.prod(np.array(targets[-1][2:])-targets[-1][:2])/18,6)

    def test_exit_never_links_new_object_or_invalidates_old_track(self):
        observations=[{'box':[100,100,150,150],'state':'unconfirmed','confirmed':False},
                      {'box':[200,100,250,150],'state':'visible','confirmed':True,'confirms_seed':0},
                      {'box':[400,100,450,150],'state':'visible','confirmed':True},
                      {'box':None,'state':'out_of_view','confirmed':False},
                      {'box':None,'state':'uncertain','confirmed':False}]
        raw=trusted_sequence(observations)
        motion=[{'valid':True,'matrix':[[1,0,0],[0,1,0]]}]*5
        filled,kinds=fill_short_gaps(raw,observations,np.arange(5),motion,self.size,self.cfg)
        self.assertTrue(all(x is not None for x in filled[:3]));self.assertEqual(filled[3:],[None,None])

    def test_explicit_exit_cannot_be_downgraded_to_hold_after_failed_repair(self):
        obs={'verdict':'ABSENT','confirmed':False,'state':'uncertain','box':None}
        self.assertEqual(resolve_exit(obs)['state'],'out_of_view')
        corrected={'verdict':'ABSENT','confirmed':True,'state':'visible','box':[10,10,20,20]}
        self.assertEqual(resolve_exit(corrected)['state'],'visible')

    def test_short_occlusion_uses_camera_motion_but_expires(self):
        obs=[{'state':'visible'}]+[{'state':'occluded'}]*4
        transforms=[{'valid':True,'matrix':[[1,0,10],[0,1,5]]}]*5
        out,kinds=fill_short_gaps([[100,100,200,200],None,None,None,None],obs,np.arange(5),transforms,self.size,self.cfg)
        self.assertEqual(out[1],[110.,105.,210.,205.]);self.assertIsNotNone(out[2]);self.assertIsNone(out[3])

    def test_camera_compensation_detects_translation(self):
        rng=np.random.default_rng(0);im=(rng.random((360,640,3))*255).astype('uint8')
        shifted=cv2.warpAffine(im,np.float32([[1,0,25],[0,1,-12]]),(640,360))
        found=camera_transform(im,shifted,(640,360));self.assertTrue(found['valid'])
        np.testing.assert_allclose(np.array(found['matrix'])[:,2],[25,-12],atol=1)

    def test_visible_target_at_top_right_survives_smoothing(self):
        targets=[[300,500,500,700],[900,200,1100,400],[1600,0,1850,200]]
        follow=focus_candidates(targets,np.arange(3),self.size,self.ratio,self.cfg)[0]
        boxes=smooth_protected(follow['boxes'],targets,self.size,self.ratio,self.cfg)
        self.assertTrue(all(coverage(b,t,self.ratio)>=.97 for b,t in zip(boxes,targets)))

    def test_unreliable_interval_is_local(self):
        targets=[[500,500,700,700],None,[800,400,1000,600]]
        candidate=focus_candidates(targets,np.arange(3),self.size,self.ratio,self.cfg)[0]
        self.assertEqual(candidate['boxes'][1][2],1920)
        self.assertLess(candidate['boxes'][0][2],1920);self.assertLess(candidate['boxes'][2][2],1920)
        self.assertTrue(evaluate(candidate,targets,np.arange(3),self.size,self.ratio,self.cfg)['eligible'])

    def test_variable_width_interpolation_retains_bounds_and_samples(self):
        ts=np.arange(101)/10
        anchors=[[0,0,960],[0,0,1920],[960,540,960]]
        boxes=interpolate_variable([0,50,100],anchors,list(range(101)),ts,self.size,self.ratio)
        self.assertEqual(boxes[0],anchors[0]);self.assertEqual(boxes[50],anchors[1]);self.assertEqual(boxes[-1],anchors[-1])
        for x,y,w in boxes:self.assertTrue(0<=x<=1920-w and 0<=y<=1080-w*9/16)

    def test_empty_and_one_frame_cases(self):
        self.assertEqual(sample_frames([12],np.arange(20)/30,1),[12])
        pool=focus_candidates([None],[0],self.size,self.ratio,self.cfg)
        self.assertEqual(pool[0]['boxes'],[[0,0,1920]])


if __name__=='__main__':unittest.main()
