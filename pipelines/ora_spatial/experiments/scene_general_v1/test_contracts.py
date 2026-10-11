import unittest
from pipeline import *

class Contracts(unittest.TestCase):
    def setUp(self):self.cfg=read_json(Path(__file__).with_name('config.json'))
    def test_bounds_and_speed(self):
        times=np.array([0.,.1,.2,1.,3.]);size=(1920,1080);ratio=[16,9]
        boxes,_=crop_path([[0,0],[1920,1080],[0,1080],[1920,0],[960,540]],960,times,size,ratio,self.cfg)
        for x,y,w in boxes:self.assertTrue(x>=0 and y>=0 and x+w<=1920 and y+w*9/16<=1080)
        centers=np.array([[x+w/2,y+w*9/32] for x,y,w in boxes]);distance=np.linalg.norm(np.diff(centers,axis=0),axis=1)
        self.assertTrue(np.all(distance<=np.diff(times)*960*.18+2))
    def test_interpolation_original_frames(self):
        ts=np.arange(100)/30;fs=[10,11,45,99];boxes=interpolate_crops([10,99],[[0,0,960],[500,200,960]],fs,ts,(1920,1080),[16,9])
        self.assertEqual(boxes[0],[0,0,960]);self.assertEqual(boxes[-1],[500,200,960]);self.assertEqual(len(boxes),len(fs))
    def test_map_focus_and_identical_inputs(self):
        mp=np.zeros((90,160),np.float32);mp[30:45,115:130]=1;mp=normalize_map(mp)
        points=map_centers([mp]*5,960,(1920,1080),[16,9],self.cfg)
        self.assertTrue(np.all(points[:,0]>960))
        im=np.zeros((180,320,3),np.uint8);cv2.circle(im,(240,80),15,(255,255,255),-1)
        args=([im]*5,[None]*5,[],np.arange(5.),(1920,1080),[16,9],self.cfg)
        a,_=candidates(*args);b,_=candidates(*args);self.assertEqual(a,b)
        self.assertEqual(a[0]['width'],1920)
    def test_no_hardcoded_targets(self):
        # A checked config invariant: no manual per-video routing or named scene subjects.
        text=json.dumps(self.cfg).lower()
        for prohibited in ['volcano','eruption','sun disc','sunlit','treeline','257','276','291']:
            self.assertNotIn(prohibited,text)
        self.assertNotIn('refined_anchors',Path(__file__).with_name('pipeline.py').read_text())

if __name__=='__main__':unittest.main()
