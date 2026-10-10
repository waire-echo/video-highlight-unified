import unittest
from guarded import *

class Guards(unittest.TestCase):
    def setUp(self):
        self.cfg=read_json(Path(__file__).with_name('config.json'))
        self.base=read_json(ROOT/'experiments/scene_general_v1/config.json')
    def test_rejects_large_different_object(self):
        proposals=[{'box':[0,200,1920,500]}]+[{'box':[1500-50*i,800-20*i,1550-50*i,850-20*i]} for i in range(4)]
        image=np.full((540,960,3),180,np.uint8)
        result=consensus(proposals,[image]*5,np.arange(5)*4,(1920,1080),self.cfg)
        self.assertTrue(result['trusted']);self.assertEqual(result['accepted_indices'],[1,2,3,4]);self.assertIn(0,result['rejected_indices'])
    def test_two_disagreeing_targets_are_not_trusted(self):
        image=np.full((540,960,3),180,np.uint8)
        boxes=[[100,100,140,140],[1500,800,1550,850],[100,100,140,140],[1500,800,1550,850]]
        result=consensus([{'box':b} for b in boxes],[image]*4,np.arange(4),(1920,1080),self.cfg)
        self.assertFalse(result['trusted'])
    def test_hard_rejection_of_cut_target(self):
        target=np.array([[1500,820,1560,880]]*5)
        bad={'ratio':[16,9],'width':1440,'boxes':[[430,3,1440]]*5}
        d=evaluate(bad,target,list(range(5)),np.arange(5.),(1920,1080),self.base,self.cfg)
        self.assertFalse(d['eligible']);self.assertEqual(d['minimum_observed_coverage'],0)
    def test_new_target_candidates_include_valid_crop(self):
        boxes=np.array([[730,146,1320,698],[870,315,1160,609],[586,566,1415,768]],float)
        times=np.array([0.,10.,20.]);pool=target_candidates(boxes,times,(1920,1080),[16,9],self.base,self.cfg)
        self.assertTrue(any(evaluate(c,boxes,[0,1,2],times,(1920,1080),self.base,self.cfg)['eligible'] for c in pool))
    def test_disagreement_ignores_letter_preference(self):
        selected,reason=arbitrate([4,5],[2,4,5]);self.assertEqual(selected,2);self.assertTrue(reason.startswith('conflict_rejected'))
        selected,_=arbitrate([4,4],[2,4,5]);self.assertEqual(selected,4)
        selected,_=arbitrate([8,8],[2,4,5]);self.assertEqual(selected,2)

if __name__=='__main__':unittest.main()
