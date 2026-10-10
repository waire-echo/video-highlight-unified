import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from ora_verify import verify_anchor,context_crop
from sam_stage import has_verified_support


class Client:
    def __init__(self,answers):self.answers=iter(answers);self.prompts=[]
    def ask(self,content,prompt,audit,tag):
        self.prompts.append(prompt);return next(self.answers)


class Verification(unittest.TestCase):
    def run_check(self,answers):
        c=Client(answers)
        r=verify_anchor(c,np.zeros((100,200,3),np.uint8),[300,0,400,50],(400,200),
                        'the small white dog',[],0)
        return c,r

    def test_overlay_cannot_become_a_sam_seed(self):
        c,r=self.run_check(['{"kind":"graphic_or_text","description":"An animal watermark"}'])
        self.assertFalse(r['passed'])
        self.assertEqual(len(c.prompts),1)
        self.assertNotIn('the small white dog',c.prompts[0])

    def test_correct_crop_caption_does_not_override_full_frame_rejection(self):
        c,r=self.run_check(['{"kind":"physical_subject","description":"A cat"}',
                            '{"status":"wrong_object","reason":"This is a cat, not the requested dog"}'])
        self.assertFalse(r['passed'])
        self.assertEqual(r['status'],'wrong_object')

    def test_valid_evidence_is_accepted_but_unknown_is_not(self):
        for status,expected in [('valid',True),('uncertain',False),('target_absent',False),('bad_extent',False)]:
            _,r=self.run_check(['{"kind":"physical_subject","description":"A small white dog"}',
                                '{"status":"'+status+'","reason":"visual check"}'])
            self.assertEqual(r['passed'],expected)

    def test_invalid_response_fails_closed(self):
        with self.assertRaises(ValueError):self.run_check(['{"kind":"valid","description":"invalid enum"}'])

    def test_original_to_resized_crop_mapping_keeps_context_inside_image(self):
        image=np.arange(100*200*3,dtype=np.uint8).reshape(100,200,3)
        crop=context_crop(image,[380,0,400,20],(400,200))
        self.assertTrue(np.array_equal(crop,image[0:69,131:200]))

    def test_sam_cannot_bridge_a_rejected_frame(self):
        anchors=np.array([0.,4.]);barriers=np.array([1.,3.])
        self.assertTrue(has_verified_support(.5,anchors,barriers,3))
        self.assertFalse(has_verified_support(1.,anchors,barriers,3))
        self.assertFalse(has_verified_support(2.,anchors,barriers,3))
        self.assertTrue(has_verified_support(3.5,anchors,barriers,3))
        self.assertFalse(has_verified_support(8.,anchors,barriers,3))


if __name__=='__main__':unittest.main()
