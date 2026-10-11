import json,tempfile,unittest,sys
from pathlib import Path
from unittest.mock import Mock,patch
import numpy as np
from engine import span,frame_ids,parse,visible_text,Client,windows,sampling_rate,sha,save
from settings import DEFAULT_CONFIG
from run import validate_core,validate_plan,validate_boundary,Editor,pilot_ids,export_temporal,resolve_reference
import run as runner
import prompts as P

class Contracts(unittest.TestCase):
    def test_pts_not_nominal_fps(self):
        self.assertEqual(frame_ids([0,.03,.09,.15,.24],[[.04,.15]]),[2])
        self.assertEqual(frame_ids([0,.03,.09,.15,.24],[[.03,.15],[.09,.25]]),[1,2,3,4])
    def test_bad_pts(self):
        with self.assertRaises(ValueError):frame_ids([0,.03,.03],[])
    def test_bounds(self):
        for a,b in [(False,1),(0,float('nan')),(-1,1),(2,2),(0,3)]:
            with self.assertRaises(ValueError):span(a,b,2)
    def test_evidence_not_timestamp_list(self):
        with self.assertRaises(ValueError):visible_text('0.0, 1.0, 2.0')
        visible_text('The hand releases the object.')
    def test_no_times_for_uncertain(self):
        x={'decision':'review','start':0,'end':1,'evidence_times':[.5],'observed_event':'Visible subject motion',**{k:'Not clearly visible' for k in ('reason','evidence_before','evidence_core','evidence_after')}}
        with self.assertRaises(ValueError):validate_core(x,2)
    def test_evidence_must_be_inside_core(self):
        x={'decision':'keep','start':1,'end':2,'evidence_times':[.2],'observed_event':'Visible subject motion',**{k:'Visible action change' for k in ('reason','evidence_before','evidence_core','evidence_after')}}
        with self.assertRaises(ValueError):validate_core(x,3)
    def test_global_no_cycle_or_missing(self):
        with self.assertRaises(ValueError):validate_plan({'plans':[{'candidate_id':'c0','possible_duplicate_of':'c1','review_question':'Compare visible event'}]},['c0','c1'])
        with self.assertRaises(ValueError):validate_plan({'plans':[{'candidate_id':'c0','possible_duplicate_of':'c0','review_question':'Compare visible event'}]},['c0'])
    def test_unchanged_boundary_cannot_move(self):
        with self.assertRaises(ValueError):validate_boundary({'decision':'unchanged','time':2,'evidence_time':1,'reason':'Clear visible transition'},4,1)
    def test_parse_rejects_trailing_or_nan(self):
        for s in ['{"x":NaN}','{"x":1} extra']:
            with self.assertRaises(ValueError):parse(s)
    def test_window_full_coverage(self):
        w=list(windows(60,24,20));self.assertEqual(w,[[0.,24.],[20.,44.],[40.,60]])
    def test_local_endpoint_only(self):
        with self.assertRaises(ValueError):Client({'endpoint':'https://remote.example/v1'},'.','sig')
    def test_duplicate_replacement_chain(self):
        self.assertEqual(resolve_reference('a',{'c':[1,2]},[{'candidate_id':'a','decision':'replaced_by_better_visible_instance','replacement_id':'b'},{'candidate_id':'b','decision':'replaced_by_better_visible_instance','replacement_id':'c'}]),'c')
    def test_malformed_response_no_silent_keep(self):
        with tempfile.TemporaryDirectory() as d:
            cfg={'endpoint':'http://127.0.0.1:8010/v1','discovery_fps':2,'max_frames':64,'side':384,'model':'x','seed':0,'timeout_seconds':1}
            c=Client(cfg,d,'sig');response=Mock();response.json.return_value={'choices':[{'finish_reason':'length','message':{'content':'{}'}}]};c.session.post=Mock(return_value=response)
            with self.assertRaises(RuntimeError):c.ask('test','prompt',P.VERIFY_SCHEMA,[],lambda o:None)
            self.assertEqual(len(list((Path(d)/'raw').glob('*.json'))),2)
            self.assertFalse((Path(d)/'responses').exists())

class FakeClient:
    calls=hits=0
    def __init__(self,uncertain=False,duplicate=False,prefer_second=False):self.uncertain=uncertain;self.duplicate=duplicate;self.tags=[];self.prefer_second=prefer_second;self.requests=[]
    def clip(self,source,fingerprint,a,b,fps):return {'start':a,'end':b,'effective_fps':fps,'path':'mock'}
    def ask(self,tag,prompt,schema,media,validator,tokens):
        self.tags.append(tag)
        self.requests.append((tag,prompt))
        if tag=='discovery':
            cs=[{'start':1,'end':3,'event':'An object changes position','evidence':'Visible movement and stop','distinct_change':'Object reaches its destination','uncertainty':'None observed'}]
            if self.duplicate:cs.append({**cs[0],'start':5,'end':7})
            o={'candidates':cs,'overflow':False,'empty_reason':'Candidates visibly present'}
        elif tag=='global_plan':
            o={'plans':[{'candidate_id':'c000','possible_duplicate_of':None,'review_question':'Verify visible core'}]}
            if self.duplicate:o['plans'].append({'candidate_id':'c001','possible_duplicate_of':'c000','review_question':'Check repeated action'})
        elif tag=='blind_observation':o={'subject':'Visible object','observed_event':'An object visibly reaches its destination','uncertainty':'Identity not fully visible','evidence_times':[1.5]}
        elif tag.startswith('core_'):
            o={'decision':'review' if self.uncertain else 'keep','start':None if self.uncertain else 1,'end':None if self.uncertain else 3,'evidence_times':[1.5],
               'observed_event':'Corrected visible object movement','hypothesis_match':'contradicted',**{k:'Visible object motion' for k in ('reason','evidence_before','evidence_core','evidence_after')}}
        elif tag.startswith('boundary_'):
            data=json.loads(prompt.split('\n')[-1]);o={'decision':'unchanged','time':data['proposed_time'],'evidence_time':data['proposed_time'],'reason':'Visible supported boundary'}
        elif tag=='final_verify':o={'decision':'keep','reason':'Distinct visible motion','evidence_times':[2]}
        elif tag=='visual_duplicate':o={'decision':'duplicate','representative':'second' if self.prefer_second else 'first','reason':'Same visible action repeated without a new outcome'}
        else:raise AssertionError(tag)
        validator(o);return o,'mock_'+tag

class Workflow(unittest.TestCase):
    def setUp(self):self.cfg=json.loads(DEFAULT_CONFIG.read_text())
    def test_blind_pass_has_no_candidate_and_corrected_event_is_used(self):
        c=FakeClient();r=Editor(c,self.cfg,lambda **kw:None).process('x',Path('mock'),{'duration':10,'source_sha256':'x'})
        for tag,prompt in c.requests:
            if tag=='blind_observation':self.assertNotIn('An object changes position',prompt)
            if tag.startswith('boundary_') or tag=='final_verify':
                self.assertIn('Corrected visible object movement',prompt);self.assertNotIn('An object changes position',prompt)
        self.assertEqual(r['decisions'][0]['core_review']['hypothesis_match'],'contradicted')
        self.assertEqual(r['decisions'][0]['decision'],'keep')
    def test_dense_recheck_adds_sampling_on_long_clip(self):
        self.assertEqual(sampling_rate(4,64,27),sampling_rate(6,64,27))
        self.assertGreater(sampling_rate(6,self.cfg['max_frames'],27),sampling_rate(4,self.cfg['max_frames'],27))
    def test_uncertain_never_auto_kept(self):
        c=FakeClient(uncertain=True);r=Editor(c,self.cfg,lambda **kw:None).process('x',Path('mock'),{'duration':10,'source_sha256':'x'})
        self.assertEqual(r['segments'],[]);self.assertTrue(r['unresolved']);self.assertIn('core_recheck',c.tags)
    def test_visual_duplicate_verification_required(self):
        c=FakeClient(duplicate=True);r=Editor(c,self.cfg,lambda **kw:None).process('x',Path('mock'),{'duration':10,'source_sha256':'x'})
        self.assertIn('visual_duplicate',c.tags);self.assertEqual(r['decisions'][1]['decision'],'drop_duplicate');self.assertEqual(len(r['segments']),1)
    def test_no_fallback_geometry(self):
        with tempfile.TemporaryDirectory() as d:
            r=export_temporal([{'video_id':'0','segments':[[.02,.08]],'unresolved':[]}],{'videos':{'0':{'targetRatioWH':[16,9]}}},{'0':np.array([0,.03,.09])},d)
            row=json.loads((Path(d)/'temporal_handoff.jsonl').read_text());self.assertEqual(row['highlight_frames'],[1]);self.assertNotIn('predictions',row);self.assertEqual(r['frames'],1)
    def test_better_later_instance_can_replace_first(self):
        c=FakeClient(duplicate=True,prefer_second=True);r=Editor(c,self.cfg,lambda **kw:None).process('x',Path('mock'),{'duration':10,'source_sha256':'x'})
        self.assertEqual(r['decisions'][0]['decision'],'replaced_by_better_visible_instance')
        self.assertEqual(r['decisions'][1]['decision'],'keep');self.assertEqual(len(r['segments']),1);self.assertGreater(r['segments'][0][0],3)

    @unittest.skipUnless(sys.platform=='linux','Runner uses Linux flock on the inference server')
    def test_bad_video_does_not_stop_remaining_ids_or_become_empty_success(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);videos=root/'videos';videos.mkdir();out=root/'output/job'
            for vid in ('1','2'):(videos/(vid+'.mp4')).write_bytes(b'test media bytes')
            (root/'inputs').mkdir();np.savez(root/'inputs/timestamps.npz',**{'1':np.array([0.,.1,.2]),'2':np.array([0.,.1,.2])})
            save(root/'inputs/manifest.json',{'timestamps_sha256':sha(root/'inputs/timestamps.npz'),'videos':{
                vid:{'source_sha256':sha(videos/(vid+'.mp4')),'frame_count':3,'duration':.3,'targetRatioWH':[16,9]}
                for vid in ('1','2')}})
            save(root/'inputs/ora_model_provenance.json',{'model':'mock'})
            save(root/'output/parameter_ledger.json',{'total_upper_bound':1})
            client=Mock(calls=0,hits=0);client.session.get.return_value.json.return_value={'data':[{'id':self.cfg['model']}]}
            editor=Mock();editor.process.side_effect=[RuntimeError('broken first video'),{'segments':[[0,.2]],'unresolved':[],'quality_flags':[]}]
            argv=['run.py','--data-root',str(root),'--video-dir',str(videos),'--output',str(out),'--ids','1,2']
            with patch.object(sys,'argv',argv),patch.object(runner,'Client',return_value=client),patch.object(runner,'Editor',return_value=editor),patch.object(runner,'git_commit',return_value='test'):
                with self.assertRaisesRegex(RuntimeError,'remaining IDs were attempted'):runner.main()
            self.assertEqual(editor.process.call_count,2)
            self.assertFalse((out/'records/1.json').exists());self.assertTrue((out/'errors/1.json').exists())
            rows=[json.loads(line) for line in (out/'temporal_handoff.jsonl').read_text().splitlines()]
            self.assertEqual([row['video_id'] for row in rows],['2'])
            self.assertEqual(json.loads((out/'status.json').read_text())['status'],'failed')

if __name__=='__main__':unittest.main()
