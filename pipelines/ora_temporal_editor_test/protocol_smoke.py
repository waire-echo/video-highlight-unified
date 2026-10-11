"""Small real-GPU interface check before the pilot; not an accuracy evaluation."""
import argparse,json
from pathlib import Path
from engine import Client,read,digest,save,visible_text
from run import validate_discovery,validate_duplicate
import prompts as P
from settings import data_arguments

def main():
    p=argparse.ArgumentParser();data_arguments(p);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    here=Path(__file__).resolve().parent;root=a.data_root.resolve();cfg=read(a.config)
    manifest=read(root/'inputs/manifest.json');vid=min(manifest['videos'],key=lambda i:(manifest['videos'][i]['duration'],int(i)));m=manifest['videos'][vid]
    from engine import sha
    c=Client(cfg,a.output,digest(['temporal_editor_protocol_v1',cfg,{p.name:sha(p) for p in here.glob('*.py')}]))
    duration=min(m['duration'],4.0)
    media=c.clip(a.video_dir/(vid+'.mp4'),m['source_sha256'],0,duration,2)
    discovery,_=c.ask('protocol_one_video',P.DISCOVER,P.DISCOVERY_SCHEMA,[media],lambda o:validate_discovery(o,duration),2400)
    duplicate,_=c.ask('protocol_two_videos',P.DUPLICATE,P.DUPLICATE_SCHEMA,[media,media],validate_duplicate,700)
    report={'video_id':vid,'one_video_schema_passed':True,'two_video_schema_passed':True,'identical_clip_control':duplicate,'discovery':discovery,'not_an_accuracy_evaluation':True}
    save(a.output/'protocol_report.json',report)
    if duplicate['decision']!='duplicate':raise RuntimeError('Identical-clip control failed. Do not start the pilot before inspecting this interface/model behavior.')
    # Test the larger single-video frame budget before paying for the pilot.
    long_id=max(manifest['videos'],key=lambda i:manifest['videos'][i]['duration']);lm=manifest['videos'][long_id]
    length=min(lm['duration'],cfg['window_seconds']+2*cfg['context_seconds'])
    large=c.clip(a.video_dir/(long_id+'.mp4'),lm['source_sha256'],0,length,cfg['boundary_fps'])
    def validate_observe(o):
        from engine import check_times
        for k in ('subject','observed_event','uncertainty'):visible_text(o[k])
        check_times(o['evidence_times'],length)
    observed,_=c.ask('protocol_frame_budget',P.OBSERVE,P.OBSERVE_SCHEMA,[large],validate_observe,900)
    report.update(frame_budget_schema_passed=True,frame_budget_clip=large,observation=observed)
    save(a.output/'protocol_report.json',report)
    print(json.dumps(report,ensure_ascii=False))

if __name__=='__main__':main()
