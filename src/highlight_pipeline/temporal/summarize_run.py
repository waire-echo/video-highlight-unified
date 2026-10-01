#!/usr/bin/env python3
"""Summarize automatic inference diagnostics without treating predictions as GT."""
import argparse
from collections import Counter
from pathlib import Path
import statistics

from core import read_json, write_json


def summarize(root):
    manifest=read_json(root/'manifest.json')
    complete=[]
    errors={}
    missing=[]
    for vid in manifest['input_ids']:
        path=root/'records'/f'{vid}.json'
        if not path.exists():
            missing.append(vid)
            continue
        record=read_json(path)
        if record.get('status')!='complete':
            errors[vid]=record.get('error','Unknown failure')
        elif record.get('signature')!=manifest['signature']:
            errors[vid]='Record signature differs from manifest'
        else:
            complete.append(record)
    variants={}
    for variant in ('recall','grounded'):
        coverages=[]
        segments=0
        for row in complete:
            spans=row['segments_'+variant]
            coverages.append(sum(b-a for a,b in spans)/row['duration'])
            segments+=len(spans)
        variants[variant]={
            'segments':segments,
            'mean_time_coverage':statistics.mean(coverages) if coverages else None,
            'median_time_coverage':statistics.median(coverages) if coverages else None,
            'empty_videos':sum(x==0 for x in coverages),
            'at_least_99_percent_coverage':sum(x>=.99 for x in coverages),
        }
    events=[e for row in complete for e in row['events']]
    observations=[o for row in complete for o in row['observations']]
    reasons=Counter()
    input_tokens=output_tokens=0
    for path in (root/'raw_responses').glob('*.json'):
        data=read_json(path)['response']
        reasons.update([data['choices'][0].get('finish_reason','unknown')])
        usage=data.get('usage') or {}
        input_tokens+=usage.get('prompt_tokens',0)
        output_tokens+=usage.get('completion_tokens',0)
    return {
        'version':manifest['version'],'signature':manifest['signature'],
        'requested_videos':len(manifest['input_ids']), 'complete_videos':len(complete),
        'missing_ids':missing,'errors':errors,'variants':variants,
        'event_discovery_decisions':dict(Counter(e['value'] for e in events)),
        'event_review_decisions':dict(Counter(e['selection'] for e in events)),
        'grounding_status':dict(Counter(e.get('grounding_status','not_grounded') for e in events)),
        'subject_review_changes':sum(o.get('discovery_subject',o['main_subject'])!=o['main_subject'] for o in observations),
        'subject_review_failures':sum(bool(o.get('subject_warning')) for o in observations),
        'answer_finish_reasons':dict(reasons),
        'prompt_tokens':input_tokens,'completion_tokens':output_tokens,
        'note':'These are execution and coverage diagnostics, not precision, recall or competition score.',
    }


if __name__=='__main__':
    import json
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    args=p.parse_args()
    report=summarize(args.run)
    write_json(args.run/'diagnostics.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2))
