"""Class-agnostic local association control, without persistent identity claims."""
import copy,math

def box_iou(a,b):
 inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
 return inter/max(1e-8,(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter)

def selected(candidate,frame):
 box=candidate['box'];assert len(box)==4 and all(math.isfinite(v) for v in box)
 return dict(frame=int(frame),detected=True,box=list(box),center=[(box[0]+box[2])/2,(box[1]+box[3])/2],class_name=candidate['class_name'],confidence=candidate['confidence'])

def associate(records,seed_frame,seed_id,min_iou=.2,max_missing=3):
 assert records and [r['frame'] for r in records]==sorted(set(r['frame'] for r in records))
 k=next(i for i,r in enumerate(records) if r['frame']==seed_frame)
 seed=next(c for c in records[k]['eligible_candidates'] if c['target_id']==seed_id)
 result={seed_frame:dict(selected(seed,seed_frame),target_id=seed_id,origin='anchor_seed')};resets=0
 for direction in [1,-1]:
  previous=seed['box'];misses=0
  for j in range(k+direction,len(records) if direction==1 else -1,direction):
   row=records[j];frame=row['frame'];candidates=row['eligible_candidates'];best=max(candidates,key=lambda d:box_iou(previous,d['box']),default=None)
   if best is not None and box_iou(previous,best['box'])>=min_iou:
    result[frame]=dict(selected(best,frame),target_id=best['target_id'],origin='overlap');previous=best['box'];misses=0
   else:
    misses+=1
    if misses<=max_missing:result[frame]=dict(frame=frame,detected=False,target_id=None,origin='bounded_missing_interpolate')
    else:
     fallback=row['original_rule_selection'];result[frame]=dict(**copy.deepcopy(fallback),target_id=row['original_rule_choice_id'],origin='original_rule_reinitialization')
     if fallback.get('detected'):previous=fallback['box'];misses=0;resets+=1
 return [result[row['frame']] for row in records],dict(resets=resets,seed_frame=seed_frame,seed_id=seed_id,min_iou=min_iou,max_missing_sample_positions=max_missing)

def apply_control(records):
 indices=list(dict.fromkeys(round((len(records)-1)*p) for p in [.1,.35,.65,.9]))
 seed=next((records[i] for i in indices if records[i]['original_rule_choice_id'] is not None),None)
 if seed is None:return [copy.deepcopy(r['original_rule_selection']) for r in records],dict(fallback='no_supported_anchor',resets=0)
 return associate(records,seed['frame'],seed['original_rule_choice_id'])
