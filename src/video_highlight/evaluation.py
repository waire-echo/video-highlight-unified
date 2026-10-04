"""PDF-derived development metric, not the organizer scoring script."""
import math

def number(x):
    return type(x) in (int, float) and math.isfinite(x)


def valid_crop(box,width,height,ratio):
    if not isinstance(box,list) or len(box)!=3 or not all(number(v) for v in box):return False
    x,y,w=box;h=w*ratio[1]/ratio[0]
    return x>=0 and y>=0 and w>0 and x+w<=width+1e-5 and y+h<=height+1e-5


def iou(a,b):
    inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/union if union>0 else 0.


def weighted_f1(predictions,ground_truth,width,height,ratio):
    """Count every submitted attempt, including duplicates/invalid boxes, as Npred."""
    truth={x['frame']:x['bboxes'] for x in ground_truth};used=set();total=0.
    if len(truth)!=len(ground_truth):raise ValueError('Duplicate ground truth')
    def xyxy(b):return [b[0],b[1],b[0]+b[2],b[1]+b[2]*ratio[1]/ratio[0]]
    for p in predictions:
        f=p.get('frame');b=p.get('bboxes')
        if type(f) is not int or f<0 or not valid_crop(b,width,height,ratio) or f in used:continue
        used.add(f)
        if f in truth: total+=iou(xyxy(b),xyxy(truth[f]))
    npred,ngt=len(predictions),len(ground_truth)
    return {'n_pred':npred,'n_gt':ngt,'sum_iou':total,'f1':2*total/(npred+ngt) if npred+ngt else 1.}
