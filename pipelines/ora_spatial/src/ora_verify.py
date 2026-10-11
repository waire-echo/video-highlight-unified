"""Conservative visual checks; model agreement is not semantic ground truth."""
import numpy as np
from ora_tools import image_data,parse_answer


EVIDENCE_PROMPT = '''Describe what is actually visible in this image crop, without guessing a story.
Pay attention to the central item and its surrounding context. Is it a physical object/person/animal,
an environmental region, an overlaid logo/icon/subtitle, or too unclear to identify?
An animal drawing inside a watermark is a graphic, not a live animal.
Return JSON with kind (physical_subject, environment, graphic_or_text, or unclear) and a short description.
Describe only visible evidence, and mention any ambiguity.'''


def context_crop(image, box, original_size):
    W,H=original_size;h,w=image.shape[:2]
    b=np.asarray(box,float)*[w/W,h/H,w/W,h/H]
    cx,cy=(b[:2]+b[2:])/2
    bw,bh=np.maximum(b[2:]-b[:2],1)
    # Context is necessary to distinguish an icon from an actual small subject.
    cw,ch=max(128,bw*1.8),max(128,bh*1.8)
    x1,y1=max(0,int(cx-cw/2)),max(0,int(cy-ch/2))
    x2,y2=min(w,int(np.ceil(cx+cw/2))),min(h,int(np.ceil(cy+ch/2)))
    return image[y1:y2,x1:x2]


def verify_anchor(client,image,box,original_size,target,audit,frame):
    """Caption without the target hint, then judge presence/identity/extent in the full frame."""
    crop=context_crop(image,box,original_size)
    raw=client.ask({'type':'image_url','image_url':{'url':image_data(crop)}},
                   EVIDENCE_PROMPT,audit,f'evidence_{frame}')
    evidence=parse_answer(raw)
    if (not isinstance(evidence,dict) or evidence.get('kind') not in
            ('physical_subject','environment','graphic_or_text','unclear')
            or not isinstance(evidence.get('description'),str)):
        raise ValueError('Malformed blind visual evidence')
    if evidence['kind'] in ('graphic_or_text','unclear'):
        return {'passed':False,'status':evidence['kind'],'evidence':evidence,
                'reason':'Candidate crop is a graphic or lacks identifiable visual evidence.'}
    W,H=original_size
    norm=[round(v/(W if i%2==0 else H)*1000) for i,v in enumerate(box)]
    prompt=f'''Check this candidate bounding box against the supplied FULL image.
Target identity to check: {target}
Candidate xyxy box, normalized to 0..1000: {norm}
A separate description of the candidate's context crop was: {evidence['description']}
This description may also be wrong: decide from the actual image. Do not assume the target exists.
The candidate is valid only if it contains the actual visible target and reasonably fits its visible extent.
For a group, it must cover the named group, not just one member. A partially visible subject at an image
edge is allowed if identifiable. Text, logos, and icons are not physical subjects. The sky or glow alone
does not count as a visible sun disc. A scene description or expected action is not proof of visibility.
Return JSON with status (valid, target_absent, wrong_object, bad_extent, or uncertain) and a brief reason.
Use target_absent only if the target is not visible anywhere in the full image. When unsure, use uncertain.'''
    raw=client.ask({'type':'image_url','image_url':{'url':image_data(image)}},prompt,audit,f'verify_{frame}')
    verdict=parse_answer(raw)
    if (not isinstance(verdict,dict) or verdict.get('status') not in
            ('valid','target_absent','wrong_object','bad_extent','uncertain')
            or not isinstance(verdict.get('reason'),str)):
        raise ValueError('Malformed full-frame verification')
    return {'passed':verdict['status']=='valid','evidence':evidence,**verdict}


def verify_result(client,result,images,original_size):
    result['verification_barriers']=[]
    for anchor in result['anchors']:
        if not anchor.get('accepted'):continue
        frame=anchor['frame']
        anchor['agreement_passed_before_verification']=True
        try:
            check=verify_anchor(client,images[frame],anchor['box'],original_size,
                                result['route']['target'],result['audit'],frame)
        except (ValueError,TypeError,KeyError) as e:
            check={'passed':False,'status':'parse_error','reason':str(e)}
            result['errors'].append(f'verification_{frame}:{e}')
        anchor['verification']=check
        anchor['accepted']=check['passed']
        if not check['passed']:
            # Rejection need not mean the target is absent elsewhere in the image.
            result['verification_barriers'].append({'frame':frame,'reason':check['status']})
    return result
