"""Audit complete stored tensors conservatively, never infer count from model names."""
import argparse
import json
import math
import struct
from pathlib import Path
from common import read_json,write_json,sha,digest


def safetensor_elements(path):
    with open(path,'rb') as f:
        size=struct.unpack('<Q',f.read(8))[0]
        if size>100_000_000:raise ValueError('Unreasonable safetensors header')
        header=json.loads(f.read(size))
    return sum(math.prod(v['shape']) for k,v in header.items() if k!='__metadata__')


def checkpoint_elements(path):
    import torch
    obj=torch.load(path,map_location='cpu',weights_only=True)
    if isinstance(obj,dict) and 'model' in obj:obj=obj['model']
    if not isinstance(obj,dict):raise ValueError('Expected safe state dictionary')
    return sum(v.numel() for v in obj.values() if isinstance(v,torch.Tensor))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--ora-dir',required=True);p.add_argument('--sam-checkpoint',required=True)
    p.add_argument('--transnet',default='models/transnetv2-pytorch-weights.pth')
    p.add_argument('--config',default='config/default.json');p.add_argument('--output',default='output/parameter_ledger.json')
    p.add_argument('--ora-provenance',default='inputs/ora_model_provenance.json')
    a=p.parse_args();cfg=read_json(a.config)
    modeldir=Path(a.ora_dir)
    files=sorted(modeldir.glob('*.safetensors'))
    if not files:raise ValueError('No ORA safetensors found')
    index=modeldir/'model.safetensors.index.json'
    if index.exists():
        expected=set(read_json(index)['weight_map'].values())
        if expected!={f.name for f in files}:raise ValueError('Incomplete/extra ORA shards')
    ora=sum(safetensor_elements(f) for f in files)
    if ora!=cfg['ora_parameters']:raise ValueError(f'Wrong ORA checkpoint/count: {ora}')
    provenance=read_json(a.ora_provenance)
    if provenance['revision']!=cfg['ora_revision']:raise ValueError('Unexpected ORA revision')
    reference={v['filename']:v['sha256'] for v in provenance['weight_files']}
    shards=[{'name':f.name,'sha256':sha(f)} for f in files]
    if {v['name']:v['sha256'] for v in shards}!=reference:raise ValueError('ORA weight checksum mismatch')
    sam=checkpoint_elements(a.sam_checkpoint);trans=checkpoint_elements(a.transnet)
    reserve=cfg['inherited_spatial_parameter_reserve']
    total=ora+sam+trans+reserve
    ledger={'ora_tensor_elements':ora,'ora_shards':shards,
        'sam_tensor_elements':sam,'sam_checkpoint_sha256':sha(a.sam_checkpoint),
        'transnet_tensor_elements':trans,'transnet_sha256':sha(a.transnet),
        'inherited_spatial_reserved':reserve,'total_upper_bound':total,
        'parameter_limit':cfg['parameter_limit'],'pass':total<=cfg['parameter_limit'],
        'note':'Includes buffers as a conservative upper bound. One shared ORA checkpoint. No learned saliency/flow network. 0.1B reserved for inherited YOLO/Pose/ReID; audit inherited weights separately for complete historical reproduction.'}
    ledger['signature']=digest(ledger)
    write_json(a.output,ledger)
    if not ledger['pass']:raise ValueError('Total exceeds 9B')
    print(json.dumps({k:v for k,v in ledger.items() if k not in ['ora_shards','note']},indent=2))


if __name__=='__main__':main()
