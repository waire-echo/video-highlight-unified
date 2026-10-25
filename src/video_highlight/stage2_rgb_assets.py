from pathlib import Path
import json,hashlib

def asset_contract(folder):
    """Validate actual bytes, not only declared names; safe before loading a GPU model."""
    folder=Path(folder)
    manifest=json.loads((folder/'ASSETS.json').read_text())
    if manifest['version']!='B_RGBRank_DP_v1':raise ValueError('Unexpected RGB backend asset version')
    required={'visual.safetensors','config.json','processor_config.json','checkpoint_0001.pt'}
    if set(manifest['files'])!=required:raise ValueError('Incomplete RGB backend assets')
    for name,entry in manifest['files'].items():
        p=folder/name;h=hashlib.sha256()
        with p.open('rb') as f:
            for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
        if h.hexdigest()!=entry['sha256'] or p.stat().st_size!=entry['bytes']:raise ValueError('RGB backend asset changed: '+name)
    return manifest
