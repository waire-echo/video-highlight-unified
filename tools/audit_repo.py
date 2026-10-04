#!/usr/bin/env python3
"""Reject generated data, model binaries and credentials in the Git index."""
import re
from pathlib import Path
import subprocess
import sys

repo = Path(__file__).resolve().parents[1]
files = subprocess.run(['git', '-C', str(repo), 'ls-files', '-z'], check=True, capture_output=True).stdout.decode().split('\0')
forbidden = {'.jsonl', '.mp4', '.mov', '.avi', '.pt', '.pth', '.safetensors', '.ckpt', '.bin', '.zip'}
errors = []
for name in filter(None, files):
    path = Path(name)
    if path.suffix.lower() in forbidden or path.name in {'AGENTS.md', 'PROJECT_HISTORY.md', 'PROJECT_LINKS.md', 'TEMPSAMP_R1_ANALYSIS.md'}:
        errors.append('Forbidden artifact: ' + name)
    if (
        path.parts[0] in {'data', 'models', 'runs', 'results', 'outputs', 'logs', 'previews', '.codex', 'artifacts', 'attachments'}
        or path.parts[:2] in {('references', 'local'), ('datasets', 'local')}
        or name in {'references/assets.local.json', 'experiments/artifact_locations.local.json'}
    ):
        errors.append('Generated/private directory: ' + name)
    data = (repo / path).read_text(encoding='utf-8', errors='replace')
    if re.search(r'(?:gh[pousr]_[A-Za-z0-9]{25,}|sk-[A-Za-z0-9]{25,}|BEGIN (?:OPENSSH|RSA|EC) PRIVATE KEY)', data):
        errors.append('Possible credential: ' + name)
if errors:
    print('\n'.join(errors), file=sys.stderr)
    raise SystemExit(1)
print('Git index audit passed: source, configuration examples and documentation only.')
