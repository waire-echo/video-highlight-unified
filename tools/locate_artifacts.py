#!/usr/bin/env python3
"""Locate historical artifacts through local paths and verify their SHA256."""
import argparse
import json
from pathlib import Path
import sys

repo = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo / 'src'))
from video_highlight.experiments import experiment_path, read_json, sha256

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--experiment', required=True)
args = parser.parse_args()
record = read_json(experiment_path(args.experiment) / 'experiment.json')
locations_path = repo / 'experiments/artifact_locations.local.json'
locations = read_json(locations_path) if locations_path.exists() else {}
for item in record.get('artifacts', []):
    path = Path(locations[item['location_key']]) if item['location_key'] in locations else None
    print(json.dumps({'name':item['name'], 'path':str(path) if path else None,
                      'expected_sha256':item['sha256'],
                      'hash_matches':sha256(path) == item['sha256'] if path and path.is_file() else None}, ensure_ascii=False))
