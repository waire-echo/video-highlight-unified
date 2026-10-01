"""Resolve one user configuration for both pipeline stages."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import sys

PATH_FIELDS = ('run_root', 'video_dir', 'test_index', 'temporal_model', 'yolo_model',
               'pose_model', 'temporal_run_dir', 'work_dir', 'input_dir', 'log_dir',
               'output_dir', 'media_root', 'temporal_handoff')


def load_config(path: Path) -> dict:
    path = path.expanduser().resolve()
    config = json.loads(path.read_text(encoding='utf-8-sig'))
    if not isinstance(config, dict):
        raise ValueError('Configuration must be a JSON object')
    for field in ('run_root', 'video_dir', 'test_index'):
        if not isinstance(config.get(field), str) or not config[field].strip():
            raise ValueError('Missing configuration path: ' + field)
    for field in PATH_FIELDS:
        if config.get(field):
            value = Path(config[field]).expanduser()
            config[field] = str((path.parent / value).resolve() if not value.is_absolute() else value.resolve())
    for field in ('python_executable', 'temporal_python'):
        value = config.get(field, sys.executable)
        found = shutil.which(value)
        # Keep the venv executable path: resolving its symlink would select the base interpreter.
        candidate = Path(found or value).expanduser()
        config[field] = os.path.abspath(path.parent / candidate if not candidate.is_absolute() else candidate)
    run = Path(config['run_root'])
    defaults = {'temporal_run_dir': run / 'temporal', 'work_dir': run / 'spatial/work',
                'input_dir': run / 'input', 'log_dir': run / 'logs', 'output_dir': run / 'output'}
    for field, value in defaults.items():
        config.setdefault(field, str(value))
    config.setdefault('model_revision', '01850297d5ab2adaaf130f700ed7cec52993d956')
    config.setdefault('model_port', 8000)
    config.setdefault('profile', 'auto')
    config['_config_file'] = str(path)
    return config


def write_resolved(config: dict) -> Path:
    run = Path(config['run_root'])
    run.mkdir(parents=True, exist_ok=True)
    path = run / 'config.resolved.json'
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(config, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)
    return path
