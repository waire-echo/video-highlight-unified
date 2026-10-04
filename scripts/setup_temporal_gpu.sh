#!/usr/bin/env bash
# Run on the rented Linux host. Does not access any video or start a paid instance.
set -euo pipefail
TASK_ROOT="${1:?Usage: bash scripts/setup_temporal_gpu.sh /absolute/runtime/directory}"
mkdir -p "$TASK_ROOT"
TASK_ROOT="$(cd "$TASK_ROOT" && pwd)"
command -v nvidia-smi >/dev/null
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
if ! command -v ffmpeg >/dev/null || ! command -v ffprobe >/dev/null; then
  if command -v apt-get >/dev/null && [ "$(id -u)" -eq 0 ]; then
    apt-get update -qq
    apt-get install -y ffmpeg
  else
    echo 'Install ffmpeg and ffprobe first, then rerun.' >&2
    exit 1
  fi
fi
PYTHON_BIN="${PYTHON_BIN:-python3}"
"$PYTHON_BIN" -c 'import sys; assert sys.version_info >= (3,11), "Python 3.11+ required; set PYTHON_BIN"'
"$PYTHON_BIN" -m venv "$TASK_ROOT/temporal-env"
source "$TASK_ROOT/temporal-env/bin/activate"
python -m pip install --upgrade pip
# Start with the official serving versions; install the small client dependencies.
python -m pip install 'vllm==0.19.1' 'transformers==5.5.4' 'qwen-vl-utils==0.0.14' huggingface_hub pillow
python -m pip check
mkdir -p "$TASK_ROOT/models"
python - "$TASK_ROOT" <<'PY'
import json, sys
from pathlib import Path
from huggingface_hub import model_info, snapshot_download
root=Path(sys.argv[1])
repo_id='OraRL/Video-ORA-4B'
revision='01850297d5ab2adaaf130f700ed7cec52993d956'
info=model_info(repo_id,revision=revision)
if info.sha!=revision:
    raise RuntimeError('Model metadata does not match the pinned revision')
parameter_count=getattr(info.safetensors,'total',None)
if not isinstance(parameter_count,int) or not 0 < parameter_count <= 9_000_000_000:
    raise RuntimeError(f'Cannot verify the <=9B parameter budget: {parameter_count}')
snapshot_download(repo_id,revision=revision,local_dir=root/'models/Video-ORA-4B')
(root/'model_provenance.json').write_text(json.dumps({
    'repo_id':repo_id,'revision':revision,
    'safetensors_total_parameters':parameter_count,'parameter_limit':9_000_000_000,
},indent=2))
PY
python -m pip freeze > "$TASK_ROOT/temporal.environment.freeze.txt"
echo "Ready. Edit configs/pipeline.local.json, then run pipeline.py --config ... serve"
