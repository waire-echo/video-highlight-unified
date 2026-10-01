#!/usr/bin/env bash
set -euo pipefail
RUNTIME_ROOT="${1:?Usage: bash scripts/setup_spatial_gpu.sh /absolute/runtime/directory}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$RUNTIME_ROOT"
RUNTIME_ROOT="$(cd "$RUNTIME_ROOT" && pwd)"
"$PYTHON_BIN" -c 'import sys; assert sys.version_info >= (3,11), "Python 3.11+ required; set PYTHON_BIN"'
command -v nvidia-smi >/dev/null
"$PYTHON_BIN" -m venv "$RUNTIME_ROOT/spatial-env"
source "$RUNTIME_ROOT/spatial-env/bin/activate"
python -m pip install --upgrade pip
python -m pip install 'torch==2.10.0' 'torchvision==0.25.0' --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r "$SCRIPT_DIR/../requirements-spatial.txt"
python -m pip check
python "$SCRIPT_DIR/../tools/download_spatial_models.py" --output "$RUNTIME_ROOT/models"
python -m pip freeze > "$RUNTIME_ROOT/spatial.environment.freeze.txt"
python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.cuda.get_device_name(0))'
