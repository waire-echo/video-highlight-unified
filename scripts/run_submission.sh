#!/usr/bin/env bash
set -euo pipefail
SUBMISSION_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUBMISSION_ASSETS="$SUBMISSION_ROOT/../AIC-2026-37717297-模型文件与运行环境"
if [[ ! -f "$SUBMISSION_ROOT/models/spatial/assets/yolo11m.pt" && -f "$SUBMISSION_ASSETS/models/spatial/assets/yolo11m.pt" ]]; then
  mkdir -p "$SUBMISSION_ROOT/models/spatial/assets"
  cp -- "$SUBMISSION_ASSETS/models/spatial/assets/yolo11m.pt" "$SUBMISSION_ROOT/models/spatial/assets/yolo11m.pt"
fi
SUBMISSION_VENV="${VH_SUBMISSION_VENV:-$SUBMISSION_ROOT/.venv}"
SUBMISSION_PYTHON="$SUBMISSION_VENV/bin/python"
if [[ -x "$SUBMISSION_ROOT/runtime/python/bin/python3.11" ]]; then
  SUBMISSION_PYTHON="$SUBMISSION_ROOT/runtime/python/bin/python3.11"
  export PATH="$SUBMISSION_ROOT/runtime/bin:/usr/bin:/bin"
  export LD_LIBRARY_PATH="$SUBMISSION_ROOT/runtime/python/lib:$SUBMISSION_ROOT/runtime/native/lib"
fi
if [[ ! -x "$SUBMISSION_PYTHON" ]]; then
  echo 'Prepare the documented Linux Python environment before running inference.' >&2
  exit 1
fi
for arg in "$@"; do
  case "$arg" in
    --materialized|--materialized=*|--existing-run-root|--existing-run-root=*|--seed-strict-merged|--seed-strict-merged=*|--start-stage|--start-stage=*|--stop-after|--stop-after=*|--prepare-only|--dry-run|--output|--output=*|--pipeline-dir|--pipeline-dir=*)
      echo 'The submission entry always starts a new complete run.' >&2
      exit 2;;
  esac
done
RUN_ID="$(date -u +%Y%m%dT%H%M%S)-$$"
SUBMISSION_WORK="$SUBMISSION_ROOT/.runtime/$RUN_ID"
mkdir -p "$SUBMISSION_ROOT/.runtime"
mkdir "$SUBMISSION_WORK"
mkdir "$SUBMISSION_WORK/tmp" "$SUBMISSION_WORK/cache"
unset PYTHONPATH PYTHONHOME VIRTUAL_ENV CONDA_PREFIX CONDA_DEFAULT_ENV LD_PRELOAD
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1 WANDB_DISABLED=true YOLO_AUTOINSTALL=false
export TMPDIR="$SUBMISSION_WORK/tmp" XDG_CACHE_HOME="$SUBMISSION_WORK/cache"
export HF_HOME="$SUBMISSION_WORK/cache/huggingface" TORCH_HOME="$SUBMISSION_WORK/cache/torch"
export TRITON_CACHE_DIR="$SUBMISSION_WORK/cache/triton" TORCHINDUCTOR_CACHE_DIR="$SUBMISSION_WORK/cache/torchinductor"
export MPLCONFIGDIR="$SUBMISSION_WORK/cache/matplotlib" YOLO_CONFIG_DIR="$SUBMISSION_WORK/cache/ultralytics"
SUBMISSION_CONFIG="$SUBMISSION_ROOT/configs/submission.local.json"
if [[ ! -f "$SUBMISSION_CONFIG" ]]; then SUBMISSION_CONFIG="$SUBMISSION_ROOT/configs/submission.json"; fi
exec "$SUBMISSION_PYTHON" -I -B "$SUBMISSION_ROOT/scripts/run_submission_pipeline.py" \
  --config "$SUBMISSION_CONFIG" --python "$SUBMISSION_PYTHON" --spatial-python "$SUBMISSION_PYTHON" \
  "$@" --output "$SUBMISSION_ROOT/outputs/$RUN_ID"
