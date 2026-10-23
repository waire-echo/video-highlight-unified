#!/usr/bin/env bash
# One entry for a source checkout or a relocated self-contained release.
set -euo pipefail
SUBMISSION_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUBMISSION_VENV="${VH_SUBMISSION_VENV:-$SUBMISSION_ROOT/.venv}"
SUBMISSION_PYTHON="$SUBMISSION_VENV/bin/python"
if [[ ! -x "$SUBMISSION_PYTHON" ]]; then
  echo 'Run scripts/setup_submission_env.py with Python 3.11 before inference.' >&2
  exit 1
fi
SUBMISSION_CONFIG="$SUBMISSION_ROOT/configs/submission.local.json"
if [[ ! -f "$SUBMISSION_CONFIG" ]]; then
  SUBMISSION_CONFIG="$SUBMISSION_ROOT/configs/submission.json"
fi
if [[ ! -f "$SUBMISSION_CONFIG" ]]; then
  SUBMISSION_CONFIG="$SUBMISSION_ROOT/configs/submission.example.json"
fi
exec "$SUBMISSION_PYTHON" -I -B "$SUBMISSION_ROOT/scripts/run_submission_pipeline.py" \
  --config "$SUBMISSION_CONFIG" --python "$SUBMISSION_PYTHON" --spatial-python "$SUBMISSION_PYTHON" "$@"
