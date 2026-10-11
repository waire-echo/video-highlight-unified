#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
OLD_ROOT="${OLD_ROOT:-/root/rivermind-data/ora4b_v5_20261001}"
PY="$ROOT/env/bin/python"
VIDEO_DIR="${VIDEO_DIR:-$OLD_ROOT/videos}"
SCOPE="${1:-all}"
if [[ "$SCOPE" != all && "$SCOPE" != partial ]]; then echo 'Use all or partial' >&2; exit 2; fi
mkdir -p output/logs
exec 9>output/run.lock
flock -n 9 || { echo 'Another run holds output/run.lock' >&2; exit 1; }
"$PY" -c 'import torch; assert torch.cuda.is_available(), "GPU not attached"'
"$PY" src/budget.py --ora-dir "$OLD_ROOT/models/Video-ORA-4B" --sam-checkpoint models/sam2.1_hiera_large.pt
"$PY" scripts/check_processor.py --ora-dir "$OLD_ROOT/models/Video-ORA-4B"
"$PY" -m unittest discover -s tests -v
"$PY" src/shots.py --video-dir "$VIDEO_DIR" --device cuda
"$PY" src/stage1.py
# Only stop the server started by this script; never touch old experiments' servers.
"$PY" -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",8008)); s.close()'
bash scripts/serve_ora.sh > output/logs/ora_server.log 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT
"$PY" scripts/wait_server.py --pid "$SERVER_PID"
"$PY" src/ora_stage.py --video-dir "$VIDEO_DIR" --scope all --ids 0,32,269
"$PY" scripts/check_ora_smoke.py --ids 0,32,269
"$PY" src/ora_stage.py --video-dir "$VIDEO_DIR" --scope "$SCOPE" 2>&1 | tee output/logs/ora_inference.log
kill "$SERVER_PID"
wait "$SERVER_PID" || true
trap - EXIT
"$PY" scripts/wait_gpu_free.py
"$PY" src/sam_stage.py --video-dir "$VIDEO_DIR" --checkpoint models/sam2.1_hiera_large.pt --scope "$SCOPE" 2>&1 | tee output/logs/sam_inference.log
"$PY" src/export_spatial.py --scope partial --video-dir "$VIDEO_DIR" --output output/E2_partial
if [[ "$SCOPE" == all ]]; then
  "$PY" src/export_spatial.py --scope all --video-dir "$VIDEO_DIR" --output output/E3_all
fi
echo 'Complete. Score is unknown until official evaluation. Read *.report.json before submission.'
