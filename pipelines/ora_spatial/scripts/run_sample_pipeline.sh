#!/usr/bin/env bash
# Three-video end-to-end preview. Never starts full-dataset inference.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
OLD_ROOT="${OLD_ROOT:-/root/rivermind-data/ora4b_v5_20261001}"
PY="$ROOT/env/bin/python"
mkdir -p output/logs
exec 9>output/run.lock
flock -n 9 || { echo 'Another run holds output/run.lock' >&2; exit 1; }
"$PY" -c 'import torch; assert torch.cuda.is_available(), "GPU not attached"'
"$PY" src/budget.py --ora-dir "$OLD_ROOT/models/Video-ORA-4B" --sam-checkpoint models/sam2.1_hiera_large.pt
"$PY" -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",8008)); s.close()'
bash scripts/serve_ora.sh > output/logs/ora_sample_server.log 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT
"$PY" scripts/wait_server.py --pid "$SERVER_PID"
"$PY" src/ora_stage.py --video-dir "$OLD_ROOT/videos" --scope all --ids 0,32,269
"$PY" scripts/check_ora_smoke.py --ids 0,32,269
kill "$SERVER_PID"
wait "$SERVER_PID" || true
trap - EXIT
"$PY" scripts/wait_gpu_free.py
"$PY" src/sam_stage.py --video-dir "$OLD_ROOT/videos" --checkpoint models/sam2.1_hiera_large.pt --scope all --ids 0,32,269
"$PY" src/export_spatial.py --scope all --ids 0,32,269 --video-dir "$OLD_ROOT/videos" --output output/ORA_SAM_adaptive3_on_E1
echo 'SAMPLE_PIPELINE_COMPLETE: only 0,32,269 replaced; other rows remain E1.'
