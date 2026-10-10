#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
OLD_ROOT="${OLD_ROOT:-/root/rivermind-data/ora4b_v5_20261001}"
PY="$ROOT/env/bin/python"
mkdir -p output/logs
exec 9>output/run.lock
flock -n 9 || { echo 'Another run holds output/run.lock' >&2; exit 1; }
"$PY" -c 'import torch; assert torch.cuda.is_available(), "GPU not attached"'
"$PY" -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",8008)); s.close()'
bash scripts/serve_ora.sh > output/logs/ora_verify_server.log 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT
"$PY" scripts/wait_server.py --pid "$SERVER_PID"
"$PY" scripts/recheck_anchors.py --video-dir "$OLD_ROOT/videos"
kill "$SERVER_PID"
wait "$SERVER_PID" || true
trap - EXIT
echo 'Paired verification experiment finished. Original anchors unchanged. No SAM or full run.'
