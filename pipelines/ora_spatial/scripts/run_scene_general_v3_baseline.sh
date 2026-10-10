#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_ROOT="${ORA_RUNTIME_ROOT:?Set ORA_RUNTIME_ROOT to the directory containing env/bin/vllm and models/Video-ORA-4B}"
VIDEO_DIR="${VIDEO_DIR:?Set VIDEO_DIR to the directory containing 0.mp4 through 425.mp4}"
MODEL_DIR="${MODEL_DIR:-$RUNTIME_ROOT/models/Video-ORA-4B}"
CLIENT_PYTHON="${CLIENT_PYTHON:-$ROOT/env/bin/python}"
VLLM_BIN="${VLLM_BIN:-$RUNTIME_ROOT/env/bin/vllm}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/output/scene_temporal_v3_full_426}"
PORT="${ORA_PORT:-8008}"
IDS="${VIDEO_IDS:-$(seq -s, 0 425)}"
SERVICE_LOG="${SERVICE_LOG:-$OUTPUT_DIR/vllm.log}"

mkdir -p "$OUTPUT_DIR"

"$CLIENT_PYTHON" "$ROOT/scripts/verify_scene_general_v3_baseline.py" \
  --root "$ROOT" --model-dir "$MODEL_DIR" --check-environment

if curl -fsS --max-time 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
  echo "Port $PORT already has a model service. Stop it before running the frozen baseline." >&2
  exit 2
fi

export LD_LIBRARY_PATH="$RUNTIME_ROOT/python311/lib:${LD_LIBRARY_PATH:-}"
export VLLM_NO_USAGE_STATS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

"$VLLM_BIN" serve "$MODEL_DIR" \
  --served-model-name Video-ORA-4B --host 127.0.0.1 --port "$PORT" \
  --dtype bfloat16 --tensor-parallel-size 1 --max-model-len 16384 \
  --max-num-seqs 1 --max-num-batched-tokens 8192 --enforce-eager \
  --gpu-memory-utilization 0.85 --mm-processor-cache-gb 0 \
  --limit-mm-per-prompt '{"image":3,"video":0}' \
  --mm-processor-kwargs '{"size":{"shortest_edge":65536,"longest_edge":1900000}}' \
  >>"$SERVICE_LOG" 2>&1 &
SERVICE_PID=$!

cleanup() {
  kill "$SERVICE_PID" 2>/dev/null || true
  wait "$SERVICE_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

for _ in $(seq 1 180); do
  if curl -fsS --max-time 2 "http://127.0.0.1:$PORT/v1/models" | grep -q 'Video-ORA-4B'; then
    break
  fi
  if ! kill -0 "$SERVICE_PID" 2>/dev/null; then
    echo "ORA service stopped during startup. See $SERVICE_LOG" >&2
    exit 3
  fi
  sleep 1
done

curl -fsS --max-time 5 "http://127.0.0.1:$PORT/v1/models" | grep -q 'Video-ORA-4B'

"$CLIENT_PYTHON" "$ROOT/experiments/scene_general_v3/preflight.py" \
  --ids "$IDS" --video-dir "$VIDEO_DIR"

"$CLIENT_PYTHON" "$ROOT/experiments/scene_general_v3/temporal.py" \
  --ids "$IDS" --video-dir "$VIDEO_DIR" --output "$OUTPUT_DIR"
