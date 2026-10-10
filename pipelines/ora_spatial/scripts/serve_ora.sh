#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OLD_ROOT="${OLD_ROOT:-/root/rivermind-data/ora4b_v5_20261001}"
if [[ -d "$OLD_ROOT/python311/lib" ]]; then
  export LD_LIBRARY_PATH="$OLD_ROOT/python311/lib:${LD_LIBRARY_PATH:-}"
fi
export VLLM_NO_USAGE_STATS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_VIDEO_LOADER_BACKEND=opencv
exec "$OLD_ROOT/env/bin/vllm" serve "$OLD_ROOT/models/Video-ORA-4B" \
  --served-model-name Video-ORA-4B --host 127.0.0.1 --port 8008 \
  --dtype bfloat16 --tensor-parallel-size 1 --max-model-len 16384 \
  --max-num-seqs 1 --max-num-batched-tokens 8192 --enforce-eager \
  --gpu-memory-utilization 0.85 --mm-processor-cache-gb 0 \
  --limit-mm-per-prompt '{"image":1,"video":1}' \
  --media-io-kwargs '{"video":{"num_frames":-1}}' \
  --mm-processor-kwargs '{"max_frames":64,"size":{"shortest_edge":65536,"longest_edge":16777216}}'
