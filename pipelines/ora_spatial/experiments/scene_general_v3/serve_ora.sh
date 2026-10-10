#!/usr/bin/env bash
set -euo pipefail
OLD_ROOT=/root/rivermind-data/ora4b_v5_20261001
export LD_LIBRARY_PATH="$OLD_ROOT/python311/lib:${LD_LIBRARY_PATH:-}"
export VLLM_NO_USAGE_STATS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
exec "$OLD_ROOT/env/bin/vllm" serve "$OLD_ROOT/models/Video-ORA-4B" \
 --served-model-name Video-ORA-4B --host 127.0.0.1 --port 8008 \
 --dtype bfloat16 --tensor-parallel-size 1 --max-model-len 16384 \
 --max-num-seqs 1 --max-num-batched-tokens 8192 --enforce-eager \
 --gpu-memory-utilization 0.85 --mm-processor-cache-gb 0 \
 --limit-mm-per-prompt '{"image":3,"video":0}' \
 --mm-processor-kwargs '{"size":{"shortest_edge":65536,"longest_edge":1900000}}'
