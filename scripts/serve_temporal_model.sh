#!/usr/bin/env bash
set -euo pipefail
CONFIG="${1:?Usage: bash scripts/serve_temporal_model.sh /absolute/pipeline.local.json}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
exec "$PYTHON_BIN" "$SCRIPT_DIR/../pipeline.py" --config "$CONFIG" serve
