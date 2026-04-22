#!/bin/bash
set -euo pipefail

# Quick test - single round, skip warmup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN=python3
fi

"$PYTHON_BIN" "$SCRIPT_DIR/claude_api_bench.py" --rounds 1 --no-warmup
