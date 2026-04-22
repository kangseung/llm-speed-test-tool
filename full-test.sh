#!/bin/bash
set -euo pipefail

# Full test - 3 rounds with warmup, generate charts
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN=python3
fi

"$PYTHON_BIN" "$SCRIPT_DIR/claude_api_bench.py" --rounds 3 --charts --output-dir "$SCRIPT_DIR/output"
