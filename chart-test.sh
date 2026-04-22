#!/bin/bash
set -euo pipefail

# Test with charts generation
# Usage: ./chart-test.sh configs.json
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE=${1:-"$SCRIPT_DIR/example_config.json"}
PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN=python3
fi

"$PYTHON_BIN" "$SCRIPT_DIR/claude_api_bench.py" --config "$CONFIG_FILE" --charts --output-dir "$SCRIPT_DIR/output"
