#!/bin/bash
set -euo pipefail

# Multi-config test with configurable rounds
# Usage: ./multi-test.sh configs.json 5
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE=${1:-"$SCRIPT_DIR/example_config.json"}
ROUNDS=${2:-3}
PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN=python3
fi

"$PYTHON_BIN" "$SCRIPT_DIR/claude_api_bench.py" --config "$CONFIG_FILE" --rounds "$ROUNDS"
