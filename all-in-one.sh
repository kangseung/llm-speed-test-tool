#!/bin/bash
set -euo pipefail

# All-in-one test - full benchmark with charts and JSON report
# Usage: ./all-in-one.sh configs.json 5 ./output
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE=${1:-"${HOME}/.claude/settings.json"}
ROUNDS=${2:-3}
OUTPUT_DIR=${3:-"$SCRIPT_DIR/output"}

python3 "$SCRIPT_DIR/claude_api_bench.py" --config "$CONFIG_FILE" --rounds "$ROUNDS" --charts --output-dir "$OUTPUT_DIR" --json
