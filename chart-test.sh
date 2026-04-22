#!/bin/bash
set -euo pipefail

# Test with charts generation
# Usage: ./chart-test.sh configs.json
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE=${1:-"$SCRIPT_DIR/example_config.json"}

python3 "$SCRIPT_DIR/claude_api_bench.py" --config "$CONFIG_FILE" --charts --output-dir "$SCRIPT_DIR/output"
