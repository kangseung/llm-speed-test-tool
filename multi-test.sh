#!/bin/bash
set -euo pipefail

# Multi-config test with configurable rounds
# Usage: ./multi-test.sh configs.json 5
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE=${1:-"${HOME}/.claude/settings.json"}
ROUNDS=${2:-3}

python3 "$SCRIPT_DIR/claude_api_bench.py" --config "$CONFIG_FILE" --rounds "$ROUNDS"
