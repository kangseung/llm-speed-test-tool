#!/bin/bash
set -euo pipefail

# Full test - 3 rounds with warmup, generate charts
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 "$SCRIPT_DIR/claude_api_bench.py" --rounds 3 --charts --output-dir "$SCRIPT_DIR/output"
