#!/bin/bash
set -euo pipefail

# Quick test - single round, skip warmup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 "$SCRIPT_DIR/claude_api_bench.py" --rounds 1 --no-warmup
