#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Run tests — suppress verbose output, only show errors
python -m pytest tests/ -x -q --tb=short 2>&1 | tail -30
