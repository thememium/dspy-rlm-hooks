#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Run tests — suppress success output, only show errors
uv run pytest tests/ -v --tb=short 2>&1 | tail -50
