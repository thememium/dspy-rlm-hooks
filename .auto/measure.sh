#!/bin/bash
set -euo pipefail

# Run tests with coverage and capture output
OUTPUT=$(uv run poe test-cov 2>&1) || {
    echo "TEST FAILURE:"
    echo "$OUTPUT" | tail -30
    exit 1
}

# Extract coverage line (e.g., "TOTAL                                        335    137    59%")
COVERAGE_LINE=$(echo "$OUTPUT" | grep "^TOTAL" | tail -1)

if [ -z "$COVERAGE_LINE" ]; then
    echo "ERROR: Could not find coverage line in output"
    echo "$OUTPUT" | tail -20
    exit 1
fi

# Parse TOTAL line: TOTAL  <stmts>  <miss>  <pct>%
TOTAL_STMTS=$(echo "$COVERAGE_LINE" | awk '{print $2}')
TOTAL_MISS=$(echo "$COVERAGE_LINE" | awk '{print $3}')
COVERAGE_PCT=$(echo "$COVERAGE_LINE" | grep -oE '[0-9]+%' | tr -d '%')

echo "METRIC coverage_pct=$COVERAGE_PCT"
echo "METRIC total_lines=$TOTAL_STMTS"
echo "METRIC missed_lines=$TOTAL_MISS"

# Print per-file coverage for debugging
echo ""
echo "--- Per-file coverage ---"
echo "$OUTPUT" | grep -E "^src/" | while read -r line; do
    FILE=$(echo "$line" | awk '{print $1}')
    PCT=$(echo "$line" | grep -oE '[0-9]+%' | head -1)
    echo "  $FILE: $PCT"
done

# Print test summary
echo ""
echo "$OUTPUT" | tail -3
