#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Run the overlap benchmark (3 iterations for stability) and extract metrics.
# Primary metric: total speculated wall time (lower is better).
# This captures the end-to-end overhead of the speculation engine.

OUTPUT=$(uv run python benchmarks/overlap_sim.py 2>&1)

# Parse metrics from the benchmark output
BASELINE=$(echo "$OUTPUT" | grep "baseline real call" | awk '{print $NF}' | sed 's/ms//')
SPECULATED=$(echo "$OUTPUT" | grep "speculated: streamed" | awk '{print $NF}' | sed 's/ms//')
STREAM=$(echo "$OUTPUT" | grep "stream window" | awk '{print $NF}' | sed 's/ms//')
DRAIN=$(echo "$OUTPUT" | grep "drain:" | awk '{print $NF}' | sed 's/ms//')
CLAIM_WAIT=$(echo "$OUTPUT" | grep "real-path claim wait" | awk '{print $NF}' | sed 's/ms//')
HIDDEN=$(echo "$OUTPUT" | grep "tool latency hidden" | awk '{print $NF}' | sed 's/ms//')
HITS=$(echo "$OUTPUT" | grep "claim hits/misses" | awk '{print $NF}' | sed 's/,/\//')

# Run the shadow overhead benchmark and extract construction time
OVERHEAD_OUTPUT=$(uv run python benchmarks/shadow_overhead.py 2>&1)
CONSTRUCT_5MB=$(echo "$OVERHEAD_OUTPUT" | grep "ShadowRunner(5 MB" | awk '{print $(NF-1)}')
DRAIN_BARRIER=$(echo "$OVERHEAD_OUTPUT" | grep "drain barrier" | awk '{print $(NF-1)}')

# Primary metric: total speculated time (ms) - lower is better
echo "METRIC speculated_ms=${SPECULATED}"
echo "METRIC baseline_ms=${BASELINE}"
echo "METRIC stream_ms=${STREAM}"
echo "METRIC drain_ms=${DRAIN}"
echo "METRIC claim_wait_ms=${CLAIM_WAIT}"
echo "METRIC hidden_ms=${HIDDEN}"
echo "METRIC claim_hits=${HITS}"
echo "METRIC construct_5mb_ms=${CONSTRUCT_5MB}"
echo "METRIC drain_barrier_ms=${DRAIN_BARRIER}"
