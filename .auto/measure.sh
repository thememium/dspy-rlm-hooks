#!/bin/bash
set -euo pipefail

# Run the deterministic A/B benchmark with enough repeats for stable medians.
# pace-ms=0 removes simulated token-arrival delay (pure framework overhead).
# tool-ms=60, llm-ms=150 match the default scenario.
cd "$(dirname "$0")/.."

# Run benchmark, capture JSON output
python -m dspy_rlm_hooks.benchmark \
    --variants default spec \
    --repeats 5 \
    --pace-ms 0.1 \
    --json-out .auto/bench-result.json 2>/dev/null

# Parse medians from JSON
python3 -c "
import json, sys
with open('.auto/bench-result.json') as f:
    r = json.load(f)
for vname, vdata in r['variants'].items():
    w = vdata['wall_s']['median']
    e = vdata['execute_ms_total']['median']
    g = vdata['generate_ms_total']['median']
    tc = vdata['tool_critical_ms']['median']
    ts = vdata['tool_serial_ms']['median']
    print(f'[{vname}] wall={w:.3f}s exec={e:.1f}ms gen={g:.1f}ms tool_crit={tc:.1f}ms tool_serial={ts:.1f}ms')

# Primary metric: spec variant wall time (lower is better)
spec_wall = r['variants']['spec']['wall_s']['median']
default_wall = r['variants']['default']['wall_s']['median']
speedup = default_wall / spec_wall if spec_wall > 0 else 0

print(f'METRIC wall_s={spec_wall:.3f}')
print(f'METRIC default_wall_s={default_wall:.3f}')
print(f'METRIC speedup={speedup:.3f}')
print(f'METRIC exec_ms={r[\"variants\"][\"spec\"][\"execute_ms_total\"][\"median\"]:.1f}')
print(f'METRIC gen_ms={r[\"variants\"][\"spec\"][\"generate_ms_total\"][\"median\"]:.1f}')
print(f'METRIC tool_crit_ms={r[\"variants\"][\"spec\"][\"tool_critical_ms\"][\"median\"]:.1f}')

# Check step equivalence
equiv = r.get('equivalent_steps', True)
print(f'STEPS_EQUIVALENT={equiv}')
if not equiv:
    print('WARNING: Steps differ across variants!', file=sys.stderr)
"
