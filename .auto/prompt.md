# Autoresearch: RLM Performance Optimization

## Objective
Optimize the full execution time of dspy-rlm-hooks' RLM execution pipeline — both the speculation-enabled path (`spec` variant) and the plain path (`default` variant). Focus on reducing framework overhead in the hot path: iteration loop, code execution, speculation integration, streaming, and subprocess management.

The benchmarks use deterministic scripted scenarios (fixed tool latencies, no real LLM) so the ONLY thing that differs between runs is framework scheduling overhead. Wall-clock improvements here translate directly to real-world speedups.

## Metrics
- **Primary**: wall_s (seconds, lower is better) — end-to-end `forward()` execution time
- **Secondary**: execute_ms_total, generate_ms_total, tool_critical_ms, tool_serial_ms

## How to Run
`./.auto/measure.sh` — runs the deterministic A/B benchmark and outputs `METRIC name=number` lines.

## Files in Scope
- `src/dspy_rlm_hooks/speculation_integration.py` — wires speculation into RLM's execution path. Per-execution overhead: `_sync_registry_fns`, `_live_state_seed`, `_install_claim_hooks`, streaming turn management
- `src/dspy_rlm_hooks/speculation/streaming.py` — `StreamSegmenter` statement segmentation, `plan_peeks`/`plan_peeks_with_chains` AST analysis, `safe_eval`, `repair_tail`
- `src/dspy_rlm_hooks/speculation/shadow.py` — `ShadowRunner` subprocess lifecycle, `classify_ns`/`load_ns` serialization, segment execution, peek handling
- `src/dspy_rlm_hooks/speculation/session.py` — `SpecSession`, `Launcher`, `StreamTurn`, `LatencyStats`
- `src/dspy_rlm_hooks/speculation/hooks.py` — hook factories: `make_real_hooks`, `make_shadow_hooks`, claim-or-run logic
- `src/dspy_rlm_hooks/speculation/store.py` — `SpecStore`, `Speculation` future
- `src/dspy_rlm_hooks/speculation/tool.py` — `ToolSpec`, `spec_key`, `canonical_hash`, `split_batch_call`
- `src/dspy_rlm_hooks/speculation/budget.py` — `Budget` concurrency/dispatch caps
- `src/dspy_rlm_hooks/speculation/guards.py` — claim hook tagging, `current_spec()` tracking
- `src/dspy_rlm_hooks/patcher.py` — `enable_rlm_hooks`, `_execute_iteration`, `_execute_code`
- `src/dspy_rlm_hooks/utils.py` — `_assemble_execution_code`, `_strip_code_fences`
- `src/dspy_rlm_hooks/speculator.py` — `Speculator` facade
- `src/dspy_rlm_hooks/benchmark.py` — deterministic A/B benchmark (DO NOT MODIFY benchmark logic)

## Off Limits
- Do NOT modify `benchmark.py`'s scenario definitions, timing instrumentation, or variant logic
- Do NOT change the external API surface (`enable_rlm_speculation`, `enable_rlm_hooks`, etc.)
- Do NOT add new dependencies
- Do NOT change correctness guarantees (taint safety, claim identity, eviction semantics)

## Constraints
- Steps must remain IDENTICAL across variants (benchmark asserts this)
- All existing tests must pass
- Speculation claimed/evicted counts should stay the same or improve

## What's Been Tried
(Session starts fresh — no prior experiments yet.)
