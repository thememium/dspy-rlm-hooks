# Ideas Backlog

## Applied Optimizations (Cumulative ~2-3% improvement)
- **Snapshot import recognition**: `_snapshot_reads` treats `import X` as binding
- **Pure-assigned name filter**: `_pure_assigned_names` filters non-read assignments
- **First-iteration skip**: Skip snapshot on first `_execute_code` per `forward()`
- **Faster snapshot probe**: repr-based instead of json-based (24% faster per probe)
- **Skip redundant tool registration**: Track tool signature hashes, only re-register when changed

## Remaining Bottlenecks (Not Optimizable)
- **DSPy REPL subprocess startup**: 777ms on iteration 0 — Deno subprocess creation. Requires DSPy-level changes.
- **repl.execute() for snapshot**: ~1ms per iteration — subprocess IPC overhead. Unavoidable for cross-iteration state.
- **Tool latencies**: ~395ms (spec) / ~882ms (default) — already optimized by speculation

## Explored & Discarded
- **REPL reuse across forward()**: Changes RLM semantics (state leakage)
- **For-loop target in pure_assigned**: Empty loops don't overwrite — test failure
- **Batch pipe messages**: Subprocess IPC is already µs-level
- **AST parse cache**: Each segment has unique source — no cache hits
- **canonical_hash optimization**: Already ~1µs — not worth optimizing
- **classify_ns caching**: Namespace is stable — re-seeding never happens in benchmark

## If Starting Fresh
- **Pre-compute snapshot at generate_action time**: Overlap snapshot with LLM generation. Complex but could hide the 1ms snapshot cost.
- **Instrument PythonInterpreter startup**: Profile the 777ms Deno startup to find optimization opportunities in DSPy itself.
