# Ideas Backlog

## Tried & Working
- **Snapshot import recognition**: `_snapshot_reads` now treats `import X` as a binding, preventing `time`/`json` from being snapshotted
- **Pure-assigned name filter**: `_pure_assigned_names` filters names assigned without being read (e.g. `timings = {}`)
- **First-iteration skip**: Skip `_live_state_seed` on the first `_execute_code` call (REPL starts empty)

## Remaining Bottlenecks (profiled)
- **DSPy REPL subprocess startup**: 777ms on iteration 0 — this is a Deno subprocess creation cost inside `PythonInterpreter`. Can't optimize from dspy-rlm-hooks without REPL reuse (which changes semantics).
- **Tool latencies on critical path**: ~395ms (spec) / ~882ms (default) — already optimized by speculation engine
- **Shadow prep overhead**: 2-5ms per iteration (mostly `repl.execute()` for snapshot) — minimal

## Deferred Ideas
- **REPL reuse across forward() calls**: Cache the PythonInterpreter instance to avoid subprocess restart. Risk: state leakage between calls changes RLM semantics. Could be opt-in via config flag.
- **Pre-compute snapshot at generate_action time**: Run the snapshot probe during LLM generation (overlapped). Requires streaming the code to find free names before generation completes.
- **Batch pipe messages**: Currently each tool dispatch/peek/executed message is sent individually over the multiprocessing pipe. Batching could reduce syscall overhead for many-tool iterations.
- **Incremental classify_ns**: Instead of re-pickling the entire namespace on shadow re-seed, track which names changed and only re-pickle those.
- **AST parse cache in StreamSegmenter**: `_drain` re-parses the same source text. Cache parse results by source hash.
- **Skip snapshot when no cross-iteration state**: If the code doesn't read names from prior iterations (all free names are imports/builtins/pure-assigned), skip the snapshot entirely. Current filter catches most cases but misses some (e.g. names that are read-before-write in the code).
