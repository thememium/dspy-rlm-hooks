# Ideas Backlog

## Applied Optimizations (8 total, ~2.4% improvement)
1. **Snapshot import recognition**: `_snapshot_reads` treats `import X` as binding
2. **Pure-assigned name filter**: `_pure_assigned_names` filters non-read assignments
3. **First-iteration skip**: Skip snapshot on first `_execute_code` per `forward()`
4. **Faster snapshot probe**: repr-based instead of json-based (24% faster per probe)
5. **Skip redundant tool registration**: Track tool signature hashes, only re-register when changed
6. **Skip redundant registry sync**: `_sync_registry_fns` called once per iteration instead of twice
7. **Loop-target snapshot skip**: Skip snapshot for loop targets over non-empty iterables
8. **Skip snapshot in streaming path**: Shadow already processed code during generate_action

## Final State
- **Framework overhead**: 0.02% (0.3ms/1750ms) — effectively zero
- **Remaining bottleneck**: DSPy REPL subprocess execution (99.98%)
- **Speculation speedup**: 1.39× over default (tool call overlap)

## Key Finding
The snapshot probe was entirely wasted in the streaming path. The shadow processes code during `generate_action` (streaming), so by the time the snapshot runs in `_speculation_execute_code`, the shadow has already processed everything. The snapshot only helps in the Lazy/JIT path (non-streaming or stream failure).

## No Further Optimizations Available
All speculation engine internals are at sub-0.1ms. The remaining 99.98% is DSPy's PythonInterpreter subprocess execution — requires DSPy-level changes.
