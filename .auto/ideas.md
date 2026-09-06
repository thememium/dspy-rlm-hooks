# Ideas Backlog

- **Pre-pickle tool functions once**: `classify_ns` pickles tool functions every turn. Since tools are stable across turns, cache their pickled bytes and skip re-serialization.
- **Share shadow namespace across turns without re-pickle**: `_last_turn_seed` comparison uses `dict.__eq__` on the full namespace. Hash the pickle bytes instead for cheaper equality check.
- **Batch pipe sends in ShadowRunner**: Each segment, peek, and tool dispatch sends individually over the multiprocessing pipe. Batching multiple messages into one `send()` could reduce syscall overhead.
- **Incremental classify_ns**: Instead of re-pickling the entire namespace each turn, track which names changed and only re-pickle those. The seed dict diff is cheap.
- **AST cache in StreamSegmenter**: `_drain` re-parses the same source text via `ast.parse`. Cache parse results by source hash.
- **Lazy `_snapshot_reads`**: The `_snapshot_reads` AST walk happens every `_speculation_execute_code` call. For iterations with no free names beyond input_args, skip it entirely.
- **Thread-local `_sync_registry_fns` cache**: The registry sync walks all tools every execution. Cache the last-synced version and skip when unchanged.
- **Pre-compute `_SHADOW_IMPORT_WHITELIST` as a frozenset**: Minor, but set lookup is faster than frozenset in some cases; profile to confirm.
