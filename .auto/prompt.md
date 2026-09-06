# Autoresearch: RLM Speculation Speed

## Objective
Optimize the speculative execution engine for DSPy's RLM (Recursive Language Model) to minimize total inference overhead. The goal is to make speculation a net win — the speculated path should be FASTER than the baseline (no speculation), not slower.

Current state: speculation ADDS ~87ms overhead (389ms vs 302ms baseline). The shadow construction is ~70ms, drain barrier ~10ms, and only 28ms of tool latency is hidden under streaming. The claim wait is 274ms (the real execution waits for speculations to finish).

## Metrics
- **Primary**: `speculated_ms` (ms, lower is better) — total wall time with speculation enabled
- **Secondary**: `baseline_ms`, `stream_ms`, `drain_ms`, `claim_wait_ms`, `hidden_ms`, `claim_hits`, `construct_5mb_ms`, `drain_barrier_ms`

## How to Run
`./.auto/measure.sh` — outputs `METRIC name=number` lines.

## Files in Scope
- `src/dspy_rlm_hooks/speculation/shadow.py` — Subprocess-based speculative execution (ShadowRunner, worker process, namespace classification)
- `src/dspy_rlm_hooks/speculation/session.py` — Session management, StreamTurn, Launcher, EventBus, LatencyStats
- `src/dspy_rlm_hooks/speculation/streaming.py` — Token stream segmentation, tail peeking, plan_peeks
- `src/dspy_rlm_hooks/speculation/hooks.py` — Hook factories for claiming/dispatching (make_real_hooks, make_shadow_hooks)
- `src/dspy_rlm_hooks/speculation/store.py` — SpecStore, Speculation future management
- `src/dspy_rlm_hooks/speculation/budget.py` — Budget enforcement
- `src/dspy_rlm_hooks/speculation_integration.py` — Integration with DSPy's RLM execution path
- `benchmarks/overlap_sim.py` — Overlap benchmark (simulates one RLM iteration)
- `benchmarks/shadow_overhead.py` — Shadow construction overhead benchmark

## Off Limits
- `src/dspy_rlm_hooks/patcher.py` — Core hook patching (stable, don't break)
- `src/dspy_rlm_hooks/types.py` — Type definitions
- `src/dspy_rlm_hooks/__init__.py` — Public API surface
- Test files — only modify if a test is genuinely wrong

## Constraints
- All 700 tests must pass (`uv run pytest tests/ -v`)
- No new dependencies (use only stdlib + existing deps)
- Speculation correctness: claimed results must be byte-identical to real execution
- The shadow subprocess isolation boundary must be preserved (security)
- Async tools must keep working (asyncio event loop integration)

## What's Been Tried
(Starting fresh — no experiments yet)

## Key Architectural Insights
1. **Shadow subprocess cost**: ~70ms to spawn + classify + transfer namespace. Persistent shadow reuses across turns but still spawns once per RLM run.
2. **Drain barrier**: StreamTurn.end() waits for worker to process all messages (~10ms). This blocks real execution.
3. **Claim wait dominance**: 274ms of the 389ms total is waiting for speculations to resolve. This means the speculation is NOT hiding latency — it's adding it.
4. **Low hit rate**: Only 1 claim hit in the benchmark. The shadow dispatches but the real execution doesn't benefit.
5. **Pipe overhead**: Every segment/peek goes through multiprocessing.Pipe — serialization + context switch per message.

## Optimization Ideas (to explore)
1. **Thread-based shadow** instead of subprocess: eliminates spawn overhead, uses shared memory directly. Trade-off: weaker isolation (but tools are pure by definition).
2. **Batch pipe messages**: send multiple segments/peek plans in one message instead of one-at-a-time.
3. **Overlap drain with real execution start**: begin real execution before drain completes (risky but could save 10ms).
4. **Reduce namespace serialization**: only pickle values that changed since last turn.
5. **Pre-dispatch before streaming starts**: begin speculation as soon as the RLM iteration starts, not after first tokens arrive.
6. **Increase speculation parallelism**: dispatch more speculations concurrently (currently capped at max_inflight=8).
7. **Speculative claim without wait**: return immediately if speculation is still running, let the real execution proceed and claim later.
8. **Cache-friendly namespace transfer**: use shared memory (multiprocessing.shared_memory) instead of pickle+pipe for large namespaces.
