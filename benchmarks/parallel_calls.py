"""Parallel tool call benchmark: does speculation hide latency when multiple
tools are called in sequence?

This tests the real-world scenario where the RLM code calls llm_query
multiple times in a loop. Speculation should dispatch all calls during
streaming, so the real execution can claim them all without waiting.

Run:  uv run python benchmarks/parallel_calls.py
"""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dspy_rlm_hooks.speculation.session import SpecSession, ToolRegistry

TOOL_LATENCY_S = float(os.environ.get("TOOL_LATENCY_S", "0.1"))  # 100ms per call
STREAM_PACE_S = float(os.environ.get("STREAM_PACE_S", "0.004"))  # per chunk
NUM_CALLS = int(os.environ.get("NUM_CALLS", "6"))  # number of llm_query calls

CODE_TEMPLATE = r"""```repl
import re
sections = context.split("\n\n")
scores = []
{calls}
best = sorted(scores)[-1]
print(best)
```
"""


def make_registry() -> ToolRegistry:
    reg = ToolRegistry()
    lock = threading.Lock()
    execs = [0]

    def llm_query(prompt: str) -> str:
        with lock:
            execs[0] += 1
        time.sleep(TOOL_LATENCY_S)
        return f"result:{prompt}"

    reg.register(
        "llm_query",
        llm_query,
        speculatable=True,
        pure=True,
        deterministic=False,
        latency_hint_ms=TOOL_LATENCY_S * 1000,
    )
    return reg


def make_code() -> str:
    calls = "\n".join(
        f'scores.append(llm_query("score: " + sections[{i}][:80]))'
        for i in range(NUM_CALLS)
    )
    return CODE_TEMPLATE.format(calls=calls)


def stream_code(turn, code: str) -> None:
    for i in range(0, len(code), 8):
        turn.feed(code[i : i + 8])
        time.sleep(STREAM_PACE_S)


def run_speculated() -> dict:
    reg = make_registry()
    session = SpecSession(reg)
    context = "\n\n".join(f"section_{i}" for i in range(NUM_CALLS))
    turn = session.begin_stream_turn({"context": context}, {}, peek=True)
    code = make_code()

    t0 = time.perf_counter()
    stream_code(turn, code)
    t_stream = time.perf_counter() - t0

    turn.end(timeout=5.0)
    t_drain = time.perf_counter() - t0 - t_stream

    hooks = session.real_hooks()
    # Make all real calls (simulating the RLM's execution)
    results = []
    t_exec_start = time.perf_counter()
    for i in range(NUM_CALLS):
        result = hooks["llm_query"](prompt=f"score: section_{i}")
        results.append(result)
    t_exec = time.perf_counter() - t_exec_start

    total = time.perf_counter() - t0
    session.end_turn()
    session.close()

    hits = sum(1 for e in session.bus.history if e[0] == "claim_hit")
    misses = sum(1 for e in session.bus.history if e[0] == "claim_miss")
    return {
        "stream_s": t_stream,
        "drain_s": t_drain,
        "exec_s": t_exec,
        "total_s": total,
        "hits": hits,
        "misses": misses,
    }


def run_baseline() -> float:
    reg = make_registry()
    session = SpecSession(reg)
    hooks = session.baseline_hooks()
    t0 = time.perf_counter()
    for i in range(NUM_CALLS):
        hooks["llm_query"](prompt=f"score: section_{i}")
    total = time.perf_counter() - t0
    session.close()
    return total


def main() -> None:
    base = run_baseline()
    spec = run_speculated()
    print(f"baseline ({NUM_CALLS} sequential calls):    {base * 1000:8.1f} ms")
    print(f"speculated total:                    {spec['total_s'] * 1000:8.1f} ms")
    print(f"  stream window:                     {spec['stream_s'] * 1000:8.1f} ms")
    print(f"  drain:                             {spec['drain_s'] * 1000:8.1f} ms")
    print(f"  execution (all claims):            {spec['exec_s'] * 1000:8.1f} ms")
    print(f"  claim hits/misses:                 {spec['hits']}/{spec['misses']}")
    speedup = base / spec["total_s"] if spec["total_s"] > 0 else 0
    print(f"  speedup:                           {speedup:.2f}x")


if __name__ == "__main__":
    main()
