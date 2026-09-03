"""Overlap benchmark: does speculation actually hide tool latency under
main-model generation?

Simulates one RLM iteration against the REAL machinery (StreamSegmenter,
StreamTurn, Launcher, SpecStore, claiming hooks):

- a scripted fake streaming LM feeds a canned code block chunk-by-chunk with
  small delays (token-arrival pacing);
- fake tools sleep a fixed latency;
- "real execution" then claims through make_real_hooks, exactly as the
  integration layer does.

Reported: baseline (no speculation) vs speculated wall time, claim hit rate,
and tool-latency hidden under generation.

Run:  uv run python benchmarks/overlap_sim.py
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dspy_rlm_hooks.speculation.session import SpecSession, ToolRegistry

TOOL_LATENCY_S = 0.3
STREAM_PACE_S = 0.004  # per chunk, simulates token arrival

CODE = r"""```repl
import re
sections = context.split("\n\n")
scores = []
for s in sections[:12]:
    scores.append(llm_query("score: " + s[:80]))
best = sorted(scores)[-1]
print(best)
```
"""


def make_registry(calls: list) -> ToolRegistry:
    reg = ToolRegistry()
    lock = threading.Lock()

    def llm_query(prompt: str) -> str:
        with lock:
            calls.append(time.perf_counter())
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


def stream_code(turn, code: str) -> None:
    """Feed the code like a token stream: 8-char chunks, paced."""
    for i in range(0, len(code), 8):
        turn.feed(code[i : i + 8])
        time.sleep(STREAM_PACE_S)


def run_speculated() -> dict:
    reg = make_registry([])
    session = SpecSession(reg)
    turn = session.begin_stream_turn({"context": "a\n\nb\n\nc"}, {}, peek=True)
    t0 = time.perf_counter()
    stream_code(turn, CODE)
    t_streamed = time.perf_counter() - t0
    turn.end(timeout=5.0)
    t_gen_end = time.perf_counter()

    hooks = session.real_hooks()
    t0 = time.perf_counter()
    hooks["llm_query"](prompt="score: a")
    t_real = time.perf_counter() - t0
    session.end_turn()
    session.close()

    hits = sum(1 for e in session.bus.history if e[0] == "claim_hit")
    misses = sum(1 for e in session.bus.history if e[0] == "claim_miss")
    return {
        "stream_s": t_streamed,
        "gen_end_to_real_done_s": t_real,
        "total_s": t_gen_end - t0 + t_real,
        "hits": hits,
        "misses": misses,
        "first_call_wait_s": t_real,
    }


def run_baseline() -> float:
    reg = make_registry([])
    session = SpecSession(reg)
    hooks = session.baseline_hooks()
    t0 = time.perf_counter()
    hooks["llm_query"]("score: a")
    total = time.perf_counter() - t0
    session.close()
    return total


def main() -> None:
    base = run_baseline()
    spec = run_speculated()
    hidden_ms = (base - spec["first_call_wait_s"]) * 1000
    print(f"baseline real call (no speculation):   {base * 1000:8.1f} ms")
    print(f"speculated: streamed+drain+claim:      {spec['total_s'] * 1000:8.1f} ms")
    print(f"  stream window:                       {spec['stream_s'] * 1000:8.1f} ms")
    print(
        f"  real-path claim wait:                {spec['first_call_wait_s'] * 1000:8.1f} ms"
    )
    print(f"  tool latency hidden under streaming: {hidden_ms:8.1f} ms")
    print(f"  claim hits/misses:                   {spec['hits']}/{spec['misses']}")


if __name__ == "__main__":
    main()
