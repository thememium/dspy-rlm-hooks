"""Shadow startup + teardown overhead benchmark.

Measures where the per-iteration shadow cost goes for the speculative
execution engine:

1. ``ShadowRunner`` construction (spawn + namespace classification + payload
   transfer) vs context size — this runs once per RLM iteration today.
2. The decomposed parent-side costs: ``deepcopy`` (snapshot_ns), probe-pickle
   (_picklable_ns), and the spawn handoff.
3. ``StreamTurn.end()`` drain barrier after streaming a realistic code block.
4. Parent-side feed throughput (segments + tail peeks).

Run:  uv run python benchmarks/shadow_overhead.py
"""

from __future__ import annotations

import copy
import pickle
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dspy_rlm_hooks.speculation.session import StreamTurn
from dspy_rlm_hooks.speculation.shadow import ShadowRunner, _picklable_ns, snapshot_ns
from dspy_rlm_hooks.speculation.store import SpecStore
from dspy_rlm_hooks.speculation.streaming import StreamSegmenter


def _timeit(fn, repeats: int = 3) -> list[float]:
    out = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        out.append((time.perf_counter() - t0) * 1000)
    return out


def _report(label: str, ms: list[float]) -> None:
    print(f"{label:44s} {statistics.median(ms):8.1f} ms  (n={len(ms)})")


def bench_construction(ns: dict) -> None:
    r = ShadowRunner(ns, {}, SpecStore(), {}, launcher=None, registry=None)
    r.finish()
    r.join(5)


def bench_decomposed(ns: dict) -> None:
    """Attribute construction cost to its stages (parent side only)."""
    _report("  snapshot_ns deepcopy", _timeit(lambda: snapshot_ns(ns)))
    snap = snapshot_ns(ns)
    _report("  _picklable_ns probe-pickle", _timeit(lambda: _picklable_ns(snap)))
    _report("  plain deepcopy of ns", _timeit(lambda: copy.deepcopy(ns)))
    _report("  full pickle.dumps(ns)", _timeit(lambda: pickle.dumps(ns, -1)))


STMT = (
    "import re",
    'data = llm_query("summarize section {i}")',
    "clean = data.strip().lower()",
    "",
)


def bench_turn_end(ns: dict, n_stmts: int = 30) -> None:
    def run() -> tuple[float, float]:
        shadow = ShadowRunner(ns, {}, SpecStore(), {}, launcher=None, registry=None)
        turn = StreamTurn(StreamSegmenter(), shadow, peek=True)
        t0 = time.perf_counter()
        turn.feed("```repl\n")
        for i in range(n_stmts):
            for tmpl in STMT:
                turn.feed(tmpl.replace("{i}", str(i)) + "\n")
        t_feed = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        turn.end(timeout=5.0)
        t_end = (time.perf_counter() - t0) * 1000
        return t_feed, t_end

    results = [run() for _ in range(3)]
    _report(f"  feed {n_stmts} stmts (parent side)", [r[0] for r in results])
    _report("  StreamTurn.end() drain barrier", [r[1] for r in results])


def main() -> None:
    print("== Shadow construction vs context size (spawn + classify + transfer) ==")
    for mb in (0.1, 1.0, 5.0, 25.0):
        ns = {"context": "x" * int(mb * 1024 * 1024), "question": "q"}
        ms = []
        for _ in range(3):
            t0 = time.perf_counter()
            bench_construction(ns)
            ms.append((time.perf_counter() - t0) * 1000)
        _report(f"ShadowRunner({mb:g} MB ctx)", ms)

    print("\n== Decomposed parent-side cost (5 MB ctx) ==")
    ns5 = {"context": "x" * (5 * 1024 * 1024), "question": "q"}
    bench_decomposed(ns5)

    print("\n== StreamTurn lifecycle (5 MB ctx) ==")
    bench_turn_end(ns5)

    print(
        "\n== Warm-worker upside: 20 iterations x construction above = per-RLM saving =="
    )


if __name__ == "__main__":
    main()
