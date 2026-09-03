"""Chained continuations: calls whose args depend on a speculated
predecessor's result fire automatically when the producer resolves, pipelining
whole dataflow chains under the model's still-streaming output."""

from __future__ import annotations

import time

from dspy_rlm_hooks.speculation.session import SpecSession, ToolRegistry

CODE_CHAIN = """```repl
doc = fetch("auth")
summary = llm_query("summarize: " + doc)
```
"""


def _registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(
        "fetch",
        lambda q: f"doc:{q}",
        speculatable=True,
        pure=True,
        deterministic=True,
        latency_hint_ms=50.0,
    )
    reg.register(
        "llm_query",
        lambda prompt: f"r:{prompt}",
        speculatable=True,
        pure=True,
        deterministic=False,
        latency_hint_ms=50.0,
    )
    return reg


def _feed(turn, code: str) -> None:
    for i in range(0, len(code), 6):
        turn.feed(code[i : i + 6])
        time.sleep(0.001)


def _wait_for(predicate, timeout: float = 10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        out = predicate()
        if out:
            return out
        time.sleep(0.02)
    return predicate()


def test_chain_fires_when_producer_resolves():
    session = SpecSession(_registry())
    try:
        turn = session.begin_stream_turn({}, {}, peek=True)
        _feed(turn, CODE_CHAIN)
        turn.end(timeout=5.0)

        hooks = session.real_hooks()
        # the producer resolves when the real run claims it
        out_doc = hooks["fetch"]("auth")
        assert out_doc == "doc:auth"

        # the chained llm_query("summarize: doc:auth") must have been
        # dispatched automatically; the real run claims it without re-calling
        summary_spec = _wait_for(
            lambda: next(
                (
                    s
                    for s in session.store.all
                    if s.key[0] == "llm_query"
                    and s.args == ("summarize: doc:auth",)
                    and s.state in ("pending", "running", "ready", "claimed")
                ),
                None,
            )
        )
        assert summary_spec is not None, (
            f"chained call never dispatched: {[(s.key[0], s.args) for s in session.store.all]}"
        )
        out_summary = hooks["llm_query"](prompt="summarize: doc:auth")
        assert out_summary == "r:summarize: doc:auth"

        llm_executions = [s for s in session.store.all if s.key[0] == "llm_query"]
        # the chained call dispatched exactly once (no duplicate from re-peeks)
        assert len(llm_executions) == 1
    finally:
        session.close()


def test_three_level_chain():
    reg = _registry()
    reg.register(
        "rank",
        lambda text: f"ranked:{text}",
        speculatable=True,
        pure=True,
        deterministic=True,
        latency_hint_ms=50.0,
    )
    session = SpecSession(reg)
    try:
        code = """```repl
doc = fetch("auth")
summary = llm_query("summarize: " + doc)
final = rank(summary)
```
"""
        turn = session.begin_stream_turn({}, {}, peek=True)
        _feed(turn, code)
        turn.end(timeout=5.0)

        hooks = session.real_hooks()
        _wait_for(lambda: session.store.all)
        assert hooks["fetch"]("auth") == "doc:auth"
        _wait_for(
            lambda: any(
                s.key[0] == "llm_query" and s.args == ("summarize: doc:auth",)
                for s in session.store.all
            )
        )
        assert (
            hooks["llm_query"](prompt="summarize: doc:auth") == "r:summarize: doc:auth"
        )
        _wait_for(
            lambda: any(
                s.key[0] == "rank" and s.args == ("ranked:r:summarize: doc:auth",)
                for s in session.store.all
            ),
        )
        out = hooks["rank"]("r:summarize: doc:auth")
        assert out == "ranked:r:summarize: doc:auth"
    finally:
        session.close()


def test_unrelated_stale_arg_still_skipped():
    """A call whose arg reads a tail-assigned name with NO producer is still
    skipped (the staleness rail is not loosened by chain support)."""
    from dspy_rlm_hooks.speculation.streaming import plan_peeks, plan_peeks_with_chains

    tail = "x = unknown_tool()" + chr(10) + "y = llm_query(x)"
    plans, chains, metas = plan_peeks_with_chains(tail, {"llm_query"}, {})
    # unknown_tool is not a spec name -> no producer -> no plan, no chain
    assert plans == []
    assert chains == [] and metas == []
    assert plan_peeks(tail, {"llm_query"}, {}) == []
