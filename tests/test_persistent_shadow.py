"""Warm persistent shadow runner + lazy turn start.

- one shadow subprocess is reused across turns (reset per turn, respawned on
  crash) instead of spawning per iteration;
- a StreamTurn starts the shadow only when a speculatable call actually
  appears in the stream, so call-free iterations pay zero shadow cost.
"""

from __future__ import annotations

import time

from dspy_rlm_hooks.speculation.session import SpecSession, ToolRegistry


def _registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(
        "llm_query",
        lambda prompt: f"r:{prompt}",
        speculatable=True,
        pure=True,
        deterministic=False,
        latency_hint_ms=1000.0,
    )
    return reg


def _feed(turn, code: str) -> None:
    turn.feed(f"```repl\n{code}\n```\n")


def test_lazy_turn_never_spawns_without_spec_calls():
    session = SpecSession(_registry())
    try:
        turn = session.begin_stream_turn({"context": "abc"}, {}, peek=True)
        _feed(turn, "x = 1 + 2\ny = x * 3\nprint(y)")
        turn.end(timeout=5.0)
        assert session._warm_runner is None  # the shadow never started
    finally:
        session.close()


def test_lazy_turn_spawns_on_spec_call_and_claims():
    session = SpecSession(_registry())
    try:
        turn = session.begin_stream_turn({"context": "abc"}, {}, peek=True)
        _feed(turn, "x = llm_query('hello')")
        turn.end(timeout=5.0)
        assert session._warm_runner is not None
        hooks = session.real_hooks()
        out = hooks["llm_query"](prompt="hello")
        assert out == "r:hello"
        hits = [e for e in session.bus.history if e[0] == "claim_hit"]
        assert hits, "expected the real call to claim the speculated result"
    finally:
        session.close()


def test_persistent_runner_resets_ns_between_turns():
    session = SpecSession(_registry())
    try:
        # turn 1: spawn the shadow (a call is needed to trigger it) and then
        # mutate the shadow namespace
        turn1 = session.begin_stream_turn({"ctx": "seed"}, {}, peek=True)
        _feed(turn1, "x = llm_query('t1')\nctx = 'mutated in shadow'")
        turn1.end(timeout=5.0)
        assert session._warm_runner is not None

        # turn 2: the reset must restore the seed; a peek over code that reads
        # ctx must plan against the SEED value, not turn 1's mutation
        turn2 = session.begin_stream_turn({"ctx": "seed"}, {}, peek=True)
        _feed(turn2, "y = llm_query(ctx)")
        turn2.end(timeout=5.0)
        dispatches = [m for m in session.bus.history if m[0] == "dispatch"]
        assert dispatches, "expected a peek dispatch in turn 2"

        # the reader thread stores dispatches asynchronously: poll briefly
        deadline = time.monotonic() + 5.0
        specs: list = []
        while time.monotonic() < deadline:
            specs = [
                s
                for s in session.store.all
                if s.key[0] == "llm_query" and s.source == "peek"
            ]
            if any(s.args == ("seed",) for s in specs):
                break
            time.sleep(0.05)
        assert any(s.args == ("seed",) for s in specs), (
            f"peek planned against stale shadow state: {[s.args for s in specs]}"
        )
    finally:
        session.close()


def test_persistent_runner_respawns_after_crash():
    session = SpecSession(_registry())
    try:
        turn1 = session.begin_stream_turn({}, {}, peek=True)
        _feed(turn1, "x = llm_query('a')")
        turn1.end(timeout=5.0)
        runner = session._warm_runner
        assert runner is not None
        runner.abort("test-kill")
        assert not runner.is_alive

        turn2 = session.begin_stream_turn({}, {}, peek=True)
        _feed(turn2, "y = llm_query('b')")
        turn2.end(timeout=5.0)
        hooks = session.real_hooks()
        assert hooks["llm_query"](prompt="b") == "r:b"
    finally:
        session.close()


def test_turn_end_drains_worker_before_return():
    session = SpecSession(_registry())
    try:
        turn = session.begin_stream_turn({}, {}, peek=True)
        _feed(turn, "x = llm_query('drain-me')")
        # end() must leave every dispatch in the store (drained), so the real
        # run can claim it immediately
        turn.end(timeout=5.0)
        live = [
            s
            for s in session.store.all
            if s.key[0] == "llm_query" and s.state in ("pending", "running", "ready")
        ]
        assert live, "expected the speculation to be dispatched (and drained)"
    finally:
        session.close()


def test_legacy_session_spawns_eagerly():
    session = SpecSession(_registry(), persistent_shadow=False)
    try:
        turn = session.begin_stream_turn({}, {}, peek=True)
        assert turn.shadow is not None  # eager spawn, legacy behavior
        assert turn.shadow.persistent is False
        turn.end(timeout=5.0)
    finally:
        session.close()
