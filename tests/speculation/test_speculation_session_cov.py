"""Coverage-completion tests for the speculation session (session.py).

These tests exercise the branches of ``session.py`` that the main
``test_speculation_hooks.py`` suite does not reach: the ``_bind_call``
signature-fallback, ``dispatch_or_adopt`` determinism/adopt paths,
``ensure_peeked`` top-up and budget-deny, the dispatch ``cancel_fn`` /
evicted / ``wants_spec`` / error paths, loop shutdown cancellation, the
``_serve_loop`` teardown, and the ``StreamTurn`` peek/end paths. They reuse the
same ``ToolRegistry`` + ``SpecSession`` fixtures and mocking style as the
existing suite and never call a real LLM.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading
import time

import pytest

from dspy_rlm_hooks.speculation import (
    EventBus,
    Launcher,
    SpecSession,
    SpecStore,
    ToolRegistry,
)
from dspy_rlm_hooks.speculation.session import StreamTurn, _bind_call, _serve_loop
from dspy_rlm_hooks.speculation.streaming import StreamSegmenter


def _reg(**tools) -> ToolRegistry:
    reg = ToolRegistry()
    for name, (fn, kw) in tools.items():
        reg.register(name, fn, **kw)
    return reg


# -- _bind_call signature fallback (lines 52-57) ------------------------------


def test_bind_call_fallback_on_unbindable_signature():
    """A **kwargs-only tool bound positionally falls back to __argN keys."""
    out = _bind_call(lambda **kw: kw, ("x",), {"y": 2})
    assert out == {"__arg0": "x", "y": 2}


def test_dispatch_fallback_binds_positional_to_arg_keys():
    """Dispatch of a **kwargs-only tool with positional args uses the fallback."""

    def kwargs_only(**kw):
        return kw["__arg0"]

    reg = _reg(kwargs_only=(kwargs_only, {"speculatable": True, "pure": True}))
    session = SpecSession(reg)
    try:
        tool = reg.get("kwargs_only")
        assert tool is not None
        spec = session.launcher.dispatch(tool, ("x",), {}, "shadow")
        assert spec is not None
        assert spec.result(timeout=5) == "x"
    finally:
        session.close()


# -- dispatch_or_adopt deterministic reuse (lines 149-152) --------------------


def test_dispatch_or_adopt_reuses_deterministic_existing():
    """A deterministic tool's existing speculation is adopted, not re-dispatched."""

    def det(x: str) -> str:
        return f"r:{x}"

    reg = _reg(det=(det, {"speculatable": True, "pure": True, "deterministic": True}))
    session = SpecSession(reg)
    try:
        tool = reg.get("det")
        assert tool is not None
        spec = session.launcher.dispatch(tool, ("a",), {}, "shadow")
        assert spec is not None
        assert spec.result(timeout=5) == "r:a"
        adopted = session.launcher.dispatch_or_adopt(tool, ("a",), {}, "shadow")
        assert adopted is spec
        assert adopted.adopted is True
    finally:
        session.close()


# -- dispatch_or_adopt adopt path (lines 155-156) -----------------------------


def test_dispatch_or_adopt_takes_peek():
    """A peek speculation is adopted (not re-dispatched) by dispatch_or_adopt."""

    def tool(x: str) -> str:
        return f"r:{x}"

    reg = _reg(tool=(tool, {"speculatable": True, "pure": True}))
    session = SpecSession(reg)
    try:
        t = reg.get("tool")
        assert t is not None
        peek = session.launcher.dispatch(t, ("a",), {}, "peek")
        assert peek is not None
        assert peek.result(timeout=5) == "r:a"
        adopted = session.launcher.dispatch_or_adopt(t, ("a",), {}, "shadow")
        assert adopted is peek
        assert adopted.adopted is True
    finally:
        session.close()


# -- ensure_peeked deterministic (lines 167-169) ------------------------------


def test_ensure_peeked_deterministic_existing_returns_zero():
    def det(x: str) -> str:
        return f"r:{x}"

    reg = _reg(det=(det, {"speculatable": True, "pure": True, "deterministic": True}))
    session = SpecSession(reg)
    try:
        tool = reg.get("det")
        assert tool is not None
        spec = session.launcher.dispatch(tool, ("a",), {}, "shadow")
        assert spec is not None
        assert spec.result(timeout=5) == "r:a"
        assert session.launcher.ensure_peeked(tool, ("a",), {}, 5) == 0
    finally:
        session.close()


def test_ensure_peeked_deterministic_missing_dispatches_one():
    def det(x: str) -> str:
        return f"r:{x}"

    reg = _reg(det=(det, {"speculatable": True, "pure": True, "deterministic": True}))
    session = SpecSession(reg)
    try:
        tool = reg.get("det")
        assert tool is not None
        # no existing spec for ("b",) -> needed forced to 1, one dispatch
        n = session.launcher.ensure_peeked(tool, ("b",), {}, 5)
        assert n == 1
    finally:
        session.close()


# -- ensure_peeked budget denied (line 173) -----------------------------------


def test_ensure_peeked_budget_denied_breaks():
    def tool(x: str) -> str:
        return f"r:{x}"

    reg = _reg(tool=(tool, {"speculatable": True, "pure": True}))
    session = SpecSession(reg, max_dispatches_per_turn=0)
    try:
        t = reg.get("tool")
        assert t is not None
        assert session.launcher.ensure_peeked(t, ("a",), {}, 1) == 0
    finally:
        session.close()


# -- dispatch cancel_fn (lines 212-213) ---------------------------------------


def test_dispatch_wires_cancel_fn():
    cancelled: list[int] = []

    def cancel(spec):
        cancelled.append(spec.seq)

    def tool(x: str) -> str:
        return f"r:{x}"

    reg = _reg(tool=(tool, {"speculatable": True, "pure": True, "cancel_fn": cancel}))
    session = SpecSession(reg)
    try:
        t = reg.get("tool")
        assert t is not None
        spec = session.launcher.dispatch(t, ("a",), {}, "shadow")
        assert spec is not None
        assert spec.cancel is not None
        assert spec.result(timeout=5) == "r:a"
    finally:
        session.close()


# -- dispatch evicted-before-run (lines 219-220) ------------------------------


def test_dispatch_evicted_before_run_short_circuits():
    """A spec evicted before its worker runs resolves immediately as evicted."""

    def tool(x: str) -> str:
        return f"r:{x}"

    reg = _reg(tool=(tool, {"speculatable": True, "pure": True}))
    session = SpecSession(reg, max_inflight=1)
    try:
        launcher = session.launcher
        # occupy the single worker so the next dispatch is queued
        block = threading.Event()

        def blocker():
            block.wait()
            return "blocked"

        launcher._pool.submit(blocker)
        t = reg.get("tool")
        assert t is not None
        spec = launcher.dispatch(t, ("a",), {}, "shadow")
        assert spec is not None
        session.store.evict_unclaimed("test")  # evict before the worker runs
        block.set()
        assert spec.wait(timeout=5) is True
        assert spec.state == "evicted"
    finally:
        session.close()


# -- dispatch wants_spec (line 233) -------------------------------------------


def test_dispatch_wants_spec_passes_spec():
    def spec_fn(*args, **kwargs):
        return f"spec:{kwargs['_spec'].seq}"

    setattr(spec_fn, "wants_spec", True)

    reg = _reg(tool=(spec_fn, {"speculatable": True, "pure": True, "spec_fn": spec_fn}))
    session = SpecSession(reg)
    try:
        t = reg.get("tool")
        assert t is not None
        spec = session.launcher.dispatch(t, ("a",), {}, "shadow")
        assert spec is not None
        assert spec.result(timeout=5) == "spec:1"
    finally:
        session.close()


# -- dispatch error path (lines 245-247) --------------------------------------


def test_dispatch_records_error_and_fails():
    def boom(x: str) -> str:
        raise ValueError("boom")

    reg = _reg(boom=(boom, {"speculatable": True, "pure": True}))
    session = SpecSession(reg)
    try:
        t = reg.get("boom")
        assert t is not None
        spec = session.launcher.dispatch(t, ("a",), {}, "shadow")
        assert spec is not None
        with pytest.raises(ValueError, match="boom"):
            spec.result(timeout=5)
        assert spec.state == "failed"
    finally:
        session.close()


# -- shutdown cancels in-flight coroutines (lines 275, 277) -------------------


def test_shutdown_cancels_pending_loop_task():
    launcher = Launcher(SpecStore(), EventBus())
    loop = launcher.loop()

    async def forever():
        await asyncio.sleep(3600)

    fut = asyncio.run_coroutine_threadsafe(forever(), loop)
    time.sleep(0.1)  # let the task register on the loop
    launcher.shutdown()
    with pytest.raises(concurrent.futures.CancelledError):
        fut.result(timeout=2)


# -- _serve_loop teardown (lines 289-290) --------------------------------------


def test_serve_loop_teardown_on_closed_loop():
    """A closed loop makes run_forever raise; the finally still cleans up."""
    import warnings

    loop = asyncio.new_event_loop()
    loop.close()  # run_forever raises immediately -> finally runs
    t = threading.Thread(target=_serve_loop, args=(loop,))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        t.start()
        t.join(timeout=2)
    assert not t.is_alive()


# -- StreamTurn.feed peek path (lines 309-310) --------------------------------


def test_stream_turn_feed_peeks_tail():
    from unittest.mock import MagicMock

    seg = StreamSegmenter()
    shadow = MagicMock()
    shadow.hooks = {"llm_query": object()}
    shadow._last_peek_tally = {}
    turn = StreamTurn(seg, shadow, peek=True)
    # an incomplete statement leaves a non-empty pending tail containing a hook
    turn.feed("```repl\nx = llm_query('a")
    shadow.feed_peek.assert_called_once()
    assert turn._last_tail == "x = llm_query('a"


# -- _peek_worthwhile branches (lines 313-320) --------------------------------


def test_peek_worthwhile_too_long():
    from unittest.mock import MagicMock

    shadow = MagicMock()
    shadow.hooks = {"llm_query": object()}
    shadow._last_peek_tally = {}
    turn = StreamTurn(StreamSegmenter(), shadow, peek=True)
    assert turn._peek_worthwhile("x" * 12_001) is False


def test_peek_worthwhile_hook_in_changed():
    from unittest.mock import MagicMock

    shadow = MagicMock()
    shadow.hooks = {"llm_query": object()}
    shadow._last_peek_tally = {}
    turn = StreamTurn(StreamSegmenter(), shadow, peek=True)
    turn._last_tail = "x = "
    # tail starts with last_tail -> changed is the suffix containing the hook
    assert turn._peek_worthwhile("x = llm_query('a") is True


def test_peek_worthwhile_falls_back_to_tally():
    from unittest.mock import MagicMock

    shadow = MagicMock()
    shadow.hooks = {}  # no hook name matches
    shadow._last_peek_tally = {"k": 1}
    turn = StreamTurn(StreamSegmenter(), shadow, peek=True)
    assert turn._peek_worthwhile("plain = 1") is True


def test_peek_worthwhile_false_when_no_hook_and_no_tally():
    from unittest.mock import MagicMock

    shadow = MagicMock()
    shadow.hooks = {}
    shadow._last_peek_tally = {}
    turn = StreamTurn(StreamSegmenter(), shadow, peek=True)
    assert turn._peek_worthwhile("plain = 1") is False


# -- StreamTurn.end (line 324) ------------------------------------------------


def test_stream_turn_end_feeds_finished_segments():
    from unittest.mock import MagicMock

    seg = StreamSegmenter()
    shadow = MagicMock()
    turn = StreamTurn(seg, shadow, peek=False)
    turn.feed("```repl\nx = 1")  # open block, no trailing newline
    turn.end()
    shadow.feed.assert_called()
    shadow.finish.assert_called_once()
    shadow.join.assert_called_once()
    shadow.abort.assert_called_once_with("turn_end")
