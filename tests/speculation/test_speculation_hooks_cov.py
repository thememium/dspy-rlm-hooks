"""Coverage-completion tests for the speculation hook factories (hooks.py).

These tests exercise the branches of ``hooks.py`` that the main
``test_speculation_hooks.py`` suite does not reach: the ``_NullBus`` no-op
path, ``deep_force`` container handling, the ``_single_of`` fallback, the
self-claim guard, the async non-speculatable / batched paths, the
not-yet-done ``_await_spec`` branch, shadow taint, and the shadow batched
dispatch. They reuse the same ``ToolRegistry`` + ``SpecSession`` fixtures and
mocking style as the existing suite and never call a real LLM.
"""

from __future__ import annotations

import asyncio

import pytest

from dspy_rlm_hooks.speculation import (
    NonSpeculated,
    SpecSession,
    SpecValue,
    ToolRegistry,
    make_real_hooks,
    make_shadow_hooks,
)
from dspy_rlm_hooks.speculation.guards import mark_current
from dspy_rlm_hooks.speculation.hooks import deep_force


def _reg(**tools) -> ToolRegistry:
    reg = ToolRegistry()
    for name, (fn, kw) in tools.items():
        reg.register(name, fn, **kw)
    return reg


# -- _NullBus.emit (line 47) --------------------------------------------------


def test_real_hook_without_bus_uses_null_bus():
    """A real hook built with no ``bus`` emits through the no-op ``_NullBus``."""
    calls: list[str] = []

    def llm(prompt: str) -> str:
        calls.append(prompt)
        return f"r:{prompt}"

    reg = _reg(llm_query=(llm, {"speculatable": True, "pure": True}))
    session = SpecSession(reg)
    try:
        # no bus passed -> _NULL_BUS; a claim miss emits through it (line 47)
        hooks = make_real_hooks(reg, session.store, session.launcher)
        assert hooks["llm_query"]("hello") == "r:hello"
        assert calls == ["hello"]
    finally:
        session.close()


# -- deep_force (lines 57, 59, 61, 63, 65) ------------------------------------


def test_deep_force_resolves_spec_value():
    """deep_force resolves a top-level SpecValue to its concrete value."""
    calls: list[str] = []

    def llm(prompt: str) -> str:
        calls.append(prompt)
        return f"r:{prompt}"

    reg = _reg(llm_query=(llm, {"speculatable": True, "pure": True}))
    session = SpecSession(reg)
    try:
        tool = reg.get("llm_query")
        assert tool is not None
        spec = session.launcher.dispatch(tool, ("x",), {}, "shadow")
        assert spec is not None
        assert spec.result(timeout=5) == "r:x"
        assert deep_force(SpecValue(spec)) == "r:x"
    finally:
        session.close()


def test_deep_force_depth_zero_returns_obj():
    """At depth 0 deep_force returns the object untouched."""
    assert deep_force("plain", depth=0) == "plain"


def test_deep_force_list_tuple_dict():
    """deep_force recurses through list/tuple/dict containers."""
    assert deep_force([1, [2]]) == [1, [2]]
    assert deep_force((1, (2,))) == (1, (2,))
    assert deep_force({"a": {"b": 1}}) == {"a": {"b": 1}}


# -- _single_of fallback (line 82) --------------------------------------------


def test_batched_single_of_falls_back_to_batched_tool():
    """A batched tool with no matching single tool falls back to itself."""
    calls: list[list[str]] = []

    def llm_batched(prompts: list[str]) -> list[str]:
        calls.append(prompts)
        return [f"r:{p}" for p in prompts]

    # only the batched tool is registered; there is no single "llm_query"
    reg = _reg(llm_query_batched=(llm_batched, {"speculatable": True, "pure": True}))
    session = SpecSession(reg)
    try:
        hooks = session.real_hooks()
        out = hooks["llm_query_batched"](["a", "b"])
        assert out == ["r:a", "r:b"]
        assert calls == [["a", "b"]]
    finally:
        session.close()


# -- self-claim guard in _claim_or_run (line 172) -----------------------------


def test_real_hook_self_claim_runs_raw_tool():
    """When the claimed spec is the current worker's own, run the raw tool."""
    calls: list[str] = []

    def llm(prompt: str) -> str:
        calls.append(prompt)
        return f"r:{prompt}"

    reg = _reg(llm_query=(llm, {"speculatable": True, "pure": True}))
    session = SpecSession(reg)
    try:
        hooks = session.real_hooks()
        tool = reg.get("llm_query")
        assert tool is not None
        spec = session.launcher.dispatch(tool, ("hello",), {}, "shadow")
        assert spec is not None
        assert spec.result(timeout=5) == "r:hello"
        # simulate the worker that must resolve `spec` claiming it
        mark_current(spec)
        try:
            assert hooks["llm_query"]("hello") == "r:hello"
        finally:
            mark_current(None)
        # the dispatch already ran the tool once; the self-claim guard runs it
        # again via the raw fn instead of waiting on its own future
        assert calls == ["hello", "hello"]
    finally:
        session.close()


# -- async non-speculatable passthrough (line 201) ----------------------------


async def test_async_non_speculatable_passthrough():
    """An async tool that is not speculatable is awaited directly."""
    calls: list[str] = []

    async def side_effect(x: str) -> str:
        calls.append(x)
        await asyncio.sleep(0.01)
        return f"ran:{x}"

    reg = _reg(side_effect=(side_effect, {}))  # not speculatable
    session = SpecSession(reg)
    try:
        hooks = session.real_hooks()
        assert await hooks["side_effect"]("x") == "ran:x"
        assert calls == ["x"]
    finally:
        session.close()


# -- async batched claim path (lines 203-241) ---------------------------------


async def test_async_batched_claim_and_miss():
    """Async batched hook claims ready elements and runs the batched fn for misses."""
    single_calls: list[str] = []
    batch_calls: list[list[str]] = []

    async def llm(prompt: str) -> str:
        single_calls.append(prompt)
        await asyncio.sleep(0.01)
        return f"r:{prompt}"

    async def llm_batched(prompts: list[str]) -> list[str]:
        batch_calls.append(prompts)
        await asyncio.sleep(0.01)
        return [f"r:{p}" for p in prompts]

    reg = _reg(
        llm_query=(llm, {"speculatable": True, "pure": True}),
        llm_query_batched=(llm_batched, {"speculatable": True, "pure": True}),
    )
    session = SpecSession(reg)
    try:
        hooks = session.real_hooks()
        single = reg.get("llm_query")
        assert single is not None
        # only "a" is speculated; "b" misses -> batched fn runs for the miss
        spec = session.launcher.dispatch(single, ("a",), {}, "shadow")
        assert spec is not None
        assert spec.result(timeout=5) == "r:a"
        out = await hooks["llm_query_batched"](["a", "b"])
        assert out == ["r:a", "r:b"]
        assert single_calls == ["a"]
        assert batch_calls == [["b"]]
    finally:
        session.close()


async def test_async_batched_all_claimed_no_misses():
    """Async batched hook with every element claimed skips the miss fill."""
    single_calls: list[str] = []
    batch_calls: list[list[str]] = []

    async def llm(prompt: str) -> str:
        single_calls.append(prompt)
        await asyncio.sleep(0.01)
        return f"r:{prompt}"

    async def llm_batched(prompts: list[str]) -> list[str]:
        batch_calls.append(prompts)
        await asyncio.sleep(0.01)
        return [f"r:{p}" for p in prompts]

    reg = _reg(
        llm_query=(llm, {"speculatable": True, "pure": True}),
        llm_query_batched=(llm_batched, {"speculatable": True, "pure": True}),
    )
    session = SpecSession(reg)
    try:
        hooks = session.real_hooks()
        single = reg.get("llm_query")
        assert single is not None
        for p in ("a", "b"):
            spec = session.launcher.dispatch(single, (p,), {}, "shadow")
            assert spec is not None
            assert spec.result(timeout=5) == f"r:{p}"
        # both elements claimed -> no misses -> batched fn never called
        out = await hooks["llm_query_batched"](["a", "b"])
        assert out == ["r:a", "r:b"]
        assert single_calls == ["a", "b"]
        assert batch_calls == []
    finally:
        session.close()


# -- _await_spec not-yet-done branch (line 266) -------------------------------


async def test_async_claim_waits_for_inflight_spec():
    """Claiming an in-flight async spec waits via to_thread (not-yet-done)."""
    calls: list[str] = []

    async def llm(prompt: str) -> str:
        calls.append(prompt)
        await asyncio.sleep(0.2)
        return f"r:{prompt}"

    reg = _reg(llm_query=(llm, {"speculatable": True, "pure": True}))
    session = SpecSession(reg)
    try:
        hooks = session.real_hooks()
        tool = reg.get("llm_query")
        assert tool is not None
        spec = session.launcher.dispatch(tool, ("hello",), {}, "shadow")
        assert spec is not None
        # do NOT wait for the spec first: the hook must block on the in-flight
        # future via _await_spec's to_thread branch
        assert await hooks["llm_query"]("hello") == "r:hello"
        assert calls == ["hello"]
    finally:
        session.close()


# -- shadow taint (line 294) --------------------------------------------------


def test_shadow_hook_taint_raises():
    """A shadow call whose args depend on a non-speculated result raises."""

    def llm(prompt: str) -> str:
        return f"r:{prompt}"

    reg = _reg(llm_query=(llm, {"speculatable": True, "pure": True}))
    session = SpecSession(reg)
    try:
        shadow = make_shadow_hooks(reg, session.store, session.launcher, session.bus)
        with pytest.raises(RuntimeError, match="depend on a non-speculated result"):
            shadow["llm_query"](NonSpeculated("other"))
    finally:
        session.close()


# -- shadow batched dispatch (lines 300-307) ----------------------------------


def test_shadow_batched_dispatches_per_element():
    """Shadow batched hook dispatches one speculation per element."""

    def llm(prompt: str) -> str:
        return f"r:{prompt}"

    def llm_batched(prompts: list[str]) -> list[str]:
        return [f"r:{p}" for p in prompts]

    reg = _reg(
        llm_query=(llm, {"speculatable": True, "pure": True}),
        llm_query_batched=(llm_batched, {"speculatable": True, "pure": True}),
    )
    session = SpecSession(reg)
    try:
        shadow = make_shadow_hooks(reg, session.store, session.launcher, session.bus)
        out = shadow["llm_query_batched"](["a", "b"])
        assert isinstance(out, list)
        assert all(isinstance(v, SpecValue) for v in out)
        assert [v.resolve(timeout=5) for v in out] == ["r:a", "r:b"]
    finally:
        session.close()


def test_shadow_batched_budget_denied_yields_none():
    """When the per-turn budget is exhausted, a batched element yields None."""

    def llm(prompt: str) -> str:
        return f"r:{prompt}"

    def llm_batched(prompts: list[str]) -> list[str]:
        return [f"r:{p}" for p in prompts]

    reg = _reg(
        llm_query=(llm, {"speculatable": True, "pure": True}),
        llm_query_batched=(llm_batched, {"speculatable": True, "pure": True}),
    )
    session = SpecSession(reg, max_dispatches_per_turn=1)
    try:
        shadow = make_shadow_hooks(reg, session.store, session.launcher, session.bus)
        out = shadow["llm_query_batched"](["a", "b"])
        assert isinstance(out[0], SpecValue)
        assert out[1] is None
    finally:
        session.close()


# -- zero claim-wait budget: sync batched hedge (line 409) ---------------------


class _SlowQueueLauncher:
    """Latency-aware launcher whose deep queue zeroes every claim wait budget."""

    latency_aware = True
    max_claim_wait_s = 30.0

    def __init__(self) -> None:
        from types import SimpleNamespace

        self.budget = SimpleNamespace(max_inflight=1)

    def ewma_ms(self, name: str, hint: float) -> float:
        return 5000.0

    def queued_depth(self) -> int:
        return 100


def _pending_spec(session, single, prompt):
    """Insert a never-started (pending) speculation for one prompt."""
    from dspy_rlm_hooks.speculation.store import Speculation
    from dspy_rlm_hooks.speculation.tool import spec_key

    spec = Speculation(
        key=spec_key(single, (prompt,), {}),
        seq=session.launcher.next_seq(),
        args=(prompt,),
        kwargs={},
        source="shadow",
        state="pending",
    )
    session.store.put(spec)
    return spec


def test_sync_batched_claim_zero_budget_hedges_element():
    """A claimed element whose wait budget is 0 raises TimeoutError inside
    _wait_one, is evicted, and is hedged through the real batched fn."""
    batch_calls: list[list[str]] = []

    def llm(prompt: str) -> str:
        return f"r:{prompt}"

    def llm_batched(prompts: list[str]) -> list[str]:
        batch_calls.append(list(prompts))
        return [f"r:{p}" for p in prompts]

    reg = _reg(
        llm_query=(llm, {"speculatable": True, "pure": True}),
        llm_query_batched=(llm_batched, {"speculatable": True, "pure": True}),
    )
    session = SpecSession(reg)
    try:
        single = reg.get("llm_query")
        assert single is not None
        spec = _pending_spec(session, single, "a")
        hooks = make_real_hooks(reg, session.store, _SlowQueueLauncher())
        out = hooks["llm_query_batched"](["a", "b"])
        assert out == ["r:a", "r:b"]
        assert spec.state == "evicted"
        # misses run first ("b"), then the hedged claimed element ("a")
        assert batch_calls == [["b"], ["a"]]
    finally:
        session.close()


async def test_async_batched_claim_zero_budget_hedges_element():
    """The async batched hook hedges an element whose claim budget is 0."""
    batch_calls: list[list[str]] = []

    async def llm(prompt: str) -> str:
        return f"r:{prompt}"

    async def llm_batched(prompts: list[str]) -> list[str]:
        batch_calls.append(list(prompts))
        return [f"r:{p}" for p in prompts]

    reg = _reg(
        llm_query=(llm, {"speculatable": True, "pure": True}),
        llm_query_batched=(llm_batched, {"speculatable": True, "pure": True}),
    )
    session = SpecSession(reg)
    try:
        single = reg.get("llm_query")
        assert single is not None
        spec = _pending_spec(session, single, "a")
        hooks = make_real_hooks(reg, session.store, _SlowQueueLauncher())
        out = await hooks["llm_query_batched"](["a", "b"])
        assert out == ["r:a", "r:b"]
        assert spec.state == "evicted"
        assert batch_calls == [["b"], ["a"]]
    finally:
        session.close()
