"""Tests for the SpecSession + hook factories (Task 5)."""

from __future__ import annotations

import asyncio

import pytest

from dspy_rlm_hooks.speculation import (
    NonSpeculated,
    ShadowBudgetDenied,
    SpecSession,
    SpecValue,
    ToolRegistry,
    make_shadow_hooks,
)


def _reg(**tools) -> ToolRegistry:
    reg = ToolRegistry()
    for name, (fn, kw) in tools.items():
        reg.register(name, fn, **kw)
    return reg


# -- claim hit: real call reuses the future, no re-call ------------------------


def test_claim_hit_reuses_future_no_recall():
    calls: list[str] = []

    def llm(prompt: str) -> str:
        calls.append(prompt)
        return f"result:{prompt}"

    reg = _reg(llm_query=(llm, {"speculatable": True, "pure": True}))
    session = SpecSession(reg, max_inflight=4)
    try:
        hooks = session.real_hooks()
        tool = reg.get("llm_query")
        assert tool is not None
        spec = session.launcher.dispatch(tool, ("hello",), {}, "shadow")
        assert spec is not None
        assert spec.result(timeout=5) == "result:hello"
        # the real call claims the future; fn is NOT called again
        out = hooks["llm_query"]("hello")
        assert out == "result:hello"
        assert calls == ["hello"]
        assert spec.state == "claimed"
    finally:
        session.close()


def test_claim_hit_returns_speculated_value():
    calls: list[str] = []

    def llm(prompt: str) -> str:
        calls.append(prompt)
        return f"speculated:{prompt}"

    reg = _reg(llm_query=(llm, {"speculatable": True, "pure": True}))
    session = SpecSession(reg)
    try:
        hooks = session.real_hooks()
        tool = reg.get("llm_query")
        assert tool is not None
        spec = session.launcher.dispatch(tool, ("x",), {}, "shadow")
        assert spec is not None
        spec.result(timeout=5)
        assert hooks["llm_query"]("x") == "speculated:x"
        assert calls == ["x"]
    finally:
        session.close()


# -- claim miss: runs the real tool --------------------------------------------


def test_claim_miss_runs_real():
    calls: list[str] = []

    def llm(prompt: str) -> str:
        calls.append(prompt)
        return f"result:{prompt}"

    reg = _reg(llm_query=(llm, {"speculatable": True, "pure": True}))
    session = SpecSession(reg)
    try:
        hooks = session.real_hooks()
        # no speculation dispatched -> miss -> runs the real tool
        assert hooks["llm_query"]("hello") == "result:hello"
        assert calls == ["hello"]
    finally:
        session.close()


def test_non_speculatable_passthrough():
    calls: list[str] = []

    def side_effect(x: int) -> int:
        calls.append(x)
        return x * 2

    reg = _reg(side_effect=(side_effect, {}))  # not speculatable
    session = SpecSession(reg)
    try:
        hooks = session.real_hooks()
        assert hooks["side_effect"](21) == 42
        assert calls == [21]
    finally:
        session.close()


# -- async tool ----------------------------------------------------------------


async def test_async_tool_claim_hit():
    calls: list[str] = []

    async def llm(prompt: str) -> str:
        calls.append(prompt)
        await asyncio.sleep(0.01)
        return f"result:{prompt}"

    reg = _reg(llm_query=(llm, {"speculatable": True, "pure": True}))
    session = SpecSession(reg)
    try:
        hooks = session.real_hooks()
        tool = reg.get("llm_query")
        assert tool is not None
        spec = session.launcher.dispatch(tool, ("hello",), {}, "shadow")
        assert spec is not None
        assert spec.result(timeout=5) == "result:hello"
        out = await hooks["llm_query"]("hello")
        assert out == "result:hello"
        assert calls == ["hello"]
    finally:
        session.close()


async def test_async_tool_claim_miss_runs_real():
    calls: list[str] = []

    async def llm(prompt: str) -> str:
        calls.append(prompt)
        await asyncio.sleep(0.01)
        return f"result:{prompt}"

    reg = _reg(llm_query=(llm, {"speculatable": True, "pure": True}))
    session = SpecSession(reg)
    try:
        hooks = session.real_hooks()
        out = await hooks["llm_query"]("hello")
        assert out == "result:hello"
        assert calls == ["hello"]
    finally:
        session.close()


# -- gate: gate-False -> NonSpeculated in shadow, real still runs --------------


def test_gate_false_shadow_nonspec_real_runs():
    calls: list[str] = []

    def llm(prompt: str) -> str:
        calls.append(prompt)
        return f"result:{prompt}"

    def gate(args, kwargs) -> bool:
        return args[0] != "skip"

    reg = _reg(llm_query=(llm, {"speculatable": True, "pure": True, "gate_fn": gate}))
    session = SpecSession(reg)
    try:
        shadow_hooks = make_shadow_hooks(
            reg, session.store, session.launcher, session.bus
        )
        # gate-False -> inert marker in the shadow, nothing dispatched
        marker = shadow_hooks["llm_query"]("skip")
        assert isinstance(marker, NonSpeculated)
        assert calls == []
        # the real hook still runs the real tool
        real_hooks = session.real_hooks()
        assert real_hooks["llm_query"]("skip") == "result:skip"
        assert calls == ["skip"]
    finally:
        session.close()


def test_gate_true_shadow_dispatches():
    calls: list[str] = []

    def llm(prompt: str) -> str:
        calls.append(prompt)
        return f"result:{prompt}"

    def gate(args, kwargs) -> bool:
        return args[0] != "skip"

    reg = _reg(llm_query=(llm, {"speculatable": True, "pure": True, "gate_fn": gate}))
    session = SpecSession(reg)
    try:
        shadow_hooks = make_shadow_hooks(
            reg, session.store, session.launcher, session.bus
        )
        v = shadow_hooks["llm_query"]("go")
        assert isinstance(v, SpecValue)
        assert v.resolve(timeout=5) == "result:go"
        assert calls == ["go"]
    finally:
        session.close()


# -- batched per-element claim -------------------------------------------------


def test_batched_per_element_claim():
    single_calls: list[str] = []
    batch_calls: list[list[str]] = []

    def llm(prompt: str) -> str:
        single_calls.append(prompt)
        return f"r:{prompt}"

    def llm_batched(prompts: list[str]) -> list[str]:
        batch_calls.append(prompts)
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
        # real batched call claims per-element; no misses -> batched fn not called
        out = hooks["llm_query_batched"](["a", "b"])
        assert out == ["r:a", "r:b"]
        assert single_calls == ["a", "b"]
        assert batch_calls == []
    finally:
        session.close()


def test_batched_miss_runs_real_batched_path():
    single_calls: list[str] = []
    batch_calls: list[list[str]] = []

    def llm(prompt: str) -> str:
        single_calls.append(prompt)
        return f"r:{prompt}"

    def llm_batched(prompts: list[str]) -> list[str]:
        batch_calls.append(prompts)
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
        # only "a" was speculated; "b" misses -> batched fn runs for the miss
        spec = session.launcher.dispatch(single, ("a",), {}, "shadow")
        assert spec is not None
        assert spec.result(timeout=5) == "r:a"
        out = hooks["llm_query_batched"](["a", "b"])
        assert out == ["r:a", "r:b"]
        assert single_calls == ["a"]
        assert batch_calls == [["b"]]
    finally:
        session.close()


# -- budget enforcement --------------------------------------------------------


def test_budget_dispatch_cap_denies_shadow():
    calls: list[str] = []

    def llm(prompt: str) -> str:
        calls.append(prompt)
        return f"r:{prompt}"

    reg = _reg(llm_query=(llm, {"speculatable": True, "pure": True}))
    session = SpecSession(reg, max_dispatches_per_turn=1)
    try:
        shadow_hooks = make_shadow_hooks(
            reg, session.store, session.launcher, session.bus
        )
        v = shadow_hooks["llm_query"]("a")
        assert isinstance(v, SpecValue)
        # second dispatch exceeds the per-turn cap -> hard deny
        with pytest.raises(ShadowBudgetDenied):
            shadow_hooks["llm_query"]("b")
        assert calls == ["a"]
    finally:
        session.close()


def test_budget_reset_allows_new_turn():
    calls: list[str] = []

    def llm(prompt: str) -> str:
        calls.append(prompt)
        return f"r:{prompt}"

    reg = _reg(llm_query=(llm, {"speculatable": True, "pure": True}))
    session = SpecSession(reg, max_dispatches_per_turn=1)
    try:
        shadow_hooks = make_shadow_hooks(
            reg, session.store, session.launcher, session.bus
        )
        assert isinstance(shadow_hooks["llm_query"]("a"), SpecValue)
        with pytest.raises(ShadowBudgetDenied):
            shadow_hooks["llm_query"]("b")
        session.end_turn()  # resets the per-turn dispatch counter
        v = shadow_hooks["llm_query"]("c")
        assert isinstance(v, SpecValue)
        assert v.resolve(timeout=5) == "r:c"  # force the async dispatch to finish
        assert calls == ["a", "c"]
    finally:
        session.close()


# -- session surface -----------------------------------------------------------


def test_baseline_hooks_are_unmodified():
    calls: list[str] = []

    def llm(prompt: str) -> str:
        calls.append(prompt)
        return f"result:{prompt}"

    reg = _reg(llm_query=(llm, {"speculatable": True, "pure": True}))
    session = SpecSession(reg)
    try:
        baseline = session.baseline_hooks()
        assert baseline["llm_query"]("x") == "result:x"
        assert calls == ["x"]
        # baseline never touches the store
        assert len(session.store) == 0
    finally:
        session.close()


def test_session_holds_store_launcher_budget():
    reg = _reg()
    session = SpecSession(reg, max_inflight=3, max_dispatches_per_turn=7)
    try:
        assert session.launcher.budget.max_inflight == 3
        assert session.launcher.budget.max_dispatches_per_turn == 7
        assert session.store is session.launcher.store
    finally:
        session.close()
