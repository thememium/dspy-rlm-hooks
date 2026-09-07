"""Tests for the Speculator facade (Task 6)."""

from __future__ import annotations

import pytest

from dspy_rlm_hooks.speculation import (
    SpeculativeTool,
    Speculator,
    StreamTurn,
    speculate,
)


def test_tool_decorator_registers_plain_fn():
    spec = Speculator()

    @spec.tool()
    def plain(x: int) -> int:
        return x * 2

    assert spec.registry.names() == ["plain"]
    ts = spec.registry.get("plain")
    assert ts is not None
    assert ts.fn is plain
    assert ts.speculatable is False
    assert ts.pure is False


def test_tool_decorator_speculatable_requires_pure():
    spec = Speculator()

    with pytest.raises(ValueError, match="speculatable=True requires pure=True"):

        @spec.tool(speculatable=True)
        def bad(x: int) -> int:
            return x


def test_tool_decorator_speculatable_pure_ok():
    spec = Speculator()

    @spec.tool(speculatable=True, pure=True, deterministic=True, latency_hint_ms=5.0)
    def llm(prompt: str) -> str:
        return f"r:{prompt}"

    ts = spec.registry.get("llm")
    assert ts is not None
    assert ts.speculatable is True
    assert ts.pure is True
    assert ts.deterministic is True
    assert ts.latency_hint_ms == 5.0


def test_tool_decorator_custom_name_and_gate():
    spec = Speculator()

    def gate(args, kwargs) -> bool:
        return True

    @spec.tool(name="renamed", speculatable=True, pure=True, gate=gate)
    def fn(x: int) -> int:
        return x

    assert spec.registry.names() == ["renamed"]
    ts = spec.registry.get("renamed")
    assert ts is not None
    assert ts.gate_fn is gate


def test_add_tool_spec():
    spec = Speculator()
    ts = speculate(lambda x: x + 1, speculatable=True, pure=True)
    spec.add(ts)
    assert spec.registry.names() == [ts.name]
    assert spec.registry.get(ts.name) is not None


def test_add_speculative_tool_instance():
    class Doubler(SpeculativeTool):
        name = "doubler"
        speculatable = True
        pure = True

        def execute(self, x: int) -> int:
            return x * 2

    spec = Speculator()
    spec.add(Doubler())
    assert spec.registry.names() == ["doubler"]
    ts = spec.registry.get("doubler")
    assert ts is not None
    assert ts.speculatable is True
    assert ts.pure is True


def test_hooks_returns_claiming_hooks():
    spec = Speculator()

    @spec.tool()
    def side_effect(x: int) -> int:
        return x

    hooks = spec.hooks()
    assert "side_effect" in hooks
    assert callable(hooks["side_effect"])


def test_turn_yields_stream_turn_and_ends():
    spec = Speculator()

    @spec.tool(speculatable=True, pure=True)
    def llm(prompt: str) -> str:
        return f"r:{prompt}"

    with spec.turn(repl_locals={}) as t:
        assert isinstance(t, StreamTurn)
        t.feed("x = llm('hello')\n")
    # after exit the turn is ended: unclaimed speculations evicted, budget reset
    assert spec.session.launcher.budget.dispatched_this_turn == 0


def test_turn_feeds_shadow_and_dispatches():
    spec = Speculator()

    @spec.tool(speculatable=True, pure=True)
    def llm(prompt: str) -> str:
        return f"r:{prompt}"

    with spec.turn(repl_locals={}) as t:
        t.feed("```repl\nx = llm('hello')\n```\n")
    # the shadow dispatched the call; it was never claimed -> evicted at end_turn
    stats = spec.stats()
    assert stats["speculated"] >= 1
    assert stats["evicted"] >= 1


def test_stats_counts_claimed_and_evicted():
    spec = Speculator()

    @spec.tool(speculatable=True, pure=True)
    def llm(prompt: str) -> str:
        return f"r:{prompt}"

    # dispatch a speculation and claim it via the real hook
    tool = spec.registry.get("llm")
    assert tool is not None
    spec.session.launcher.dispatch(tool, ("a",), {}, "shadow")
    hooks = spec.hooks()
    hooks["llm"]("a")
    stats = spec.stats()
    assert stats["speculated"] == 1
    assert stats["claimed"] == 1


def test_end_turn_evicts_unclaimed():
    spec = Speculator()

    @spec.tool(speculatable=True, pure=True)
    def llm(prompt: str) -> str:
        return f"r:{prompt}"

    tool = spec.registry.get("llm")
    assert tool is not None
    spec.session.launcher.dispatch(tool, ("a",), {}, "shadow")
    spec.end_turn()
    stats = spec.stats()
    assert stats["evicted"] == 1


def test_close_is_idempotent_and_noop_without_session():
    spec = Speculator()
    spec.close()  # session never built -> no-op
    spec.close()  # idempotent


def test_close_shuts_down_built_session():
    spec = Speculator()

    @spec.tool()
    def plain(x: int) -> int:
        return x

    spec.hooks()  # force session build
    spec.close()
