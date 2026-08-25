"""Tests for the speculative execution engine contracts (Task 1)."""

from __future__ import annotations

import asyncio

import pytest

from dspy_rlm_hooks.speculation import (
    NonSpeculated,
    SpecKey,
    SpeculationConfig,
    SpeculationPolicy,
    SpeculativeTool,
    SpecValue,
    ToolSpec,
    canonical_hash,
    contains_nonspec,
    spec_key,
    speculate,
)

# -- SpeculationConfig defaults ----------------------------------------------


def test_speculation_config_defaults():
    cfg = SpeculationConfig()
    assert cfg.enabled is True
    assert cfg.max_inflight == 8
    assert cfg.max_dispatches_per_turn == 2048
    assert cfg.speculate_llm_query is True
    assert cfg.speculate_llm_query_batched is True
    assert cfg.speculate_user_tools is False
    assert cfg.timeout_s == 5.0


def test_speculation_config_overrides():
    cfg = SpeculationConfig(enabled=False, max_inflight=2, timeout_s=1.5)
    assert cfg.enabled is False
    assert cfg.max_inflight == 2
    assert cfg.timeout_s == 1.5


# -- SpeculationPolicy -------------------------------------------------------


def test_speculation_policy_defaults():
    p = SpeculationPolicy()
    assert p.speculatable is False
    assert p.pure is False
    assert p.deterministic is False
    assert p.latency_hint_ms == 1000.0
    assert p.gate is None


def test_speculation_policy_fields():
    gate = lambda args, kwargs: True  # noqa: E731
    p = SpeculationPolicy(
        speculatable=True,
        pure=True,
        deterministic=True,
        latency_hint_ms=42.0,
        gate=gate,
    )
    assert p.speculatable is True
    assert p.pure is True
    assert p.deterministic is True
    assert p.latency_hint_ms == 42.0
    assert p.gate is gate


# -- speculatable without pure raises ----------------------------------------


def test_tool_spec_speculatable_without_pure_raises():
    with pytest.raises(ValueError):
        ToolSpec(name="t", fn=lambda: 1, speculatable=True, pure=False)


def test_tool_spec_speculatable_with_pure_ok():
    spec = ToolSpec(name="t", fn=lambda: 1, speculatable=True, pure=True)
    assert spec.speculatable is True
    assert spec.pure is True


def test_speculate_speculatable_without_pure_raises():
    with pytest.raises(ValueError):
        speculate(lambda: 1, speculatable=True, pure=False)


# -- canonical_hash ----------------------------------------------------------


def test_canonical_hash_deterministic():
    a = canonical_hash("tool", (1, 2), {"x": 3})
    b = canonical_hash("tool", (1, 2), {"x": 3})
    assert a == b
    assert len(a) == 16


def test_canonical_hash_differs_on_inputs():
    assert canonical_hash("tool", (1,), {}) != canonical_hash("tool", (2,), {})
    assert canonical_hash("tool", (1,), {}) != canonical_hash("tool", (1,), {"x": 1})


def test_canonical_hash_kwarg_order_irrelevant():
    assert canonical_hash("tool", (), {"a": 1, "b": 2}) == canonical_hash(
        "tool", (), {"b": 2, "a": 1}
    )


# -- spec_key ----------------------------------------------------------------


def test_spec_key_deterministic():
    spec = ToolSpec(name="t", fn=lambda: None)
    k1 = spec_key(spec, (1, 2), {"x": 3})
    k2 = spec_key(spec, (1, 2), {"x": 3})
    assert k1 == k2
    assert isinstance(k1, tuple)
    assert k1[0] == "t"
    assert len(k1[1]) == 16


def test_spec_key_uses_key_fn():
    def key_fn(args, kwargs):
        return args[0]

    spec = ToolSpec(name="t", fn=lambda: None, key_fn=key_fn)
    assert spec_key(spec, (1, "ignored"), {}) == spec_key(spec, (1, "different"), {})


def test_spec_key_type_alias():
    k: SpecKey = ("t", "abc")
    assert k == ("t", "abc")


# -- NonSpeculated taint -----------------------------------------------------


def test_nonspeculated_storing_is_safe():
    m = NonSpeculated("slow_tool")
    assert isinstance(m, NonSpeculated)


def test_nonspeculated_use_raises():
    m = NonSpeculated("slow_tool")
    with pytest.raises(RuntimeError):
        str(m)
    with pytest.raises(RuntimeError):
        bool(m)
    with pytest.raises(RuntimeError):
        m + 1
    with pytest.raises(RuntimeError):
        m == 1


def test_contains_nonspec_deep():
    m = NonSpeculated("slow_tool")
    assert contains_nonspec(m) is True
    assert contains_nonspec([1, {"a": m}]) is True
    assert contains_nonspec({"k": (m,)}) is True
    assert contains_nonspec([1, 2, 3]) is False
    assert contains_nonspec({"a": 1}) is False


# -- speculate() helper ------------------------------------------------------


def test_speculate_builds_tool_spec():
    def my_tool(x):
        return x

    spec = speculate(my_tool, speculatable=True, pure=True)
    assert isinstance(spec, ToolSpec)
    assert spec.name == "my_tool"
    assert spec.fn is my_tool
    assert spec.speculatable is True
    assert spec.pure is True


def test_speculate_with_policy():
    def my_tool(x):
        return x

    policy = SpeculationPolicy(speculatable=True, pure=True, deterministic=True)
    spec = speculate(my_tool, policy=policy)
    assert spec.speculatable is True
    assert spec.pure is True
    assert spec.deterministic is True


def test_speculate_defaults_not_speculatable():
    def my_tool(x):
        return x

    spec = speculate(my_tool)
    assert spec.speculatable is False
    assert spec.pure is False


# -- SpecValue lazy proxy ----------------------------------------------------


class _FakeSpec:
    def __init__(self, value):
        self._value = value
        self.state = "done"

    def wait(self, timeout=None):
        return True

    def result(self, timeout=None):
        return self._value


def test_spec_value_resolves_on_use():
    sv = SpecValue(_FakeSpec(42))
    assert sv.resolve() == 42
    assert sv + 1 == 43
    assert sv == 42


def test_spec_value_await():
    sv = SpecValue(_FakeSpec(7))
    assert asyncio.run(_await_it(sv)) == 7


async def _await_it(sv):
    return await sv


# -- SpeculativeTool ---------------------------------------------------------


def test_speculative_tool_to_spec():
    class MyTool(SpeculativeTool):
        name = "my_tool"
        speculatable = True
        pure = True

        def execute(self, x):
            return x

    spec = MyTool().to_spec()
    assert spec.name == "my_tool"
    assert spec.speculatable is True
    assert spec.pure is True
    assert spec.fn is not None
