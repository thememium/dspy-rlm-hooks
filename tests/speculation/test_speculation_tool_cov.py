"""Coverage-completion tests for the tool contracts (Task 1).

These tests exercise the remaining uncovered branches of ``tool.py``:
``SpeculativeTool`` base-class defaults, the ``NonSpeculated`` taint marker's
remaining dunder traps, ``contains_nonspec`` depth cutoff, ``SpecValue``
delegation operators, and the ``speculate()`` policy-override path.
"""

from __future__ import annotations

import asyncio

import pytest

from dspy_rlm_hooks.speculation.config import SpeculationPolicy
from dspy_rlm_hooks.speculation.tool import (
    NonSpeculated,
    SpeculativeTool,
    SpecValue,
    contains_nonspec,
    speculate,
)


class _FakeSpec:
    """Minimal stand-in for a ``Speculation`` future (state + wait/result)."""

    def __init__(self, value):
        self._value = value
        self.state = "done"

    def wait(self, timeout=None):
        return True

    def result(self, timeout=None):
        return self._value


# -- SpeculativeTool base hooks ------------------------------------------------


def test_speculative_tool_execute_not_implemented():
    tool = SpeculativeTool()
    with pytest.raises(NotImplementedError):
        tool.execute(1)


def test_speculative_tool_speculative_execute_delegates_to_execute():
    class MyTool(SpeculativeTool):
        def execute(self, *args, **kwargs):
            return ("exec", args, kwargs)

    tool = MyTool()
    assert tool.speculative_execute(1, 2, x=3) == ("exec", (1, 2), {"x": 3})


def test_speculative_tool_cancel_is_noop():
    tool = SpeculativeTool()
    assert tool.cancel("spec") is None


def test_speculative_tool_claim_key_returns_args_kwargs():
    tool = SpeculativeTool()
    assert tool.claim_key((1,), {"a": 2}) == ((1,), {"a": 2})


def test_speculative_tool_speculatable_call_defaults_true():
    tool = SpeculativeTool()
    assert tool.speculatable_call((), {}) is True


# -- NonSpeculated remaining dunder traps --------------------------------------


def test_nonspeculated_getattr_raises():
    m = NonSpeculated("slow")
    with pytest.raises(RuntimeError):
        m.some_attribute


async def _await_nonspec(m):
    return await m


def test_nonspeculated_await_returns_marker():
    m = NonSpeculated("slow")
    result = asyncio.run(_await_nonspec(m))
    assert isinstance(result, NonSpeculated)


def test_nonspeculated_format_raises():
    m = NonSpeculated("slow")
    with pytest.raises(RuntimeError):
        format(m)


def test_nonspeculated_iter_raises():
    m = NonSpeculated("slow")
    with pytest.raises(RuntimeError):
        iter(m)


def test_nonspeculated_getitem_raises():
    m = NonSpeculated("slow")
    with pytest.raises(RuntimeError):
        m[0]


def test_nonspeculated_radd_raises():
    m = NonSpeculated("slow")
    with pytest.raises(RuntimeError):
        1 + m


def test_nonspeculated_hash_raises():
    m = NonSpeculated("slow")
    with pytest.raises(RuntimeError):
        hash(m)


# -- contains_nonspec depth cutoff ---------------------------------------------


def test_contains_nonspec_depth_zero_returns_false():
    assert contains_nonspec([1, 2], depth=0) is False
    assert contains_nonspec(5, depth=0) is False


# -- SpecValue delegation operators --------------------------------------------


def test_spec_value_done_checks_underlying_state():
    sv = SpecValue(_FakeSpec(42))
    assert sv.done() is True

    class _Pending:
        state = "pending"

        def wait(self, timeout=None):
            return True

        def result(self, timeout=None):
            return 1

    assert SpecValue(_Pending()).done() is False


def test_spec_value_getattr_delegates_to_resolved():
    sv = SpecValue(_FakeSpec("hello"))
    assert sv.upper() == "HELLO"


def test_spec_value_str():
    sv = SpecValue(_FakeSpec(42))
    assert str(sv) == "42"


def test_spec_value_repr():
    sv = SpecValue(_FakeSpec(42))
    assert repr(sv) == "SpecValue(42)"


def test_spec_value_bool():
    assert bool(SpecValue(_FakeSpec(1))) is True
    assert bool(SpecValue(_FakeSpec(0))) is False


def test_spec_value_iter():
    sv = SpecValue(_FakeSpec([1, 2, 3]))
    assert list(sv) == [1, 2, 3]


def test_spec_value_getitem():
    sv = SpecValue(_FakeSpec([10, 20]))
    assert sv[1] == 20


def test_spec_value_radd():
    sv = SpecValue(_FakeSpec(5))
    assert 1 + sv == 6


def test_spec_value_hash():
    sv = SpecValue(_FakeSpec("abc"))
    assert hash(sv) == hash("abc")


# -- speculate() policy override path ------------------------------------------


def test_speculate_with_policy_and_kwarg_overrides():
    def my_tool(x):
        return x

    policy = SpeculationPolicy(speculatable=True, pure=True, deterministic=False)
    spec = speculate(my_tool, policy=policy, deterministic=True)
    assert spec.speculatable is True
    assert spec.pure is True
    assert spec.deterministic is True
