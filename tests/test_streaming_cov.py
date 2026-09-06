"""Coverage tests for streaming.py defensive/fallback branches.

These target the remaining branches in ``streaming.py`` that are reachable
through mocking, direct private-helper calls, or hand-built segmenter state:

- ``plan_peeks``'s ``except SyntaxError`` fallback
- ``_resolve_call_or_chain``'s non-``Name`` func guard
- ``_resolve_call_or_chain``'s keyword-arg-reads-earlier-assigned-name rail
- the simple-statement break in ``_next_closed``, exercised on a buffer where
  a closed simple statement is followed by a still-buffered line

The formerly "unreachable" empty-body skip and the two post-loop ``return
None`` guards were deleted outright: no input to ``_next_closed`` could ever
produce an empty-body parse or an empty slice, so they guarded nothing.
"""

from __future__ import annotations

import ast
from unittest.mock import patch

from dspy_rlm_hooks.speculation.streaming import (
    StreamSegmenter,
    _BlockState,
    _ContCounter,
    _resolve_call_or_chain,
    plan_peeks,
)

SPEC = {"llm_query"}


def test_plan_peeks_handles_unparseable_repaired_tail():
    """``repair_tail`` normally only returns parseable strings, but the
    defensive ``except SyntaxError`` branch is exercised by forcing it to
    return garbage."""
    with patch(
        "dspy_rlm_hooks.speculation.streaming.repair_tail",
        return_value="if :",
    ):
        assert plan_peeks("anything", SPEC, {}) == []


def test_resolve_call_rejects_non_name_func():
    """``_resolve_call_or_chain`` is only reached with ``Name`` funcs via
    ``_hooked_calls``, but the defensive guard is exercised directly."""
    call = ast.Call(
        func=ast.Attribute(
            value=ast.Name(id="obj", ctx=ast.Load()),
            attr="method",
            ctx=ast.Load(),
        ),
        args=[],
        keywords=[],
    )
    assert (
        _resolve_call_or_chain(call, {}, set(), "obj.method()", {}, _ContCounter())
        is None
    )


def test_plan_peeks_skips_kwarg_reading_earlier_assigned_name():
    """A keyword argument that reads a name assigned earlier in the tail is
    skipped (the assignment hasn't reached the shadow namespace yet)."""
    assert plan_peeks("y = 5\nllm_query('a', flag=y)\n", SPEC, {}) == []


def test_simple_statement_followed_by_buffered_line_emits_alone():
    """A closed simple statement with a following buffered line emits alone.

    White-box: builds a buffer where a closed simple statement is followed by
    a still-buffered line. The statement is emitted without consuming the
    follower, which is retained for the next drain.
    """
    seg = StreamSegmenter()
    blk = _BlockState(
        buf="x = 1\ny = 2\n",
        emitted_upto=0,
        stmt_index=0,
        lines=["x = 1", "y = 2"],
        scan=None,
    )
    src = seg._next_closed(blk, final=False)
    assert src == "x = 1"
    assert blk.lines == ["y = 2"]
    assert blk.emitted_upto == 6
