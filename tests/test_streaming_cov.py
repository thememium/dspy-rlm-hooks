"""Coverage tests for streaming.py defensive/fallback branches.

These target the remaining uncovered lines in ``streaming.py`` that are
reachable through mocking or direct private-helper calls:

- ``plan_peeks``'s ``except SyntaxError`` fallback (lines 599-600)
- ``_resolve_call``'s non-``Name`` func guard (line 691)
- ``_resolve_call``'s keyword-arg-reads-earlier-assigned-name rail (line 713)

The following lines are **genuinely unreachable** defensive branches and are
intentionally left uncovered (see the justification block at the bottom):

  * 173  ``if not tree.body: continue`` — ``_next_closed`` skips leading
          blank/comment lines and always returns a source whose first line is a
          real statement, so ``ast.parse`` always yields a non-empty body.
  * 234  ``if not is_compound: break`` — ``feed`` drains after every line, so a
          simple statement is always the last line of its ``_drain`` buffer and
          never has a following line to break past.
  * 262  ``if is_compound: return None`` — when ``is_compound`` and not final,
          the loop already returns at the ``open_phys or (is_compound and not
          final)`` guard.
  * 264  ``if not lines[j - 1:]: return None`` — ``j`` only advances while
          ``j < len(lines)``, so ``lines[j - 1:]`` is always non-empty.
"""

from __future__ import annotations

import ast
from unittest.mock import patch

from dspy_rlm_hooks.speculation.streaming import _resolve_call, plan_peeks

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
    """``_resolve_call`` is only reached with ``Name`` funcs via
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
    assert _resolve_call(call, {}, set(), "obj.method()") is None


def test_plan_peeks_skips_kwarg_reading_earlier_assigned_name():
    """A keyword argument that reads a name assigned earlier in the tail is
    skipped (the assignment hasn't reached the shadow namespace yet)."""
    assert plan_peeks("y = 5\nllm_query('a', flag=y)\n", SPEC, {}) == []
