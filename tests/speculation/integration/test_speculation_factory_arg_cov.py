"""Coverage tests for the DSPy >=3.4 ``interpreter_factory`` forward branches.

DSPy 3.4 appends a trailing ``interpreter_factory`` argument to the calls into
``_execute_iteration``/``_aexecute_iteration``. The speculation wrappers accept
it and forward it to the saved original only when present. Covers the
forward-when-present branch on both the sync and async wrappers.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from dspy_rlm_hooks.speculation.integration.execute import (
    _speculation_aexecute_iteration,
    _speculation_execute_iteration,
)

_FACTORY = lambda: "interpreter"  # noqa: E731


def test_sync_wrapper_forwards_seventh_positional():
    seen = {}

    def orig_execute(self, *args):
        # Plain function attribute: unbound, so "repl" arrives as ``self``.
        seen["call"] = (self, *args)
        return "sync-result"

    rlm = SimpleNamespace(_speculation_original_execute_iteration=orig_execute)
    out = _speculation_execute_iteration(
        rlm, "repl", [], "hist", 0, {}, ["answer"], _FACTORY
    )
    assert out == "sync-result"
    assert len(seen["call"]) == 7
    assert seen["call"][0] == "repl" and seen["call"][-1] is _FACTORY


@pytest.mark.asyncio
async def test_async_wrapper_forwards_seventh_positional():
    seen = {}

    async def orig_aexecute(self, *args):
        # Plain function attribute: unbound, so "repl" arrives as ``self``.
        seen["call"] = (self, *args)
        return "async-result"

    rlm = SimpleNamespace(_speculation_original_aexecute_iteration=orig_aexecute)
    out = await _speculation_aexecute_iteration(
        rlm, "repl", [], "hist", 0, {}, ["answer"], _FACTORY
    )
    assert out == "async-result"
    assert len(seen["call"]) == 7
    assert seen["call"][0] == "repl" and seen["call"][-1] is _FACTORY
