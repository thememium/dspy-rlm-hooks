"""Malformed model output must not trigger a silent non-streaming regeneration."""

import asyncio
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from dspy.signatures import make_signature
from dspy.streaming import StreamResponse
from dspy.utils.exceptions import AdapterParseError

from dspy_rlm_hooks.speculation.integration.streaming_turn import (
    _StreamingGenerateAction,
)


@pytest.mark.parametrize("async_call", [False, True])
@pytest.mark.parametrize("grouped", [False, True])
def test_stream_parse_failure_propagates_without_original_predict_retry(
    monkeypatch, async_call, grouped
):
    failure = AdapterParseError(
        adapter_name="ChatAdapter",
        signature=cast(Any, make_signature("task -> reasoning, code")),
        lm_response="malformed fields",
    )

    async def stream(*args, **kwargs):
        yield StreamResponse("generate_action", "code", "```python\nprint(1)\n", False)
        if grouped:
            raise ExceptionGroup("stream program failed", [failure])
        raise failure

    monkeypatch.setattr("dspy.streaming.streamify", lambda *args, **kwargs: stream)
    original = MagicMock()
    original.acall = AsyncMock()
    turn = MagicMock()
    rlm = SimpleNamespace(
        _speculation_original_generate_action=original,
        _active_stream_turn=turn,
        _streaming_fed_any=False,
    )
    wrapper = _StreamingGenerateAction(rlm)
    with pytest.raises(AdapterParseError) as caught:
        if async_call:
            asyncio.run(wrapper.acall(task="test"))
        else:
            wrapper(task="test")
    assert caught.value is failure
    assert rlm._active_stream_turn is None
    turn.feed.assert_called_once()
    original.assert_not_called()
    original.acall.assert_not_called()
