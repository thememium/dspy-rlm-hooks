"""Custom adapters must retain live streaming instead of silently retrying."""

from types import SimpleNamespace

import dspy
import pytest
from dspy.adapters import ChatAdapter, JSONAdapter, XMLAdapter

from dspy_rlm_hooks.speculation.integration.streaming_turn import (
    _StreamingGenerateAction,
)


def _chunk(text: str) -> SimpleNamespace:
    """A provider chunk in the shape ``StreamListener.receive`` reads.

    Built locally so the test does not depend on litellm (DSPy >=3.5 no longer
    ships it); ``receive`` only touches ``choices[0].delta.content`` and
    ``choices[0].finish_reason``.
    """
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                index=0,
                delta=SimpleNamespace(content=text),
                finish_reason=None,
            )
        ]
    )


@pytest.mark.parametrize(
    ("base", "response"),
    [
        (ChatAdapter, "[[ ## code ## ]]\nprint(1)\n[[ ## completed ## ]]"),
        (JSONAdapter, '{"code": "print(1)"}'),
        (XMLAdapter, "<code>print(1)</code>"),
    ],
)
def test_listener_accepts_adapter_subclasses(monkeypatch, base, response):
    from dspy.streaming import StreamListener

    captured = []

    def streamify(program, *, stream_listeners, **kwargs):
        captured.extend(stream_listeners)
        return lambda: None

    monkeypatch.setattr("dspy.streaming.streamify", streamify)
    predictor = dspy.Predict("question -> code")
    wrapper = _StreamingGenerateAction(
        SimpleNamespace(_speculation_original_generate_action=predictor)
    )
    assert wrapper._ensure() is not None
    listener = captured[0]
    adapter = type("CustomAdapter", (base,), {})()

    def collect(target):
        output = []
        for start in range(0, len(response), 3):
            chunk = _chunk(response[start : start + 3])
            if item := target.receive(chunk):
                output.append(item.chunk)
        if item := target.finalize():
            output.append(item.chunk)
        return "".join(output).strip()

    # DSPy versions differ in JSON string unescaping. A subclass must keep
    # exactly the same wire behavior as its built-in base on either version.
    with dspy.context(adapter=base()):
        expected = collect(StreamListener("code", predict=predictor))
    assert "print(1)" in expected
    with dspy.context(adapter=adapter):
        for _ in range(2):
            assert collect(listener) == expected


def test_listener_does_not_mutate_other_listeners(monkeypatch):
    from dspy.streaming import StreamListener

    captured = []

    def streamify(program, *, stream_listeners, **kwargs):
        captured.extend(stream_listeners)
        return lambda: None

    monkeypatch.setattr("dspy.streaming.streamify", streamify)
    wrapper = _StreamingGenerateAction(
        SimpleNamespace(_speculation_original_generate_action=dspy.Predict("q -> code"))
    )
    wrapper._ensure()
    adapter = type("RepairingChatAdapter", (ChatAdapter,), {})()
    chunk = _chunk("[[ ## code ## ]]\nprint(1)")
    with dspy.context(adapter=adapter):
        captured[0].receive(chunk)
    assert "RepairingChatAdapter" not in StreamListener("code").adapter_identifiers
