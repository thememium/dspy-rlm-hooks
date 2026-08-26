"""Integration tests for token-streaming speculative execution (sPTC).

Covers the streaming path that feeds the model's streamed ``code`` output into
the speculation shadow during ``generate_action`` (the RLM's ``dspy.Predict``):

- begin_streaming_turn stashes an active turn before generation
- streamed code deltas -> shadow dispatches -> real exec claims (no re-call)
- streaming turn begun but no content -> top-up with the assembled block
- sync/async iteration wrappers begin the turn and delegate
- _StreamingGenerateAction streams code deltas (sync + async) into the turn
- streaming disabled leaves generate_action unwrapped (Lazy/JIT preserved)
- disable restores generate_action + iteration methods
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from dspy.primitives.prediction import Prediction
from dspy.streaming import StreamResponse

from dspy_rlm_hooks import enable_rlm_speculation
from dspy_rlm_hooks.speculation_integration import (
    _maybe_begin_streaming_turn,
    _StreamingGenerateAction,
    disable_rlm_speculation,
)


def _make_execute(tools):
    def execute(code, variables=None):
        ns = dict(tools)
        ns.update(variables or {})
        exec(code, ns, ns)
        return "ok"

    return execute


def _real_execute_code(repl, code, input_args):
    return repl.execute(code, variables=dict(input_args))


def _setup_real(mock_rlm, mock_repl, tools):
    mock_rlm.max_llm_calls = 50
    mock_repl.tools = tools
    mock_repl.execute = _make_execute(tools)
    mock_rlm._execute_code = _real_execute_code
    return mock_rlm, mock_repl


# ---------------------------------------------------------------------------
# begin_streaming_turn
# ---------------------------------------------------------------------------


def test_begin_streaming_turn_stashes_turn(mock_rlm, mock_repl):
    _setup_real(mock_rlm, mock_repl, {"llm_query": lambda p: f"r:{p}"})
    enable_rlm_speculation(mock_rlm)
    assert mock_rlm._active_stream_turn is None

    _maybe_begin_streaming_turn(mock_rlm, mock_repl, {"doc": "x"})
    assert mock_rlm._active_stream_turn is not None
    assert mock_rlm._streaming_fed_any is False


def test_begin_streaming_turn_noop_when_disabled(mock_rlm, mock_repl):
    _setup_real(mock_rlm, mock_repl, {"llm_query": lambda p: f"r:{p}"})
    enable_rlm_speculation(mock_rlm, streaming=False)
    _maybe_begin_streaming_turn(mock_rlm, mock_repl, {})
    assert mock_rlm._active_stream_turn is None


# ---------------------------------------------------------------------------
# streaming claim
# ---------------------------------------------------------------------------


def test_streaming_claim_no_recall(mock_rlm, mock_repl):
    real_calls = []

    def llm_query(prompt):
        real_calls.append(prompt)
        return f"r:{prompt}"

    _setup_real(mock_rlm, mock_repl, {"llm_query": llm_query})
    enable_rlm_speculation(mock_rlm, max_inflight=4)

    # Simulate the generate wrapper having streamed the code field into the turn.
    _maybe_begin_streaming_turn(mock_rlm, mock_repl, {})
    turn = mock_rlm._active_stream_turn
    turn.feed("```repl\nx = llm_query('hello')\n```\n")
    mock_rlm._streaming_fed_any = True

    mock_rlm._execute_code(mock_repl, "x = llm_query('hello')\n", {})
    # Dispatched once by the shadow during streaming, claimed (not re-called).
    assert real_calls == ["hello"]


def test_streaming_top_up_when_no_content(mock_rlm, mock_repl):
    real_calls = []

    def llm_query(prompt):
        real_calls.append(prompt)
        return f"r:{prompt}"

    _setup_real(mock_rlm, mock_repl, {"llm_query": llm_query})
    enable_rlm_speculation(mock_rlm, max_inflight=4)

    # Turn begun but no code deltas streamed (cache-hit / no-fence): the
    # execute path tops up with the full assembled block -> still speculates.
    _maybe_begin_streaming_turn(mock_rlm, mock_repl, {})
    mock_rlm._execute_code(mock_repl, "x = llm_query('hello')\n", {})
    assert real_calls == ["hello"]


def test_streaming_parallel_dispatch(mock_rlm, mock_repl):
    import time

    latency = 0.3
    real_calls = []

    def llm_query(prompt):
        real_calls.append(prompt)
        time.sleep(latency)
        return f"r:{prompt}"

    _setup_real(mock_rlm, mock_repl, {"llm_query": llm_query})
    enable_rlm_speculation(mock_rlm, max_inflight=8)

    _maybe_begin_streaming_turn(mock_rlm, mock_repl, {})
    turn = mock_rlm._active_stream_turn
    code = "\n".join(f"x{i} = llm_query('q{i}')" for i in range(8))
    turn.feed(f"```repl\n{code}\n```\n")
    mock_rlm._streaming_fed_any = True

    t0 = time.perf_counter()
    mock_rlm._execute_code(mock_repl, code, {})
    elapsed = time.perf_counter() - t0

    assert len(real_calls) == 8
    # Dispatched in parallel during streaming -> well under the 8x serial time.
    assert elapsed < 8 * latency


# ---------------------------------------------------------------------------
# iteration wrappers
# ---------------------------------------------------------------------------


def test_sync_iteration_wrapper_begins_turn(
    mock_rlm, mock_repl, mock_variables, mock_history
):
    _setup_real(mock_rlm, mock_repl, {"llm_query": lambda p: f"r:{p}"})
    enable_rlm_speculation(mock_rlm, streaming=True)
    inner = MagicMock(return_value=mock_history)
    mock_rlm._speculation_original_execute_iteration = inner

    mock_rlm._execute_iteration(
        mock_repl, mock_variables, mock_history, 0, {"question": "t"}, ["answer"]
    )
    assert mock_rlm._active_stream_turn is not None
    inner.assert_called_once()


@pytest.mark.asyncio
async def test_async_iteration_wrapper_begins_turn(
    mock_rlm, mock_repl, mock_variables, mock_history
):
    from unittest.mock import AsyncMock

    _setup_real(mock_rlm, mock_repl, {"llm_query": lambda p: f"r:{p}"})
    enable_rlm_speculation(mock_rlm, streaming=True)
    inner = AsyncMock(return_value=mock_history)
    mock_rlm._speculation_original_aexecute_iteration = inner

    await mock_rlm._aexecute_iteration(
        mock_repl, mock_variables, mock_history, 0, {"question": "t"}, ["answer"]
    )
    assert mock_rlm._active_stream_turn is not None
    inner.assert_awaited_once()


# ---------------------------------------------------------------------------
# _StreamingGenerateAction
# ---------------------------------------------------------------------------


def _fake_streamify(items, pred):
    def _factory(program, *, stream_listeners, async_streaming):
        if async_streaming:

            async def astr(*a, **k):
                for it in items:
                    yield it
                yield pred

            return astr

        def sstr(*a, **k):
            yield from items
            yield pred

        return sstr

    return _factory


def test_generate_action_streams_code_deltas(monkeypatch, mock_rlm):
    fed = []

    class FakeTurn:
        def feed(self, chunk):
            fed.append(chunk)

    pred = Prediction(reasoning="r", code="x = llm_query('hi')")
    items = [
        StreamResponse("ga", "code", "```repl\n", False),
        StreamResponse("ga", "code", "x = llm_query('hi')\n", False),
        StreamResponse("ga", "code", "```\n", True),
    ]
    monkeypatch.setattr("dspy.streaming.streamify", _fake_streamify(items, pred))

    def orig(*a, **k):
        raise AssertionError("original should not be called when streaming works")

    mock_rlm._speculation_original_generate_action = orig
    mock_rlm._active_stream_turn = FakeTurn()
    mock_rlm._streaming_fed_any = False

    wrapper = _StreamingGenerateAction(mock_rlm)
    result = wrapper("variables_info", "hist", "1")
    assert result is pred
    assert "".join(fed) == "```repl\nx = llm_query('hi')\n```\n"
    assert mock_rlm._streaming_fed_any is True


@pytest.mark.asyncio
async def test_generate_action_acall_streams_code_deltas(monkeypatch, mock_rlm):
    fed = []

    class FakeTurn:
        def feed(self, chunk):
            fed.append(chunk)

    pred = Prediction(reasoning="r", code="x = llm_query('hi')")
    items = [
        StreamResponse("ga", "code", "```repl\n", False),
        StreamResponse("ga", "code", "x = llm_query('hi')\n", False),
        StreamResponse("ga", "code", "```\n", True),
    ]
    monkeypatch.setattr("dspy.streaming.streamify", _fake_streamify(items, pred))

    async def orig_acall(*a, **k):
        raise AssertionError("original acall should not be called when streaming works")

    mock_rlm._speculation_original_generate_action = MagicMock()
    mock_rlm._speculation_original_generate_action.acall = orig_acall
    mock_rlm._active_stream_turn = FakeTurn()
    mock_rlm._streaming_fed_any = False

    wrapper = _StreamingGenerateAction(mock_rlm)
    result = await wrapper.acall("variables_info", "hist", "1")
    assert result is pred
    assert "".join(fed) == "```repl\nx = llm_query('hi')\n```\n"
    assert mock_rlm._streaming_fed_any is True


def test_generate_action_falls_back_on_stream_failure(monkeypatch, mock_rlm):
    def orig(*a, **k):
        return "orig-result"

    mock_rlm._speculation_original_generate_action = orig
    mock_rlm._active_stream_turn = MagicMock()
    mock_rlm._streaming_fed_any = False

    def boom(*a, **k):
        raise RuntimeError("streamify boom")

    monkeypatch.setattr("dspy.streaming.streamify", boom)
    wrapper = _StreamingGenerateAction(mock_rlm)
    assert wrapper("variables_info", "hist", "1") == "orig-result"
    assert mock_rlm._active_stream_turn is None  # cleared -> Lazy/JIT fallback


# ---------------------------------------------------------------------------
# enable/disable wiring
# ---------------------------------------------------------------------------


def test_streaming_disabled_leaves_generate_action_unwrapped(mock_rlm, mock_repl):
    _setup_real(mock_rlm, mock_repl, {"llm_query": lambda p: f"r:{p}"})
    ga = mock_rlm.generate_action
    enable_rlm_speculation(mock_rlm, streaming=False)
    assert mock_rlm.generate_action is ga
    assert mock_rlm._speculation_config.streaming is False


def test_streaming_enabled_wraps_generate_action(mock_rlm, mock_repl):
    _setup_real(mock_rlm, mock_repl, {"llm_query": lambda p: f"r:{p}"})
    ga = mock_rlm.generate_action
    enable_rlm_speculation(mock_rlm, streaming=True)
    assert mock_rlm.generate_action is not ga
    assert isinstance(mock_rlm.generate_action, _StreamingGenerateAction)


def test_disable_restores_generate_action_and_iteration(mock_rlm, mock_repl):
    _setup_real(mock_rlm, mock_repl, {"llm_query": lambda p: f"r:{p}"})
    orig_ga = mock_rlm.generate_action
    orig_iter = mock_rlm._execute_iteration
    orig_aiter = mock_rlm._aexecute_iteration

    enable_rlm_speculation(mock_rlm, streaming=True)
    assert mock_rlm.generate_action is not orig_ga
    assert mock_rlm._execute_iteration is not orig_iter

    disable_rlm_speculation(mock_rlm)
    assert mock_rlm.generate_action is orig_ga
    assert mock_rlm._execute_iteration is orig_iter
    assert mock_rlm._aexecute_iteration is orig_aiter
    assert not hasattr(mock_rlm, "_active_stream_turn")
    assert not hasattr(mock_rlm, "_streaming_fed_any")
