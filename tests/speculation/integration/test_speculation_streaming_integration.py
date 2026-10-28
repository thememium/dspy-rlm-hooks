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

from unittest.mock import AsyncMock, MagicMock

import pytest
from dspy.primitives.prediction import Prediction
from dspy.streaming import StreamResponse

from dspy_rlm_hooks import enable_rlm_speculation
from dspy_rlm_hooks.speculation.integration import (
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


class _SubLMRouter:
    """sub_lm stand-in: routes prompts to the recorded tool double (counter-
    free speculative llm executions call sub_lm directly)."""

    def __init__(self, tools: dict) -> None:
        self._tools = dict(tools)  # snapshot the RAW fns (repl.tools gets hooks)

    def __call__(self, prompt: str):
        fn = self._tools.get("llm_query")
        if fn is None:
            return [{"text": f"[no llm_query] {prompt}"}]
        return [{"text": fn(prompt)}]


def _setup_real(mock_rlm, mock_repl, tools):
    mock_rlm.max_llm_calls = 50
    mock_repl.tools = tools
    mock_repl.execute = _make_execute(tools)
    mock_rlm._execute_code = _real_execute_code
    mock_rlm.sub_lm = _SubLMRouter(tools)
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
    print(
        "STORE:",
        [
            (s.key[0], s.args, s.state, repr(s._result)[:60], s.error)
            for s in mock_rlm._speculator.session.store.all
        ],
    )
    print("sub_lm type:", type(mock_rlm.sub_lm).__name__)
    print(
        "BUS:",
        [
            (k, v.get("tool") or v.get("msg", ""))
            for k, v in mock_rlm._speculator.session.bus.history
        ],
    )
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
# _run_async_to_completion
# ---------------------------------------------------------------------------


def test_run_async_to_completion_from_sync_context(monkeypatch):
    """The sync path owns the loop and preserves the active context."""
    import contextvars

    from dspy_rlm_hooks.speculation.integration.streaming_turn import (
        _run_async_to_completion,
    )

    marker = contextvars.ContextVar("marker")
    marker.set("kept")

    async def main():
        return marker.get()

    assert _run_async_to_completion(main) == "kept"


@pytest.mark.asyncio
async def test_run_async_to_completion_from_running_loop():
    """A sync call inside an async caller runs on a joined private loop."""
    import contextvars

    from dspy_rlm_hooks.speculation.integration.streaming_turn import (
        _run_async_to_completion,
    )

    marker = contextvars.ContextVar("marker")
    marker.set("kept")

    async def main():
        return marker.get()

    assert _run_async_to_completion(main) == "kept"


# ---------------------------------------------------------------------------
# _ThreadedProgram
# ---------------------------------------------------------------------------


def test_threaded_program_exposes_wrapped_predictor():
    from dspy_rlm_hooks.speculation.integration.streaming_turn import _ThreadedProgram

    def predictor():
        return "prediction"

    assert _ThreadedProgram(predictor).named_predictors() == [
        ("generate_action", predictor)
    ]


@pytest.mark.asyncio
async def test_threaded_program_runs_sync_program_with_overrides():
    from dspy.dsp.utils.settings import settings

    from dspy_rlm_hooks.speculation.integration.streaming_turn import _ThreadedProgram

    seen = {}

    def program(value):
        seen["marker"] = settings.get("threaded_program_marker")
        return value + 1

    with settings.context(threaded_program_marker="kept"):
        assert await _ThreadedProgram(program).acall(1) == 2
    assert seen["marker"] == "kept"


@pytest.mark.asyncio
async def test_threaded_program_does_not_abandon_worker(monkeypatch):
    from anyio import to_thread

    from dspy_rlm_hooks.speculation.integration.streaming_turn import _ThreadedProgram

    kwargs_seen = {}
    real_run_sync = to_thread.run_sync

    async def spy(func, *args, **kwargs):
        kwargs_seen.update(kwargs)
        return await real_run_sync(func, *args, **kwargs)

    monkeypatch.setattr(to_thread, "run_sync", spy)

    assert await _ThreadedProgram(lambda: "ok").acall() == "ok"
    assert kwargs_seen["abandon_on_cancel"] is False


@pytest.mark.asyncio
async def test_threaded_program_propagates_program_error():
    from dspy_rlm_hooks.speculation.integration.streaming_turn import _ThreadedProgram

    def boom():
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        await _ThreadedProgram(boom).acall()


def test_threaded_program_does_not_raise_run_finished_on_cancel():
    """A cancelled stream must not leave the worker pointing at a dead loop.

    ``abandon_on_cancel=False`` keeps the loop alive until the worker returns.
    The worker may observe a normal cancellation, but never AnyIO's
    ``RunFinishedError`` from a bridge call into a closed loop.
    """
    import asyncio
    import contextlib
    import threading
    import time

    from anyio import RunFinishedError
    from anyio.from_thread import run as from_thread_run
    from dspy.streaming import streamify

    from dspy_rlm_hooks.speculation.integration.streaming_turn import _ThreadedProgram

    async def noop():
        await asyncio.sleep(0)

    started = threading.Event()
    errors = []

    def program():
        try:
            for _ in range(40):
                started.set()
                from_thread_run(noop)
                time.sleep(0.005)
        except BaseException as exc:
            errors.append(exc)
            raise
        return Prediction(output="done")

    async def consume(stream):
        async for _ in stream():
            pass

    async def main():
        task = asyncio.create_task(consume(stream))
        while not started.is_set():
            await asyncio.sleep(0.001)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    stream = streamify(
        _ThreadedProgram(program), async_streaming=True, is_async_program=True
    )
    asyncio.run(main())

    assert not any(isinstance(exc, RunFinishedError) for exc in errors)


# ---------------------------------------------------------------------------
# _StreamingGenerateAction
# ---------------------------------------------------------------------------


def _fake_streamify(items, pred):
    def _factory(program, *, stream_listeners, async_streaming, **kwargs):
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


def test_generate_action_drains_stream_after_prediction(monkeypatch, mock_rlm):
    """The stream is consumed past the Prediction and not pumped by DSPy's
    sync bridge, whose producer loop outlives the consumer's return."""
    pred = Prediction(reasoning="r", code="print(1)")
    consumed = []

    async def astr(*a, **k):
        consumed.append("code")
        yield StreamResponse("ga", "code", "print(1)", False)
        consumed.append("prediction")
        yield pred
        consumed.append("trailer")
        yield StreamResponse("ga", "reasoning", "ignored", False)
        consumed.append("end")

    def factory(program, *, stream_listeners, async_streaming, **kwargs):
        assert async_streaming is True
        return astr

    monkeypatch.setattr("dspy.streaming.streamify", factory)
    mock_rlm._speculation_original_generate_action = MagicMock()
    mock_rlm._active_stream_turn = MagicMock()
    mock_rlm._streaming_fed_any = False

    result = _StreamingGenerateAction(mock_rlm)("variables_info", "hist", "1")

    assert result is pred
    assert consumed == ["code", "prediction", "trailer", "end"]


def test_generate_action_keeps_prediction_when_stream_teardown_fails(
    monkeypatch, mock_rlm
):
    """A failure after a complete Prediction is teardown, not a failed call."""
    pred = Prediction(reasoning="r", code="print(1)")

    async def astr(*a, **k):
        yield pred
        raise RuntimeError("event loop closed during teardown")

    def factory(program, *, stream_listeners, async_streaming, **kwargs):
        assert async_streaming is True
        return astr

    monkeypatch.setattr("dspy.streaming.streamify", factory)
    orig = MagicMock(return_value="orig-result")
    mock_rlm._speculation_original_generate_action = orig
    mock_rlm._active_stream_turn = MagicMock()
    mock_rlm._streaming_fed_any = False

    result = _StreamingGenerateAction(mock_rlm)("variables_info", "hist", "1")

    assert result is pred
    orig.assert_not_called()


@pytest.mark.asyncio
async def test_generate_action_drains_async_stream_after_prediction(
    monkeypatch, mock_rlm
):
    """The async iteration path also consumes past the final Prediction."""
    pred = Prediction(reasoning="r", code="print(1)")
    consumed = []

    async def astr(*a, **k):
        consumed.append("code")
        yield StreamResponse("ga", "code", "print(1)", False)
        consumed.append("prediction")
        yield pred
        consumed.append("trailer")
        yield StreamResponse("ga", "reasoning", "ignored", False)
        consumed.append("end")

    def factory(program, *, stream_listeners, async_streaming, **kwargs):
        if async_streaming:
            return astr

        def sstr(*a, **k):
            return
            yield

        return sstr

    monkeypatch.setattr("dspy.streaming.streamify", factory)
    mock_rlm._speculation_original_generate_action = MagicMock()
    mock_rlm._active_stream_turn = MagicMock()
    mock_rlm._streaming_fed_any = False

    result = await _StreamingGenerateAction(mock_rlm).acall(
        "variables_info", "hist", "1"
    )

    assert result is pred
    assert consumed == ["code", "prediction", "trailer", "end"]


class _RaisingCloseStream:
    """Async iterable that delivers its items, then fails on aclose()."""

    def __init__(self, items) -> None:
        self._items = items

    def __aiter__(self):
        async def iterator():
            for item in self._items:
                yield item

        return iterator()

    async def aclose(self) -> None:
        raise RuntimeError("close boom")


@pytest.mark.asyncio
async def test_run_async_to_completion_propagates_from_running_loop():
    """The private-thread branch re-raises the factory's exception."""
    from dspy_rlm_hooks.speculation.integration.streaming_turn import (
        _run_async_to_completion,
    )

    async def boom():
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        _run_async_to_completion(boom)


def test_generate_action_falls_back_when_stream_has_no_prediction(
    monkeypatch, mock_rlm
):
    """A stream that ends without a Prediction falls back to the original."""

    async def astr(*a, **k):
        yield "not a prediction"

    def factory(program, *, stream_listeners, async_streaming, **kwargs):
        assert async_streaming is True
        return astr

    monkeypatch.setattr("dspy.streaming.streamify", factory)
    orig = MagicMock(return_value="orig-result")
    mock_rlm._speculation_original_generate_action = orig
    mock_rlm._active_stream_turn = MagicMock()
    mock_rlm._streaming_fed_any = False

    result = _StreamingGenerateAction(mock_rlm)("variables_info", "hist", "1")

    assert result == "orig-result"
    orig.assert_called_once_with("variables_info", "hist", "1")
    assert mock_rlm._active_stream_turn is None


def test_generate_action_ignores_close_failure_after_prediction(monkeypatch, mock_rlm):
    """A failing aclose() does not replace an already-complete Prediction."""
    pred = Prediction(reasoning="r", code="print(1)")

    def factory(program, *, stream_listeners, async_streaming, **kwargs):
        assert async_streaming is True

        def astream(*a, **k):
            return _RaisingCloseStream([pred])

        return astream

    monkeypatch.setattr("dspy.streaming.streamify", factory)
    mock_rlm._speculation_original_generate_action = MagicMock()
    mock_rlm._active_stream_turn = MagicMock()
    mock_rlm._streaming_fed_any = False

    result = _StreamingGenerateAction(mock_rlm)("variables_info", "hist", "1")

    assert result is pred


@pytest.mark.asyncio
async def test_generate_action_acall_keeps_prediction_when_stream_teardown_fails(
    monkeypatch, mock_rlm
):
    """acall returns the Prediction even if the stream fails after yielding it."""
    pred = Prediction(reasoning="r", code="print(1)")

    async def astr(*a, **k):
        yield pred
        raise RuntimeError("event loop closed during teardown")

    def factory(program, *, stream_listeners, async_streaming, **kwargs):
        assert async_streaming is True
        return astr

    monkeypatch.setattr("dspy.streaming.streamify", factory)
    mock_rlm._speculation_original_generate_action = MagicMock()
    mock_rlm._speculation_original_generate_action.acall = AsyncMock()
    mock_rlm._active_stream_turn = MagicMock()
    mock_rlm._streaming_fed_any = False

    result = await _StreamingGenerateAction(mock_rlm).acall(
        "variables_info", "hist", "1"
    )

    assert result is pred
    mock_rlm._speculation_original_generate_action.acall.assert_not_awaited()


@pytest.mark.asyncio
async def test_generate_action_acall_ignores_close_failure_after_prediction(
    monkeypatch, mock_rlm
):
    """acall keeps the Prediction when aclose() raises during teardown."""
    pred = Prediction(reasoning="r", code="print(1)")

    def factory(program, *, stream_listeners, async_streaming, **kwargs):
        assert async_streaming is True

        def astream(*a, **k):
            return _RaisingCloseStream([pred])

        return astream

    monkeypatch.setattr("dspy.streaming.streamify", factory)
    mock_rlm._speculation_original_generate_action = MagicMock()
    mock_rlm._speculation_original_generate_action.acall = AsyncMock()
    mock_rlm._active_stream_turn = MagicMock()
    mock_rlm._streaming_fed_any = False

    result = await _StreamingGenerateAction(mock_rlm).acall(
        "variables_info", "hist", "1"
    )

    assert result is pred
    mock_rlm._speculation_original_generate_action.acall.assert_not_awaited()


@pytest.mark.asyncio
async def test_generate_action_acall_falls_back_when_stream_has_no_prediction(
    monkeypatch, mock_rlm
):
    """acall falls back to the original call when the stream has no Prediction."""

    async def astr(*a, **k):
        yield "not a prediction"

    def factory(program, *, stream_listeners, async_streaming, **kwargs):
        assert async_streaming is True
        return astr

    monkeypatch.setattr("dspy.streaming.streamify", factory)
    orig = MagicMock()
    orig.acall = AsyncMock(return_value="orig-result")
    mock_rlm._speculation_original_generate_action = orig
    mock_rlm._active_stream_turn = MagicMock()
    mock_rlm._streaming_fed_any = False

    result = await _StreamingGenerateAction(mock_rlm).acall(
        "variables_info", "hist", "1"
    )

    assert result == "orig-result"
    orig.acall.assert_awaited_once_with("variables_info", "hist", "1")
    assert mock_rlm._active_stream_turn is None


@pytest.mark.asyncio
async def test_threaded_program_translates_concurrent_cancellation(monkeypatch):
    """AnyIO's empty CancelledError becomes a real cancellation, not an LM error."""
    import asyncio
    import concurrent.futures

    from anyio import to_thread

    from dspy_rlm_hooks.speculation.integration.streaming_turn import _ThreadedProgram

    async def cancelled(func, *args, **kwargs):
        raise concurrent.futures.CancelledError()

    monkeypatch.setattr(to_thread, "run_sync", cancelled)

    with pytest.raises(asyncio.CancelledError):
        await _ThreadedProgram(lambda: None).acall()


def test_generate_action_propagates_sync_cancellation(monkeypatch, mock_rlm):
    """A cancelled stream must not be treated as a parse failure/fallback."""
    import concurrent.futures

    from dspy_rlm_hooks.speculation.integration import streaming_turn

    def cancel(factory):
        raise concurrent.futures.CancelledError()

    monkeypatch.setattr(streaming_turn, "_run_async_to_completion", cancel)
    mock_rlm._active_stream_turn = MagicMock()
    wrapper = _StreamingGenerateAction(mock_rlm)
    wrapper._stream = MagicMock()

    with pytest.raises(concurrent.futures.CancelledError):
        wrapper("variables_info", "hist", "1")


@pytest.mark.asyncio
async def test_generate_action_acall_propagates_cancellation(monkeypatch, mock_rlm):
    """acall re-raises cancellation instead of falling back to the original."""
    import concurrent.futures

    async def astr(*a, **k):
        yield "not a prediction"
        raise concurrent.futures.CancelledError()

    def factory(program, *, stream_listeners, async_streaming, **kwargs):
        return astr

    monkeypatch.setattr("dspy.streaming.streamify", factory)
    mock_rlm._speculation_original_generate_action = MagicMock()
    mock_rlm._speculation_original_generate_action.acall = AsyncMock()
    mock_rlm._active_stream_turn = MagicMock()

    with pytest.raises(concurrent.futures.CancelledError):
        await _StreamingGenerateAction(mock_rlm).acall("variables_info", "hist", "1")
    mock_rlm._speculation_original_generate_action.acall.assert_not_awaited()
