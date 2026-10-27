"""Streaming turn: overlap shadow dispatch with the model's code generation.

Begins a :class:`~dspy_rlm_hooks.speculation.session.StreamTurn` before
``generate_action`` runs and replaces ``generate_action`` with a
``dspy.streamify`` wrapper that feeds streamed ``code`` deltas into the active
turn so the shadow can dispatch tool calls while the model is still generating.
"""

from __future__ import annotations

import asyncio
import builtins
import concurrent.futures
import contextvars
import functools
import threading
from collections.abc import Callable, Coroutine
from typing import Any

from dspy.primitives.module import Module

from dspy_rlm_hooks.speculation.integration.registry import (
    _has_speculatable,
    _prediction_type,
    _sync_registry_fns,
)
from dspy_rlm_hooks.speculation.shadow import shadow_builtins


def _maybe_begin_streaming_turn(
    rlm: Any, repl: Any, input_args: dict[str, Any]
) -> None:
    """Begin a streaming :class:`StreamTurn` before ``generate_action`` runs.

    Called from the patched ``_execute_iteration``/``_aexecute_iteration`` (the
    only call sites with both ``repl`` and ``input_args``). Syncs the fresh tool
    closures (they must be current BEFORE the first peek dispatch, which happens
    during generation), seeds the shadow with the persistent prelude, and stashes
    the turn on the RLM for the generate wrapper and ``_speculation_execute_code``
    to share. Silent on failure so the Lazy/JIT fallback takes over.
    """
    spec = getattr(rlm, "_speculator", None)
    config = getattr(rlm, "_speculation_config", None)
    if spec is None or config is None or not config.enabled or not config.streaming:
        return
    if not _has_speculatable(spec):
        return
    try:
        _sync_registry_fns(spec, repl)
        rlm._spec_synced_this_iter = True
        # Reset per-forward() exec counter when a fresh REPL is detected.
        # The REPL is created anew each forward() call, so a different object
        # means we're starting a new run.
        if getattr(rlm, "_spec_last_repl", None) is not repl:
            rlm._spec_exec_count = 0
            rlm._spec_last_repl = repl
        turn = spec.session.begin_stream_turn(
            dict(input_args), shadow_builtins(dict(builtins.__dict__))
        )
        prelude = getattr(repl, "repl_globals", "") or ""
        if prelude:
            turn.feed(f"```repl\n{prelude}\n```\n")
        rlm._active_stream_turn = turn
        rlm._streaming_fed_any = False
    except Exception:
        rlm._active_stream_turn = None


def _run_async_to_completion(factory: Callable[[], Coroutine[Any, Any, Any]]) -> Any:
    """Drive an async stream generator until it has fully shut down.

    DSPy's sync ``streamify`` bridge pumps the async generator through a
    daemon producer thread and signals completion before ``asyncio.run`` has
    finished closing its event loop.  AnyIO worker threads can then hop back
    with ``anyio.from_thread.run`` after the loop is gone, which surfaces as
    ``RunFinishedError`` ("The event loop associated with the given token has
    already finished") on a later LM call.  Here the loop belongs to this call:
    by the time ``asyncio.run`` returns, the stream, its task group, and the
    loop have all finished.  If the caller already runs a loop (for example a
    sync RLM invoked from async code), the work runs in a private thread with
    its own loop, and this function joins that thread before returning.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(factory())

    context = contextvars.copy_context()
    outcome: dict[str, Any] = {}

    def runner() -> None:
        try:
            outcome["value"] = asyncio.run(factory())
        except BaseException as exc:  # noqa: BLE001 - re-raised in the caller
            outcome["error"] = exc

    thread = threading.Thread(target=context.run, args=(runner,), daemon=True)
    thread.start()
    thread.join()
    if "error" in outcome:
        raise outcome["error"]
    return outcome["value"]


class _ThreadedProgram(Module):
    """Async streamify program that runs a sync predictor without abandoning it.

    ``dspy.streamify`` normally wraps a sync program with
    ``dspy.utils.asyncify``, which calls
    ``anyio.to_thread.run_sync(..., abandon_on_cancel=True)``.  If the stream
    is cancelled (parse failure, teardown, caller shutdown), the worker thread
    keeps running after the event loop has closed; its next
    ``anyio.from_thread.run`` then raises AnyIO's ``RunFinishedError`` or the
    loop reports cancelled bridge tasks during ``asyncio.run`` shutdown.

    Running the predictor with ``abandon_on_cancel=False`` keeps the event
    loop alive until the sync call has actually returned, so no LM call can
    outlive the loop it sends stream chunks to.
    """

    def __init__(self, program: Callable[..., Any]) -> None:
        super().__init__()
        self._program = program

    def named_predictors(self) -> list[tuple[str, Any]]:
        """Expose the wrapped predictor to ``find_predictor_for_stream_listeners``."""
        return [("generate_action", self._program)]

    async def acall(self, *args: Any, **kwargs: Any) -> Any:
        from anyio import to_thread
        from dspy.dsp.utils.settings import thread_local_overrides
        from dspy.dsp.utils.utils import dotdict

        parent_overrides = thread_local_overrides.get().copy()

        def wrapped_program(*a: Any, **kw: Any) -> Any:
            original_overrides = thread_local_overrides.get()
            token = thread_local_overrides.set(
                dotdict({**original_overrides, **parent_overrides})
            )
            try:
                return self._program(*a, **kw)
            finally:
                thread_local_overrides.reset(token)

        partial_f = functools.partial(wrapped_program, *args, **kwargs)
        try:
            return await to_thread.run_sync(partial_f, abandon_on_cancel=False)
        except concurrent.futures.CancelledError as exc:
            # AnyIO bridges a cancelled loop-side send back to the worker as
            # concurrent.futures.CancelledError (empty message).  DSPy would
            # wrap that as an unexpected LM error; cancellation must stay a
            # BaseException so it propagates as cancellation instead.
            raise asyncio.CancelledError() from exc


class _StreamingGenerateAction:
    """Stand-in for ``rlm.generate_action`` that streams the ``code`` output.

    Wraps the original ``dspy.Predict`` in an async ``dspy.streamify`` and
    feeds the streamed ``code`` field deltas into the active
    :class:`StreamTurn` so the shadow can dispatch tool calls while the model
    is still generating. Exposes both ``__call__`` (sync path, which drives the
    async stream in an owned loop) and ``acall`` (async path). If streaming is
    unavailable (non-streaming adapter/LM, cache hit) or fails partway, it
    falls back to the original predict and clears the active turn so the
    Lazy/JIT shadow in ``_speculation_execute_code`` takes over.
    """

    def __init__(self, rlm: Any) -> None:
        self._rlm = rlm
        self._orig = rlm._speculation_original_generate_action
        self._stream: Callable | None = None

    def _ensure(self) -> Callable | None:
        """Lazily build the async streamify wrapper around the original predict.

        The async form is used by both ``__call__`` and ``acall``: the sync
        form's daemon-thread bridge is exactly what leaves the loop teardown
        race described in ``_run_async_to_completion``.  Returns the wrapper,
        or None if streaming is unavailable.
        """
        if self._stream is not None:
            return self._stream
        try:
            from dspy.streaming import StreamListener, streamify

            class CodeStreamListener(StreamListener):
                def receive(self, chunk: Any) -> Any:
                    # DSPy chooses delimiters by the adapter's exact class
                    # name, although its parser/flush support subclasses.
                    # Register the inherited wire format on this listener
                    # only; do not mutate global settings or DSPy classes.
                    from dspy import settings

                    adapter = settings.adapter
                    if adapter is not None:
                        name = type(adapter).__name__
                        if name not in self.adapter_identifiers:
                            for base in type(adapter).__mro__[1:]:
                                if base.__name__ in self.adapter_identifiers:
                                    self.adapter_identifiers[name] = (
                                        self.adapter_identifiers[base.__name__]
                                    )
                                    break
                    return super().receive(chunk)

            self._stream = streamify(
                _ThreadedProgram(self._orig),
                stream_listeners=[
                    CodeStreamListener("code", predict=self._orig, allow_reuse=True)
                ],
                async_streaming=True,
                is_async_program=True,
            )
            return self._stream
        except Exception:
            self._stream = None
            return None

    def _feed_item(self, item: Any) -> bool:
        """Feed one streamed item into the active turn. Returns True if the item
        was a ``code`` delta (streaming produced content)."""
        from dspy.streaming import StreamResponse

        if isinstance(item, StreamResponse) and item.chunk:
            if item.signature_field_name == "code":
                turn = getattr(self._rlm, "_active_stream_turn", None)
                if turn is not None:
                    turn.feed(item.chunk)
                    self._rlm._streaming_fed_any = True
            return True
        return False

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        turn = getattr(self._rlm, "_active_stream_turn", None)
        if turn is None:
            return self._orig(*args, **kwargs)
        astream = self._ensure()
        if astream is None:
            self._rlm._active_stream_turn = None
            return self._orig(*args, **kwargs)
        prediction = None

        async def consume() -> Any:
            nonlocal prediction
            stream = astream(*args, **kwargs)
            try:
                async for item in stream:
                    if prediction is not None:
                        # Consume past the final Prediction so the stream's
                        # task group exits before the loop closes.
                        continue
                    if isinstance(item, _prediction_type()):
                        prediction = item
                        continue
                    self._feed_item(item)
            finally:
                close = getattr(stream, "aclose", None)
                if close is not None:
                    try:
                        await close()
                    except Exception:
                        # Teardown failures must not replace the prediction
                        # or the original-predict fallback.
                        pass
            return prediction

        try:
            result = _run_async_to_completion(consume)
        except concurrent.futures.CancelledError:
            raise
        except Exception:
            if prediction is not None:
                # The prediction is complete; a teardown failure after it
                # must not discard it or re-run the LM call.
                return prediction
            self._rlm._active_stream_turn = None
            return self._orig(*args, **kwargs)
        if result is not None:
            return result
        self._rlm._active_stream_turn = None
        return self._orig(*args, **kwargs)

    async def acall(self, *args: Any, **kwargs: Any) -> Any:
        turn = getattr(self._rlm, "_active_stream_turn", None)
        if turn is None:
            return await self._orig.acall(*args, **kwargs)
        astream = self._ensure()
        if astream is None:
            self._rlm._active_stream_turn = None
            return await self._orig.acall(*args, **kwargs)
        prediction = None
        stream = None
        try:
            stream = astream(*args, **kwargs)
            async for item in stream:
                if prediction is not None:
                    # Same reason as the sync path: consume past the final
                    # Prediction so the async generator exits its task group
                    # instead of being abandoned while suspended inside it.
                    continue
                if isinstance(item, _prediction_type()):
                    prediction = item
                    continue
                self._feed_item(item)
        except concurrent.futures.CancelledError:
            raise
        except Exception:
            if prediction is not None:
                return prediction
            self._rlm._active_stream_turn = None
            return await self._orig.acall(*args, **kwargs)
        finally:
            close = getattr(stream, "aclose", None)
            if close is not None:
                try:
                    await close()
                except Exception:
                    # Teardown failures must not replace the prediction or
                    # the original-predict fallback below.
                    pass
        if prediction is not None:
            return prediction
        self._rlm._active_stream_turn = None
        return await self._orig.acall(*args, **kwargs)
