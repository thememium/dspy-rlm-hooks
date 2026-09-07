"""Streaming turn: overlap shadow dispatch with the model's code generation.

Begins a :class:`~dspy_rlm_hooks.speculation.session.StreamTurn` before
``generate_action`` runs and replaces ``generate_action`` with a
``dspy.streamify`` wrapper that feeds streamed ``code`` deltas into the active
turn so the shadow can dispatch tool calls while the model is still generating.
"""

from __future__ import annotations

import builtins
from collections.abc import Callable
from typing import Any

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


class _StreamingGenerateAction:
    """Stand-in for ``rlm.generate_action`` that streams the ``code`` output.

    Wraps the original ``dspy.Predict`` in ``dspy.streamify`` and feeds the
    streamed ``code`` field deltas into the active :class:`StreamTurn` so the
    shadow can dispatch tool calls while the model is still generating. Exposes
    both ``__call__`` (sync path) and ``acall`` (async path). If streaming is
    unavailable (non-streaming adapter/LM, cache hit) or fails partway, it falls
    back to the original predict and clears the active turn so the Lazy/JIT
    shadow in ``_speculation_execute_code`` takes over.
    """

    def __init__(self, rlm: Any) -> None:
        self._rlm = rlm
        self._orig = rlm._speculation_original_generate_action
        self._sync: Callable | None = None
        self._async: Callable | None = None

    def _ensure(self) -> tuple[Callable, Callable] | None:
        """Lazily build the sync/async streamify wrappers around the original
        predict. Returns ``(sync, async)``, or None if streaming is unavailable."""
        if self._sync is not None and self._async is not None:
            return self._sync, self._async
        try:
            from dspy.streaming import StreamListener, streamify

            self._sync = streamify(
                self._orig,
                stream_listeners=[
                    StreamListener("code", predict=self._orig, allow_reuse=True)
                ],
                async_streaming=False,
            )
            self._async = streamify(
                self._orig,
                stream_listeners=[
                    StreamListener("code", predict=self._orig, allow_reuse=True)
                ],
                async_streaming=True,
            )
            return self._sync, self._async
        except Exception:
            self._sync = None
            self._async = None
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
        streams = self._ensure()
        if streams is None:
            self._rlm._active_stream_turn = None
            return self._orig(*args, **kwargs)
        sync, _ = streams
        try:
            for item in sync(*args, **kwargs):
                if isinstance(item, _prediction_type()):
                    return item
                self._feed_item(item)
        except Exception:
            self._rlm._active_stream_turn = None
        return self._orig(*args, **kwargs)

    async def acall(self, *args: Any, **kwargs: Any) -> Any:
        turn = getattr(self._rlm, "_active_stream_turn", None)
        if turn is None:
            return await self._orig.acall(*args, **kwargs)
        streams = self._ensure()
        if streams is None:
            self._rlm._active_stream_turn = None
            return await self._orig.acall(*args, **kwargs)
        _, astream = streams
        try:
            async for item in astream(*args, **kwargs):
                if isinstance(item, _prediction_type()):
                    return item
                self._feed_item(item)
        except Exception:
            self._rlm._active_stream_turn = None
        return await self._orig.acall(*args, **kwargs)
