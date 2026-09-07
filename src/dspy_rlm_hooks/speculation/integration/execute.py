"""Wrapped execution: shadow pre-pass + claim hooks + real execution.

The patched ``_execute_iteration``/``_aexecute_iteration`` and the wrapped
``_execute_code`` choke point shared by both the sync and async paths.
"""

from __future__ import annotations

import builtins
import sys
from typing import Any

from dspy_rlm_hooks.core.utils import _assemble_execution_code
from dspy_rlm_hooks.speculation.integration.live_state import _live_state_seed
from dspy_rlm_hooks.speculation.integration.registry import (
    _has_speculatable,
    _sync_registry_fns,
)
from dspy_rlm_hooks.speculation.integration.streaming_turn import (
    _maybe_begin_streaming_turn,
)
from dspy_rlm_hooks.speculation.shadow import shadow_builtins


def _speculation_execute_iteration(
    self: Any,
    repl: Any,
    variables: list[Any],
    history: Any,
    iteration: int,
    input_args: dict[str, Any],
    output_field_names: list[str],
) -> Any:
    """Wrapped sync iteration: begin the streaming turn before generation."""
    _maybe_begin_streaming_turn(self, repl, input_args)
    inner = self._speculation_original_execute_iteration
    return inner(repl, variables, history, iteration, input_args, output_field_names)


async def _speculation_aexecute_iteration(
    self: Any,
    repl: Any,
    variables: list[Any],
    history: Any,
    iteration: int,
    input_args: dict[str, Any],
    output_field_names: list[str],
) -> Any:
    """Wrapped async iteration: begin the streaming turn before generation."""
    _maybe_begin_streaming_turn(self, repl, input_args)
    inner = self._speculation_original_aexecute_iteration
    return await inner(
        repl, variables, history, iteration, input_args, output_field_names
    )


def _speculation_execute_code(
    self: Any, repl: Any, code: str, input_args: dict[str, Any]
) -> Any:
    """The wrapped ``_execute_code``: shadow pre-pass + claim hooks + real exec.

    Sits between ``pre_execution`` and ``post_execution`` in the hook lifecycle
    (it wraps the execute step), so ``pre_execution`` can still rewrite code
    before speculation and ``post_execution`` sees the real result.
    """
    spec = getattr(self, "_speculator", None)
    config = getattr(self, "_speculation_config", None)
    inner = getattr(self, "_speculation_original_execute_code", None)
    if spec is None or config is None or inner is None or not config.enabled:
        return inner(repl, code, input_args) if inner is not None else None

    # The FINAL code the real interpreter runs (persistent prelude + injected
    # vars), NOT the raw un-assembled code.
    assembled = _assemble_execution_code(repl, code)
    # Skip redundant sync when _maybe_begin_streaming_turn already synced this iteration.
    # Use explicit `in __dict__` check: getattr on MagicMock would auto-create the attr.
    if "_spec_synced_this_iter" not in self.__dict__ or not self._spec_synced_this_iter:
        _sync_registry_fns(spec, repl)
    self._spec_synced_this_iter = False

    # Reset per-forward() exec counter when a fresh REPL is detected
    # (non-streaming path: _maybe_begin_streaming_turn handles the streaming path).
    if getattr(self, "_spec_last_repl", None) is not repl:
        self._spec_exec_count = 0
        self._spec_last_repl = repl

    # First-iteration fast path: the REPL starts empty (no prior iterations),
    # so the live-state snapshot is guaranteed to return nothing.  Skip the
    # expensive repl.execute() round-trip (~775ms on Deno subprocess).
    first_exec = not getattr(self, "_spec_exec_count", 0)
    self._spec_exec_count = getattr(self, "_spec_exec_count", 0) + 1

    # A streaming turn is active when it was begun by the patched iteration
    # method (streaming mode). Otherwise fall back to the Lazy/JIT one-shot pass.
    turn = getattr(self, "_active_stream_turn", None)
    if turn is None:
        # --- Lazy/JIT shadow pre-pass over the assembled code -----------------
        if _has_speculatable(spec):
            t = None
            try:
                seed = (
                    dict(input_args)
                    if first_exec
                    else _live_state_seed(repl, code, input_args, spec)
                )
                t = spec.session.begin_stream_turn(
                    seed,
                    shadow_builtins(dict(builtins.__dict__)),
                )
                # CRITICAL: StreamSegmenter only emits inside ```repl fences.
                t.feed(f"```repl\n{assembled}\n```\n")
            except Exception:
                pass  # shadow errors are SAFE: fall through to real execution
            finally:
                if t is not None:
                    try:
                        t.end(timeout=config.timeout_s)
                    except Exception:
                        pass
    if turn is not None:
        # Streaming turn is active (begun during generate_action). If it
        # produced no code deltas (cache hit, stream failure, or unfenced
        # output), top up with the live-state snapshot and the full assembled
        # block so the turn still speculates over what the real interpreter
        # will run.
        if not getattr(self, "_streaming_fed_any", False):
            try:
                if not first_exec:
                    snap = _live_state_seed(repl, code, input_args, spec)
                    assigns = "".join(
                        f"{name} = {value!r}\n"
                        for name, value in snap.items()
                        if name not in input_args
                    )
                    if assigns:
                        turn.feed(f"```repl\n{assigns}\n```\n")
                turn.feed(f"```repl\n{assembled}\n```\n")
            except Exception:
                pass
        # Drain BEFORE real execution so the shadow has queued every dispatch
        # and the no-recall invariant holds (a claim must never re-dispatch a
        # call the shadow is about to dispatch). With the persistent warm
        # worker this drain is a cheap pipe round-trip, not a process teardown.
        try:
            turn.end(timeout=config.timeout_s)
        except Exception:
            pass
        self._active_stream_turn = None

    # --- install claiming hooks into the real tool path ---------------------
    try:
        # Resolved through the integration package namespace at call time —
        # exactly as in the pre-split single module, where this was a module-
        # global lookup — so rebinding
        # ``dspy_rlm_hooks.speculation.integration._install_claim_hooks``
        # keeps affecting this path.
        sys.modules[__name__.rsplit(".", 1)[0]]._install_claim_hooks(
            repl, spec, config, self
        )
    except Exception:
        pass  # safe fallback: real execution runs un-claimed

    # --- real execution -----------------------------------------------------
    try:
        return inner(repl, code, input_args)
    finally:
        try:
            spec.end_turn()  # evict unclaimed, reset per-turn budget
        except Exception:
            pass
