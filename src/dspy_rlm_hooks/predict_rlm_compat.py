"""PredictRLM compatibility layer for dspy-rlm-hooks.

Provides hook support for predict-rlm's :class:`~predict_rlm.PredictRLM`
instances, which extend :class:`dspy.RLM` with additional validation,
telemetry, and a callback dispatch system.

All ``predict_rlm`` imports are **lazy** — they happen inside functions so
this module can be imported even when predict-rlm is not installed.

Lifecycle (same 4-hook contract as :mod:`dspy_rlm_hooks.patcher`)::

    pre_iteration_hook → generate_action → pre_execution_hook → execute → post_execution_hook → post_iteration_hook

All four hooks support **full mutation** for PredictRLM:

- ``pre_iteration_hook`` — inject variables and persistent code
- ``pre_execution_hook`` — rewrite generated code
- ``post_execution_hook`` — transform the raw execution result
- ``post_iteration_hook`` — modify history or set ``stop=True`` to halt
"""

from __future__ import annotations

import asyncio
import functools
import logging
from types import MethodType
from typing import Any, cast

from dspy_rlm_hooks.types import (
    PostExecutionOutput,
    PostIterationOutput,
    PreExecutionOutput,
    PreIterationOutput,
)
from dspy_rlm_hooks.utils import _strip_code_fences

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


class _StopIteration(Exception):
    """Raised when ``post_iteration_hook`` sets ``stop=True``.

    Caught by the ``_execute_iteration`` wrapper so that the caller
    (PredictRLM's iteration loop) sees a normal return with a
    ``Prediction`` result.
    """


def _run_async(coroutine: Any) -> Any:
    """Run an async coroutine, handling both sync and async contexts.

    Mirrors :func:`dspy_rlm_hooks.patcher._run_async`.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)
    else:
        new_loop = asyncio.new_event_loop()
        try:
            return new_loop.run_until_complete(coroutine)
        finally:
            new_loop.close()


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def _is_predict_rlm(rlm: Any) -> bool:
    """Return ``True`` if *rlm* is a :class:`~predict_rlm.PredictRLM` instance.

    The import is lazy — returns ``False`` when predict-rlm is not installed.
    """
    try:
        from predict_rlm import PredictRLM

        return isinstance(rlm, PredictRLM)
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Enable / Disable
# ---------------------------------------------------------------------------


def enable_predict_rlm_hooks(
    rlm: Any,
    *,
    pre_iteration_hook: Any | None = None,
    pre_execution_hook: Any | None = None,
    post_execution_hook: Any | None = None,
    post_iteration_hook: Any | None = None,
) -> None:
    """Install lifecycle hooks on a :class:`~predict_rlm.PredictRLM` instance.

    This is the PredictRLM counterpart of
    :func:`dspy_rlm_hooks.patcher.enable_rlm_hooks`.  All patches are
    **instance-level** — no class attributes are modified.

    All four hooks support **full mutation**:

    - ``pre_iteration_hook`` — inject variables and persistent code.
    - ``pre_execution_hook`` — rewrite the generated code.
    - ``post_execution_hook`` — transform the raw execution result before
      it is processed into history.
    - ``post_iteration_hook`` — modify history or set ``stop=True`` to
      halt the iteration loop early.

    Args:
        rlm: A ``PredictRLM`` (or compatible) instance.
        pre_iteration_hook: Called before action generation.
        pre_execution_hook: Called after code is generated, before execution.
        post_execution_hook: Called after code executes, before the result
            is processed into history.  May transform ``result``.
        post_iteration_hook: Called after the result is processed into
            history.  May modify history or set ``stop=True``.
    """
    # ------------------------------------------------------------------
    # Step 1 — Idempotency guard
    # ------------------------------------------------------------------
    if hasattr(rlm, "_hook_originals"):
        disable_predict_rlm_hooks(rlm)

    # ------------------------------------------------------------------
    # Step 2 — Wrap ``_execute_iteration`` (sync) for pre_iteration_hook
    # ------------------------------------------------------------------
    orig_execute = rlm._execute_iteration

    @functools.wraps(orig_execute)
    def _wrapped_execute_iteration(
        self: Any,
        repl: Any,
        variables: Any,
        history: Any,
        iteration: int,
        input_args: dict[str, Any],
        output_field_names: Any,
        **kw: Any,
    ) -> Any:
        # --- pre-iteration hook ---
        if self._hook_pre_iteration:
            pre_iter_out = self._hook_pre_iteration(
                iteration,
                variables,
                history,
                input_args,
            )
            if asyncio.iscoroutine(pre_iter_out):
                pre_iter_out = _run_async(pre_iter_out)
            pre_iter_out = cast(PreIterationOutput, pre_iter_out)
            input_args = {**input_args, **pre_iter_out.extra_vars}
            if pre_iter_out.python_code:
                current_globals = getattr(repl, "repl_globals", "") or ""
                repl.repl_globals = current_globals + "\n" + pre_iter_out.python_code

        # Publish iteration context for other wrappers.
        self._hook_current_context = {
            "iteration": iteration,
            "variables": variables,
            "history": history,
            "input_args": input_args,
        }
        try:
            return orig_execute(
                self,
                repl,
                variables,
                history,
                iteration,
                input_args,
                output_field_names,
                **kw,
            )
        except _StopIteration as exc:
            # post_iteration_hook set stop=True — return the stored result.
            return exc.args[0]

    rlm._execute_iteration = MethodType(_wrapped_execute_iteration, rlm)

    # ------------------------------------------------------------------
    # Step 3 — Wrap ``_aexecute_iteration`` (async) for pre_iteration_hook
    # ------------------------------------------------------------------
    orig_aexecute = rlm._aexecute_iteration

    @functools.wraps(orig_aexecute)
    async def _wrapped_aexecute_iteration(
        self: Any,
        repl: Any,
        variables: Any,
        history: Any,
        iteration: int,
        input_args: dict[str, Any],
        output_field_names: Any,
        **kw: Any,
    ) -> Any:
        # --- pre-iteration hook (async) ---
        if self._hook_pre_iteration:
            pre_iter_out = self._hook_pre_iteration(
                iteration,
                variables,
                history,
                input_args,
            )
            if asyncio.iscoroutine(pre_iter_out):
                pre_iter_out = await pre_iter_out
            pre_iter_out = cast(PreIterationOutput, pre_iter_out)
            input_args = {**input_args, **pre_iter_out.extra_vars}
            if pre_iter_out.python_code:
                current_globals = getattr(repl, "repl_globals", "") or ""
                repl.repl_globals = current_globals + "\n" + pre_iter_out.python_code

        self._hook_current_context = {
            "iteration": iteration,
            "variables": variables,
            "history": history,
            "input_args": input_args,
        }
        try:
            return await orig_aexecute(
                self,
                repl,
                variables,
                history,
                iteration,
                input_args,
                output_field_names,
                **kw,
            )
        except _StopIteration as exc:
            return exc.args[0]

    rlm._aexecute_iteration = MethodType(_wrapped_aexecute_iteration, rlm)

    # ------------------------------------------------------------------
    # Step 4 — Wrap ``generate_action.forward`` for pre_execution_hook
    # ------------------------------------------------------------------
    orig_forward = rlm.generate_action.forward

    @functools.wraps(orig_forward)
    def _wrapped_forward(**kwargs: Any) -> Any:
        result = orig_forward(**kwargs)
        if rlm._hook_pre_execution:
            ctx: dict[str, Any] = getattr(rlm, "_hook_current_context", {})
            raw_code = result.code  # Preserve original before stripping
            code = _strip_code_fences(result.code)
            pre_exec_out = rlm._hook_pre_execution(
                ctx.get("iteration", 0),
                code,
                ctx.get("variables", []),
                ctx.get("history", []),
                ctx.get("input_args", {}),
                raw_code=raw_code,
            )
            if asyncio.iscoroutine(pre_exec_out):
                pre_exec_out = _run_async(pre_exec_out)
            pre_exec_out = cast(PreExecutionOutput, pre_exec_out)
            result.code = pre_exec_out.code
            # Store raw_code in context for post_execution_hook
            rlm._hook_current_context = {**ctx, "raw_code": raw_code}
        return result

    rlm.generate_action.forward = _wrapped_forward

    # ------------------------------------------------------------------
    # Step 5 — Wrap ``_process_execution_result`` for post hooks
    # ------------------------------------------------------------------
    # This fires BETWEEN code execution and history processing, enabling
    # full mutation of the raw result (post_execution_hook) and the
    # processed history (post_iteration_hook).
    orig_process = rlm._process_execution_result

    @functools.wraps(orig_process)
    def _wrapped_process_execution_result(
        self: Any,
        pred: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        ctx: dict[str, Any] = getattr(self, "_hook_current_context", {})
        iteration = ctx.get("iteration", 0)
        variables = ctx.get("variables", [])
        history = ctx.get("history", [])
        input_args = ctx.get("input_args", {})

        # Determine which arg is ``code`` (PredictRLM passes it; dspy.RLM may not).
        # PredictRLM signature: _process_execution_result(pred, code, result, history, output_field_names)
        # dspy.RLM signature:   _process_execution_result(pred, result, history, output_field_names)
        if args and isinstance(args[0], str):
            code_arg, raw_result = args[0], args[1]
            rest_args = args[2:]
        else:
            code_arg = ctx.get("code", "")
            raw_result = args[0] if args else kwargs.get("result")
            rest_args = args[1:] if len(args) > 1 else ()

        # --- post-execution hook (mutation) ---
        if self._hook_post_execution:
            raw_code = ctx.get("raw_code", code_arg)
            post_exec_out = self._hook_post_execution(
                iteration,
                code_arg,
                raw_result,
                variables,
                history,
                input_args,
                raw_code=raw_code,
            )
            if asyncio.iscoroutine(post_exec_out):
                post_exec_out = _run_async(post_exec_out)
            post_exec_out = cast(PostExecutionOutput, post_exec_out)
            raw_result = post_exec_out.result
            # Rebuild args with transformed result.
            if args and isinstance(args[0], str):
                args = (args[0], raw_result, *rest_args)
            else:
                args = (raw_result, *rest_args)

        # Call original to process result into history.
        processed = orig_process(self, pred, *args, **kwargs)

        # --- post-iteration hook (mutation) ---
        if self._hook_post_iteration:
            post_iter_out = self._hook_post_iteration(
                iteration,
                pred,
                code_arg,
                raw_result,
                processed,
            )
            if asyncio.iscoroutine(post_iter_out):
                post_iter_out = _run_async(post_iter_out)
            post_iter_out = cast(PostIterationOutput, post_iter_out)
            processed = post_iter_out.history
            if post_iter_out.stop:
                raise _StopIteration(processed)

        return processed

    rlm._process_execution_result = MethodType(_wrapped_process_execution_result, rlm)

    # ------------------------------------------------------------------
    # Step 6 — Store originals and hook references
    # ------------------------------------------------------------------
    rlm._hook_originals = {
        "_execute_iteration": orig_execute,
        "_aexecute_iteration": orig_aexecute,
        "generate_action_forward": orig_forward,
        "_process_execution_result": orig_process,
    }
    rlm._hook_pre_iteration = pre_iteration_hook
    rlm._hook_pre_execution = pre_execution_hook
    rlm._hook_post_execution = post_execution_hook
    rlm._hook_post_iteration = post_iteration_hook


def disable_predict_rlm_hooks(rlm: Any) -> None:
    """Remove lifecycle hooks from a PredictRLM instance.

    Restores all original methods and removes hook attributes that were
    added by :func:`enable_predict_rlm_hooks`.  Safe to call even if
    hooks were never enabled (no-op).
    """
    originals: dict[str, Any] = getattr(rlm, "_hook_originals", {})
    if not originals:
        return  # nothing to disable

    # Restore original methods
    if "_execute_iteration" in originals:
        rlm._execute_iteration = originals["_execute_iteration"]
    if "_aexecute_iteration" in originals:
        rlm._aexecute_iteration = originals["_aexecute_iteration"]
    if "generate_action_forward" in originals:
        rlm.generate_action.forward = originals["generate_action_forward"]
    if "_process_execution_result" in originals:
        rlm._process_execution_result = originals["_process_execution_result"]

    # Cleanup all hook attributes
    for attr in (
        "_hook_originals",
        "_hook_pre_iteration",
        "_hook_pre_execution",
        "_hook_post_execution",
        "_hook_post_iteration",
        "_hook_current_context",
    ):
        if hasattr(rlm, attr):
            delattr(rlm, attr)
