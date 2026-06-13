"""Monkey-patch DSPy :class:`~dspy.RLM` instances to support lifecycle hooks.

This module provides :func:`enable_rlm_hooks`, which injects custom behaviour
into the RLM's private iteration loop.  Hooks may be sync *or* async; the
system auto-detects coroutines via :func:`asyncio.iscoroutine` and handles both
paths transparently.

.. warning::
    This module uses monkey-patching to instrument DSPy internals.  It is
    designed for DSPy **3.1+** and may require updates if the internal RLM
    API changes in future releases.
"""

from __future__ import annotations

import asyncio
import logging
from types import MethodType
from typing import Any, cast

from dspy.primitives.prediction import Prediction
from dspy.primitives.repl_types import REPLHistory, REPLVariable

from dspy_rlm_hooks.types import (
    PostExecutionHook,
    PostExecutionOutput,
    PostIterationHook,
    PostIterationOutput,
    PreExecutionHook,
    PreExecutionOutput,
    PreIterationHook,
    PreIterationOutput,
)
from dspy_rlm_hooks.utils import _strip_code_fences

logger = logging.getLogger(__name__)


def _run_async(coroutine):
    """Run an async coroutine, handling both sync and async contexts."""
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


_REQUIRED_METHODS = (
    "_execute_iteration",
    "_aexecute_iteration",
    "_process_execution_result",
    "generate_action",
    "max_iterations",
    "verbose",
)


def _validate_rlm(rlm: Any) -> None:
    """Ensure *rlm* exposes the internal methods we need to patch."""
    missing = [name for name in _REQUIRED_METHODS if not hasattr(rlm, name)]
    if missing:
        raise AttributeError(
            f"RLM instance missing required attributes: {', '.join(missing)}. "
            "Ensure you are passing a dspy.RLM instance from dspy>=3.1."
        )


def _execute_code(self: Any, repl: Any, code: str, input_args: dict[str, Any]) -> Any:
    """Run ``code`` inside the REPL, prepending any persisted globals.

    Mirrors the original ``RLM._execute_code`` logic and is bound as a
    replacement method during :func:`enable_rlm_hooks`.
    """
    if hasattr(repl, "repl_globals") and repl.repl_globals:
        code = repl.repl_globals + "\n" + code
    try:
        return repl.execute(code, variables=dict(input_args))
    except Exception as exc:  # noqa: BLE001
        return f"[Error] {exc}"


def _execute_iteration(
    self: Any,
    repl: Any,
    variables: list[REPLVariable],
    history: REPLHistory,
    iteration: int,
    input_args: dict[str, Any],
    output_field_names: list[str],
) -> Prediction | REPLHistory:
    """Synchronous RLM iteration with hook support.

    Lifecycle::

        pre_iteration → generate_action → pre_execution → execute → post_execution → post_iteration
    """
    # --- pre-iteration hook ---
    if self._hook_pre_iteration:
        pre_iter_out = self._hook_pre_iteration(
            iteration, variables, history, input_args
        )
        if asyncio.iscoroutine(pre_iter_out):
            pre_iter_out = _run_async(pre_iter_out)
        pre_iter_out = cast(PreIterationOutput, pre_iter_out)
        input_args = {**input_args, **pre_iter_out.extra_vars}
        if pre_iter_out.python_code:
            current_globals = getattr(repl, "repl_globals", "") or ""
            repl.repl_globals = current_globals + "\n" + pre_iter_out.python_code

    # --- action generation ---
    variables_info = [variable.format() for variable in variables]
    action = self.generate_action(
        variables_info=variables_info,
        repl_history=history,
        iteration=f"{iteration + 1}/{self.max_iterations}",
    )

    if self.verbose:
        logger.info(
            "RLM iteration %d/%d\nReasoning: %s\nCode:\n%s",
            iteration + 1,
            self.max_iterations,
            action.reasoning,
            action.code,
        )

    # --- strip fences ---
    try:
        code = _strip_code_fences(action.code)
    except SyntaxError as exc:
        code = action.code
        result = f"[Error] {exc}"
        return self._process_execution_result(
            action, code, result, history, output_field_names
        )

    # --- pre-execution hook ---
    if self._hook_pre_execution:
        pre_exec_out = self._hook_pre_execution(
            iteration, code, variables, history, input_args
        )
        if asyncio.iscoroutine(pre_exec_out):
            pre_exec_out = _run_async(pre_exec_out)
        pre_exec_out = cast(PreExecutionOutput, pre_exec_out)
        code = pre_exec_out.code

    # --- execute ---
    result = self._execute_code(repl, code, input_args)

    # --- post-execution hook ---
    if self._hook_post_execution:
        post_exec_out = self._hook_post_execution(
            iteration, code, result, variables, history, input_args
        )
        if asyncio.iscoroutine(post_exec_out):
            post_exec_out = _run_async(post_exec_out)
        post_exec_out = cast(PostExecutionOutput, post_exec_out)
        result = post_exec_out.result

    # --- process result ---
    processed = self._process_execution_result(
        action, code, result, history, output_field_names
    )

    # --- post-iteration hook ---
    if self._hook_post_iteration and isinstance(processed, REPLHistory):
        post_iter_out = self._hook_post_iteration(
            iteration, action, code, result, processed
        )
        if asyncio.iscoroutine(post_iter_out):
            post_iter_out = _run_async(post_iter_out)
        post_iter_out = cast(PostIterationOutput, post_iter_out)
        processed = post_iter_out.history

        if post_iter_out.stop:
            return self._extract_fallback(variables, processed, output_field_names)

    return processed


async def _aexecute_iteration(
    self: Any,
    repl: Any,
    variables: list[REPLVariable],
    history: REPLHistory,
    iteration: int,
    input_args: dict[str, Any],
    output_field_names: list[str],
) -> Prediction | REPLHistory:
    """Asynchronous RLM iteration with hook support.

    Mirrors :func:`_execute_iteration` but uses ``await`` for async hooks
    and ``generate_action.acall``.
    """
    # --- pre-iteration hook ---
    if self._hook_pre_iteration:
        pre_iter_out = self._hook_pre_iteration(
            iteration, variables, history, input_args
        )
        if asyncio.iscoroutine(pre_iter_out):
            pre_iter_out = await pre_iter_out
        pre_iter_out = cast(PreIterationOutput, pre_iter_out)
        input_args = {**input_args, **pre_iter_out.extra_vars}
        if pre_iter_out.python_code:
            current_globals = getattr(repl, "repl_globals", "") or ""
            repl.repl_globals = current_globals + "\n" + pre_iter_out.python_code

    # --- action generation ---
    variables_info = [variable.format() for variable in variables]
    pred = await self.generate_action.acall(
        variables_info=variables_info,
        repl_history=history,
        iteration=f"{iteration + 1}/{self.max_iterations}",
    )

    if self.verbose:
        logger.info(
            "RLM iteration %d/%d\nReasoning: %s\nCode:\n%s",
            iteration + 1,
            self.max_iterations,
            pred.reasoning,
            pred.code,
        )

    # --- strip fences ---
    try:
        code = _strip_code_fences(pred.code)
    except SyntaxError as exc:
        code = pred.code
        result = f"[Error] {exc}"
        return self._process_execution_result(
            pred, code, result, history, output_field_names
        )

    # --- pre-execution hook ---
    if self._hook_pre_execution:
        pre_exec_out = self._hook_pre_execution(
            iteration, code, variables, history, input_args
        )
        if asyncio.iscoroutine(pre_exec_out):
            pre_exec_out = await pre_exec_out
        pre_exec_out = cast(PreExecutionOutput, pre_exec_out)
        code = pre_exec_out.code

    # --- execute ---
    result = self._execute_code(repl, code, input_args)

    # --- post-execution hook ---
    if self._hook_post_execution:
        post_exec_out = self._hook_post_execution(
            iteration, code, result, variables, history, input_args
        )
        if asyncio.iscoroutine(post_exec_out):
            post_exec_out = await post_exec_out
        post_exec_out = cast(PostExecutionOutput, post_exec_out)
        result = post_exec_out.result

    # --- process result ---
    processed = self._process_execution_result(
        pred, code, result, history, output_field_names
    )

    # --- post-iteration hook ---
    if self._hook_post_iteration and isinstance(processed, REPLHistory):
        post_iter_out = self._hook_post_iteration(
            iteration, pred, code, result, processed
        )
        if asyncio.iscoroutine(post_iter_out):
            post_iter_out = await post_iter_out
        post_iter_out = cast(PostIterationOutput, post_iter_out)
        processed = post_iter_out.history

        if post_iter_out.stop:
            return await self._aextract_fallback(
                variables, processed, output_field_names
            )

    return processed


def enable_rlm_hooks(
    rlm: Any,
    *,
    pre_iteration_hook: PreIterationHook | None = None,
    pre_execution_hook: PreExecutionHook | None = None,
    post_execution_hook: PostExecutionHook | None = None,
    post_iteration_hook: PostIterationHook | None = None,
) -> None:
    """Inject lifecycle hooks into a :class:`~dspy.RLM` instance.

    Monkey-patches the RLM's private ``_execute_iteration`` and
    ``_aexecute_iteration`` methods so that user-provided hooks are invoked
    at each stage of the iteration loop.

    Hooks can be **sync** or **async** — the system automatically detects
    coroutine return values via :func:`asyncio.iscoroutine` and handles both
    paths.

    Lifecycle order::

        pre_iteration_hook → generate_action → pre_execution_hook → execute → post_execution_hook → post_iteration_hook

    Args:
        rlm: The RLM instance to patch.  Must expose the internal methods
            ``_execute_iteration``, ``_aexecute_iteration``,
            ``_process_execution_result``, ``generate_action``/``generate_action.acall``,
            ``max_iterations``, and ``verbose``.
        pre_iteration_hook: Called before action generation.  May inject
            variables via :attr:`PreIterationOutput.extra_vars` or prepend
            persistent code via :attr:`PreIterationOutput.python_code`.
        pre_execution_hook: Called after code is generated, before execution.
            May rewrite the generated ``code`` string.
        post_execution_hook: Called after code executes, before the result is
            processed into history.  May transform or audit ``result``.
        post_iteration_hook: Called after the result is processed into history.
            May save learnings, trigger side effects, or modify history.

    Raises:
        AttributeError: If *rlm* does not expose the expected internal API.

    Example:
        >>> def my_pre_iter(iteration, variables, history, input_args):
        ...     return PreIterationOutput(extra_vars={"debug": True})
        ...
        >>> enable_rlm_hooks(rlm, pre_iteration_hook=my_pre_iter)
    """
    from dspy_rlm_hooks.predict_rlm_compat import _is_predict_rlm

    if _is_predict_rlm(rlm):
        from dspy_rlm_hooks.predict_rlm_compat import enable_predict_rlm_hooks

        enable_predict_rlm_hooks(
            rlm,
            pre_iteration_hook=pre_iteration_hook,
            pre_execution_hook=pre_execution_hook,
            post_execution_hook=post_execution_hook,
            post_iteration_hook=post_iteration_hook,
        )
        return

    _validate_rlm(rlm)

    # Store hook references on the instance
    rlm._hook_pre_iteration = pre_iteration_hook
    rlm._hook_pre_execution = pre_execution_hook
    rlm._hook_post_execution = post_execution_hook
    rlm._hook_post_iteration = post_iteration_hook

    # Bind patched methods
    rlm._execute_iteration = MethodType(_execute_iteration, rlm)
    rlm._aexecute_iteration = MethodType(_aexecute_iteration, rlm)
    rlm._execute_code = MethodType(_execute_code, rlm)


def disable_rlm_hooks(rlm: Any) -> None:
    """Remove lifecycle hooks from an RLM instance.

    Deletes the monkey-patched overrides and hook attributes that were added
    by :func:`enable_rlm_hooks`.  After calling this, the instance reverts
    to its original behaviour.

    Args:
        rlm: A previously patched RLM instance.

    Example:
        >>> disable_rlm_hooks(rlm)
    """
    from dspy_rlm_hooks.predict_rlm_compat import _is_predict_rlm

    if _is_predict_rlm(rlm):
        from dspy_rlm_hooks.predict_rlm_compat import disable_predict_rlm_hooks

        disable_predict_rlm_hooks(rlm)
        return

    for attr in (
        "_hook_pre_iteration",
        "_hook_pre_execution",
        "_hook_post_execution",
        "_hook_post_iteration",
        "_execute_iteration",
        "_aexecute_iteration",
        "_execute_code",
    ):
        if hasattr(rlm, attr):
            delattr(rlm, attr)
