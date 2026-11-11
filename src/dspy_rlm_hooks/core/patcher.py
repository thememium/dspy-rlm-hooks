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
import contextlib
import inspect
import logging
from types import MethodType
from typing import TYPE_CHECKING, Any, cast

from dspy_rlm_hooks.core.types import (
    PostExecutionHook,
    PostExecutionOutput,
    PostIterationHook,
    PostIterationOutput,
    PreExecutionHook,
    PreExecutionOutput,
    PreIterationHook,
    PreIterationOutput,
)
from dspy_rlm_hooks.core.utils import (
    _assemble_execution_code,
    _prepend_python_code,
    _strip_code_fences,
    _with_prompt_context,
)

if TYPE_CHECKING:
    # Imported lazily so importing this module (e.g. from the speculation
    # engine's shadow subprocess) never pays the dspy import cost.
    from dspy.primitives.prediction import Prediction
    from dspy.primitives.repl_types import REPLHistory, REPLVariable


def _repl_history_type() -> type:
    """Runtime ``REPLHistory`` class, imported on first use (kept off the
    module import path so the shadow subprocess stays lightweight)."""
    from dspy.primitives.repl_types import REPLHistory

    return REPLHistory


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


# ``max_iters`` is intentionally NOT in this tuple: its attribute name changed
# across DSPy versions (``max_iterations`` in <3.3, ``max_iters`` in >=3.3) and
# is validated separately in :func:`_validate_rlm`.
_REQUIRED_METHODS = (
    "_execute_iteration",
    "_aexecute_iteration",
    "_process_execution_result",
    "generate_action",
    "verbose",
)


def _max_iterations(rlm: Any) -> int:
    for attr in ("max_iters", "max_iterations"):
        value = getattr(rlm, attr, None)
        if value is not None:
            return value
    raise AttributeError(
        "RLM instance missing required attributes: max_iters. "
        "Ensure you are passing a dspy.RLM instance from dspy>=3.1."
    )


def _validate_rlm(rlm: Any) -> None:
    """Ensure *rlm* exposes the internal methods we need to patch."""
    missing = [name for name in _REQUIRED_METHODS if not hasattr(rlm, name)]
    if not (hasattr(rlm, "max_iters") or hasattr(rlm, "max_iterations")):
        missing.append("max_iters")
    if missing:
        raise AttributeError(
            f"RLM instance missing required attributes: {', '.join(missing)}. "
            "Ensure you are passing a dspy.RLM instance from dspy>=3.1."
        )


def _with_rlm_prelude(code: str) -> str:
    """Apply DSPy >=3.5's per-execution sandbox prelude when it exists.

    DSPy 3.5 runs every REPL execution inside the active sandbox session and
    prefixes a rebind prelude once the interpreter has imported ``dspy``. The
    hook system owns ``_execute_code``, so it must reproduce both; on older
    releases there is nothing to import and the code passes through.
    """
    try:
        from dspy.predict.rlm import _repl_prelude, _with_prelude
    except ImportError:
        return code
    return _with_prelude(_repl_prelude.get(), code)


def _repl_session_scope():
    """DSPy >=3.5's active REPL session scope, or a no-op context."""
    try:
        from dspy.predict.rlm import _repl_session
    except ImportError:
        return contextlib.nullcontext()
    session = _repl_session.get()
    return session.scope() if session is not None else contextlib.nullcontext()


def _execute_code(self: Any, repl: Any, code: str, input_args: dict[str, Any]) -> Any:
    """Run ``code`` inside the REPL, prepending any persisted globals.

    Mirrors the original ``RLM._execute_code`` logic and is bound as a
    replacement method during :func:`enable_rlm_hooks`.
    """
    code = _with_rlm_prelude(_assemble_execution_code(repl, code))
    try:
        with _repl_session_scope():
            return repl.execute(code, variables=dict(input_args))
    except Exception as exc:  # noqa: BLE001
        # DSPy >=3.4 separates repairable submitted-code errors from fatal
        # process/protocol failures. A dead interpreter cannot execute the
        # model's repairs; let the owner create a new session rather than
        # spending every remaining iteration on the same terminal error.
        from dspy.primitives import code_interpreter

        recoverable = getattr(code_interpreter, "CodeExecutionError", None)
        # Older DSPy uses CodeInterpreterError for both categories, so keep
        # its original feedback behavior when no distinction is available.
        if (
            recoverable is not None
            and isinstance(exc, code_interpreter.CodeInterpreterError)
            and not isinstance(exc, recoverable)
        ):
            raise
        return f"[Error] {exc}"


def _uses_entry_tuple_contract(rlm: Any) -> bool:
    """Whether the RLM returns ``(REPLEntry, final_outputs)`` per iteration.

    DSPy >=3.5 replaced the "processed history or final ``Prediction``" return
    of ``_execute_iteration``/``_process_execution_result`` with that tuple and
    removed ``history`` from ``_process_execution_result``'s parameters. Older
    releases take history explicitly, which is the structural signal used here
    (the class, not the instance, so mock-based callers keep the old contract).
    """
    method = getattr(type(rlm), "_process_execution_result", None)
    if method is None:
        return False
    try:
        parameters = inspect.signature(method).parameters
    except (TypeError, ValueError):
        return False
    return "history" not in parameters


def _action_history_kwarg(rlm: Any) -> str:
    """``generate_action``'s history input name: ``history`` on DSPy >=3.5."""
    return "history" if _uses_entry_tuple_contract(rlm) else "repl_history"


def _generate_action_signature_kwargs(
    rlm: Any, interpreter_factory: Any, sub_agent_rules: Any = None
) -> dict[str, Any]:
    """Per-call ``signature=`` kwargs for ``generate_action``.

    DSPy >=3.4 passes ``interpreter_factory`` down to ``_execute_iteration``/
    ``_aexecute_iteration`` and derives a per-call action signature from it via
    ``RLM._action_signature_for_current_factory``. DSPy >=3.4.1 passes the
    interpreter's sub-agent guidance (``sub_agent_rules``) alongside it, and
    the helper requires both. Older DSPy has neither, so this returns ``{}``
    and the call stays exactly as it was on DSPy <=3.3.
    """
    helper = getattr(rlm, "_action_signature_for_current_factory", None)
    if helper is None or interpreter_factory is None:
        return {}
    try:
        parameters = inspect.signature(helper).parameters
    except (TypeError, ValueError):
        parameters = {}
    if "sub_agent_rules" in parameters:
        rules = sub_agent_rules
        if rules is None:
            rules = getattr(rlm, "_initial_sub_agent_rules", "")
        return {"signature": helper(interpreter_factory, rules)}
    return {"signature": helper(interpreter_factory)}


def _process_result(
    rlm: Any,
    action: Any,
    code: str,
    result: Any,
    history: Any,
    output_field_names: list[str],
) -> Any:
    """Call the release-appropriate ``_process_execution_result`` signature.

    DSPy 3.1 takes ``(pred, result, history, output_field_names)`` — the code
    lives on ``pred``. 3.2–3.4 add a separate ``code`` argument, and 3.5 drops
    ``history`` and returns ``(REPLEntry, final_outputs)``.
    """
    if _uses_entry_tuple_contract(rlm):
        return rlm._process_execution_result(action, code, result, output_field_names)

    method = getattr(type(rlm), "_process_execution_result", None)
    try:
        parameters = inspect.signature(method).parameters if method else {}
    except (TypeError, ValueError):
        parameters = {}
    if parameters and "code" not in parameters:
        return rlm._process_execution_result(
            action, result, history, output_field_names
        )
    return rlm._process_execution_result(
        action, code, result, history, output_field_names
    )


def _hook_history(rlm: Any, history: Any, iteration: int) -> Any:
    """History view passed to pre/post-execution hooks.

    DSPy >=3.5 hands iterations a ``dspy.History`` and keeps the turn's
    ``REPLEntry`` list in the caller's loop. The hook API has always exposed
    ``REPLHistory`` (``append``/``format``), so build that view from the
    entries the hook system tracks for the current turn.
    """
    if not _uses_entry_tuple_contract(rlm):
        return history
    entries = [] if iteration == 0 else list(getattr(rlm, "_hook_turn_entries", []))
    return _repl_history_type()(entries=entries)


def _track_entry_tuple(
    rlm: Any, iteration: int, processed: tuple[Any, Any]
) -> list[Any]:
    """Record this iteration's entry for the hook-visible history."""
    entries = [] if iteration == 0 else list(getattr(rlm, "_hook_turn_entries", []))
    entries.append(processed[0])
    rlm._hook_turn_entries = entries
    return entries


def _post_iteration_entry_tuple(
    rlm: Any,
    iteration: int,
    action: Any,
    code: str,
    result: Any,
    processed: tuple[Any, dict[str, Any] | None],
    variables: list[REPLVariable],
    history: Any,
    output_field_names: list[str],
) -> tuple[Any, dict[str, Any] | None]:
    """Run the post-iteration hook against DSPy >=3.5's tuple contract.

    The hook API exposes the accumulated ``REPLHistory``; 3.5 keeps its entries
    in the caller's loop, so the hook system tracks them per turn and returns
    the (possibly replaced) last entry for that loop to record. ``stop=True``
    feeds the accumulated history to the extract fallback and returns its
    outputs as this iteration's final outputs.
    """
    entries = _track_entry_tuple(rlm, iteration, processed)
    if processed[1] is not None or not rlm._hook_post_iteration:
        return processed
    repl_entry, final_outputs = processed

    post_iter_out = rlm._hook_post_iteration(
        iteration, action, code, result, _repl_history_type()(entries=entries)
    )
    if asyncio.iscoroutine(post_iter_out):
        post_iter_out = _run_async(post_iter_out)
    post_iter_out = cast(PostIterationOutput, post_iter_out)

    entries = list(post_iter_out.history.entries)
    rlm._hook_turn_entries = entries
    repl_entry = entries[-1] if entries else repl_entry

    if post_iter_out.stop:
        fallback = rlm._extract_fallback(
            variables, history, entries, output_field_names
        )
        return repl_entry, {
            name: getattr(fallback, name) for name in output_field_names
        }
    return repl_entry, final_outputs


async def _apost_iteration_entry_tuple(
    rlm: Any,
    iteration: int,
    action: Any,
    code: str,
    result: Any,
    processed: tuple[Any, dict[str, Any] | None],
    variables: list[REPLVariable],
    history: Any,
    output_field_names: list[str],
) -> tuple[Any, dict[str, Any] | None]:
    """Async :func:`_post_iteration_entry_tuple`."""
    entries = _track_entry_tuple(rlm, iteration, processed)
    if processed[1] is not None or not rlm._hook_post_iteration:
        return processed
    repl_entry, final_outputs = processed

    post_iter_out = rlm._hook_post_iteration(
        iteration, action, code, result, _repl_history_type()(entries=entries)
    )
    if asyncio.iscoroutine(post_iter_out):
        post_iter_out = await post_iter_out
    post_iter_out = cast(PostIterationOutput, post_iter_out)

    entries = list(post_iter_out.history.entries)
    rlm._hook_turn_entries = entries
    repl_entry = entries[-1] if entries else repl_entry

    if post_iter_out.stop:
        fallback = rlm._extract_fallback(
            variables, history, entries, output_field_names
        )
        return repl_entry, {
            name: getattr(fallback, name) for name in output_field_names
        }
    return repl_entry, final_outputs


def _execute_iteration(
    self: Any,
    repl: Any,
    variables: list[REPLVariable],
    history: REPLHistory,
    iteration: int,
    input_args: dict[str, Any],
    output_field_names: list[str],
    interpreter_factory: Any = None,
    sub_agent_rules: Any = None,
) -> Prediction | REPLHistory | tuple[Any, dict[str, Any] | None]:
    """Synchronous RLM iteration with hook support.

    Lifecycle::

        pre_iteration → generate_action → pre_execution → execute → post_execution → post_iteration
    """
    # --- pre-iteration hook ---
    iteration_python_code = ""
    prompt_context = ""
    # DSPy >=3.5 passes dspy.History; hooks keep the REPLHistory contract.
    hook_history = _hook_history(self, history, iteration)
    if self._hook_pre_iteration:
        pre_iter_out = self._hook_pre_iteration(
            iteration, variables, hook_history, input_args
        )
        if asyncio.iscoroutine(pre_iter_out):
            pre_iter_out = _run_async(pre_iter_out)
        pre_iter_out = cast(PreIterationOutput, pre_iter_out)
        input_args = {**input_args, **pre_iter_out.extra_vars}
        iteration_python_code = pre_iter_out.python_code
        prompt_context = pre_iter_out.prompt_context
        if pre_iter_out.persistent_python_code is not None:
            repl.repl_globals = pre_iter_out.persistent_python_code

    # --- action generation ---
    variables_info = _with_prompt_context(
        [variable.format() for variable in variables], prompt_context
    )
    action = self.generate_action(
        **_generate_action_signature_kwargs(self, interpreter_factory, sub_agent_rules),
        variables_info=variables_info,
        **{_action_history_kwarg(self): history},
        iteration=f"{iteration + 1}/{_max_iterations(self)}",
    )

    if self.verbose:
        logger.info(
            "RLM iteration %d/%d\nReasoning: %s\nCode:\n%s",
            iteration + 1,
            _max_iterations(self),
            action.reasoning,
            action.code,
        )

    # --- strip fences ---
    try:
        code = _strip_code_fences(action.code)
    except SyntaxError as exc:
        code = action.code
        result = f"[Error] {exc}"
        return _process_result(self, action, code, result, history, output_field_names)

    # --- pre-execution hook ---
    if self._hook_pre_execution:
        pre_exec_out = self._hook_pre_execution(
            iteration, code, variables, hook_history, input_args
        )
        if asyncio.iscoroutine(pre_exec_out):
            pre_exec_out = _run_async(pre_exec_out)
        pre_exec_out = cast(PreExecutionOutput, pre_exec_out)
        code = pre_exec_out.code

    # --- execute ---
    code = _prepend_python_code(code, iteration_python_code)
    result = self._execute_code(repl, code, input_args)

    # --- post-execution hook ---
    if self._hook_post_execution:
        post_exec_out = self._hook_post_execution(
            iteration, code, result, variables, hook_history, input_args
        )
        if asyncio.iscoroutine(post_exec_out):
            post_exec_out = _run_async(post_exec_out)
        post_exec_out = cast(PostExecutionOutput, post_exec_out)
        result = post_exec_out.result

    # --- process result ---
    processed = _process_result(self, action, code, result, history, output_field_names)

    # --- post-iteration hook ---
    if _uses_entry_tuple_contract(self):
        return _post_iteration_entry_tuple(
            self,
            iteration,
            action,
            code,
            result,
            processed,
            variables,
            history,
            output_field_names,
        )

    if self._hook_post_iteration and isinstance(processed, _repl_history_type()):
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
    interpreter_factory: Any = None,
    sub_agent_rules: Any = None,
) -> Prediction | REPLHistory | tuple[Any, dict[str, Any] | None]:
    """Asynchronous RLM iteration with hook support.

    Mirrors :func:`_execute_iteration` but uses ``await`` for async hooks
    and ``generate_action.acall``.
    """
    # --- pre-iteration hook ---
    iteration_python_code = ""
    prompt_context = ""
    # DSPy >=3.5 passes dspy.History; hooks keep the REPLHistory contract.
    hook_history = _hook_history(self, history, iteration)
    if self._hook_pre_iteration:
        pre_iter_out = self._hook_pre_iteration(
            iteration, variables, hook_history, input_args
        )
        if asyncio.iscoroutine(pre_iter_out):
            pre_iter_out = await pre_iter_out
        pre_iter_out = cast(PreIterationOutput, pre_iter_out)
        input_args = {**input_args, **pre_iter_out.extra_vars}
        iteration_python_code = pre_iter_out.python_code
        prompt_context = pre_iter_out.prompt_context
        if pre_iter_out.persistent_python_code is not None:
            repl.repl_globals = pre_iter_out.persistent_python_code

    # --- action generation ---
    variables_info = _with_prompt_context(
        [variable.format() for variable in variables], prompt_context
    )
    pred = await self.generate_action.acall(
        **_generate_action_signature_kwargs(self, interpreter_factory, sub_agent_rules),
        variables_info=variables_info,
        **{_action_history_kwarg(self): history},
        iteration=f"{iteration + 1}/{_max_iterations(self)}",
    )

    if self.verbose:
        logger.info(
            "RLM iteration %d/%d\nReasoning: %s\nCode:\n%s",
            iteration + 1,
            _max_iterations(self),
            pred.reasoning,
            pred.code,
        )

    # --- strip fences ---
    try:
        code = _strip_code_fences(pred.code)
    except SyntaxError as exc:
        code = pred.code
        result = f"[Error] {exc}"
        return _process_result(self, pred, code, result, history, output_field_names)

    # --- pre-execution hook ---
    if self._hook_pre_execution:
        pre_exec_out = self._hook_pre_execution(
            iteration, code, variables, hook_history, input_args
        )
        if asyncio.iscoroutine(pre_exec_out):
            pre_exec_out = await pre_exec_out
        pre_exec_out = cast(PreExecutionOutput, pre_exec_out)
        code = pre_exec_out.code

    # --- execute ---
    code = _prepend_python_code(code, iteration_python_code)
    result = self._execute_code(repl, code, input_args)

    # --- post-execution hook ---
    if self._hook_post_execution:
        post_exec_out = self._hook_post_execution(
            iteration, code, result, variables, hook_history, input_args
        )
        if asyncio.iscoroutine(post_exec_out):
            post_exec_out = await post_exec_out
        post_exec_out = cast(PostExecutionOutput, post_exec_out)
        result = post_exec_out.result

    # --- process result ---
    processed = _process_result(self, pred, code, result, history, output_field_names)

    # --- post-iteration hook ---
    if _uses_entry_tuple_contract(self):
        return await _apost_iteration_entry_tuple(
            self,
            iteration,
            pred,
            code,
            result,
            processed,
            variables,
            history,
            output_field_names,
        )

    if self._hook_post_iteration and isinstance(processed, _repl_history_type()):
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
            ``max_iters`` (or ``max_iterations`` in DSPy <3.3), and ``verbose``.
        pre_iteration_hook: Called before action generation. May inject
            interpreter variables, current or persistent Python preludes, and
            iteration-local action-generation context.
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
    from dspy_rlm_hooks.core.predict_rlm_compat import _is_predict_rlm

    if _is_predict_rlm(rlm):
        from dspy_rlm_hooks.core.predict_rlm_compat import enable_predict_rlm_hooks

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
    from dspy_rlm_hooks.core.predict_rlm_compat import _is_predict_rlm

    if _is_predict_rlm(rlm):
        from dspy_rlm_hooks.core.predict_rlm_compat import disable_predict_rlm_hooks

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
