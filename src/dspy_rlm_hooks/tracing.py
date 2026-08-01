"""MLflow tracing wrapper for DSPy RLM lifecycle hooks.

This module provides :func:`enable_rlm_hooks_with_tracing`, a drop-in
replacement for :func:`enable_rlm_hooks` that wraps each hook with MLflow
span tracking.  Hook inputs and outputs are recorded as span attributes,
making them visible in the MLflow UI alongside DSPy's native tracing.

MLflow is an **optional** dependency — install with::

    pip install mlflow

or::

    uv add mlflow

Example::

    import mlflow
    import dspy
    from dspy_rlm_hooks import enable_rlm_hooks_with_tracing, PreIterationOutput

    mlflow.dspy.autolog()  # enable DSPy-native tracing

    rlm = dspy.RLM(signature="question -> answer")

    def inject_context(iteration, variables, history, input_args):
        return PreIterationOutput(extra_vars={"context": "some data"})

    enable_rlm_hooks_with_tracing(rlm, pre_iteration_hook=inject_context)
    result = rlm(question="What is 2 + 2?")
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from dspy_rlm_hooks.patcher import enable_rlm_hooks
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

logger = logging.getLogger(__name__)


def _import_mlflow():
    """Import mlflow, raising a clear error if not installed."""
    try:
        import mlflow
    except ImportError as exc:
        raise ImportError(
            "mlflow is required for tracing. Install it with: pip install mlflow"
        ) from exc
    return mlflow


def _safe_serialize(value: Any) -> Any:
    """Convert a value to something MLflow span attributes can store."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): _safe_serialize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_serialize(v) for v in value]
    # For complex objects (REPLVariable, REPLHistory, Prediction, etc.), use repr
    return repr(value)


def _make_traced_pre_iteration(hook: PreIterationHook) -> PreIterationHook:
    """Wrap a pre_iteration hook with MLflow span tracking."""

    def traced(
        iteration: int,
        variables: list[Any],
        history: list[Any],
        input_args: dict[str, Any],
    ) -> PreIterationOutput:
        mlflow = _import_mlflow()
        with mlflow.start_span(name=f"rlm_hook/pre_iteration/{iteration}") as span:
            span.set_inputs(
                {
                    "iteration": iteration,
                    "input_args": _safe_serialize(input_args),
                }
            )
            result = hook(iteration, variables, history, input_args)
            if asyncio.iscoroutine(result):
                result = asyncio.get_event_loop().run_until_complete(result)
            result = _ensure_type(result, PreIterationOutput)
            span.set_outputs(
                {
                    "extra_vars": _safe_serialize(result.extra_vars),
                    "python_code": result.python_code,
                }
            )
            return result

    async def async_traced(
        iteration: int,
        variables: list[Any],
        history: list[Any],
        input_args: dict[str, Any],
    ) -> PreIterationOutput:
        mlflow = _import_mlflow()
        with mlflow.start_span(name=f"rlm_hook/pre_iteration/{iteration}") as span:
            span.set_inputs(
                {
                    "iteration": iteration,
                    "input_args": _safe_serialize(input_args),
                }
            )
            result = hook(iteration, variables, history, input_args)
            if asyncio.iscoroutine(result):
                result = await result
            result = _ensure_type(result, PreIterationOutput)
            span.set_outputs(
                {
                    "extra_vars": _safe_serialize(result.extra_vars),
                    "python_code": result.python_code,
                }
            )
            return result

    # Return async version if the original hook is async
    if asyncio.iscoroutinefunction(hook):
        return async_traced
    return traced


def _make_traced_pre_execution(hook: PreExecutionHook) -> PreExecutionHook:
    """Wrap a pre_execution hook with MLflow span tracking."""

    def traced(
        iteration: int,
        code: str,
        variables: list[Any],
        history: list[Any],
        input_args: dict[str, Any],
    ) -> PreExecutionOutput:
        mlflow = _import_mlflow()
        with mlflow.start_span(name=f"rlm_hook/pre_execution/{iteration}") as span:
            span.set_inputs(
                {
                    "iteration": iteration,
                    "original_code": code,
                }
            )
            result = hook(iteration, code, variables, history, input_args)
            if asyncio.iscoroutine(result):
                result = asyncio.get_event_loop().run_until_complete(result)
            result = _ensure_type(result, PreExecutionOutput)
            span.set_outputs({"modified_code": result.code})
            return result

    async def async_traced(
        iteration: int,
        code: str,
        variables: list[Any],
        history: list[Any],
        input_args: dict[str, Any],
    ) -> PreExecutionOutput:
        mlflow = _import_mlflow()
        with mlflow.start_span(name=f"rlm_hook/pre_execution/{iteration}") as span:
            span.set_inputs(
                {
                    "iteration": iteration,
                    "original_code": code,
                }
            )
            result = hook(iteration, code, variables, history, input_args)
            if asyncio.iscoroutine(result):
                result = await result
            result = _ensure_type(result, PreExecutionOutput)
            span.set_outputs({"modified_code": result.code})
            return result

    if asyncio.iscoroutinefunction(hook):
        return async_traced
    return traced


def _make_traced_post_execution(hook: PostExecutionHook) -> PostExecutionHook:
    """Wrap a post_execution hook with MLflow span tracking."""

    def traced(
        iteration: int,
        code: str,
        result: Any,
        variables: list[Any],
        history: list[Any],
        input_args: dict[str, Any],
    ) -> PostExecutionOutput:
        mlflow = _import_mlflow()
        with mlflow.start_span(name=f"rlm_hook/post_execution/{iteration}") as span:
            span.set_inputs(
                {
                    "iteration": iteration,
                    "code": code,
                    "original_result": _safe_serialize(result),
                }
            )
            hook_result = hook(iteration, code, result, variables, history, input_args)
            if asyncio.iscoroutine(hook_result):
                hook_result = asyncio.get_event_loop().run_until_complete(hook_result)
            hook_result = _ensure_type(hook_result, PostExecutionOutput)
            span.set_outputs({"final_result": _safe_serialize(hook_result.result)})
            return hook_result

    async def async_traced(
        iteration: int,
        code: str,
        result: Any,
        variables: list[Any],
        history: list[Any],
        input_args: dict[str, Any],
    ) -> PostExecutionOutput:
        mlflow = _import_mlflow()
        with mlflow.start_span(name=f"rlm_hook/post_execution/{iteration}") as span:
            span.set_inputs(
                {
                    "iteration": iteration,
                    "code": code,
                    "original_result": _safe_serialize(result),
                }
            )
            hook_result = hook(iteration, code, result, variables, history, input_args)
            if asyncio.iscoroutine(hook_result):
                hook_result = await hook_result
            hook_result = _ensure_type(hook_result, PostExecutionOutput)
            span.set_outputs({"final_result": _safe_serialize(hook_result.result)})
            return hook_result

    if asyncio.iscoroutinefunction(hook):
        return async_traced
    return traced


def _make_traced_post_iteration(hook: PostIterationHook) -> PostIterationHook:
    """Wrap a post_iteration hook with MLflow span tracking."""

    def traced(
        iteration: int,
        pred: Any,
        code: str,
        result: Any,
        history: Any,
    ) -> PostIterationOutput:
        mlflow = _import_mlflow()
        with mlflow.start_span(name=f"rlm_hook/post_iteration/{iteration}") as span:
            span.set_inputs(
                {
                    "iteration": iteration,
                    "code": code,
                    "result": _safe_serialize(result),
                }
            )
            hook_result = hook(iteration, pred, code, result, history)
            if asyncio.iscoroutine(hook_result):
                hook_result = asyncio.get_event_loop().run_until_complete(hook_result)
            hook_result = _ensure_type(hook_result, PostIterationOutput)
            span.set_outputs({"stop": hook_result.stop})
            return hook_result

    async def async_traced(
        iteration: int,
        pred: Any,
        code: str,
        result: Any,
        history: Any,
    ) -> PostIterationOutput:
        mlflow = _import_mlflow()
        with mlflow.start_span(name=f"rlm_hook/post_iteration/{iteration}") as span:
            span.set_inputs(
                {
                    "iteration": iteration,
                    "code": code,
                    "result": _safe_serialize(result),
                }
            )
            hook_result = hook(iteration, pred, code, result, history)
            if asyncio.iscoroutine(hook_result):
                hook_result = await hook_result
            hook_result = _ensure_type(hook_result, PostIterationOutput)
            span.set_outputs({"stop": hook_result.stop})
            return hook_result

    if asyncio.iscoroutinefunction(hook):
        return async_traced
    return traced


def _ensure_type(value: Any, expected_type: type) -> Any:
    """Validate that a hook returned the expected type, or log a warning."""
    if isinstance(value, expected_type):
        return value
    logger.warning(
        "Hook returned %s, expected %s. Returning as-is.",
        type(value).__name__,
        expected_type.__name__,
    )
    return value


def enable_rlm_hooks_with_tracing(
    rlm: Any,
    *,
    pre_iteration_hook: PreIterationHook | None = None,
    pre_execution_hook: PreExecutionHook | None = None,
    post_execution_hook: PostExecutionHook | None = None,
    post_iteration_hook: PostIterationHook | None = None,
) -> None:
    """Inject lifecycle hooks with MLflow span tracking.

    Drop-in replacement for :func:`enable_rlm_hooks` that wraps each hook
    with MLflow span instrumentation.  Hook inputs and outputs are recorded
    as span attributes, making them visible in the MLflow UI.

    Requires MLflow to be installed (``pip install mlflow``).

    Args:
        rlm: The RLM instance to patch.
        pre_iteration_hook: Called before action generation.
        pre_execution_hook: Called after code generation, before execution.
        post_execution_hook: Called after code runs, before result processing.
        post_iteration_hook: Called after the iteration result is processed.

    Raises:
        ImportError: If MLflow is not installed.

    Example::

        import mlflow
        import dspy
        from dspy_rlm_hooks import enable_rlm_hooks_with_tracing, PreIterationOutput

        mlflow.dspy.autolog()

        rlm = dspy.RLM(signature="question -> answer")

        def inject_context(iteration, variables, history, input_args):
            return PreIterationOutput(extra_vars={"context": "some data"})

        enable_rlm_hooks_with_tracing(rlm, pre_iteration_hook=inject_context)
        result = rlm(question="What is 2 + 2?")
    """
    # Validate MLflow is available before patching
    _import_mlflow()

    enable_rlm_hooks(
        rlm,
        pre_iteration_hook=_make_traced_pre_iteration(pre_iteration_hook)
        if pre_iteration_hook
        else None,
        pre_execution_hook=_make_traced_pre_execution(pre_execution_hook)
        if pre_execution_hook
        else None,
        post_execution_hook=_make_traced_post_execution(post_execution_hook)
        if post_execution_hook
        else None,
        post_iteration_hook=_make_traced_post_iteration(post_iteration_hook)
        if post_iteration_hook
        else None,
    )
