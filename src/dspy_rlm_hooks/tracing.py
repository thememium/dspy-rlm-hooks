"""MLflow tracing wrapper for DSPy RLM lifecycle hooks.

The public :func:`dspy_rlm_hooks.enable_rlm_hooks` function automatically
uses these wrappers when MLflow is available.  The explicit
:func:`enable_rlm_hooks_with_tracing` entry point is retained for callers that
want missing MLflow to raise an error.  Hook inputs and outputs are recorded
on spans, making them visible alongside DSPy's native tracing.

MLflow is an **optional** dependency — install with::

    pip install mlflow

or::

    uv add mlflow

Example::

    import mlflow
    import dspy
    from dspy_rlm_hooks import enable_rlm_hooks, PreIterationOutput

    mlflow.dspy.autolog()  # enable DSPy-native tracing

    rlm = dspy.RLM(signature="question -> answer")

    def inject_context(iteration, variables, history, input_args):
        return PreIterationOutput(extra_vars={"context": "some data"})

    enable_rlm_hooks(rlm, pre_iteration_hook=inject_context)
    result = rlm(question="What is 2 + 2?")
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

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


def _load_mlflow() -> Any | None:
    """Return the user's MLflow module, or ``None`` when it is not installed.

    Only a genuinely missing top-level ``mlflow`` package is treated as an
    optional-dependency miss.  Import failures from inside an installed MLflow
    package are allowed to surface instead of silently disabling tracing.
    """
    try:
        import mlflow
    except ModuleNotFoundError as exc:
        if exc.name == "mlflow":
            return None
        raise
    return mlflow


def _import_mlflow() -> Any:
    """Import MLflow, raising a clear error if it is not installed."""
    mlflow = _load_mlflow()
    if mlflow is None:
        raise ImportError(
            "mlflow is required for tracing. Install it with: pip install mlflow"
        )
    if not callable(getattr(mlflow, "start_span", None)):
        raise ImportError(
            "MLflow tracing requires mlflow>=2.14.0 with the start_span API."
        )
    return mlflow


def _is_mlflow_tracing_available() -> bool:
    """Return whether MLflow's span API is available in the user environment."""
    mlflow = _load_mlflow()
    return mlflow is not None and callable(getattr(mlflow, "start_span", None))


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
        with mlflow.start_span(name="rlm_hook/pre_iteration") as span:
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
        with mlflow.start_span(name="rlm_hook/pre_iteration") as span:
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
        with mlflow.start_span(name="rlm_hook/pre_execution") as span:
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
        with mlflow.start_span(name="rlm_hook/pre_execution") as span:
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
        with mlflow.start_span(name="rlm_hook/post_execution") as span:
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
        with mlflow.start_span(name="rlm_hook/post_execution") as span:
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
        with mlflow.start_span(name="rlm_hook/post_iteration") as span:
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
        with mlflow.start_span(name="rlm_hook/post_iteration") as span:
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

    # Import the non-dispatching patcher here to avoid routing back through
    # automatic MLflow detection.
    from dspy_rlm_hooks.patcher import enable_rlm_hooks

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
