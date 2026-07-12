"""Type definitions for the DSPy RLM hook system.



Usage::

    from dspy_rlm_hooks import (
        PreIterationHook,
        PreIterationOutput,
        enable_rlm_hooks,
    )
"""

from __future__ import annotations

from typing import Any, Awaitable, Protocol, runtime_checkable

import pydantic
from dspy.primitives.repl_types import REPLHistory


class PreIterationOutput(pydantic.BaseModel):
    """Data produced by a ``pre_iteration`` hook.

    Allows injecting variables and persistent Python code into the interpreter
    namespace before the LLM generates the next action.

    Attributes:
        extra_vars: Variables to inject into the interpreter namespace before
            the next code generation.  These are merged into ``input_args``.
        python_code: Code to prepend to generated code on every execution.
            Persists across iterations via ``repl.repl_globals``.
    """

    extra_vars: dict[str, Any] = pydantic.Field(default_factory=dict)
    python_code: str = ""


class PreExecutionOutput(pydantic.BaseModel):
    """Data produced by a ``pre_execution`` hook.

    Allows rewriting or augmenting the LLM-generated code before it is
    sent to the code interpreter.

    Attributes:
        code: The code to execute.  Usually the LLM-generated code, but can
            be rewritten (e.g. to add imports, fix patterns, inject helpers).
    """

    code: str


class PostExecutionOutput(pydantic.BaseModel):
    """Data produced by a ``post_execution`` hook.

    Allows transforming, auditing, or replacing the raw execution result
    before it is processed into history.

    Attributes:
        result: The result to pass to history.  Can transform or replace the
            raw execution result.
    """

    result: Any


class PostIterationOutput(pydantic.BaseModel):
    """Data produced by a ``post_iteration`` hook.

    Allows modifying or persisting the REPL history after the iteration
    result has been processed.

    Attributes:
        history: The updated history for the next iteration.  Return the same
            history to persist it, or a modified copy to change it.
        stop: If ``True``, immediately stop iterating and force-extract a final
            answer using the same extract-fallback path as max-iterations.
    """

    history: REPLHistory
    stop: bool = False


@runtime_checkable
class PreIterationHook(Protocol):
    """Called before action generation in each RLM iteration.

    Receives the current iteration index, variable state, history, and input
    arguments.  May return :class:`PreIterationOutput` (sync) or an
    ``Awaitable[PreIterationOutput]`` (async).
    """

    def __call__(
        self,
        iteration: int,
        variables: list[Any],
        history: list[Any],
        input_args: dict[str, Any],
    ) -> PreIterationOutput | Awaitable[PreIterationOutput]: ...


@runtime_checkable
class PreExecutionHook(Protocol):
    """Called after code generation, before execution.

    Receives the generated code and may rewrite it.

    Args:
        iteration: Current iteration index (0-based).
        code: The code after markdown fence stripping. May be truncated
            if the LLM output multiple code blocks.
        variables: Current REPL variables.
        history: REPL interaction history.
        input_args: Original input arguments.
        raw_code: The full, untruncated code as returned by the LLM,
            before any fence stripping. Use this to access code blocks
            that would otherwise be discarded by fence stripping.
    """

    def __call__(
        self,
        iteration: int,
        code: str,
        variables: list[Any],
        history: list[Any],
        input_args: dict[str, Any],
        *,
        raw_code: str = "",
    ) -> PreExecutionOutput | Awaitable[PreExecutionOutput]: ...


@runtime_checkable
class PostExecutionHook(Protocol):
    """Called after code execution, before result processing.

    Receives the raw execution result and may transform or audit it.

    Args:
        iteration: Current iteration index (0-based).
        code: The code that was executed (after fence stripping).
        result: The raw execution result.
        variables: Current REPL variables.
        history: REPL interaction history.
        input_args: Original input arguments.
        raw_code: The full, untruncated code as returned by the LLM,
            before any fence stripping. Use this to see what the LLM
            originally generated, including any discarded code blocks.
    """

    def __call__(
        self,
        iteration: int,
        code: str,
        result: Any,
        variables: list[Any],
        history: list[Any],
        input_args: dict[str, Any],
        *,
        raw_code: str = "",
    ) -> PostExecutionOutput | Awaitable[PostExecutionOutput]: ...


@runtime_checkable
class PostIterationHook(Protocol):
    """Called after the iteration result is processed into history.

    Receives the updated history and may modify or persist it.
    """

    def __call__(
        self,
        iteration: int,
        pred: Any,
        code: str,
        result: Any,
        history: REPLHistory,
    ) -> PostIterationOutput | Awaitable[PostIterationOutput]: ...


RLMHook = PreIterationHook | PreExecutionHook | PostExecutionHook | PostIterationHook
"""Union of all hook protocols.  Used for type narrowing."""
