"""Configuration for the speculative execution engine.

This module freezes the shared runtime configuration (Task 1). The store,
budget, and streaming engines (Tasks 2/3) import these types and build against
them. No engine implementation lives here.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pydantic


class SpeculationConfig(pydantic.BaseModel):
    """Runtime configuration for the speculative execution engine.

    Attributes:
        enabled: Master switch. When ``False`` the engine is inert and every
            call falls through to the real tool.
        max_inflight: Maximum number of speculative executions allowed to be
            in flight (dispatched but not yet claimed) at once.
        max_dispatches_per_turn: Hard cap on speculative dispatches issued in
            a single RLM turn.
        speculate_llm_query: Speculate the built-in ``llm_query`` tool.
        speculate_llm_query_batched: Speculate the built-in
            ``llm_query_batched`` tool.
        speculate_user_tools: Master switch for user-registered tools.
            Defaults to ``False`` — user tools are NOT speculatively executed
            unless explicitly marked via
            :func:`~dspy_rlm_hooks.speculation.tool.speculate`.
        timeout_s: How long (seconds) to wait on a speculative future before
            falling back to the real call.
        streaming: When ``True`` (default), the shadow feeds the model's
            streamed ``code`` output during ``generate_action`` so tool calls
            overlap with main-context token generation. When ``False``, the
            Lazy/JIT one-shot shadow runs over the fully assembled code block
            after generation (the pre-streaming behaviour).
    """

    enabled: bool = True
    max_inflight: int = 8
    max_dispatches_per_turn: int = 2048
    speculate_llm_query: bool = True
    speculate_llm_query_batched: bool = True
    speculate_user_tools: bool = False
    timeout_s: float = 5.0
    streaming: bool = True


@dataclass
class SpeculationPolicy:
    """Public classification a user fills in to mark a tool as speculatable.

    This is the *public* surface; :func:`~dspy_rlm_hooks.speculation.tool.speculate`
    folds it into an internal :class:`~dspy_rlm_hooks.speculation.tool.ToolSpec`.

    Attributes:
        speculatable: Opt-in to speculative execution. Requires ``pure=True``.
        pure: True when the tool has no observable side effects and may run
            early.
        deterministic: True when identical inputs always produce identical
            output.
        latency_hint_ms: Expected latency, used by the budget/scheduler.
        gate: Optional per-call predicate ``(args, kwargs) -> bool`` deciding
            whether a *specific* call may be speculated.
    """

    speculatable: bool = False
    pure: bool = False
    deterministic: bool = False
    latency_hint_ms: float = 1000.0
    gate: Callable[..., bool] | None = None
