"""Integration layer: wire the speculation engine into DSPy's RLM execution path.

Task 7 — builds against the :class:`~dspy_rlm_hooks.speculation.speculator.Speculator`
facade (Task 6) and composes with the existing
:func:`~dspy_rlm_hooks.core.patcher.enable_rlm_hooks` (unchanged).

:func:`enable_rlm_speculation` wraps the single ``_execute_code`` choke point
(called by BOTH the sync ``_execute_iteration`` and async ``_aexecute_iteration``
paths) so that, per execution:

1. A **Lazy/JIT shadow pre-pass** runs over the FINAL assembled code the real
   interpreter actually executes (persistent prelude + injected vars via
   :func:`~dspy_rlm_hooks.core.utils._assemble_execution_code`), wrapped in
   `` ```repl `` fences — :class:`StreamSegmenter` only emits segments inside
   fences (Task 6 finding).
2. **Claiming hooks** are installed into ``repl.tools`` per-execution (fresh
   closures each ``forward()``), wrapping the real tool and re-implementing the
   closure-local ``llm_query`` counter + ``max_llm_calls`` limit so a claimed
   call still counts as real usage.
3. After real execution, ``end_turn()`` evicts unclaimed speculations and resets
   the per-turn budget.

Composition
-----------
Speculation wraps the CURRENT ``_execute_code`` (which may be the hooks-patched
one). ``enable_rlm_hooks`` then ``enable_rlm_speculation`` (Order 1) composes:
hooks patch ``_execute_code`` first, speculation wraps it. The reverse order
(Order 2) leaves speculation inactive because ``enable_rlm_hooks`` overwrites
``_execute_code`` — the documented, expected behaviour given ``enable_rlm_hooks``
is unchanged. ``disable_rlm_speculation`` restores whatever ``_execute_code`` was
active before speculation (including a hooks-patched one), and is safe to call
even when hooks later overwrote the wrapper.
"""

from __future__ import annotations

from types import MethodType
from typing import TYPE_CHECKING, Any

from dspy_rlm_hooks.core.patcher import _validate_rlm
from dspy_rlm_hooks.speculation.config import SpeculationConfig
from dspy_rlm_hooks.speculation.integration.execute import (
    _speculation_aexecute_iteration,
    _speculation_execute_code,
    _speculation_execute_iteration,
)
from dspy_rlm_hooks.speculation.integration.registry import (
    _register_classifications,
    _register_speculator,
    _unregister_speculator,
)
from dspy_rlm_hooks.speculation.integration.streaming_turn import (
    _StreamingGenerateAction,
)
from dspy_rlm_hooks.speculation.speculator import Speculator

if TYPE_CHECKING:
    # Kept off the module import path: importing this module must stay cheap
    # because the speculation engine's shadow subprocess imports this package.
    pass


def enable_rlm_speculation(
    rlm: Any,
    *,
    tools: Any = None,
    max_inflight: int = 8,
    max_dispatches_per_turn: int = 2048,
    speculate_llm_query: bool = True,
    speculate_llm_query_batched: bool = True,
    speculate_user_tools: bool = False,
    timeout_s: float = 5.0,
    streaming: bool = True,
    persistent_shadow: bool = True,
    latency_aware: bool = True,
) -> None:
    """Enable speculative execution on a :class:`~dspy.RLM` instance.

    Builds a :class:`Speculator` once per RLM, registers the built-in LLM tool
    classifications (plus any classified user tools), and wraps the current
    ``_execute_code`` (which may be the hooks-patched one) so every execution
    runs a shadow pre-pass and installs claiming hooks.

    When ``streaming`` (default) the shadow feeds the model's streamed ``code``
    output during ``generate_action`` so sub-LLM tool calls overlap with
    main-context token generation (speculative programmatic tool calling). The
    ``generate_action`` ``dspy.Predict`` is wrapped in ``dspy.streamify`` and the
    patched iteration methods begin the streaming turn. When ``streaming=False``
    the original Lazy/JIT one-shot shadow runs over the fully assembled code
    block after generation. If streaming is unavailable or fails, execution
    transparently falls back to Lazy/JIT.

    Composes with :func:`~dspy_rlm_hooks.core.patcher.enable_rlm_hooks`: call hooks
    first, then speculation, for both to be active.

    Args:
        rlm: The RLM instance to patch (must expose the internal API validated
            by :func:`~dspy_rlm_hooks.core.patcher._validate_rlm`).
        tools: Optional mapping of user tool name -> callable (or ``Tool``) to
            classify. Only speculated when ``speculate_user_tools=True``.
        max_inflight: Max speculative executions in flight at once.
        max_dispatches_per_turn: Hard cap on speculative dispatches per turn.
        speculate_llm_query: Speculate the built-in ``llm_query`` tool.
        speculate_llm_query_batched: Speculate ``llm_query_batched``.
        speculate_user_tools: Master switch for user-registered tools.
        timeout_s: How long to wait on the shadow pre-pass before falling back
            to real execution.
        persistent_shadow: Keep one shadow subprocess warm across iterations
            (default True) instead of spawning per iteration.
        latency_aware: Track per-tool latency and let claims on in-flight
            speculations hedge (run the real tool) when waiting would cost
            more than duplicating the call (default True).
        streaming: Stream the ``code`` output during generation (default True).
            When False, use the Lazy/JIT one-shot shadow over the assembled code.
    """
    _validate_rlm(rlm)

    config = SpeculationConfig(
        enabled=True,
        max_inflight=max_inflight,
        max_dispatches_per_turn=max_dispatches_per_turn,
        speculate_llm_query=speculate_llm_query,
        speculate_llm_query_batched=speculate_llm_query_batched,
        speculate_user_tools=speculate_user_tools,
        timeout_s=timeout_s,
        streaming=streaming,
        persistent_shadow=persistent_shadow,
        latency_aware=latency_aware,
    )
    spec = Speculator(
        max_inflight=max_inflight,
        max_dispatches_per_turn=max_dispatches_per_turn,
        persistent_shadow=persistent_shadow,
        latency_aware=latency_aware,
    )
    _register_classifications(spec, config, tools, rlm=rlm)
    _register_speculator(spec)

    original = rlm._execute_code
    rlm._speculator = spec
    rlm._speculation_config = config
    rlm._speculation_original_execute_code = original
    rlm._active_stream_turn = None
    rlm._streaming_fed_any = False
    rlm._execute_code = MethodType(_speculation_execute_code, rlm)
    rlm._speculation_wrapper = rlm._execute_code

    if streaming:
        # Wrap the iteration methods (begin the streaming turn before generation)
        # and replace generate_action with a streaming wrapper.
        rlm._speculation_original_execute_iteration = rlm._execute_iteration
        rlm._speculation_original_aexecute_iteration = rlm._aexecute_iteration
        rlm._speculation_execute_iteration_wrapper = MethodType(
            _speculation_execute_iteration, rlm
        )
        rlm._speculation_aexecute_iteration_wrapper = MethodType(
            _speculation_aexecute_iteration, rlm
        )
        rlm._execute_iteration = rlm._speculation_execute_iteration_wrapper
        rlm._aexecute_iteration = rlm._speculation_aexecute_iteration_wrapper

        rlm._speculation_original_generate_action = rlm.generate_action
        rlm._speculation_generate_action_wrapper = _StreamingGenerateAction(rlm)
        rlm.generate_action = rlm._speculation_generate_action_wrapper


def disable_rlm_speculation(rlm: Any) -> None:
    """Remove speculation from an RLM instance.

    Restores whatever ``_execute_code``, iteration methods, and ``generate_action``
    were active before speculation (including hooks-patched ones) and shuts down
    the speculator. Idempotent. If hooks later overwrote a wrapper (Order 2
    composition), the current value is left untouched so hooks keep working.
    """
    wrapper = getattr(rlm, "_speculation_wrapper", None)
    original = getattr(rlm, "_speculation_original_execute_code", None)
    if wrapper is not None and getattr(rlm, "_execute_code", None) is wrapper:
        if original is not None:
            rlm._execute_code = original
    for cur_attr, wrap_attr, orig_attr in (
        (
            "_execute_iteration",
            "_speculation_execute_iteration_wrapper",
            "_speculation_original_execute_iteration",
        ),
        (
            "_aexecute_iteration",
            "_speculation_aexecute_iteration_wrapper",
            "_speculation_original_aexecute_iteration",
        ),
        (
            "generate_action",
            "_speculation_generate_action_wrapper",
            "_speculation_original_generate_action",
        ),
    ):
        wrapper = getattr(rlm, wrap_attr, None)
        if wrapper is not None and getattr(rlm, cur_attr, None) is wrapper:
            orig = getattr(rlm, orig_attr, None)
            if orig is not None:
                setattr(rlm, cur_attr, orig)
    for attr in (
        "_speculation_wrapper",
        "_speculation_original_execute_code",
        "_speculation_original_execute_iteration",
        "_speculation_original_aexecute_iteration",
        "_speculation_execute_iteration_wrapper",
        "_speculation_aexecute_iteration_wrapper",
        "_speculation_original_generate_action",
        "_speculation_generate_action_wrapper",
        "_speculation_config",
        "_active_stream_turn",
        "_streaming_fed_any",
    ):
        if hasattr(rlm, attr):
            delattr(rlm, attr)
    spec = getattr(rlm, "_speculator", None)
    if spec is not None:
        try:
            spec.close()
        except Exception:
            pass
        _unregister_speculator(spec)
        if hasattr(rlm, "_speculator"):
            delattr(rlm, "_speculator")
