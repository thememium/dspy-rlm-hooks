"""Integration layer: wire the speculation engine into DSPy's RLM execution path.

Task 7 — builds against the :class:`~dspy_rlm_hooks.speculator.Speculator`
facade (Task 6) and composes with the existing
:func:`~dspy_rlm_hooks.patcher.enable_rlm_hooks` (unchanged).

:func:`enable_rlm_speculation` wraps the single ``_execute_code`` choke point
(called by BOTH the sync ``_execute_iteration`` and async ``_aexecute_iteration``
paths) so that, per execution:

1. A **Lazy/JIT shadow pre-pass** runs over the FINAL assembled code the real
   interpreter actually executes (persistent prelude + injected vars via
   :func:`~dspy_rlm_hooks.utils._assemble_execution_code`), wrapped in
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

import builtins
import inspect
from functools import wraps
from types import MethodType
from typing import Any

from dspy_rlm_hooks.patcher import _validate_rlm
from dspy_rlm_hooks.speculation.config import SpeculationConfig
from dspy_rlm_hooks.speculation.shadow import shadow_builtins
from dspy_rlm_hooks.speculator import Speculator
from dspy_rlm_hooks.utils import _assemble_execution_code

# The built-in LLM tools whose closure-local counter we re-implement on claim.
_LLM_TOOLS = ("llm_query", "llm_query_batched")


def _placeholder(*args: Any, **kwargs: Any) -> Any:
    """Stand-in fn for a classification registered before the fresh per-execution
    closure is available. Replaced by :func:`_sync_registry_fns` before the
    shadow runs, so it is never actually invoked."""
    raise RuntimeError(
        "placeholder tool fn — should be replaced per-execution from repl.tools"
    )


def _register_classifications(
    spec: Speculator, config: SpeculationConfig, tools: Any
) -> None:
    """Register tool CLASSIFICATIONS once per RLM.

    The built-in ``llm_query``/``llm_query_batched`` are registered as
    speculatable+pure (per config flags). User tools are registered with their
    classification (``speculate_user_tools`` master switch). The actual
    functions are synced per-execution from the fresh ``repl.tools``.
    """
    if config.speculate_llm_query:
        spec.registry.register(
            "llm_query",
            _placeholder,
            speculatable=True,
            pure=True,
            deterministic=False,
            latency_hint_ms=1000.0,
        )
    if config.speculate_llm_query_batched:
        spec.registry.register(
            "llm_query_batched",
            _placeholder,
            speculatable=True,
            pure=True,
            deterministic=False,
            latency_hint_ms=1000.0,
        )
    if tools:
        for name, tool in tools.items():
            fn = getattr(tool, "func", tool)
            spec.registry.register(
                name,
                fn,
                speculatable=config.speculate_user_tools,
                pure=config.speculate_user_tools,
                deterministic=False,
                latency_hint_ms=1000.0,
            )


def _has_speculatable(spec: Speculator) -> bool:
    """True if any registered tool is speculatable (i.e. the shadow is worth
    running at all)."""
    return any(
        (t is not None and t.speculatable)
        for t in (spec.registry.get(n) for n in spec.registry.names())
    )


def _sync_registry_fns(spec: Speculator, repl: Any) -> None:
    """Point each registered ToolSpec's ``fn`` at the fresh per-execution closure
    from ``repl.tools`` (``_make_llm_tools`` returns fresh closures each
    forward). The shadow and real hooks both read ``tool.fn`` at call time, so
    this must happen before the shadow pre-pass dispatches."""
    tools = getattr(repl, "tools", None)
    if tools is None:
        return
    for name in spec.registry.names():
        tool = spec.registry.get(name)
        if tool is not None and name in tools:
            tool.fn = tools[name]


def _make_claim_hook(
    real_tool: Any, claim_hook: Any, name: str, max_llm_calls: int
) -> Any:
    """Wrap a claiming hook with the real ``llm_query`` counter accounting.

    The real counter is a closure-local inside ``_make_llm_tools``; on a claim
    hit the real tool is never called, so its counter would not increment. This
    wrapper re-implements the counter + ``max_llm_calls`` limit so a claimed
    call still counts as real usage. On a miss it delegates to the claiming hook
    (which runs the real tool, incrementing the real counter too — our counter
    is authoritative for the limit). For ``llm_query_batched`` the counter is
    incremented by the number of prompts (per-element claiming still accounts
    the whole batch as real usage).
    """
    counter = {"n": 0}
    sig = inspect.signature(real_tool)

    def _check_and_increment(n: int) -> None:
        if counter["n"] + n > max_llm_calls:
            raise RuntimeError(
                f"LLM call limit exceeded: {counter['n']} + {n} > {max_llm_calls}. "
                "Use Python code for aggregation instead of making more LLM calls."
            )
        counter["n"] += n

    def _normalize(args: tuple, kwargs: dict) -> tuple[tuple, dict]:
        # The shadow records calls positionally; the Deno interpreter calls the
        # tool with keyword args. Bind to the real signature so both produce the
        # same claim key.
        try:
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            return bound.args, bound.kwargs
        except TypeError:
            return args, kwargs

    if name.endswith("_batched"):

        @wraps(real_tool)
        def hook(prompts: Any, *args: Any, **kwargs: Any) -> Any:
            n = len(prompts) if isinstance(prompts, (list, tuple)) else 1
            _check_and_increment(n)
            norm_args, norm_kwargs = _normalize((prompts,) + args, kwargs)
            return claim_hook(*norm_args, **norm_kwargs)

    else:

        @wraps(real_tool)
        def hook(*args: Any, **kwargs: Any) -> Any:
            _check_and_increment(1)
            norm_args, norm_kwargs = _normalize(args, kwargs)
            return claim_hook(*norm_args, **norm_kwargs)

    # Preserve the real tool's signature so the Deno interpreter re-registers the
    # claim hook with the SAME parameter names (a bare *args/**kwargs wrapper
    # would register bogus `args`/`kwargs` params and break the tool call).
    setattr(hook, "__signature__", sig)
    return hook


def _install_claim_hooks(
    repl: Any, spec: Speculator, config: SpeculationConfig, rlm: Any
) -> None:
    """Install claiming hooks into the real tool path per-execution.

    Per the SPIKE: wrap ``repl.tools[name]`` and set ``_tools_registered=False``
    to force re-registration with the same signature. Only speculatable tools
    are wrapped; the built-in LLM tools additionally get counter accounting.
    """
    real_hooks = spec.hooks()
    max_llm_calls = getattr(rlm, "max_llm_calls", 50)
    tools = getattr(repl, "tools", None)
    if tools is None:
        return
    for name, claim_hook in real_hooks.items():
        if name not in tools:
            continue
        tool_spec = spec.registry.get(name)
        if tool_spec is None or not tool_spec.speculatable:
            continue
        if name in _LLM_TOOLS:
            tools[name] = _make_claim_hook(tools[name], claim_hook, name, max_llm_calls)
        else:
            tools[name] = claim_hook
    if hasattr(repl, "_tools_registered"):
        repl._tools_registered = False


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
    _sync_registry_fns(spec, repl)

    # --- Lazy/JIT shadow pre-pass over the assembled code -------------------
    if _has_speculatable(spec):
        t = None
        try:
            t = spec.session.begin_stream_turn(
                dict(input_args), shadow_builtins(dict(builtins.__dict__))
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

    # --- install claiming hooks into the real tool path ---------------------
    try:
        _install_claim_hooks(repl, spec, config, self)
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
) -> None:
    """Enable speculative execution on a :class:`~dspy.RLM` instance.

    Builds a :class:`Speculator` once per RLM, registers the built-in LLM tool
    classifications (plus any classified user tools), and wraps the current
    ``_execute_code`` (which may be the hooks-patched one) so every execution
    runs a Lazy/JIT shadow pre-pass and installs claiming hooks.

    Composes with :func:`~dspy_rlm_hooks.patcher.enable_rlm_hooks`: call hooks
    first, then speculation, for both to be active.

    Args:
        rlm: The RLM instance to patch (must expose the internal API validated
            by :func:`~dspy_rlm_hooks.patcher._validate_rlm`).
        tools: Optional mapping of user tool name -> callable (or ``Tool``) to
            classify. Only speculated when ``speculate_user_tools=True``.
        max_inflight: Max speculative executions in flight at once.
        max_dispatches_per_turn: Hard cap on speculative dispatches per turn.
        speculate_llm_query: Speculate the built-in ``llm_query`` tool.
        speculate_llm_query_batched: Speculate ``llm_query_batched``.
        speculate_user_tools: Master switch for user-registered tools.
        timeout_s: How long to wait on the shadow pre-pass before falling back
            to real execution.
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
    )
    spec = Speculator(
        max_inflight=max_inflight,
        max_dispatches_per_turn=max_dispatches_per_turn,
    )
    _register_classifications(spec, config, tools)

    original = rlm._execute_code
    rlm._speculator = spec
    rlm._speculation_config = config
    rlm._speculation_original_execute_code = original
    rlm._execute_code = MethodType(_speculation_execute_code, rlm)
    rlm._speculation_wrapper = rlm._execute_code


def disable_rlm_speculation(rlm: Any) -> None:
    """Remove speculation from an RLM instance.

    Restores whatever ``_execute_code`` was active before speculation (including
    a hooks-patched one) and shuts down the speculator. Idempotent. If hooks
    later overwrote the wrapper (Order 2 composition), the current
    ``_execute_code`` is left untouched so hooks keep working.
    """
    wrapper = getattr(rlm, "_speculation_wrapper", None)
    original = getattr(rlm, "_speculation_original_execute_code", None)
    if wrapper is not None and getattr(rlm, "_execute_code", None) is wrapper:
        if original is not None:
            rlm._execute_code = original
    for attr in (
        "_speculation_wrapper",
        "_speculation_original_execute_code",
        "_speculation_config",
    ):
        if hasattr(rlm, attr):
            delattr(rlm, attr)
    spec = getattr(rlm, "_speculator", None)
    if spec is not None:
        try:
            spec.close()
        except Exception:
            pass
        if hasattr(rlm, "_speculator"):
            delattr(rlm, "_speculator")
