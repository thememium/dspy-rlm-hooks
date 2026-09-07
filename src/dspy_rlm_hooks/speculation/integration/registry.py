"""Speculator registry: live-speculator tracking and tool classifications.

Owns the module-global registry of live :class:`Speculator` instances (the
``_active_speculators`` list, drained at interpreter exit) plus the tool
classification and registry-sync helpers shared by the execution and API
layers. The mutable state lives HERE and nowhere else; other modules in this
package reach it only through these functions (or the re-exported object).
"""

from __future__ import annotations

import atexit
import weakref
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from dspy_rlm_hooks.speculation.config import SpeculationConfig
from dspy_rlm_hooks.speculation.guards import fully_raw
from dspy_rlm_hooks.speculation.speculator import Speculator


def _prediction_type() -> type:
    """Runtime ``Prediction`` class, imported on first use."""
    from dspy.primitives.prediction import Prediction

    return Prediction


# Weak refs to live Speculators, drained at interpreter exit so the non-daemon
# launcher pool never blocks process shutdown when the caller does not call
# ``disable_rlm_speculation`` explicitly.
_active_speculators: "list[weakref.ref]" = []


def _close_speculators() -> None:
    """Interpreter-exit hook: close every still-live Speculator."""
    for ref in list(_active_speculators):
        spec = ref()
        if spec is not None:
            try:
                spec.close()
            except Exception:
                pass
    _active_speculators.clear()


atexit.register(_close_speculators)


def _register_speculator(spec: Speculator) -> None:
    _active_speculators.append(weakref.ref(spec))


def _unregister_speculator(spec: Speculator) -> None:
    for ref in list(_active_speculators):
        if ref() is spec:
            _active_speculators.remove(ref)


def _placeholder(*args: Any, **kwargs: Any) -> Any:
    """Stand-in fn for a classification registered before the fresh per-execution
    closure is available. Replaced by :func:`_sync_registry_fns` before the
    shadow runs, so it is never actually invoked."""
    raise RuntimeError(
        "placeholder tool fn — should be replaced per-execution from repl.tools"
    )


def _extract_sub_lm_text(response: Any) -> str:
    """Extract the text from a sub-LM response — best-effort, mirroring dspy's
    ``_query_lm`` shapes with a ``str`` fallback.

    Speculative results are best-effort predictions: the strict response
    contract stays enforced by the REAL call (a claimed miss re-runs it), so a
    lenient fallback here never changes what the model finally receives when
    the sub-LM is genuinely misconfigured."""
    import dspy  # lazy: this module must stay importable without dspy

    lm_response = getattr(dspy, "LMResponse", None)
    if lm_response is not None and isinstance(response, lm_response):
        text = response.text
    elif isinstance(response, list) and response:
        first = response[0]
        text = first.get("text") if isinstance(first, dict) else first
    else:
        text = str(response)
    return text if isinstance(text, str) else str(text)


def _make_llm_spec_fns(rlm: Any) -> dict[str, Callable]:
    """Counter-free speculative executors for the built-in LLM tools.

    dspy's raw ``llm_query`` closure increments the ``max_llm_calls`` budget on
    EVERY execution — including speculative ones — so wasted bets (evicted
    peeks, re-plan churn) consumed the model's logical budget. Speculative
    executions call the sub-LM directly instead: they do not consume the
    logical budget (enforced by the claim-hook counter on model-requested
    calls only) and stay bounded by the speculation budget
    (``max_dispatches_per_turn``) plus per-prompt dedup.
    """

    def _query(prompt: str) -> str:
        import dspy  # lazy: this module must stay importable without dspy

        lm = getattr(rlm, "sub_lm", None) or dspy.settings.lm
        if lm is None:
            # dspy 3.2.x exposes this as RuntimeError, 3.3.x as LMNotConfiguredError
            err = getattr(dspy, "LMNotConfiguredError", RuntimeError)
            raise err(
                "No LM configured. Use dspy.configure(lm=...) or pass sub_lm to RLM."
            )
        return _extract_sub_lm_text(lm(prompt))

    def llm_query(prompt: str) -> str:
        if not prompt:
            raise ValueError("prompt cannot be empty")
        return _query(prompt)

    def llm_query_batched(prompts: list) -> list:
        if not prompts:
            return []
        with ThreadPoolExecutor(max_workers=8) as executor:
            return list(executor.map(_query, prompts))

    return {"llm_query": llm_query, "llm_query_batched": llm_query_batched}


def _register_classifications(
    spec: Speculator, config: SpeculationConfig, tools: Any, rlm: Any = None
) -> None:
    """Register tool CLASSIFICATIONS once per RLM.

    The built-in ``llm_query``/``llm_query_batched`` are registered as
    speculatable+pure (per config flags) with COUNTER-FREE speculative
    executors (see :func:`_make_llm_spec_fns`). User tools are registered with
    their classification (``speculate_user_tools`` master switch). The actual
    functions are synced per-execution from the fresh ``repl.tools``.
    """
    spec_fns = _make_llm_spec_fns(rlm) if rlm is not None else {}
    if config.speculate_llm_query:
        spec.registry.register(
            "llm_query",
            _placeholder,
            speculatable=True,
            pure=True,
            deterministic=False,
            latency_hint_ms=1000.0,
            spec_fn=spec_fns.get("llm_query"),
        )
    if config.speculate_llm_query_batched:
        spec.registry.register(
            "llm_query_batched",
            _placeholder,
            speculatable=True,
            pure=True,
            deterministic=False,
            latency_hint_ms=1000.0,
            spec_fn=spec_fns.get("llm_query_batched"),
        )
    if tools:
        for name, tool in tools.items():
            # Accepted forms: a plain callable, a dspy ``Tool`` (uses ``.func``),
            # or a ``(callable, policy_kwargs)`` pair for per-tool overrides.
            policy_kwargs: dict[str, Any] = {}
            if isinstance(tool, tuple) and len(tool) == 2 and callable(tool[0]):
                fn, policy_kwargs = tool
                policy_kwargs = dict(policy_kwargs or {})
            else:
                fn = getattr(tool, "func", tool)
            spec.registry.register(
                name,
                fn,
                speculatable=config.speculate_user_tools,
                pure=config.speculate_user_tools,
                deterministic=bool(policy_kwargs.get("deterministic", False)),
                latency_hint_ms=float(policy_kwargs.get("latency_hint_ms", 1000.0)),
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
    this must happen before the shadow pre-pass dispatches.

    A claim hook (left in ``repl.tools`` by a previous iteration's
    ``_install_claim_hooks``) is NEVER synced: executing a hook as the
    speculative fn makes it claim+wait on its own pending speculation — a
    self-claim deadlock that blocks the launcher pool (and interpreter
    shutdown). Raw fns are cached per tool so a hooked entry falls back to the
    last known raw implementation.
    """
    tools = getattr(repl, "tools", None)
    if tools is None:
        return
    for name in spec.registry.names():
        tool = spec.registry.get(name)
        if tool is not None and name in tools:
            candidate = fully_raw(tools[name], fallback=None)
            if candidate is None:
                candidate = spec._raw_fns.get(name)
            if candidate is None:
                continue  # leave the existing (raw) fn untouched
            spec._raw_fns[name] = candidate
            tool.fn = candidate
