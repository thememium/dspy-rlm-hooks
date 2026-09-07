"""Claiming hooks: wrap the real tool path so claimed speculations count.

Installed into ``repl.tools`` per-execution (fresh closures each ``forward()``).
The built-in LLM tools additionally get the closure-local ``llm_query`` counter
+ ``max_llm_calls`` limit re-implemented so a claimed call still counts as real
usage.
"""

from __future__ import annotations

import inspect
from functools import wraps
from typing import Any

from dspy_rlm_hooks.speculation.config import SpeculationConfig
from dspy_rlm_hooks.speculation.guards import fully_raw, tag_claim_hook
from dspy_rlm_hooks.speculation.speculator import Speculator

# The built-in LLM tools whose closure-local counter we re-implement on claim.
_LLM_TOOLS = ("llm_query", "llm_query_batched")


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
            raw = fully_raw(tools[name], fallback=tools[name])
            tools[name] = tag_claim_hook(
                _make_claim_hook(raw, claim_hook, name, max_llm_calls), raw_fn=raw
            )
        else:
            # Mirror the LLM branch: hide the raw hook's internal ``_tool=ToolSpec``
            # default from DSPy's tool registration (it is not JSON-serializable).
            # dspy 3.3.x wraps tools with __signature__ set; 3.2.x passes raw
            # functions whose signature must be COMPUTED here.
            raw = fully_raw(tools[name], fallback=tools[name])
            sig = getattr(raw, "__signature__", None)
            if sig is None:
                try:
                    sig = inspect.signature(raw)
                except (TypeError, ValueError):
                    sig = None
            if sig is not None:
                setattr(claim_hook, "__signature__", sig)
            tools[name] = tag_claim_hook(claim_hook, raw_fn=raw)
    if hasattr(repl, "_tools_registered"):
        # Only force tool re-registration when tool signatures actually changed.
        # _register_tools sends a JSON-RPC message to the sandbox (~0.6ms per
        # call with tools). Since claim hooks preserve the raw tool's signature,
        # re-registration is a no-op when the tool set is stable across iterations.
        try:
            sig_hash = hash(
                tuple(
                    (name, str(getattr(tools[name], "__signature__", None)))
                    for name in sorted(tools)
                )
            )
        except Exception:
            sig_hash = None
        if sig_hash is not None and sig_hash != getattr(
            rlm, "_spec_last_tool_sig_hash", None
        ):
            rlm._spec_last_tool_sig_hash = sig_hash
            repl._tools_registered = False
