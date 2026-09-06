"""Guards for the speculative engine (Task 5/7).

Small, shared helpers used by the hook factories, the launcher and the DSPy
integration to:

- distinguish a *claim hook* (a real-path wrapper installed into ``repl.tools``)
  from the *raw* tool implementation it wraps, and
- track which :class:`Speculation` a launcher worker is currently resolving, so
  a claim hook can never wait on its own (never-to-be-set) ``done`` event.

Background — the "hook leak" deadlock
-------------------------------------
``_sync_registry_fns`` re-points each registered ``ToolSpec.fn`` at the
per-execution closure from ``repl.tools``. If that dict still holds claim hooks
installed by a *previous* iteration (the repl only regenerates its tools when
``_tools_registered`` is False), a shadow dispatch executes the claim hook
*itself* as the speculative tool. The hook then claims its own freshly-
dispatched speculation (state ``pending``) and blocks on
``spec.result(timeout=600)`` — a future only its own worker (which is *us*)
would resolve. Every launcher worker stacks up behind such a self-claim, and
interpreter shutdown joining the pool blocks for the full 600s timeout.
"""

from __future__ import annotations

import threading
from typing import Any

# Attribute names used to tag hook wrappers and record the raw fn they wrap.
IS_CLAIM_HOOK = "_is_speculation_claim_hook"
RAW_FN = "_speculation_raw_fn"

# Thread-local: the Speculation the CURRENT thread is resolving. Set by the
# launcher worker around ``run_fn(...)``, read by the claim hooks.
_current = threading.local()


def tag_claim_hook(hook: Any, raw_fn: Any = None) -> Any:
    """Mark ``hook`` as a claim hook (and record the raw fn it wraps)."""
    setattr(hook, IS_CLAIM_HOOK, True)
    if raw_fn is not None:
        setattr(hook, RAW_FN, raw_fn)
    return hook


def is_claim_hook(fn: Any) -> bool:
    """True when ``fn`` is a claim hook, not the raw tool implementation."""
    return bool(getattr(fn, IS_CLAIM_HOOK, False))


def raw_of(fn: Any, fallback: Any = None) -> Any:
    """The raw tool behind a claim hook; ``fn`` itself when it is not a hook."""
    if not is_claim_hook(fn):
        return fn
    raw = getattr(fn, RAW_FN, None)
    return raw if raw is not None else fallback


def fully_raw(fn: Any, fallback: Any = None, _max_depth: int = 64) -> Any:
    """Strip EVERY layer of hook wrapping down to the true raw tool.

    Per-execution re-installation nests wrappers: generation N's tagged hook
    wraps generation N-1's, whose ``raw_fn`` is itself a counter wrapper built
    over the previous generation. ``raw_of`` unwraps only one tag layer, so a
    registry ``fn`` pointed at an intermediate wrapper re-enters the real claim
    hook on a miss and recurses (each level claims again, misses, and calls
    itself). This walks the tag chain and the ``functools.wraps`` chain until
    neither applies, so callers always reach the genuine implementation.
    """
    seen = 0
    while seen < _max_depth:
        nxt = raw_of(fn, fallback=None) if is_claim_hook(fn) else None
        if nxt is None:
            wrapped = getattr(fn, "__wrapped__", None)
            if wrapped is None or wrapped is fn:
                break
            nxt = wrapped
        fn = nxt
        seen += 1
    return fn if fn is not None else fallback


def raw_tool_fn(tool: Any) -> Any:
    """The callable to execute for ``tool`` — never a claim hook.

    Prefer the freshest raw fn (``tool.fn``); fall back to the fn recorded on
    the hook (the create-time snapshot) only if ``tool.fn`` was itself replaced
    by a claim hook (the pre-fix leak).
    """
    fn = tool.fn
    if is_claim_hook(fn):
        return raw_of(fn, fallback=fn)
    return fn


def mark_current(spec: Any) -> None:
    """Record the speculation the CURRENT worker thread is resolving."""
    _current.cur = spec


def clear_current() -> None:
    """Clear the current worker's speculation marker."""
    _current.cur = None


def current_spec() -> Any:
    """The speculation the current thread is resolving, or None."""
    return getattr(_current, "cur", None)
