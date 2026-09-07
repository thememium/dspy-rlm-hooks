"""Tool contracts for the speculative execution engine.

This module freezes the shared tool contracts (Task 1) that the store/budget
(Task 2) and streaming (Task 3) engines import and build against. It contains
only the tool classification surface — no store/budget/streaming/shadow
implementation.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

from dspy_rlm_hooks.speculation.config import SpeculationPolicy

SpecKey = tuple[str, str]  # (tool_name, canonical args hash)


# ---------------------------------------------------------------------------
# Speculation future contract (documented here, implemented in Task 2 store.py)
# ---------------------------------------------------------------------------
# The `Speculation` future dataclass is defined by the store (Task 2). This
# module only documents the fields the engine relies on so Task 2 matches:
#
#   @dataclass
#   class Speculation:
#       key: SpecKey                  # (tool_name, canonical args hash)
#       seq: int                      # monotonic dispatch sequence
#       args: tuple                   # the speculated call args
#       kwargs: dict                  # the speculated call kwargs
#       source: str                   # "shadow" | "real" | ...
#       state: str                    # "pending" | "running" | "done" | "cancelled"
#       result: Any | None = None     # resolved value once done
#       error: BaseException | None = None  # raised error once done
#       done: threading.Event         # set when the future resolves
#       adopted: bool = False         # claimed by the real REPL
#       cancel: Callable[[], None] | None = None
#       dispatched_at: float | None = None  # monotonic() timestamp
#       resolved_at: float | None = None    # monotonic() timestamp
#       def wait(self, timeout: float | None = None) -> bool: ...
#       def result(self, timeout: float | None = None) -> Any: ...
# ---------------------------------------------------------------------------


def canonical_hash(tool_name: str, args: tuple, kwargs: dict) -> str:
    """Stable 16-hex-char identity for one concrete call.

    ``sort_keys=True`` makes kwarg order irrelevant; ``default=repr`` keeps
    non-JSON-serializable arguments hashable without crashing.
    """
    payload = json.dumps([tool_name, args, kwargs], sort_keys=True, default=repr)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


@dataclass
class ToolSpec:
    """Internal registry record for a tool.

    ``speculatable`` is an explicit opt-in that requires ``pure=True``; a tool
    with observable side effects must never execute early. Unmarked tools are
    invisible to the whole speculation machinery.

    Attributes:
        name: Tool name (the key in ``repl.tools``).
        fn: The real implementation.
        is_async: ``fn`` is a coroutine function — awaited, not called.
        speculatable: Opt-in to speculative execution (requires ``pure``).
        pure: No observable side effects; safe to run early.
        deterministic: Identical inputs always produce identical output.
        latency_hint_ms: Expected latency, used by the budget/scheduler.
        spec_fn: How to RUN speculatively (defaults to ``fn``).
        cancel_fn: Abort an in-flight speculation on eviction.
        key_fn: Canonical claim identity for one call.
        gate_fn: Per-call speculatability predicate.
    """

    name: str
    fn: Callable[..., Any]  # the real implementation
    is_async: bool = False  # fn is a coroutine function: awaited, not called
    speculatable: bool = False
    pure: bool = False
    deterministic: bool = False
    latency_hint_ms: float = 1000.0
    spec_fn: Callable[..., Any] | None = None  # how to RUN speculatively
    cancel_fn: Callable[..., None] | None = None  # abort in-flight on evict
    key_fn: Callable[..., Any] | None = None  # canonical claim identity
    gate_fn: Callable[..., bool] | None = None  # per-CALL speculatability

    def __post_init__(self) -> None:
        if self.speculatable and not self.pure:
            raise ValueError(
                f"tool {self.name!r}: speculatable=True requires pure=True — a "
                "tool with observable side effects must never execute early"
            )


def _canonical_call(tool: ToolSpec, args: tuple, kwargs: dict) -> tuple[tuple, dict]:
    """Bind a concrete call to the tool's signature and apply defaults, so a
    positional call ``f(1, 2)`` and a keyword call ``f(a=1, b=2)`` hash to the
    same key. Falls back to the raw args when the signature cannot bind (no
    usable signature, too few/many args, or a ``**kwargs``-only tool)."""
    try:
        bound = inspect.signature(tool.fn).bind(*args, **kwargs)
        bound.apply_defaults()
        return bound.args, bound.kwargs
    except (TypeError, ValueError):
        return args, kwargs


def split_batch_call(
    args: tuple, kwargs: dict
) -> tuple[list, tuple, dict, str | None] | None:
    """Extract the batch prompt list from a batched tool call.

    A batched call (``llm_query_batched`` & friends) may pass its list of
    prompts either as the first positional argument or as a keyword argument
    (e.g. ``llm_query_batched(prompts=[...])``) — the interpreter's generated
    code uses both styles. Returns ``(prompts, rest_args, clean_kwargs,
    kwarg_name)`` where ``prompts`` is the extracted list, ``rest_args`` the
    remaining positional args, ``clean_kwargs`` the kwargs with the prompts
    entry removed, and ``kwarg_name`` the kwarg that held the list (``None``
    for a positional batch). Returns ``None`` when no list-shaped argument is
    found.
    """
    if args and isinstance(args[0], (list, tuple)):
        return list(args[0]), tuple(args[1:]), dict(kwargs), None
    if not kwargs:
        return None
    names = [k for k in ("prompts", "queries", "items", "inputs") if k in kwargs]
    if len(names) == 1 and isinstance(kwargs[names[0]], (list, tuple)):
        key = names[0]
    else:
        candidates = [k for k, v in kwargs.items() if isinstance(v, (list, tuple))]
        if len(candidates) != 1:
            return None
        key = candidates[0]
    prompts = list(kwargs[key])
    clean = {k: v for k, v in kwargs.items() if k != key}
    return prompts, tuple(args), clean, key


def rejoin_batch_call(
    prompts: list, rest: tuple, clean: dict, kwarg_name: str | None
) -> tuple[tuple, dict]:
    """Rebuild a real batched-tool call for a SUBSET of prompts: the inverse
    of :func:`split_batch_call` element selection, so the fallback real call
    keeps the call shape the model emitted (positional list vs. kwarg)."""
    if kwarg_name is None:
        return (prompts, *rest), dict(clean)
    return rest, {**clean, kwarg_name: prompts}


def spec_key(tool: ToolSpec, args: tuple, kwargs: dict) -> SpecKey:
    """Claim identity for one concrete call.

    Both dispatch and claim hash through here, so a shadow dispatch and the
    real REPL claim agree on the same key. ``key_fn`` (if present) reduces the
    call to a canonical material before hashing; otherwise the call is bound to
    the tool signature and defaults applied so positional and keyword forms of
    the same call produce the same key.
    """
    if tool.key_fn is not None:
        material = tool.key_fn(args, kwargs)
    else:
        norm_args, norm_kwargs = _canonical_call(tool, args, kwargs)
        material = (norm_args, norm_kwargs)
    return (tool.name, canonical_hash(tool.name, (material,), {}))


class SpeculativeTool:
    """Public base class for a speculatable tool (the modular interface).

    Subclass and override :meth:`execute` (required) plus the optional
    speculation hooks, then convert to an internal :class:`ToolSpec` via
    :meth:`to_spec`.
    """

    name: str = ""
    speculatable: bool = False
    pure: bool = False
    deterministic: bool = False
    latency_hint_ms: float = 1000.0

    # -- required ---------------------------------------------------------
    def execute(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    # -- overridable speculation behavior ----------------------------------
    def speculative_execute(
        self, *args: Any, _spec: Any | None = None, **kwargs: Any
    ) -> Any:
        return self.execute(*args, **kwargs)

    def cancel(self, spec: Any) -> None:
        pass

    def claim_key(self, args: tuple, kwargs: dict) -> Any:
        return (args, kwargs)

    def speculatable_call(self, args: tuple, kwargs: dict) -> bool:
        return True

    def to_spec(self) -> ToolSpec:
        """Build the internal :class:`ToolSpec` for this tool."""
        return ToolSpec(
            name=self.name,
            fn=self.execute,
            is_async=_is_async_callable(self.execute),
            speculatable=self.speculatable,
            pure=self.pure,
            deterministic=self.deterministic,
            latency_hint_ms=self.latency_hint_ms,
            spec_fn=self.speculative_execute,
            cancel_fn=self.cancel,
            key_fn=self.claim_key,
            gate_fn=self.speculatable_call,
        )


async def _identity(v: Any) -> Any:
    """Await-protocol shim: completes immediately with ``v``, never suspends."""
    return v


class NonSpeculated:
    """Inert marker a non-speculatable tool returns in the SHADOW.

    Storing it is fine; using it (any operation, or passing it into a
    speculatable call's arguments) raises, which opaque-aborts speculation at
    exactly the first statement that actually depended on the un-speculated
    result.
    """

    __slots__ = ("_tool",)

    def __init__(self, tool: str) -> None:
        object.__setattr__(self, "_tool", tool)

    def _boom(self):
        raise RuntimeError(
            f"result of non-speculatable tool {object.__getattribute__(self, '_tool')!r} "
            "used in shadow"
        )

    def __getattr__(self, k):
        self._boom()

    def __await__(self):
        # `x = await slow_tool()` in the shadow: stay a marker so taint flows
        # by value instead of aborting on an un-awaitable object.
        return _identity(self).__await__()

    def __str__(self):
        self._boom()

    def __format__(self, s):
        self._boom()

    def __bool__(self):
        self._boom()

    def __iter__(self):
        self._boom()

    def __getitem__(self, k):
        self._boom()

    def __add__(self, o):
        self._boom()

    def __radd__(self, o):
        self._boom()

    def __eq__(self, o):
        self._boom()

    def __hash__(self):
        self._boom()


def contains_nonspec(obj: Any, depth: int = 3) -> bool:
    """Deep check: is a :class:`NonSpeculated` marker hiding in these args?

    (repr/hash of the marker raises, but json ``default=repr`` canonicalization
    must never get that far — dispatching on marker args would be garbage.)
    """
    if isinstance(obj, NonSpeculated):
        return True
    if depth <= 0:
        return False
    if isinstance(obj, (list, tuple, set)):
        return any(contains_nonspec(x, depth - 1) for x in obj)
    if isinstance(obj, dict):
        return any(contains_nonspec(v, depth - 1) for v in obj.values())
    return False


class SpecValue:
    """Lazy proxy for a speculation future.

    Holds a :class:`Speculation` future (Task 2) and resolves it on first use,
    then delegates every operation to the resolved value. Used by the shadow
    hooks (Task 5) so a speculated call's result can be consumed transparently
    without blocking until the value is actually needed.
    """

    __slots__ = ("_spec", "_value", "_resolved")

    def __init__(self, spec: Any) -> None:
        self._spec = spec
        self._value: Any = None
        self._resolved = False

    # -- resolution -------------------------------------------------------
    def resolve(self, timeout: float | None = None) -> Any:
        """Wait for the future and return its resolved value."""
        if not self._resolved:
            self._spec.wait(timeout)
            self._value = self._spec.result()
            self._resolved = True
        return self._value

    def done(self) -> bool:
        """True once the underlying future has resolved."""
        return self._resolved or self._spec.state == "done"

    # -- delegation -------------------------------------------------------
    def __getattr__(self, name: str) -> Any:
        return getattr(self.resolve(), name)

    def __await__(self):
        return _identity(self.resolve()).__await__()

    def __str__(self) -> str:
        return str(self.resolve())

    def __repr__(self) -> str:
        return f"SpecValue({self.resolve()!r})"

    def __bool__(self) -> bool:
        return bool(self.resolve())

    def __iter__(self):
        return iter(self.resolve())

    def __getitem__(self, k: Any) -> Any:
        return self.resolve()[k]

    def __add__(self, o: Any) -> Any:
        return self.resolve() + o

    def __radd__(self, o: Any) -> Any:
        return o + self.resolve()

    def __eq__(self, o: Any) -> bool:
        return self.resolve() == o

    def __hash__(self) -> int:
        return hash(self.resolve())


def _is_async_callable(fn: Any) -> bool:
    """True for ``async def`` tools (incl. partials and objects whose
    ``__call__`` is a coroutine function)."""
    if inspect.iscoroutinefunction(fn):
        return True
    call = getattr(type(fn), "__call__", None)
    return call is not None and inspect.iscoroutinefunction(call)


def speculate(
    tool: Callable[..., Any],
    *,
    policy: SpeculationPolicy | None = None,
    **policy_kwargs: Any,
) -> ToolSpec:
    """Public helper to mark a user tool as speculatable.

    Builds an internal :class:`ToolSpec` from a plain callable plus a
    :class:`SpeculationPolicy` (or ``policy_kwargs`` overrides). Raises
    :class:`ValueError` if ``speculatable=True`` without ``pure=True``.
    """
    if policy is None:
        policy = SpeculationPolicy(**policy_kwargs)
    elif policy_kwargs:
        policy = replace(policy, **policy_kwargs)
    name = getattr(tool, "__name__", type(tool).__name__)
    return ToolSpec(
        name=name,
        fn=tool,
        is_async=_is_async_callable(tool),
        speculatable=policy.speculatable,
        pure=policy.pure,
        deterministic=policy.deterministic,
        latency_hint_ms=policy.latency_hint_ms,
        gate_fn=policy.gate,
    )


@dataclass(frozen=True)
class SpeculativeToolRequest:
    """Marks a tool for speculative execution when passed to
    :func:`~dspy_rlm_hooks.speculation.integration.api.enable_rlm_speculation`
    via the ``tools`` list.

    Attributes:
        fn: The tool implementation.
        name: Optional explicit tool name. Defaults to ``fn.__name__`` (or the
            tool's ``name`` attribute for ``dspy.Tool`` objects). The resolved
            name must match the name the tool is registered under in the REPL.
        deterministic: Identical inputs always produce identical output.
        latency_hint_ms: Expected latency, used by the budget/scheduler.
    """

    fn: Callable[..., Any]
    name: str | None = None
    deterministic: bool = False
    latency_hint_ms: float = 1000.0


def speculative(
    fn: Callable[..., Any],
    *,
    name: str | None = None,
    deterministic: bool = False,
    latency_hint_ms: float = 1000.0,
) -> SpeculativeToolRequest:
    """Mark a user tool for speculative execution.

    Wrapping a tool with :func:`speculative` is the per-tool opt-in to
    speculative execution: wrapped tools passed to
    :func:`~dspy_rlm_hooks.speculation.integration.api.enable_rlm_speculation`
    via ``tools=[...]`` are always speculated, without setting
    ``speculate_user_tools=True``. The tool must be pure — no observable side
    effects — because speculated calls run early and may run more than once.

    Example::

        enable_rlm_speculation(rlm, tools=[speculative(lookup_price)])
    """
    return SpeculativeToolRequest(
        fn=fn,
        name=name,
        deterministic=deterministic,
        latency_hint_ms=latency_hint_ms,
    )


SpecTool = speculative
Spec = speculative
