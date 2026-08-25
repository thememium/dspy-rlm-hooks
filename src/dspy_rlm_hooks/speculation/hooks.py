"""Hook factories for the speculation engine (Task 5).

Three hook modes around a host tool registry:

- :func:`make_real_hooks` — *claiming* hooks for the real REPL. Each call
  claims a matching speculation future from the store (reusing its resolved
  result) or, on a miss, runs the real tool byte-identical to calling it
  directly. Async tools get a coroutine hook so ``await``/``asyncio.gather``
  keep their natural shape.
- :func:`make_shadow_hooks` — *dispatch* hooks for the shadow. Each call fires
  the speculation early via the launcher and returns a lazy :class:`SpecValue`
  proxy (or a :class:`NonSpeculated` marker for non-speculatable / gated-off
  calls).
- :func:`make_baseline_hooks` — unmodified passthrough tools (no speculation).

Batched tools (name ending in ``_batched``, e.g. ``llm_query_batched``) are
decomposed per element: the shadow dispatches N independent single-prompt
speculations and the real hook claims them element-wise, running the real
batched path only for the elements that missed.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from dspy_rlm_hooks.speculation.store import SpecStore, Speculation
from dspy_rlm_hooks.speculation.tool import (
    NonSpeculated,
    SpecValue,
    ToolSpec,
    contains_nonspec,
    spec_key,
)


class ShadowBudgetDenied(Exception):
    """Raised by a shadow hook when the per-turn dispatch budget is exhausted."""


class _NullBus:
    """No-op event bus used when a caller omits ``bus``."""

    def emit(self, kind: str, **data: Any) -> None:
        return None


_NULL_BUS = _NullBus()


def deep_force(obj: Any, depth: int = 3) -> Any:
    """Replace :class:`SpecValue` proxies with concrete values through common
    containers (chained shadow calls force their inputs here)."""
    if isinstance(obj, SpecValue):
        return obj.resolve()
    if depth <= 0:
        return obj
    if isinstance(obj, list):
        return [deep_force(x, depth - 1) for x in obj]
    if isinstance(obj, tuple):
        return tuple(deep_force(x, depth - 1) for x in obj)
    if isinstance(obj, dict):
        return {k: deep_force(v, depth - 1) for k, v in obj.items()}
    return obj


def _is_batched(tool: ToolSpec) -> bool:
    """A batched tool is one whose name ends in ``_batched`` (e.g.
    ``llm_query_batched``). Our Task 1 ``ToolSpec`` has no ``batched`` field,
    so the name suffix is the discriminator."""
    return tool.name.endswith("_batched")


def _single_of(batched_tool: ToolSpec, reg) -> ToolSpec:
    """The single-prompt :class:`ToolSpec` a batched tool decomposes into."""
    base = batched_tool.name.removesuffix("_batched")
    single = reg.get(base)
    if single is not None:
        return single
    return batched_tool


def make_baseline_hooks(reg) -> dict[str, Any]:
    """Unmodified passthrough hooks: every call runs the real tool directly."""
    hooks: dict[str, Any] = {}
    for name in reg.names():
        tool = reg.get(name)
        assert tool is not None

        def hook(*args: Any, _tool=tool, **kwargs: Any):
            return _tool.fn(*args, **kwargs)

        hooks[name] = hook
    return hooks


def make_real_hooks(reg, store: SpecStore, launcher, bus=None) -> dict[str, Any]:
    """Claiming hooks for the real REPL.

    A speculatable call claims a matching speculation future (reusing its
    result) or, on a miss, runs the real tool — byte-identical to calling it
    directly. Non-speculatable tools pass straight through.
    """
    bus = bus or _NULL_BUS
    hooks: dict[str, Any] = {}
    for name in reg.names():
        tool = reg.get(name)
        assert tool is not None

        if tool.is_async:
            hooks[name] = _async_real_hook(tool, reg, store, bus)
            continue

        def hook(*args: Any, _tool=tool, **kwargs: Any):
            if not _tool.speculatable:
                return _tool.fn(*args, **kwargs)
            if _is_batched(_tool) and args and isinstance(args[0], (list, tuple)):
                single = _single_of(_tool, reg)
                prompts = list(args[0])
                rest = tuple(args[1:])
                out: list[Any] = [None] * len(prompts)
                misses: list[int] = []
                claimed: list[tuple[int, Speculation]] = []
                for i, p in enumerate(prompts):
                    key = spec_key(single, (p,) + rest, kwargs)
                    spec = store.claim(key, reuse=single.deterministic)
                    if spec is None:
                        bus.emit("claim_miss", key=key, tool=single.name)
                        misses.append(i)
                    else:
                        bus.emit(
                            "claim_hit",
                            key=key,
                            seq=spec.seq,
                            tool=single.name,
                            already_ready=spec.done.is_set(),
                        )
                        claimed.append((i, spec))
                if misses:
                    batch_res = _tool.fn([prompts[i] for i in misses], *rest, **kwargs)
                    for j, i in enumerate(misses):
                        out[i] = batch_res[j]
                for i, spec in claimed:
                    out[i] = spec.result(timeout=600)
                    spec.state = "claimed"
                return out
            return _claim_or_run(_tool, tuple(args), kwargs, store, bus)

        hooks[name] = hook
    return hooks


def _claim_or_run(
    tool: ToolSpec, args: tuple, kwargs: dict, store: SpecStore, bus
) -> Any:
    """Claim a speculation for one call; on hit wait for and return its result,
    on miss run the real tool (the baseline path)."""
    key = spec_key(tool, args, kwargs)
    t0 = time.perf_counter()
    spec = store.claim(key, reuse=tool.deterministic)
    if spec is not None:
        bus.emit(
            "claim_hit",
            key=key,
            seq=spec.seq,
            tool=tool.name,
            already_ready=spec.done.is_set(),
        )
        result = spec.result(timeout=600)
        spec.state = "claimed"
        bus.emit(
            "claim_done",
            key=key,
            seq=spec.seq,
            waited_ms=(time.perf_counter() - t0) * 1000,
        )
        return result
    bus.emit("claim_miss", key=key, tool=tool.name)
    return tool.fn(*args, **kwargs)


def _async_real_hook(tool: ToolSpec, reg, store: SpecStore, bus):
    """Claim-or-run for an ``async def`` tool. The hook is itself a coroutine
    function, so model code keeps its natural shape (``await llm(x)``,
    ``asyncio.gather(...)``) — and a claim never blocks the caller's event
    loop, so sibling gather branches that MISSED still run concurrently."""

    async def hook(*args: Any, _tool=tool, **kwargs: Any):
        if not _tool.speculatable:
            return await _tool.fn(*args, **kwargs)
        if _is_batched(_tool) and args and isinstance(args[0], (list, tuple)):
            single = _single_of(_tool, reg)
            prompts, rest = list(args[0]), tuple(args[1:])
            out: list[Any] = [None] * len(prompts)
            misses: list[int] = []
            claimed: list[tuple[int, Speculation]] = []
            for i, p in enumerate(prompts):
                key = spec_key(single, (p,) + rest, kwargs)
                spec = store.claim(key, reuse=single.deterministic)
                if spec is None:
                    bus.emit("claim_miss", key=key, tool=single.name)
                    misses.append(i)
                else:
                    bus.emit(
                        "claim_hit",
                        key=key,
                        seq=spec.seq,
                        tool=single.name,
                        already_ready=spec.done.is_set(),
                    )
                    claimed.append((i, spec))

            # the misses' batch call and every claim wait overlap
            async def _fill_misses():
                if not misses:
                    return
                res = await _tool.fn([prompts[i] for i in misses], *rest, **kwargs)
                for j, i in enumerate(misses):
                    out[i] = res[j]

            async def _fill_claim(i: int, spec: Speculation):
                out[i] = await _await_spec(spec)
                spec.state = "claimed"

            await asyncio.gather(
                _fill_misses(), *[_fill_claim(i, s) for i, s in claimed]
            )
            return out
        key = spec_key(_tool, tuple(args), kwargs)
        spec = store.claim(key, reuse=_tool.deterministic)
        if spec is None:
            bus.emit("claim_miss", key=key, tool=_tool.name)
            return await _tool.fn(*args, **kwargs)  # miss: the baseline path
        bus.emit(
            "claim_hit",
            key=key,
            seq=spec.seq,
            tool=_tool.name,
            already_ready=spec.done.is_set(),
        )
        result = await _await_spec(spec)
        spec.state = "claimed"
        return result

    return hook


async def _await_spec(spec: Speculation, timeout: float = 600.0) -> Any:
    """Wait for an in-flight speculation without stalling the event loop."""
    if spec.done.is_set():
        return spec.result(0)
    return await asyncio.to_thread(spec.result, timeout)


def make_shadow_hooks(reg, store: SpecStore, launcher, bus=None) -> dict[str, Any]:
    """Dispatch hooks for the shadow.

    A speculatable call fires the speculation early via the launcher and
    returns a lazy :class:`SpecValue` proxy. Non-speculatable (or per-call
    gated-off) tools return an inert :class:`NonSpeculated` marker; a call
    whose args depend on a non-speculated result raises (taint).
    """
    bus = bus or _NULL_BUS
    hooks: dict[str, Any] = {}
    for name in reg.names():
        tool = reg.get(name)
        assert tool is not None

        def hook(*args: Any, _tool=tool, **kwargs: Any):
            args = tuple(deep_force(a) for a in args)  # chained calls force here
            kwargs = {k: deep_force(v) for k, v in kwargs.items()}
            # non-speculatable (or per-call gated-off) tools do NOT run early —
            # return an inert marker; the shadow aborts only if a later
            # statement actually uses it.
            if not _tool.speculatable or (
                _tool.gate_fn and not _tool.gate_fn(args, kwargs)
            ):
                return NonSpeculated(_tool.name)
            if contains_nonspec(args) or contains_nonspec(kwargs):
                raise RuntimeError(
                    f"{_tool.name} args depend on a non-speculated result"
                )
            if _is_batched(_tool) and args and isinstance(args[0], (list, tuple)):
                # llm_query_batched: dispatch per element so the real run can
                # claim elementwise; return a list of SpecValues.
                prompts = list(args[0])
                specs = [
                    launcher.dispatch_or_adopt(
                        _single_of(_tool, reg), (p,) + args[1:], kwargs, "shadow"
                    )
                    for p in prompts
                ]
                return [SpecValue(s) if s else None for s in specs]
            spec = launcher.dispatch_or_adopt(_tool, args, kwargs, "shadow")
            if spec is None:  # budget denied
                raise ShadowBudgetDenied(_tool.name)
            bus.emit("shadow_dispatch", tool=_tool.name, key=spec.key)
            return SpecValue(spec)

        hooks[name] = hook
    return hooks
