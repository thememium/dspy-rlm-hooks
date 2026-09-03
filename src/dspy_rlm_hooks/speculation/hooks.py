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

from dspy_rlm_hooks.speculation.guards import current_spec, raw_tool_fn, tag_claim_hook
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


_MAX_CLAIM_WAIT_S = 30.0  # hard ceiling on claim waits without a launcher


def _claim_wait_budget(spec: Speculation, tool: ToolSpec, launcher: Any) -> float:
    """Seconds worth waiting for an in-flight speculation before hedging.

    Uses the launcher's per-tool latency EWMA (falling back to the tool's
    ``latency_hint_ms``): waiting is capped at ~2x the estimated remaining
    time and never past the launcher's ceiling; a queued-not-started
    speculation whose queue will drain slower than duplicating the call
    returns 0 (hedge immediately).
    """
    if launcher is None or not getattr(launcher, "latency_aware", False):
        return _MAX_CLAIM_WAIT_S
    ceiling = float(getattr(launcher, "max_claim_wait_s", _MAX_CLAIM_WAIT_S))
    ewma_s = launcher.ewma_ms(tool.name, tool.latency_hint_ms) / 1000.0
    duplicate_cost = max(ewma_s, 0.05)
    if spec.state == "running":
        elapsed = time.monotonic() - (spec.dispatched_at or time.monotonic())
        remaining = max(ewma_s - elapsed, 0.0)
        return min(max(remaining * 1.5, 0.25), duplicate_cost * 2.0, ceiling)
    # pending: queued behind other dispatches
    est_s = (launcher.queued_depth() + 1) * ewma_s
    if est_s > duplicate_cost * 1.5:
        return 0.0
    return min(max(est_s * 1.5, 0.25), duplicate_cost * 2.0, ceiling)


def _hedge(
    spec: Speculation,
    tool: ToolSpec,
    args: tuple,
    kwargs: dict,
    bus: Any,
    key: Any,
    t0: float,
) -> Any:
    """Abandon a slow speculation and run the real tool.

    Only pure tools are speculated, so duplicate execution is safe by
    definition; the abandoned future is marked evicted and cancelled.
    """
    bus.emit(
        "claim_hedge",
        key=key,
        tool=tool.name,
        waited_ms=(time.perf_counter() - t0) * 1000,
    )
    spec.state = "evicted"
    if spec.cancel is not None:
        try:
            spec.cancel()
        except Exception:
            pass
    bus.emit("claim_miss", key=key, tool=tool.name)
    return tool.fn(*args, **kwargs)


def _claim_or_run(
    tool: ToolSpec,
    args: tuple,
    kwargs: dict,
    store: SpecStore,
    bus: Any,
    launcher: Any = None,
) -> Any:
    """Claim a speculation for one call; on hit wait for and return its result,
    on miss run the real tool (the baseline path).

    Latency-aware: a claim on an in-flight speculation waits at most its
    estimated remaining time (vs. the cost of duplicating the call), then
    hedges.
    """
    key = spec_key(tool, args, kwargs)
    t0 = time.perf_counter()
    spec = store.claim(key, reuse=tool.deterministic)
    if spec is not None:
        if spec is current_spec():
            # self-claim guard: this hook is being run BY the very worker that
            # must resolve `spec`; waiting on it would block the pool (and
            # interpreter shutdown) for the full timeout. Run the raw tool.
            return raw_tool_fn(tool)(*args, **kwargs)
        if not spec.done.is_set():
            budget = _claim_wait_budget(spec, tool, launcher)
            if budget <= 0:
                return _hedge(spec, tool, args, kwargs, bus, key, t0)
            try:
                result = spec.result(timeout=budget)
            except TimeoutError:
                return _hedge(spec, tool, args, kwargs, bus, key, t0)
            spec.state = "claimed"
            bus.emit(
                "claim_hit",
                key=key,
                seq=spec.seq,
                tool=tool.name,
                already_ready=False,
            )
            bus.emit(
                "claim_done",
                key=key,
                seq=spec.seq,
                tool=tool.name,
                waited_ms=(time.perf_counter() - t0) * 1000,
            )
            return result
        bus.emit(
            "claim_hit",
            key=key,
            seq=spec.seq,
            tool=tool.name,
            already_ready=True,
        )
        result = spec.result(0)
        spec.state = "claimed"
        bus.emit(
            "claim_done",
            key=key,
            seq=spec.seq,
            tool=tool.name,
            waited_ms=(time.perf_counter() - t0) * 1000,
        )
        return result
    bus.emit("claim_miss", key=key, tool=tool.name)
    return tool.fn(*args, **kwargs)


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
            hooks[name] = tag_claim_hook(
                _async_real_hook(tool, reg, store, bus, launcher), raw_fn=tool.fn
            )
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
                cur = current_spec()
                for i, p in enumerate(prompts):
                    key = spec_key(single, (p,) + rest, kwargs)
                    spec = store.claim(key, reuse=single.deterministic)
                    if spec is None or spec is cur:
                        # self-claim guard: waiting on our own worker's done would deadlock the pool
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
                waited: list[int] = []
                for i, spec in claimed:
                    if spec.done.is_set():
                        out[i] = spec.result(0)
                        spec.state = "claimed"
                        continue
                    budget = _claim_wait_budget(spec, single, launcher)
                    try:
                        if budget <= 0:
                            raise TimeoutError
                        out[i] = spec.result(timeout=budget)
                    except TimeoutError:
                        waited.append(i)
                        spec.state = "evicted"
                        continue
                    spec.state = "claimed"
                if waited:
                    # hedge the elements whose speculations missed their budget
                    bus.emit(
                        "claim_hedge",
                        key=spec_key(single, (prompts[waited[0]],) + rest, kwargs),
                        tool=single.name,
                        waited_ms=0.0,
                    )
                    batch_res = _tool.fn([prompts[i] for i in waited], *rest, **kwargs)
                    for j, i in enumerate(waited):
                        out[i] = batch_res[j]
                return out
            return _claim_or_run(_tool, tuple(args), kwargs, store, bus, launcher)

        hooks[name] = tag_claim_hook(hook, raw_fn=tool.fn)
    return hooks


def _async_real_hook(tool: ToolSpec, reg, store: SpecStore, bus, launcher=None):
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
            cur = current_spec()
            for i, p in enumerate(prompts):
                key = spec_key(single, (p,) + rest, kwargs)
                spec = store.claim(key, reuse=single.deterministic)
                if spec is None or spec is cur:
                    # self-claim guard: waiting on our own worker's done would deadlock the pool
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

            hedged: list[int] = []

            async def _fill_claim(i: int, spec: Speculation):
                budget = (
                    _claim_wait_budget(spec, single, launcher)
                    if not spec.done.is_set() and launcher is not None
                    else 600.0
                )
                if budget <= 0:
                    hedged.append(i)
                    spec.state = "evicted"
                    return
                try:
                    out[i] = await _await_spec(spec, budget)
                    spec.state = "claimed"
                except (TimeoutError, asyncio.TimeoutError):
                    hedged.append(i)
                    spec.state = "evicted"

            await asyncio.gather(
                _fill_misses(), *[_fill_claim(i, s) for i, s in claimed]
            )
            if hedged:
                # hedge the elements whose speculations missed their budget
                res = await _tool.fn([prompts[i] for i in hedged], *rest, **kwargs)
                for j, i in enumerate(hedged):
                    out[i] = res[j]
            return out
        key = spec_key(_tool, tuple(args), kwargs)
        spec = store.claim(key, reuse=_tool.deterministic)
        if spec is None or spec is current_spec():
            # self-claim guard: waiting on our own worker's done would deadlock the pool
            bus.emit("claim_miss", key=key, tool=_tool.name)
            return await _tool.fn(*args, **kwargs)  # miss: the baseline path
        budget = (
            _claim_wait_budget(spec, _tool, launcher)
            if not spec.done.is_set() and launcher is not None
            else 600.0
        )
        if budget <= 0:
            spec.state = "evicted"
            bus.emit("claim_miss", key=key, tool=_tool.name)
            return await _tool.fn(*args, **kwargs)  # hedge: queue drain too slow
        bus.emit(
            "claim_hit",
            key=key,
            seq=spec.seq,
            tool=_tool.name,
            already_ready=spec.done.is_set(),
        )
        try:
            result = await _await_spec(spec, budget)
        except (TimeoutError, asyncio.TimeoutError):
            spec.state = "evicted"
            bus.emit("claim_miss", key=key, tool=_tool.name)
            return await _tool.fn(*args, **kwargs)  # hedge after the budget
        spec.state = "claimed"
        return result

    return hook


async def _await_spec(spec: Speculation, timeout: float = 600.0) -> Any:
    """Wait for an in-flight speculation without stalling the event loop.

    ``timeout`` doubles as the claim budget: on expiry ``spec.result`` raises
    :class:`TimeoutError` and the caller hedges.
    """
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
