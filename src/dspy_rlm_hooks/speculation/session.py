"""SpecSession + Launcher for the speculation engine (Task 5).

:class:`SpecSession` owns the :class:`SpecStore`, :class:`Launcher` and
:class:`Budget` around a host tool registry, and exposes the hook factories the
real REPL and the shadow consume. :class:`Launcher` dispatches speculative
executions on a thread pool (one shared asyncio loop for async tools) and
enforces the Budget's concurrency + per-turn dispatch caps.

This module also defines the minimal :class:`ToolRegistry` and :class:`EventBus`
the session builds against (the project had none before Task 5).
"""

from __future__ import annotations

import asyncio
import inspect
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from dspy_rlm_hooks.speculation.budget import Budget
from dspy_rlm_hooks.speculation.guards import (
    clear_current,
    is_claim_hook,
    mark_current,
    raw_of,
)
from dspy_rlm_hooks.speculation.hooks import (
    make_baseline_hooks,
    make_real_hooks,
    make_shadow_hooks,
)
from dspy_rlm_hooks.speculation.shadow import ShadowRunner
from dspy_rlm_hooks.speculation.store import SpecStore, Speculation
from dspy_rlm_hooks.speculation.streaming import Segment, StreamSegmenter
from dspy_rlm_hooks.speculation.tool import ToolSpec, _is_async_callable, spec_key


def _bind_call(fn: Any, args: tuple, kwargs: dict) -> dict:
    """Bind ``(args, kwargs)`` to ``fn``'s signature and return a single kwargs
    dict. The shadow records calls positionally (as the generated code wrote
    them) but the real interpreter invokes tools via ``fn(**kwargs)``, so the
    speculated execution must mirror that. Falls back to merging raw args by
    position when the signature cannot be bound."""
    try:
        sig = inspect.signature(fn)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        return bound.arguments
    except (TypeError, ValueError):
        out: dict[str, Any] = {}
        for i, v in enumerate(args):
            out[f"__arg{i}"] = v
        out.update(kwargs)
        return out


class EventBus:
    """Minimal observable event bus: ``emit(kind, **data)`` records to history
    and notifies subscribers (used by the shadow runner to watch speculation
    resolutions for chained continuations)."""

    def __init__(self) -> None:
        self.history: list[tuple[str, dict]] = []
        self.record = True
        self._subs: list[Callable[[str, dict], None]] = []
        self._sub_lock = threading.Lock()

    def subscribe(self, fn: Callable[[str, dict], None]) -> None:
        """Register a callback invoked as ``fn(kind, data)`` on every emit."""
        with self._sub_lock:
            self._subs.append(fn)

    def emit(self, kind: str, **data: Any) -> tuple[str, dict]:
        ev = (kind, data)
        if self.record:
            self.history.append(ev)
        with self._sub_lock:
            subs = list(self._subs)
        for fn in subs:
            try:
                fn(kind, data)
            except Exception:
                pass  # subscriber errors must never break dispatch
        return ev


class ToolRegistry:
    """Minimal registry of :class:`ToolSpec` by name (``names()``/``get()``/``register()``)."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, name: str, fn: Any, **kw: Any) -> ToolSpec:
        kw.setdefault("is_async", _is_async_callable(fn))
        spec = ToolSpec(name=name, fn=fn, **kw)
        self._tools[name] = spec
        return spec

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)


class LatencyStats:
    """Thread-safe per-tool EWMA of speculative execution latency (ms).

    Feeds latency-aware claiming: when the real interpreter reaches a call
    whose speculation is still in flight, the estimated remaining wait is
    compared against the cost of simply running the tool.
    """

    def __init__(self, alpha: float = 0.3) -> None:
        self.alpha = alpha
        self._ewma_ms: dict[str, float] = {}
        self._samples: dict[str, int] = {}
        self._lock = threading.Lock()

    def record(self, name: str, ms: float) -> None:
        with self._lock:
            prev = self._ewma_ms.get(name)
            self._ewma_ms[name] = (
                ms if prev is None else prev + self.alpha * (ms - prev)
            )
            self._samples[name] = self._samples.get(name, 0) + 1

    def ewma_ms(self, name: str, fallback_ms: float) -> float:
        with self._lock:
            v = self._ewma_ms.get(name)
        return v if v is not None else fallback_ms

    def samples(self, name: str) -> int:
        with self._lock:
            return self._samples.get(name, 0)


class Launcher:
    """Dispatches speculative executions on a thread pool.

    ``max_workers`` comes from the Budget's ``max_inflight``; each dispatch
    reserves a per-turn slot via ``budget.try_dispatch()`` (hard deny when the
    turn cap is hit). Async tools run on one shared background asyncio loop so
    loop-bound clients stay valid across calls.

    ``latency_aware=True`` records a per-tool latency EWMA and tracks the
    pending-dispatch depth so claim hooks can decide between waiting for an
    in-flight speculation and hedging (running the real tool).
    """

    def __init__(
        self,
        store: SpecStore,
        bus: EventBus | None = None,
        budget: Budget | None = None,
        latency_aware: bool = True,
    ) -> None:
        self.store = store
        self.bus = bus or EventBus()
        self.budget = budget or Budget()
        self.latency_aware = latency_aware
        self.latency = LatencyStats()
        self.max_claim_wait_s: float = 30.0  # hard ceiling on claim waits
        self._queued = 0  # dispatches accepted but not yet started
        self._queued_lock = threading.Lock()
        self._seq = 0
        self._seq_lock = threading.Lock()
        self._pool = ThreadPoolExecutor(
            max_workers=self.budget.max_inflight, thread_name_prefix="spec"
        )
        self._aloop: asyncio.AbstractEventLoop | None = None
        self._aloop_lock = threading.Lock()

    # -- latency-aware claim support ------------------------------------------
    def queued_depth(self) -> int:
        with self._queued_lock:
            return self._queued

    def ewma_ms(self, tool_name: str, fallback_ms: float) -> float:
        return self.latency.ewma_ms(tool_name, fallback_ms)

    def loop(self) -> asyncio.AbstractEventLoop:
        """One shared background event loop for every async tool."""
        with self._aloop_lock:
            if self._aloop is None:
                self._aloop = asyncio.new_event_loop()
                threading.Thread(
                    target=_serve_loop,
                    args=(self._aloop,),
                    daemon=True,
                    name="spec-aio",
                ).start()
            return self._aloop

    def run_awaitable(self, aw: Any, timeout: float = 600.0) -> Any:
        """Drive an awaitable to completion from a worker thread."""
        fut = asyncio.run_coroutine_threadsafe(_as_coro(aw), self.loop())
        return fut.result(timeout)

    def next_seq(self) -> int:
        with self._seq_lock:
            self._seq += 1
            return self._seq

    def dispatch_or_adopt(
        self, tool: ToolSpec, args: tuple, kwargs: dict, source: str
    ) -> Speculation | None:
        """Shadow hooks come through here: if a peek already fired this exact
        call, adopt the in-flight speculation instead of duplicating it."""
        key = spec_key(tool, args, kwargs)
        if tool.deterministic:
            spec = self.store.existing(key)
            if spec is not None:
                spec.adopted = True  # shield from peek retraction
                return spec
        spec = self.store.adopt(key)
        if spec is not None:
            self.bus.emit("adopt", key=key, seq=spec.seq, tool=tool.name)
            return spec
        return self.dispatch(tool, args, kwargs, source)

    def ensure_peeked(
        self, tool: ToolSpec, args: tuple, kwargs: dict, needed: int
    ) -> int:
        """Top the store up to ``needed`` un-adopted peek speculations for this
        exact call (multiplicity-safe dedup across repeated peeks of a growing
        tail). Returns how many new dispatches were made."""
        key = spec_key(tool, args, kwargs)
        if tool.deterministic:
            if self.store.existing(key):
                return 0
            needed = 1
        new = 0
        while self._unadopted_peeks(key) < needed:
            if self.dispatch(tool, args, kwargs, "peek") is None:
                break
            new += 1
        return new

    def _unadopted_peeks(self, key) -> int:
        """Count live un-adopted peek speculations for ``key``.

        The Task 2 ``SpecStore`` has no ``unadopted_peeks`` accessor and we may
        not modify it, so the launcher (the only peek dispatcher) counts by
        scanning the store's queue directly.
        """
        n = 0
        for spec in self.store._q.get(key, ()):
            if (
                spec.source == "peek"
                and not spec.adopted
                and spec.state in ("pending", "running", "ready")
            ):
                n += 1
        return n

    def dispatch(
        self, tool: ToolSpec, args: tuple, kwargs: dict, source: str
    ) -> Speculation | None:
        """Fire the tool now; returns the :class:`Speculation`, or None if
        budget-denied."""
        if not self.budget.try_dispatch():
            self.bus.emit("note", msg=f"budget denied dispatch of {tool.name}")
            return None
        key = spec_key(tool, args, kwargs)
        spec = Speculation(
            key=key,
            seq=self.next_seq(),
            args=args,
            kwargs=kwargs,
            source=source,
        )
        spec.dispatched_at = time.monotonic()
        if tool.cancel_fn is not None:
            cancel_fn = tool.cancel_fn
            spec.cancel = lambda _s=spec, _f=cancel_fn: _f(_s)
        self.store.put(spec)
        self.bus.emit("dispatch", key=key, seq=spec.seq, tool=tool.name, source=source)

        def run() -> None:
            with self._queued_lock:
                self._queued -= 1
            spec.started_at = time.monotonic()  # excludes queue time
            if spec.state == "evicted":
                spec.done.set()
                return
            spec.state = "running"
            run_fn = tool.spec_fn or tool.fn
            if is_claim_hook(run_fn):
                # A claim hook installed into repl.tools must never be executed
                # as a speculative fn: it claims+waits on in-flight speculations,
                # which inside a worker is a self-claim deadlock (see guards).
                run_fn = raw_of(run_fn, fallback=tool.fn)
            try:
                mark_current(spec)
                try:
                    call_kwargs = _bind_call(run_fn, args, kwargs)
                    if getattr(run_fn, "wants_spec", False):
                        out = run_fn(**call_kwargs, _spec=spec)
                    else:
                        out = run_fn(**call_kwargs)
                finally:
                    clear_current()
                # an async tool hands back a coroutine: drive it here so the
                # speculation stores the VALUE, not an un-awaited coroutine.
                if inspect.isawaitable(out):
                    out = self.run_awaitable(out)
                spec._result = out
                if spec.state != "evicted":
                    spec.state = "ready"
            except BaseException as e:  # surfaced at claim/force point
                spec.error = e
                spec.state = "failed" if spec.state != "evicted" else "evicted"
            finally:
                spec.resolved_at = time.monotonic()
                if self.latency_aware and spec.started_at is not None:
                    # RUN time only: including queue time would inflate the
                    # EWMA under load and make claim budgets hedge too early.
                    self.latency.record(
                        tool.name, (spec.resolved_at - spec.started_at) * 1000
                    )
            spec.done.set()
            if spec.state == "ready":
                self.bus.emit("ready", key=key, seq=spec.seq, spec=spec)

        with self._queued_lock:
            self._queued += 1
        self._pool.submit(run)
        return spec

    def shutdown(self) -> None:
        with self._aloop_lock:
            loop, self._aloop = self._aloop, None
        if loop is not None:
            # cancel in-flight coroutines FIRST: a worker parked in
            # run_awaitable() unblocks only when its future resolves, and a
            # merely-stopped loop never resolves it.
            asyncio.run_coroutine_threadsafe(_cancel_all_and_stop(), loop)
        self._pool.shutdown(wait=False, cancel_futures=True)


async def _as_coro(aw: Any) -> Any:
    return await aw


async def _cancel_all_and_stop() -> None:
    loop = asyncio.get_running_loop()
    tasks = [t for t in asyncio.all_tasks(loop) if t is not asyncio.current_task()]
    for t in tasks:
        t.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
    loop.stop()


def _serve_loop(loop: asyncio.AbstractEventLoop) -> None:
    """Own the loop end-to-end so it is CLOSED on its own thread."""
    asyncio.set_event_loop(loop)
    try:
        loop.run_forever()
    finally:
        try:
            loop.run_until_complete(loop.shutdown_asyncgens())
        except Exception:
            pass
        loop.close()


@dataclass
class StreamTurn:
    """One streaming turn's segmenter + shadow, bound to a host namespace.

    The shadow starts LAZILY: nothing spawns until the first closed statement
    (or peek-worthy tail) actually mentions a speculatable tool, so call-free
    iterations (e.g. the final-answer turn) pay zero shadow cost. Segments fed
    before the shadow exists are buffered and flushed in order on start.
    """

    segmenter: StreamSegmenter
    shadow: ShadowRunner | None = None
    peek: bool = True
    shadow_factory: Callable[[], ShadowRunner] | None = None
    spec_names: frozenset[str] = frozenset()
    _last_tail: str = ""
    _pending: list[Segment] = field(default_factory=list)

    @property
    def _hook_names(self) -> set[str] | frozenset[str]:
        """Names the lazy trigger watches. Falls back to the shadow's hook
        names when the turn was constructed with an eager shadow."""
        if self.spec_names:
            return self.spec_names
        return set(self.shadow.hooks) if self.shadow is not None else set()

    def _acquire(self) -> ShadowRunner:
        """Start (or reuse) the shadow, flushing any buffered segments first."""
        if self.shadow is None:
            if self.shadow_factory is None:
                raise RuntimeError("StreamTurn has neither shadow nor factory")
            self.shadow = self.shadow_factory()
            for seg in self._pending:
                self.shadow.feed(seg)
            self._pending.clear()
        return self.shadow

    def _feed_segment(self, seg: Segment) -> None:
        if self.shadow is not None:
            self.shadow.feed(seg)
        elif seg.has_call and any(name in seg.source for name in self._hook_names):
            self._acquire().feed(seg)
        else:
            self._pending.append(seg)

    def feed(self, delta: str) -> None:
        for seg in self.segmenter.feed(delta):
            self._feed_segment(seg)
        if self.peek and "\n" in delta:
            tail = self.segmenter.pending_tail()
            if tail.strip() and tail != self._last_tail and self._peek_worthwhile(tail):
                self._last_tail = tail
                self._acquire().feed_peek(tail)

    def _peek_worthwhile(self, tail: str) -> bool:
        if len(tail) > 12_000:
            return False
        changed = (
            tail[len(self._last_tail) :] if tail.startswith(self._last_tail) else tail
        )
        if any(name in changed for name in self._hook_names):
            return True
        return self.shadow is not None and bool(self.shadow._last_peek_tally)

    def end(self, timeout: float = 600) -> None:
        for seg in self.segmenter.finish():
            self._feed_segment(seg)
        if self.shadow is None:
            return  # nothing speculatable streamed: the shadow never started
        if self.shadow.persistent is True:  # real bool; mocks fall through to legacy
            self.shadow.end_turn(timeout)
        else:
            self.shadow.finish()
            self.shadow.join(timeout)
            self.shadow.abort("turn_end")


class SpecSession:
    """Store + launcher + hook factories around a host's tool registry."""

    def __init__(
        self,
        registry,
        bus: EventBus | None = None,
        max_inflight: int = 8,
        max_dispatches_per_turn: int = 2048,
        taint_skip: bool = True,
        persistent_shadow: bool = True,
        latency_aware: bool = True,
    ) -> None:
        self.reg = registry
        self.bus = bus or EventBus()
        self.taint_skip = taint_skip
        self.persistent_shadow = persistent_shadow
        self.latency_aware = latency_aware
        self._warm_runner: ShadowRunner | None = None
        self.store = SpecStore()
        self.launcher = Launcher(
            self.store,
            self.bus,
            Budget(
                max_inflight=max_inflight,
                max_dispatches_per_turn=max_dispatches_per_turn,
            ),
            latency_aware=latency_aware,
        )

    def _spec_names(self) -> frozenset[str]:
        """Names of speculatable tools — the lazy-start trigger."""
        return frozenset(
            name
            for name in self.reg.names()
            if (tool := self.reg.get(name)) is not None and tool.speculatable
        )

    def _new_runner(self, host_locals: dict, safe_builtins: dict) -> ShadowRunner:
        return ShadowRunner(
            host_locals,
            make_shadow_hooks(self.reg, self.store, self.launcher, self.bus),
            self.store,
            safe_builtins,
            launcher=self.launcher,
            registry=self.reg,
            taint_skip=self.taint_skip,
            persistent=self.persistent_shadow,
        )

    def _acquire_runner(self, host_locals: dict, safe_builtins: dict) -> ShadowRunner:
        """Reuse the warm runner across turns; respawn it if it crashed.

        ``input_args`` is stable across an RLM run, so the seed stays valid; a
        respawn re-seeds from the (possibly changed) host locals.
        """
        runner = self._warm_runner
        if runner is not None and not runner.is_alive:
            runner = None
        if runner is None:
            runner = self._new_runner(host_locals, safe_builtins)
            self._warm_runner = runner
        else:
            runner.begin_turn(host_locals)
        return runner

    def real_hooks(self) -> dict:
        """Claiming hooks for the real REPL (byte-identical to calling tools)."""
        return make_real_hooks(self.reg, self.store, self.launcher, self.bus)

    def baseline_hooks(self) -> dict:
        """Unmodified passthrough tools (no speculation)."""
        return make_baseline_hooks(self.reg)

    def begin_stream_turn(
        self, host_locals: dict, safe_builtins: dict, peek: bool = True
    ) -> StreamTurn:
        if not self.persistent_shadow:
            # legacy behavior: spawn eagerly, tear down at turn end
            shadow = self._new_runner(host_locals, safe_builtins)
            return StreamTurn(StreamSegmenter(), shadow, peek=peek)
        # lazy start: spawn/reuse the runner only when a speculatable call
        # actually appears in the stream. The (single-flight) acquire is
        # PREFETCHED on a background thread so the ~70ms process boot overlaps
        # with the first streamed tokens instead of stalling a mid-stream peek.
        box: dict[str, ShadowRunner] = {}
        box_lock = threading.Lock()

        def factory() -> ShadowRunner:
            with box_lock:
                if "runner" not in box:
                    box["runner"] = self._acquire_runner(host_locals, safe_builtins)
                return box["runner"]

        threading.Thread(
            target=factory, daemon=True, name="spec-shadow-prefetch"
        ).start()
        spec_names = self._spec_names()
        return StreamTurn(
            StreamSegmenter(),
            shadow=None,
            peek=peek,
            shadow_factory=factory,
            spec_names=spec_names,
        )

    def end_turn(self) -> None:
        self.store.evict_unclaimed("turn_end")
        self.launcher.budget.reset()

    def close(self) -> None:
        if self._warm_runner is not None:
            try:
                self._warm_runner.shutdown()
            except Exception:
                pass
            self._warm_runner = None
        self.launcher.shutdown()
