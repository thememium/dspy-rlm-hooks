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
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from dspy_rlm_hooks.speculation.budget import Budget
from dspy_rlm_hooks.speculation.hooks import (
    make_baseline_hooks,
    make_real_hooks,
    make_shadow_hooks,
)
from dspy_rlm_hooks.speculation.shadow import ShadowRunner
from dspy_rlm_hooks.speculation.store import SpecStore, Speculation
from dspy_rlm_hooks.speculation.streaming import StreamSegmenter
from dspy_rlm_hooks.speculation.tool import ToolSpec, _is_async_callable, spec_key


class EventBus:
    """Minimal observable event bus: ``emit(kind, **data)`` records to history."""

    def __init__(self) -> None:
        self.history: list[tuple[str, dict]] = []
        self.record = True

    def emit(self, kind: str, **data: Any) -> tuple[str, dict]:
        ev = (kind, data)
        if self.record:
            self.history.append(ev)
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


class Launcher:
    """Dispatches speculative executions on a thread pool.

    ``max_workers`` comes from the Budget's ``max_inflight``; each dispatch
    reserves a per-turn slot via ``budget.try_dispatch()`` (hard deny when the
    turn cap is hit). Async tools run on one shared background asyncio loop so
    loop-bound clients stay valid across calls.
    """

    def __init__(
        self,
        store: SpecStore,
        bus: EventBus | None = None,
        budget: Budget | None = None,
    ) -> None:
        self.store = store
        self.bus = bus or EventBus()
        self.budget = budget or Budget()
        self._seq = 0
        self._seq_lock = threading.Lock()
        self._pool = ThreadPoolExecutor(
            max_workers=self.budget.max_inflight, thread_name_prefix="spec"
        )
        self._aloop: asyncio.AbstractEventLoop | None = None
        self._aloop_lock = threading.Lock()

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

    def ensure_peeked(self, tool: ToolSpec, args: tuple, needed: int) -> int:
        """Top the store up to ``needed`` un-adopted peek speculations for this
        exact call (multiplicity-safe dedup across repeated peeks of a growing
        tail). Returns how many new dispatches were made."""
        key = spec_key(tool, args, {})
        if tool.deterministic:
            if self.store.existing(key):
                return 0
            needed = 1
        new = 0
        while self._unadopted_peeks(key) < needed:
            if self.dispatch(tool, args, {}, "peek") is None:
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
            if spec.state == "evicted":
                spec.done.set()
                return
            spec.state = "running"
            run_fn = tool.spec_fn or tool.fn
            try:
                if getattr(run_fn, "wants_spec", False):
                    out = run_fn(*args, _spec=spec, **kwargs)
                else:
                    out = run_fn(*args, **kwargs)
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
            spec.resolved_at = time.monotonic()
            spec.done.set()
            if spec.state == "ready":
                self.bus.emit("ready", key=key, seq=spec.seq)

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
    """One streaming turn's segmenter + shadow, bound to a host namespace."""

    segmenter: StreamSegmenter
    shadow: ShadowRunner
    peek: bool = True
    _last_tail: str = ""

    def feed(self, delta: str) -> None:
        for seg in self.segmenter.feed(delta):
            self.shadow.feed(seg)
        if self.peek and "\n" in delta:
            tail = self.segmenter.pending_tail()
            if tail.strip() and tail != self._last_tail and self._peek_worthwhile(tail):
                self._last_tail = tail
                self.shadow.feed_peek(tail)

    def _peek_worthwhile(self, tail: str) -> bool:
        if len(tail) > 12_000:
            return False
        changed = (
            tail[len(self._last_tail) :] if tail.startswith(self._last_tail) else tail
        )
        if any(name in changed for name in self.shadow.hooks):
            return True
        return bool(self.shadow._last_peek_tally)

    def end(self, timeout: float = 600) -> None:
        for seg in self.segmenter.finish():
            self.shadow.feed(seg)
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
    ) -> None:
        self.reg = registry
        self.bus = bus or EventBus()
        self.taint_skip = taint_skip
        self.store = SpecStore()
        self.launcher = Launcher(
            self.store,
            self.bus,
            Budget(
                max_inflight=max_inflight,
                max_dispatches_per_turn=max_dispatches_per_turn,
            ),
        )

    def real_hooks(self) -> dict:
        """Claiming hooks for the real REPL (byte-identical to calling tools)."""
        return make_real_hooks(self.reg, self.store, self.launcher, self.bus)

    def baseline_hooks(self) -> dict:
        """Unmodified passthrough tools (no speculation)."""
        return make_baseline_hooks(self.reg)

    def begin_stream_turn(
        self, host_locals: dict, safe_builtins: dict, peek: bool = True
    ) -> StreamTurn:
        shadow = ShadowRunner(
            host_locals,
            make_shadow_hooks(self.reg, self.store, self.launcher, self.bus),
            self.store,
            safe_builtins,
            launcher=self.launcher,
            registry=self.reg,
            taint_skip=self.taint_skip,
        )
        return StreamTurn(StreamSegmenter(), shadow, peek=peek)

    def end_turn(self) -> None:
        self.store.evict_unclaimed("turn_end")
        self.launcher.budget.reset()

    def close(self) -> None:
        self.launcher.shutdown()
