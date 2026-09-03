"""Public ``Speculator`` facade for the speculation engine (Task 6).

The facade is the thin entry point users interact with. It owns a
:class:`ToolRegistry` and lazily builds a :class:`SpecSession`, then delegates
every operation to it — no business logic lives here. ``enable_rlm_speculation``
(Task 7) builds on top of this facade.

Typical usage::

    spec = Speculator()
    @spec.tool(speculatable=True, pure=True)
    def llm_query(prompt: str) -> str: ...

    hooks = spec.hooks()          # install in the real REPL namespace
    with spec.turn(repl_locals=ns) as t:
        t.feed(delta)             # stream model deltas
    spec.end_turn()
    spec.close()
"""

from __future__ import annotations

import builtins
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

from dspy_rlm_hooks.speculation.session import EventBus, SpecSession, ToolRegistry
from dspy_rlm_hooks.speculation.shadow import shadow_builtins
from dspy_rlm_hooks.speculation.tool import SpeculativeTool, ToolSpec


class Speculator:
    """Thin facade over a :class:`SpecSession` + :class:`ToolRegistry`.

    Args:
        max_inflight: Max speculative executions in flight at once.
        max_dispatches_per_turn: Hard cap on dispatches per RLM turn.
        bus: Optional :class:`EventBus`; a fresh one is created if omitted.
        taint_skip: Skip taint propagation in the shadow when safe.
    """

    def __init__(
        self,
        max_inflight: int = 8,
        max_dispatches_per_turn: int = 2048,
        bus: EventBus | None = None,
        taint_skip: bool = True,
        persistent_shadow: bool = True,
        latency_aware: bool = True,
    ) -> None:
        self.registry = ToolRegistry()
        self.bus = bus or EventBus()
        self.max_inflight = max_inflight
        self.max_dispatches_per_turn = max_dispatches_per_turn
        self.taint_skip = taint_skip
        self.persistent_shadow = persistent_shadow
        self.latency_aware = latency_aware
        self._session: SpecSession | None = None
        # last known RAW fn per tool name (never a claim hook), so
        # _sync_registry_fns can recover when repl.tools holds hooks
        self._raw_fns: dict[str, Callable] = {}

    # ---------------------------------------------------------- registration
    def tool(
        self,
        *,
        speculatable: bool = False,
        pure: bool = False,
        deterministic: bool = False,
        latency_hint_ms: float = 1000.0,
        name: str | None = None,
        gate: Callable[..., bool] | None = None,
    ) -> Callable:
        """Decorator: register a plain function as a tool.

        ``speculatable=True`` requires ``pure=True`` (a tool with observable
        side effects must never execute early). Unmarked functions are never
        speculated.
        """

        def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
            fn_name = getattr(fn, "__name__", type(fn).__name__)
            if speculatable and not pure:
                raise ValueError(
                    f"tool {name or fn_name!r}: speculatable=True requires "
                    "pure=True — a tool with observable side effects must never "
                    "execute early"
                )
            self.registry.register(
                name or fn_name,
                fn,
                speculatable=speculatable,
                pure=pure,
                deterministic=deterministic,
                latency_hint_ms=latency_hint_ms,
                gate_fn=gate,
            )
            return fn

        return deco

    def add(self, tool: ToolSpec | SpeculativeTool) -> None:
        """Register a :class:`ToolSpec` or :class:`SpeculativeTool` instance."""
        spec = tool.to_spec() if isinstance(tool, SpeculativeTool) else tool
        self.registry.register(
            spec.name,
            spec.fn,
            speculatable=spec.speculatable,
            pure=spec.pure,
            deterministic=spec.deterministic,
            latency_hint_ms=spec.latency_hint_ms,
            spec_fn=spec.spec_fn,
            cancel_fn=spec.cancel_fn,
            key_fn=spec.key_fn,
            gate_fn=spec.gate_fn,
        )

    # ---------------------------------------------------------- runtime
    @property
    def session(self) -> SpecSession:
        """The lazily-built :class:`SpecSession` (built on first use)."""
        if self._session is None:
            self._session = SpecSession(
                self.registry,
                self.bus,
                max_inflight=self.max_inflight,
                max_dispatches_per_turn=self.max_dispatches_per_turn,
                taint_skip=self.taint_skip,
                persistent_shadow=self.persistent_shadow,
                latency_aware=self.latency_aware,
            )
        return self._session

    def hooks(self) -> dict[str, Callable]:
        """Claiming hooks to install in the host REPL's namespace."""
        return self.session.real_hooks()

    @contextmanager
    def turn(
        self,
        repl_locals: dict | None = None,
        safe_builtins: dict | None = None,
        peek: bool = True,
    ):
        """One streaming turn: feed model deltas inside the block.

        Yields a :class:`StreamTurn`; on exit the shadow drains and the turn is
        ended (``turn.end(timeout)`` + ``session.end_turn()``).
        """
        t = self.session.begin_stream_turn(
            dict(repl_locals or {}),
            safe_builtins or shadow_builtins(dict(builtins.__dict__)),
            peek=peek,
        )
        try:
            yield t
        finally:
            t.end()
            self.session.end_turn()

    def end_turn(self) -> None:
        """Evict unclaimed speculations and reset the per-turn budget."""
        self.session.end_turn()

    def stats(self) -> dict:
        """Aggregate speculation metrics from the store's ledger."""
        store = self.session.store
        return {
            "speculated": len(store.all),
            "claimed": sum(1 for s in store.all if s.state == "claimed"),
            "evicted": sum(1 for s in store.all if s.state == "evicted"),
        }

    def close(self) -> None:
        """Shut down the launcher (no-op if the session was never built)."""
        if self._session is not None:
            self._session.close()
