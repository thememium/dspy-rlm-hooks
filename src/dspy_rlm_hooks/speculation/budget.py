"""Budget for the speculative execution engine (Task 2).

Caps speculative spend. ``max_inflight`` bounds concurrent speculative
executions; ``max_dispatches_per_turn`` is a hard per-turn deny on total
dispatches. This budget tracks speculative concurrency and per-turn dispatch
volume — NOT the real ``max_llm_calls`` counter (that is handled in Task 7
integration). All counters are thread-safe.
"""

from __future__ import annotations

import threading


class Budget:
    """Thread-safe caps on speculative concurrency and per-turn dispatches."""

    def __init__(
        self,
        max_inflight: int = 8,
        max_dispatches_per_turn: int = 2048,
    ) -> None:
        self.max_inflight = max_inflight
        self.max_dispatches_per_turn = max_dispatches_per_turn
        self._inflight = 0
        self._dispatched = 0
        self._lock = threading.Lock()

    # -- inflight (concurrent executions) -----------------------------------

    def acquire_inflight(self) -> bool:
        """Reserve one in-flight slot; False when ``max_inflight`` is reached."""
        with self._lock:
            if self._inflight >= self.max_inflight:
                return False
            self._inflight += 1
            return True

    def release_inflight(self) -> None:
        """Free one in-flight slot (no-op if none are held)."""
        with self._lock:
            if self._inflight > 0:
                self._inflight -= 1

    # -- per-turn dispatch cap -----------------------------------------------------

    def can_dispatch(self) -> bool:
        """True if another dispatch is allowed this turn."""
        with self._lock:
            return self._dispatched < self.max_dispatches_per_turn

    def try_dispatch(self) -> bool:
        """Atomically reserve one dispatch slot; False when the turn cap is hit."""
        with self._lock:
            if self._dispatched >= self.max_dispatches_per_turn:
                return False
            self._dispatched += 1
            return True

    def reset(self) -> None:
        """Reset the per-turn dispatch counter (called at the start of a turn)."""
        with self._lock:
            self._dispatched = 0

    # -- introspection -------------------------------------------------------------

    @property
    def inflight(self) -> int:
        with self._lock:
            return self._inflight

    @property
    def dispatched_this_turn(self) -> int:
        with self._lock:
            return self._dispatched
