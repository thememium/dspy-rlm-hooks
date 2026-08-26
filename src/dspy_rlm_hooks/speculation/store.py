"""Speculation future + SpecStore (Task 2).

This module implements the ``Speculation`` future dataclass and the
:class:`SpecStore` that holds and claims them. It builds against the frozen
contracts from Task 1 (``tool.py`` / ``config.py``) and matches the documented
``Speculation`` field contract in ``tool.py`` lines 23-46 exactly, so Task 5
(hooks) and Task 6 (facade) can consume it.

Only the store/budget surface lives here — no streaming/shadow/session/hooks.
"""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from dspy_rlm_hooks.speculation.tool import SpecKey

# The lifecycle states a speculation can be in. ``ready`` means the future has
# resolved successfully and is waiting to be claimed; ``claimed`` means a real
# call has taken ownership of it.
SpecState = str  # "pending" | "running" | "ready" | "claimed" | "evicted" | "failed"


@dataclass
class Speculation:
    """A single speculative execution future.

    Matches the documented contract in ``tool.py`` (lines 23-46). ``wait``
    blocks on the ``done`` event and returns success; ``result`` waits then
    returns the resolved value or raises the stored error.

    Note: the resolved value is stored in ``_result`` and exposed through the
    ``result()`` accessor. A dataclass field and a method cannot share the name
    ``result`` in Python, so the value lives in ``_result`` and ``result()`` is
    the documented accessor (the only way to obtain the value, since ``wait``
    returns a bool).
    """

    key: SpecKey  # (tool_name, canonical args hash)
    seq: int  # monotonic dispatch sequence (== claim order)
    args: tuple  # the speculated call args
    kwargs: dict  # the speculated call kwargs
    source: str  # "shadow" | "real" | "peek" | ...
    state: str = "pending"  # pending|running|ready|claimed|evicted|failed
    _result: Any = field(default=None, init=False, repr=False)  # resolved value
    error: BaseException | None = None  # raised error once done
    done: threading.Event = field(default_factory=threading.Event)  # set on resolve
    adopted: bool = False  # a shadow hook took ownership of this peek
    cancel: Callable[[], None] | None = None  # abort in-flight on eviction
    dispatched_at: float | None = None  # monotonic() timestamp
    resolved_at: float | None = None  # monotonic() timestamp

    def wait(self, timeout: float | None = None) -> bool:
        """Block on the ``done`` event; return True if the future resolved
        successfully (no error), False on timeout or error."""
        if not self.done.wait(timeout):
            return False
        return self.error is None

    def result(self, timeout: float | None = None) -> Any:
        """Wait for resolution, then return the result or raise the error."""
        if not self.done.wait(timeout):
            raise TimeoutError(f"speculation {self.key} not ready after {timeout}s")
        if self.error is not None:
            raise self.error
        return self._result


class SpecStore:
    """Thread-safe store of speculative futures, keyed by ``SpecKey``.

    Multiplicity is preserved: N identical non-deterministic dispatches queue N
    independent speculations, and the k-th claim pops the k-th (FIFO). A
    deterministic tool reuses its one cached run instead of popping.

    The store is bounded by ``max_entries``; when exceeded the oldest
    speculation is evicted (marked ``evicted``, cancel called, removed).
    """

    def __init__(self, max_entries: int = 10_000) -> None:
        self._q: dict[SpecKey, deque[Speculation]] = {}
        self._lock = threading.Lock()
        self.all: list[Speculation] = []  # every speculation ever, for metrics
        self.max_entries = max_entries

    # -- mutation -----------------------------------------------------------

    def put(self, spec: Speculation) -> None:
        """Insert a speculation into its key's FIFO and the ``all`` ledger."""
        with self._lock:
            self._q.setdefault(spec.key, deque()).append(spec)
            self.all.append(spec)
            self._enforce_bound()

    def _enforce_bound(self) -> None:
        """Evict the oldest speculations once ``max_entries`` is exceeded."""
        while len(self.all) > self.max_entries:
            oldest = self.all.pop(0)
            q = self._q.get(oldest.key)
            if q is not None:
                try:
                    q.remove(oldest)
                except ValueError:
                    pass
                if not q:
                    del self._q[oldest.key]
            if oldest.state in ("pending", "running", "ready"):
                oldest.state = "evicted"
                if oldest.cancel:
                    try:
                        oldest.cancel()
                    except Exception:
                        pass

    # -- claim / reuse ------------------------------------------------------

    def claim(self, key: SpecKey, reuse: bool = False) -> Speculation | None:
        """Pop the oldest claimable speculation for ``key`` (hit), else None.

        ``reuse=True`` (deterministic tools) returns the one cached run without
        removing it, so repeated claims share a single result. ``reuse=False``
        pops FIFO, preserving multiplicity for non-deterministic tools.
        """
        with self._lock:
            q = self._q.get(key)
            if reuse:
                for spec in q or ():
                    if spec.state in ("pending", "running", "ready", "claimed"):
                        if spec.state == "ready":
                            spec.state = "claimed"
                        return spec
                return None
            while q:
                spec = q.popleft()
                if spec.state in ("pending", "running", "ready"):
                    if spec.state == "ready":
                        spec.state = "claimed"
                    return spec
            return None

    def existing(self, key: SpecKey) -> Speculation | None:
        """Return the first live speculation for ``key`` without consuming it."""
        with self._lock:
            for spec in self._q.get(key, ()):
                if spec.state in ("pending", "running", "ready", "claimed"):
                    return spec
            return None

    def adopt(self, key: SpecKey) -> Speculation | None:
        """Shadow-side twin of ``claim``: take ownership of the oldest
        un-adopted PEEK speculation for ``key`` without removing it from the
        claim FIFO (the real run must still claim it later, in order)."""
        with self._lock:
            for spec in self._q.get(key, ()):
                if (
                    spec.source == "peek"
                    and not spec.adopted
                    and spec.state in ("pending", "running", "ready")
                ):
                    spec.adopted = True
                    return spec
            return None

    # -- eviction -----------------------------------------------------------

    def evict_unclaimed(self, reason: str) -> int:
        """Evict every live (unclaimed) speculation; return how many."""
        return self._evict(lambda s: True, reason)

    def evict_tool(self, tool_name: str, reason: str) -> int:
        """Evict all live speculations for one tool; return how many."""
        return self._evict(lambda s: s.key[0] == tool_name, reason)

    def evict_unadopted_peeks(self, key: SpecKey, keep: int, reason: str) -> int:
        """Retract peek bets: evict un-adopted peek speculations for ``key``
        beyond the first ``keep`` (oldest stay — adoption takes oldest first)."""
        n = 0
        with self._lock:
            q = self._q.get(key)
            if not q:
                return 0
            seen = 0
            for spec in list(q):
                if (
                    spec.source == "peek"
                    and not spec.adopted
                    and spec.state in ("pending", "running", "ready")
                ):
                    seen += 1
                    if seen > keep:
                        spec.state = "evicted"
                        if spec.cancel:
                            try:
                                spec.cancel()
                            except Exception:
                                pass
                        q.remove(spec)
                        n += 1
        return n

    def _evict(self, pred: Callable[[Speculation], bool], reason: str) -> int:
        n = 0
        with self._lock:
            for q in self._q.values():
                for spec in list(q):
                    if pred(spec) and spec.state in ("pending", "running", "ready"):
                        spec.state = "evicted"
                        if spec.cancel:
                            try:
                                spec.cancel()
                            except Exception:
                                pass
                        q.remove(spec)
                        n += 1
        return n

    # -- introspection ------------------------------------------------------

    def __len__(self) -> int:
        with self._lock:
            return sum(len(q) for q in self._q.values())
