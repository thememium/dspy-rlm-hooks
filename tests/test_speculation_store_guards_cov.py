"""Coverage-completion tests for the SpecStore eviction paths and the guards.

These tests exercise the remaining uncovered branches of ``store.py`` (the
``_enforce_bound`` ValueError/empty-queue and cancel-exception paths, plus the
``evict_unadopted_peeks`` empty/cancel paths and ``_evict`` cancel-exception
path) and of ``guards.py`` (``raw_of`` non-hook passthrough and
``raw_tool_fn`` hook unwrapping).
"""

from __future__ import annotations

from types import SimpleNamespace

from dspy_rlm_hooks.speculation.guards import raw_of, raw_tool_fn, tag_claim_hook
from dspy_rlm_hooks.speculation.store import SpecKey, SpecStore, Speculation

KEY: SpecKey = ("llm_query", "abc123")


def make_spec(
    key: SpecKey = KEY,
    seq: int = 0,
    source: str = "shadow",
    state: str = "pending",
) -> Speculation:
    return Speculation(
        key=key,
        seq=seq,
        args=(),
        kwargs={},
        source=source,
        state=state,
    )


# -- SpecStore._enforce_bound --------------------------------------------------


def test_enforce_bound_swallows_spec_missing_from_queue():
    store = SpecStore(max_entries=1)
    spec = make_spec(seq=0)
    store.put(spec)
    # Manually drop the spec from its FIFO but leave it in the `all` ledger, so
    # eviction hits the ValueError branch and then removes the empty queue.
    store._q[KEY].remove(spec)
    store.put(make_spec(key=("other", "h"), seq=1))
    assert ("other", "h") in store._q
    assert KEY not in store._q
    assert spec.state == "evicted"


def test_enforce_bound_calls_cancel_on_evicted():
    cancelled = []
    store = SpecStore(max_entries=1)
    spec = make_spec(seq=0)
    spec.cancel = lambda: cancelled.append(1)
    store.put(spec)
    store.put(make_spec(seq=1))
    assert cancelled == [1]
    assert spec.state == "evicted"


def test_enforce_bound_swallows_cancel_exception():
    def bad_cancel():
        raise RuntimeError("boom")

    store = SpecStore(max_entries=1)
    spec = make_spec(seq=0)
    spec.cancel = bad_cancel
    store.put(spec)
    store.put(make_spec(seq=1))  # must not propagate
    assert spec.state == "evicted"


# -- SpecStore.evict_unadopted_peeks -------------------------------------------


def test_evict_unadopted_peeks_empty_returns_zero():
    store = SpecStore()
    assert store.evict_unadopted_peeks(KEY, keep=1, reason="x") == 0


def test_evict_unadopted_peeks_calls_cancel():
    cancelled = []
    store = SpecStore()
    specs = [make_spec(seq=i, source="peek") for i in range(3)]
    specs[1].cancel = lambda: cancelled.append(1)
    specs[2].cancel = lambda: cancelled.append(2)
    for s in specs:
        store.put(s)
    n = store.evict_unadopted_peeks(KEY, keep=1, reason="x")
    assert n == 2
    assert cancelled == [1, 2]


def test_evict_unadopted_peeks_swallows_cancel_exception():
    def bad_cancel():
        raise RuntimeError("boom")

    store = SpecStore()
    specs = [make_spec(seq=i, source="peek") for i in range(3)]
    specs[1].cancel = bad_cancel
    specs[2].cancel = bad_cancel
    for s in specs:
        store.put(s)
    n = store.evict_unadopted_peeks(KEY, keep=1, reason="x")
    assert n == 2


# -- SpecStore._evict ----------------------------------------------------------


def test_evict_swallows_cancel_exception():
    def bad_cancel():
        raise RuntimeError("boom")

    store = SpecStore()
    spec = make_spec(seq=0)
    spec.cancel = bad_cancel
    store.put(spec)
    n = store.evict_unclaimed("test")
    assert n == 1
    assert spec.state == "evicted"


# -- guards: raw_of / raw_tool_fn ----------------------------------------------


def test_raw_of_returns_fn_when_not_hook():
    def fn():
        return 1

    assert raw_of(fn) is fn


def test_raw_tool_fn_returns_fn_when_not_hook():
    tool = SimpleNamespace(fn=lambda: 1)
    assert raw_tool_fn(tool) is tool.fn


def test_raw_tool_fn_unwraps_claim_hook_to_raw():
    raw = lambda: 1  # noqa: E731
    hook = tag_claim_hook(lambda: 2, raw_fn=raw)
    tool = SimpleNamespace(fn=hook)
    assert raw_tool_fn(tool) is raw


def test_raw_tool_fn_falls_back_to_hook_when_no_raw_recorded():
    hook = tag_claim_hook(lambda: 2)
    tool = SimpleNamespace(fn=hook)
    assert raw_tool_fn(tool) is hook
