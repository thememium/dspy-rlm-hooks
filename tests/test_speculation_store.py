"""Tests for the Speculation future + SpecStore (Task 2)."""

from __future__ import annotations

import threading
import time

import pytest

from dspy_rlm_hooks.speculation import SpecKey, SpecStore, Speculation

KEY: SpecKey = ("llm_query", "abc123")


def make_spec(
    key: SpecKey = KEY,
    seq: int = 0,
    source: str = "shadow",
    state: str = "pending",
    result: object = None,
    error: BaseException | None = None,
) -> Speculation:
    spec = Speculation(
        key=key,
        seq=seq,
        args=(),
        kwargs={},
        source=source,
        state=state,
        error=error,
    )
    spec._result = result
    if state in ("ready", "failed"):
        spec.done.set()
    return spec


# -- Speculation future ------------------------------------------------------


def test_speculation_defaults():
    spec = Speculation(key=KEY, seq=1, args=(), kwargs={}, source="shadow")
    assert spec.state == "pending"
    assert spec._result is None
    assert spec.error is None
    assert spec.adopted is False
    assert spec.cancel is None
    assert spec.dispatched_at is None
    assert spec.resolved_at is None
    assert not spec.done.is_set()


def test_speculation_wait_ready_returns_true():
    spec = make_spec(state="ready", result=42)
    assert spec.wait() is True
    assert spec.result() == 42


def test_speculation_wait_timeout_returns_false():
    spec = make_spec(state="pending")
    assert spec.wait(timeout=0.01) is False


def test_speculation_result_raises_error():
    err = RuntimeError("boom")
    spec = make_spec(state="failed", error=err)
    with pytest.raises(RuntimeError, match="boom"):
        spec.result()


def test_speculation_result_timeout_raises():
    spec = make_spec(state="pending")
    with pytest.raises(TimeoutError):
        spec.result(timeout=0.01)


def test_speculation_wait_blocks_until_set():
    spec = make_spec(state="pending")

    def resolve():
        time.sleep(0.05)
        spec._result = "done"
        spec.state = "ready"
        spec.done.set()

    t = threading.Thread(target=resolve)
    t.start()
    assert spec.wait(timeout=2.0) is True
    assert spec.result() == "done"
    t.join()


# -- SpecStore: multiplicity (FIFO) ------------------------------------------


def test_put_and_claim_fifo_preserves_multiplicity():
    store = SpecStore()
    for i in range(3):
        store.put(make_spec(seq=i, result=i, state="ready"))
    # N identical non-deterministic calls -> N independent futures, FIFO claim.
    claimed = []
    for _ in range(3):
        c = store.claim(KEY)
        assert c is not None
        claimed.append(c)
    assert [c.seq for c in claimed] == [0, 1, 2]
    assert [c.result() for c in claimed] == [0, 1, 2]
    # all consumed -> miss
    assert store.claim(KEY) is None


def test_claim_miss_on_empty():
    store = SpecStore()
    assert store.claim(KEY) is None


def test_claim_skips_evicted_and_failed():
    store = SpecStore()
    store.put(make_spec(seq=0, state="evicted"))
    store.put(make_spec(seq=1, state="failed"))
    store.put(make_spec(seq=2, state="ready", result="ok"))
    spec = store.claim(KEY)
    assert spec is not None
    assert spec.seq == 2
    assert spec.state == "claimed"


def test_claim_marks_ready_as_claimed():
    store = SpecStore()
    store.put(make_spec(seq=0, state="ready", result="v"))
    spec = store.claim(KEY)
    assert spec is not None
    assert spec.state == "claimed"


# -- SpecStore: determinism reuse --------------------------------------------


def test_reuse_returns_same_spec_without_consuming():
    store = SpecStore()
    store.put(make_spec(seq=0, state="ready", result="shared"))
    a = store.claim(KEY, reuse=True)
    b = store.claim(KEY, reuse=True)
    assert a is not None
    assert b is not None
    assert a is b
    assert a.result() == "shared"
    assert a.state == "claimed"


def test_reuse_miss_returns_none():
    store = SpecStore()
    assert store.claim(KEY, reuse=True) is None


# -- SpecStore: existing / adopt -----------------------------------------------


def test_existing_returns_live_spec():
    store = SpecStore()
    store.put(make_spec(seq=0, state="running"))
    assert store.existing(KEY) is not None


def test_existing_returns_none_when_absent():
    store = SpecStore()
    assert store.existing(KEY) is None


def test_adopt_takes_oldest_unadopted_peek():
    store = SpecStore()
    store.put(make_spec(seq=0, source="peek"))
    store.put(make_spec(seq=1, source="peek"))
    adopted = store.adopt(KEY)
    assert adopted is not None
    assert adopted.seq == 0
    assert adopted.adopted is True
    # second adopt takes the next un-adopted peek
    second = store.adopt(KEY)
    assert second is not None
    assert second.seq == 1


def test_adopt_ignores_non_peek_and_adopted():
    store = SpecStore()
    store.put(make_spec(seq=0, source="shadow"))
    store.put(make_spec(seq=1, source="peek", state="evicted"))
    assert store.adopt(KEY) is None


# -- SpecStore: eviction --------------------------------------------------------


def test_evict_unclaimed():
    store = SpecStore()
    store.put(make_spec(seq=0, state="pending"))
    store.put(make_spec(seq=1, state="ready"))
    store.put(make_spec(seq=2, state="claimed"))
    n = store.evict_unclaimed("test")
    assert n == 2
    assert store.claim(KEY) is None  # claimed one was already consumed


def test_evict_tool():
    store = SpecStore()
    store.put(make_spec(key=("tool_a", "h1"), seq=0))
    store.put(make_spec(key=("tool_b", "h2"), seq=1))
    n = store.evict_tool("tool_a", "test")
    assert n == 1
    assert store.claim(("tool_a", "h1")) is None
    assert store.claim(("tool_b", "h2")) is not None


def test_evict_unadopted_peeks_keeps_first():
    store = SpecStore()
    for i in range(4):
        store.put(make_spec(seq=i, source="peek"))
    n = store.evict_unadopted_peeks(KEY, keep=1, reason="tail-invalid")
    assert n == 3
    # the kept peek is still claimable
    assert store.claim(KEY) is not None


def test_evict_calls_cancel():
    cancelled = []

    def cancel():
        cancelled.append(1)

    store = SpecStore()
    spec = make_spec(seq=0, state="pending")
    spec.cancel = cancel
    store.put(spec)
    store.evict_unclaimed("test")
    assert cancelled == [1]
    assert spec.state == "evicted"


# -- SpecStore: max-size bound ---------------------------------------------------


def test_max_entries_evicts_oldest():
    store = SpecStore(max_entries=3)
    for i in range(5):
        store.put(make_spec(seq=i))
    assert len(store) == 3
    # oldest (seq 0,1) evicted; newest (seq 2,3,4) remain
    for expected in (2, 3, 4):
        spec = store.claim(KEY)
        assert spec is not None
        assert spec.seq == expected
    assert store.claim(KEY) is None


def test_max_entries_evicted_oldest_marked_evicted():
    store = SpecStore(max_entries=2)
    specs = [make_spec(seq=i) for i in range(3)]
    for s in specs:
        store.put(s)
    assert specs[0].state == "evicted"
    assert specs[1].state == "pending"
    assert specs[2].state == "pending"


# -- SpecStore: thread-safety smoke ----------------------------------------------


def test_thread_safety_smoke():
    store = SpecStore()
    n = 200

    def producer():
        for i in range(n):
            store.put(make_spec(seq=i))

    def consumer():
        for _ in range(n):
            store.claim(KEY)

    threads = [threading.Thread(target=producer) for _ in range(2)]
    threads += [threading.Thread(target=consumer) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # no exceptions raised; store remains internally consistent
    assert isinstance(store, SpecStore)
