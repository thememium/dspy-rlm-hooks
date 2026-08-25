"""Tests for the Budget (Task 2)."""

from __future__ import annotations

import threading

from dspy_rlm_hooks.speculation import Budget


def test_budget_defaults():
    b = Budget()
    assert b.max_inflight == 8
    assert b.max_dispatches_per_turn == 2048
    assert b.inflight == 0
    assert b.dispatched_this_turn == 0


def test_acquire_inflight_cap():
    b = Budget(max_inflight=2)
    assert b.acquire_inflight() is True
    assert b.acquire_inflight() is True
    assert b.acquire_inflight() is False  # cap reached
    assert b.inflight == 2


def test_release_inflight():
    b = Budget(max_inflight=2)
    b.acquire_inflight()
    b.acquire_inflight()
    b.release_inflight()
    assert b.inflight == 1
    assert b.acquire_inflight() is True  # slot freed


def test_release_inflight_noop_when_empty():
    b = Budget()
    b.release_inflight()
    assert b.inflight == 0


def test_try_dispatch_cap():
    b = Budget(max_dispatches_per_turn=3)
    assert b.can_dispatch() is True
    assert b.try_dispatch() is True
    assert b.try_dispatch() is True
    assert b.try_dispatch() is True
    assert b.try_dispatch() is False  # hard deny
    assert b.can_dispatch() is False
    assert b.dispatched_this_turn == 3


def test_reset_clears_dispatch_counter():
    b = Budget(max_dispatches_per_turn=2)
    b.try_dispatch()
    b.try_dispatch()
    assert b.try_dispatch() is False
    b.reset()
    assert b.dispatched_this_turn == 0
    assert b.try_dispatch() is True


def test_reset_does_not_clear_inflight():
    b = Budget(max_inflight=1)
    b.acquire_inflight()
    b.reset()
    assert b.inflight == 1


def test_thread_safety_smoke():
    b = Budget(max_inflight=100, max_dispatches_per_turn=100)
    errors = []

    def worker():
        try:
            for _ in range(50):
                b.acquire_inflight()
                b.release_inflight()
                b.try_dispatch()
        except Exception as e:  # pragma: no cover
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert b.dispatched_this_turn == 100  # capped at max_dispatches_per_turn
    assert b.inflight == 0
