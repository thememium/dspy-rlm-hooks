"""Tests for the subprocess-isolated ShadowRunner (Task 4)."""

from __future__ import annotations

import builtins
import os
import threading
import time

import pytest

from dspy_rlm_hooks.speculation import (
    Opaque,
    ShadowRunner,
    shadow_builtins,
    snapshot_ns,
)
from dspy_rlm_hooks.speculation.store import SpecStore
from dspy_rlm_hooks.speculation.streaming import Segment
from dspy_rlm_hooks.speculation.tool import ToolSpec

REAL_BUILTINS = dict(builtins.__dict__)
HOOKS = {"llm_query", "llm_query_batched"}


def _seg(src: str, index: int = 0) -> Segment:
    return Segment(block_id=0, index=index, source=src, has_call=True)


def _run(code: str, real_locals: dict | None = None, **kw) -> ShadowRunner:
    runner = ShadowRunner(
        real_locals or {},
        {name: None for name in HOOKS},
        SpecStore(),
        REAL_BUILTINS,
        **kw,
    )
    runner.feed(_seg(code))
    runner.finish()
    assert runner.join(10), "shadow did not finish"
    return runner


# -- subprocess isolation ------------------------------------------------------


def test_shadow_runs_in_subprocess():
    runner = ShadowRunner(
        {}, {name: None for name in HOOKS}, SpecStore(), REAL_BUILTINS
    )
    assert runner._proc.pid != os.getpid()
    runner.finish()
    runner.join(10)


def test_import_os_system_blocked_no_host_side_effect(tmp_path):
    marker = tmp_path / "shadow_marker"
    runner = ShadowRunner(
        {}, {name: None for name in HOOKS}, SpecStore(), REAL_BUILTINS
    )
    runner.feed(_seg(f"__import__('os').system('touch {marker}')"))
    runner.finish()
    assert runner.join(10)
    assert not marker.exists()  # __import__('os') blocked -> system never ran
    assert runner.aborted is not None


def test_host_process_state_untouched_by_shadow_mutation():
    host_list = [1, 2, 3]
    runner = ShadowRunner(
        {"data": host_list}, {name: None for name in HOOKS}, SpecStore(), REAL_BUILTINS
    )
    runner.feed(_seg("data.append(999)"))
    runner.finish()
    assert runner.join(10)
    # the shadow mutated its own deepcopy fork, never the host object
    assert host_list == [1, 2, 3]


def test_introspection_escape_cannot_reach_host_objects():
    # The escape runs in the subprocess; it can only touch the subprocess's own
    # memory. It cannot mutate a host-process object passed in via real_locals.
    host_obj = {"secret": "host"}
    runner = ShadowRunner(
        {"obj": host_obj}, {name: None for name in HOOKS}, SpecStore(), REAL_BUILTINS
    )
    runner.feed(_seg("obj['secret'] = 'shadow'"))
    runner.finish()
    assert runner.join(10)
    assert host_obj == {"secret": "host"}


# -- snapshot_ns ---------------------------------------------------------------


def test_snapshot_ns_deepcopies():
    src = {"a": [1, 2], "b": {"x": 1}}
    snap = snapshot_ns(src)
    snap["a"].append(3)
    snap["b"]["x"] = 99
    assert src == {"a": [1, 2], "b": {"x": 1}}
    assert snap["a"] == [1, 2, 3]


def test_snapshot_ns_skips_dunders():
    snap = snapshot_ns({"__builtins__": object(), "x": 1})
    assert "__builtins__" not in snap
    assert snap["x"] == 1


def test_snapshot_ns_un_copyable_becomes_opaque():
    snap = snapshot_ns({"lock": threading.Lock()})
    assert isinstance(snap["lock"], Opaque)
    with pytest.raises(RuntimeError):
        snap["lock"].anything
    with pytest.raises(RuntimeError):
        snap["lock"][0]
    with pytest.raises(RuntimeError):
        iter(snap["lock"])


# -- shadow_builtins -----------------------------------------------------------


def test_shadow_builtins_blocks_dangerous_names():
    b = shadow_builtins(REAL_BUILTINS)
    for name in (
        "open",
        "eval",
        "exec",
        "compile",
        "input",
        "exit",
        "quit",
        "help",
        "breakpoint",
    ):
        with pytest.raises(RuntimeError):
            b[name]("x")


def test_shadow_builtins_restricts_import():
    b = shadow_builtins(REAL_BUILTINS)
    with pytest.raises(RuntimeError):
        b["__import__"]("os")
    with pytest.raises(RuntimeError):
        b["__import__"]("subprocess")
    # pure-stdlib whitelist still importable
    assert b["__import__"]("re") is not None


def test_shadow_builtins_print_is_noop():
    b = shadow_builtins(REAL_BUILTINS)
    assert b["print"]("hello", flush=True) is None


# -- feed / finish / join: predicted call sequence -----------------------------


def test_predicts_literal_call():
    r = _run("llm_query('summarize the doc')")
    assert r.predicted == [("llm_query", ("summarize the doc",))]


def test_predicts_assignment_call():
    r = _run("x = llm_query('analyze ' + question)", {"question": "q"})
    assert r.predicted == [("llm_query", ("analyze q",))]


def test_predicts_for_loop_calls():
    r = _run("for i in range(3):\n    llm_query('item ' + str(i))")
    assert r.predicted == [
        ("llm_query", ("item 0",)),
        ("llm_query", ("item 1",)),
        ("llm_query", ("item 2",)),
    ]


def test_predicts_concat_dependency():
    r = _run("prompt = 'Hello ' + name + '!'\nllm_query(prompt)", {"name": "Alice"})
    assert r.predicted == [("llm_query", ("Hello Alice!",))]


def test_executed_counts_statements():
    runner = ShadowRunner(
        {}, {name: None for name in HOOKS}, SpecStore(), REAL_BUILTINS
    )
    runner.feed(_seg("x = 1", 0))
    runner.feed(_seg("llm_query('q')", 1))
    runner.finish()
    assert runner.join(10)
    assert runner.executed == 2


# -- timeout: a hang never blocks real execution -------------------------------


def test_runaway_watchdog_aborts_hang():
    start = time.monotonic()
    r = _run("while True:\n    pass", stmt_budget=0.2)
    elapsed = time.monotonic() - start
    assert elapsed < 5.0  # bounded, did not hang
    assert r.aborted is not None


def test_join_returns_false_and_terminates_on_hang():
    runner = ShadowRunner(
        {}, {name: None for name in HOOKS}, SpecStore(), REAL_BUILTINS, stmt_budget=10.0
    )
    runner.feed(_seg("while True:\n    pass"))
    start = time.monotonic()
    ok = runner.join(0.2)  # parent-side bound, well under the 10s watchdog
    elapsed = time.monotonic() - start
    assert ok is False
    assert elapsed < 2.0
    runner.abort("test")


# -- taint ---------------------------------------------------------------------


def test_taint_skips_statement_and_poisons_target():
    # x = llm_query(...) stores a NonSpeculated marker in the shadow; a later
    # statement reading x is skipped and its target poisoned, so the dependent
    # call is not predicted.
    r = _run("x = llm_query('a')\ny = x + 1\nllm_query(y)")
    assert r.predicted == [("llm_query", ("a",))]


def test_taint_does_not_poison_comp_local_names():
    # x is a marker but the comprehension statement doesn't read it; its local
    # `i` must not leak/poison, so the independent call is still predicted.
    r = _run("x = llm_query('a')\ny = [i for i in range(3)]\nllm_query('done')")
    assert r.predicted == [("llm_query", ("a",)), ("llm_query", ("done",))]


# -- peek dispatch + bet retraction --------------------------------------------


class _FakeLauncher:
    def __init__(self) -> None:
        self.peeked: list[tuple[str, tuple, int]] = []

    def ensure_peeked(self, tool, args, needed) -> None:
        self.peeked.append((tool.name, args, needed))


class _FakeRegistry:
    def __init__(self, tools: dict) -> None:
        self.tools = tools

    def get(self, name: str):
        return self.tools.get(name)


def _tool(name: str, **kw) -> ToolSpec:
    return ToolSpec(name=name, fn=lambda *a, **k: None, **kw)


def test_peek_dispatch_via_launcher():
    launcher = _FakeLauncher()
    registry = _FakeRegistry(
        {"llm_query": _tool("llm_query", speculatable=True, pure=True)}
    )
    runner = ShadowRunner(
        {"questions": ["a", "b"]},
        {name: None for name in HOOKS},
        SpecStore(),
        REAL_BUILTINS,
        launcher=launcher,
        registry=registry,
    )
    runner.feed_peek("for q in questions:\n    llm_query(q)\n")
    runner.finish()
    assert runner.join(10)
    assert ("llm_query", ("a",), 1) in launcher.peeked
    assert ("llm_query", ("b",), 1) in launcher.peeked


def test_peek_skips_non_speculatable_tool():
    launcher = _FakeLauncher()
    registry = _FakeRegistry({"llm_query": _tool("llm_query")})  # not speculatable
    runner = ShadowRunner(
        {"questions": ["a"]},
        {name: None for name in HOOKS},
        SpecStore(),
        REAL_BUILTINS,
        launcher=launcher,
        registry=registry,
    )
    runner.feed_peek("for q in questions:\n    llm_query(q)\n")
    runner.finish()
    assert runner.join(10)
    assert launcher.peeked == []
