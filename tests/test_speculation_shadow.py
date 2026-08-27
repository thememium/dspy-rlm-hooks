"""Tests for the subprocess-isolated ShadowRunner (Task 4)."""

from __future__ import annotations

import ast
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
from dspy_rlm_hooks.speculation.shadow import (
    _bound_names,
    _comp_local_names,
    _make_record_hook,
    _picklable_ns,
    _read_names,
    _rebind,
    _shadow_worker,
    _worker_exec,
    _worker_peek,
)
from dspy_rlm_hooks.speculation.store import SpecStore, Speculation
from dspy_rlm_hooks.speculation.streaming import Segment
from dspy_rlm_hooks.speculation.tool import NonSpeculated, ToolSpec, spec_key

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

    def ensure_peeked(self, tool, args, kwargs, needed) -> None:
        self.peeked.append((tool.name, args, needed))


class _FakeRegistry:
    def __init__(self, tools: dict) -> None:
        self.tools = tools

    def get(self, name: str):
        return self.tools.get(name)


def _tool(name: str, **kw) -> ToolSpec:
    return ToolSpec(name=name, fn=lambda *a, **k: None, **kw)


class _FakeConn:
    """In-process stand-in for the worker's multiprocessing pipe end."""

    def __init__(self, msgs: list, raise_on_empty: type[Exception] = EOFError) -> None:
        self.msgs = list(msgs)
        self.sent: list = []
        self.closed = False
        self._raise = raise_on_empty

    def recv(self):
        if self.msgs:
            return self.msgs.pop(0)
        raise self._raise

    def send(self, msg) -> None:
        self.sent.append(msg)

    def close(self) -> None:
        self.closed = True


class _NoneConn:
    """Pipe end that yields a single None sentinel then EOF — for _read."""

    def __init__(self) -> None:
        self.calls = 0

    def recv(self):
        self.calls += 1
        if self.calls == 1:
            return None
        raise EOFError

    def send(self, msg) -> None:
        pass


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


# -- _rebind -------------------------------------------------------------------


_REBIND_GLOBAL = 1


def _rebind_target(a, b=2):
    return a + b + _REBIND_GLOBAL


def test_rebind_resolves_globals_in_new_namespace():
    g = _rebind(_rebind_target, {"_REBIND_GLOBAL": 10})
    assert g(1) == 13  # default + global both resolve in the shadow ns
    assert g.__name__ == "_rebind_target"


def test_rebind_preserves_dict_and_kwdefaults():
    def f(*, k=5):
        return k

    f.__dict__["marker"] = "kept"
    g = _rebind(f, {})
    assert g() == 5  # __kwdefaults__ carried over
    assert g.__dict__["marker"] == "kept"  # __dict__ carried over


# -- snapshot_ns fallback branches ---------------------------------------------


class _BadDict(dict):
    """A dict subclass whose deepcopy fails (un-copyable attr) but whose items
    are copyable — exercises the plain-cast fallback in snapshot_ns."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.lock = threading.Lock()


class _BadList(list):
    """A list subclass whose deepcopy fails — exercises the type(v)(...) fallback."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.lock = threading.Lock()


def test_snapshot_ns_dict_subclass_falls_back_to_plain_cast():
    snap = snapshot_ns({"d": _BadDict({"x": 1})})
    # deepcopy of the subclass fails (lock attr); the data survives as a plain dict
    assert snap["d"] == {"x": 1}
    assert type(snap["d"]) is dict


def test_snapshot_ns_list_subclass_falls_back_to_type_cast():
    snap = snapshot_ns({"l": _BadList([1, 2])})
    assert snap["l"] == [1, 2]


def test_snapshot_ns_dict_double_failure_becomes_opaque():
    # deepcopy fails AND the plain-cast fallback fails (un-copyable items too)
    snap = snapshot_ns({"d": _BadDict({"x": threading.Lock()})})
    assert isinstance(snap["d"], Opaque)


def test_snapshot_ns_list_double_failure_becomes_opaque():
    snap = snapshot_ns({"l": _BadList([threading.Lock()])})
    assert isinstance(snap["l"], Opaque)


# -- _picklable_ns --------------------------------------------------------------


def test_picklable_ns_converts_nonspec_and_unpicklable_to_opaque():
    out = _picklable_ns({"m": NonSpeculated("t"), "lock": threading.Lock(), "x": 1})
    # NonSpeculated markers can't cross the spawn boundary -> Opaque
    assert isinstance(out["m"], Opaque)
    # un-picklable values -> Opaque
    assert isinstance(out["lock"], Opaque)
    # picklable values pass through untouched
    assert out["x"] == 1


# -- _make_record_hook ----------------------------------------------------------


def test_make_record_hook_sends_call_and_returns_marker():
    conn = _FakeConn([])
    hook = _make_record_hook(conn, "llm_query")
    assert hook.__name__ == "llm_query"
    result = hook("arg1", kw=2)
    assert conn.sent == [("tool", "llm_query", ("arg1",), {"kw": 2})]
    assert isinstance(result, NonSpeculated)


# -- _worker_exec ---------------------------------------------------------------


def test_worker_exec_syntax_error_aborts():
    conn = _FakeConn([])
    _worker_exec(conn, _seg("def :"), {}, set(), True, 1.0)
    assert conn.sent == [("abort", "syntax")]


def test_worker_exec_rebinding_hooked_name_evicts_and_aborts():
    conn = _FakeConn([])
    _worker_exec(conn, _seg("llm_query = 5"), {}, {"llm_query"}, True, 1.0)
    assert conn.sent == [("evict_tool", "llm_query"), ("abort", "rebind:llm_query")]


def test_worker_exec_taint_skips_and_poisons_target():
    conn = _FakeConn([])
    ns = {"x": NonSpeculated("t")}
    _worker_exec(conn, _seg("y = x + 1"), ns, set(), True, 1.0)
    # statement reading a marker is skipped (no message) and its target poisoned
    assert conn.sent == []
    assert isinstance(ns["y"], NonSpeculated)


def test_worker_exec_success_sends_executed():
    conn = _FakeConn([])
    _worker_exec(conn, _seg("x = 1"), {}, set(), True, 1.0)
    assert conn.sent == [("executed",)]


def test_worker_exec_runtime_error_aborts():
    conn = _FakeConn([])
    _worker_exec(conn, _seg("1 / 0"), {}, set(), True, 1.0)
    assert conn.sent == [("abort", "ZeroDivisionError: division by zero")]


def test_worker_exec_runaway_watchdog_aborts():
    # the SIGALRM handler fires when a statement exceeds its wall-clock budget
    conn = _FakeConn([])
    _worker_exec(conn, _seg("while True:\n    pass"), {}, set(), True, 0.05)
    assert conn.sent == [("abort", "ShadowAborted: runaway")]


# -- _worker_peek ---------------------------------------------------------------


def test_worker_peek_sends_plans():
    conn = _FakeConn([])
    _worker_peek(conn, "llm_query('q')", {"llm_query"}, {})
    assert len(conn.sent) == 1
    kind, plans = conn.sent[0]
    assert kind == "plans"
    assert [p.tool for p in plans] == ["llm_query"]


def test_worker_peek_survives_planning_failure(monkeypatch):
    # if plan_peeks raises, the worker degrades to no plans instead of crashing
    def boom(tail, spec_names, ns):
        raise RuntimeError("boom")

    monkeypatch.setattr("dspy_rlm_hooks.speculation.shadow.plan_peeks", boom)
    conn = _FakeConn([])
    _worker_peek(conn, "llm_query('q')", {"llm_query"}, {})
    assert conn.sent == [("plans", [])]


# -- _shadow_worker -------------------------------------------------------------


def test_shadow_worker_processes_messages_then_closes():
    conn = _FakeConn(
        [
            _seg("x = 1"),
            ("peek", "llm_query('q')"),
            None,  # finish sentinel
        ]
    )
    parent = _FakeConn([])
    _shadow_worker(
        conn,
        parent,
        {
            "ns": {"x": 1, "f": lambda: 1},
            "spec_names": {"llm_query"},
            "taint_skip": True,
            "budget": 1.0,
        },
    )
    assert ("executed",) in conn.sent
    assert any(m[0] == "plans" for m in conn.sent)
    assert parent.closed  # parent end closed at start
    assert conn.closed  # worker end closed in finally


def test_worker_worker_survives_oserror():
    conn = _FakeConn([], raise_on_empty=OSError)
    parent = _FakeConn([])
    _shadow_worker(
        conn,
        parent,
        {"ns": {}, "spec_names": set(), "taint_skip": True, "budget": 1.0},
    )
    assert conn.closed


# -- name-collector helpers ------------------------------------------------------


def test_read_names_collects_load_context_only():
    assert _read_names(ast.parse("x = a + b")) == {"a", "b"}
    assert _read_names(ast.parse("x = 1")) == set()  # store ctx is not a read


def test_comp_local_names_collects_comp_and_lambda_locals():
    assert _comp_local_names(ast.parse("y = [i for i in range(3)]")) == {"i"}
    assert _comp_local_names(ast.parse("f = lambda x: x")) == {"x"}


def test_bound_names_collects_stores_and_defs():
    tree = ast.parse("x = 1\ndef f():\n    pass\nclass C:\n    pass")
    assert _bound_names(tree) == {"x", "f", "C"}


# -- parent-side: abort / _read / evict_tool ------------------------------------


def test_abort_terminates_live_process():
    runner = ShadowRunner(
        {}, {name: None for name in HOOKS}, SpecStore(), REAL_BUILTINS
    )
    assert runner._proc.is_alive()
    runner.abort("test")
    assert runner.aborted == "test"
    runner.finish()
    runner.join(10)


def test_read_breaks_on_none_sentinel():
    # The worker never sends None to the parent, but _read must still handle it
    # defensively (a None message terminates the reader loop).
    runner = ShadowRunner(
        {}, {name: None for name in HOOKS}, SpecStore(), REAL_BUILTINS
    )
    runner._conn = _NoneConn()
    runner._read()
    assert runner._done.is_set()
    runner.finish()
    runner.join(10)


def test_rebinding_hooked_name_evicts_store_speculations():
    store = SpecStore()
    store.put(
        Speculation(key=("llm_query", "h"), seq=0, args=(), kwargs={}, source="shadow")
    )
    assert len(store) == 1
    runner = ShadowRunner({}, {name: None for name in HOOKS}, store, REAL_BUILTINS)
    runner.feed(_seg("llm_query = 5"))
    runner.finish()
    assert runner.join(10)
    # the worker's evict_tool message reached the parent and evicted the bet
    assert len(store) == 0
    assert runner.aborted == "rebind:llm_query"


# -- _dispatch ------------------------------------------------------------------


def test_dispatch_calls_ensure_peeked_for_speculatable():
    launcher = _FakeLauncher()
    registry = _FakeRegistry(
        {"llm_query": _tool("llm_query", speculatable=True, pure=True)}
    )
    runner = ShadowRunner(
        {},
        {name: None for name in HOOKS},
        SpecStore(),
        REAL_BUILTINS,
        launcher=launcher,
        registry=registry,
    )
    runner.feed(_seg("llm_query('x')"))
    runner.finish()
    assert runner.join(10)
    assert ("llm_query", ("x",), 1) in launcher.peeked


def test_dispatch_skips_non_speculatable_tool():
    launcher = _FakeLauncher()
    registry = _FakeRegistry({"llm_query": _tool("llm_query")})  # not speculatable
    runner = ShadowRunner(
        {},
        {name: None for name in HOOKS},
        SpecStore(),
        REAL_BUILTINS,
        launcher=launcher,
        registry=registry,
    )
    runner.feed(_seg("llm_query('x')"))
    runner.finish()
    assert runner.join(10)
    assert launcher.peeked == []


def test_dispatch_respects_gate_fn():
    launcher = _FakeLauncher()
    registry = _FakeRegistry(
        {
            "llm_query": _tool(
                "llm_query", speculatable=True, pure=True, gate_fn=lambda a, k: False
            )
        }
    )
    runner = ShadowRunner(
        {},
        {name: None for name in HOOKS},
        SpecStore(),
        REAL_BUILTINS,
        launcher=launcher,
        registry=registry,
    )
    runner.feed(_seg("llm_query('x')"))
    runner.finish()
    assert runner.join(10)
    assert launcher.peeked == []


# -- _handle_plans --------------------------------------------------------------


def test_handle_plans_without_launcher_is_noop():
    runner = ShadowRunner(
        {}, {name: None for name in HOOKS}, SpecStore(), REAL_BUILTINS
    )
    runner.feed_peek("llm_query('a')\n")
    runner.finish()
    assert runner.join(10)  # plans arrive but no launcher -> no dispatch


def test_handle_plans_respects_gate_fn():
    launcher = _FakeLauncher()
    registry = _FakeRegistry(
        {
            "llm_query": _tool(
                "llm_query", speculatable=True, pure=True, gate_fn=lambda a, k: False
            )
        }
    )
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


def test_handle_plans_retracts_stale_peek_bets():
    launcher = _FakeLauncher()
    tool = _tool("llm_query", speculatable=True, pure=True)
    registry = _FakeRegistry({"llm_query": tool})
    store = SpecStore()
    # two un-adopted peek bets the second plan will no longer justify
    store.put(
        Speculation(
            key=spec_key(tool, ("b",), {}), seq=0, args=("b",), kwargs={}, source="peek"
        )
    )
    store.put(
        Speculation(
            key=spec_key(tool, ("c",), {}), seq=1, args=("c",), kwargs={}, source="peek"
        )
    )
    assert len(store) == 2
    runner = ShadowRunner(
        {"questions": ["a", "b", "c"]},
        {name: None for name in HOOKS},
        store,
        REAL_BUILTINS,
        launcher=launcher,
        registry=registry,
    )
    # first plan justifies a, b, c; the second only a -> b and c are retracted
    runner.feed_peek("for q in questions:\n    llm_query(q)\n")
    runner.feed_peek("llm_query('a')\n")
    runner.finish()
    assert runner.join(10)
    assert len(store) == 0  # stale peek bets evicted via evict_unadopted_peeks
