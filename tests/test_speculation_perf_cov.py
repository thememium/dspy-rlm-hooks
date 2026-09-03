"""Coverage for the perf-optimization additions: warm shadow runner,
single-pass seed classification, latency-aware claiming, and chained
continuations. Worker-side functions are exercised in-process with fake
pipes (subprocess coverage is not measured)."""

from __future__ import annotations

import ast
import pickle
import threading
import time

import pytest

from dspy_rlm_hooks.speculation.budget import Budget
from dspy_rlm_hooks.speculation.hooks import (
    _MAX_CLAIM_WAIT_S,
    _claim_or_run,
    _claim_wait_budget,
    _hedge,
    make_real_hooks,
)
from dspy_rlm_hooks.speculation.session import (
    EventBus,
    LatencyStats,
    Launcher,
    SpecSession,
    StreamTurn,
    ToolRegistry,
)
from dspy_rlm_hooks.speculation.shadow import (
    ShadowRunner,
    _register_segment_productions,
    _shadow_worker,
    _worker_fire_chain,
    _worker_peek,
    classify_ns,
    load_ns,
)
from dspy_rlm_hooks.speculation.store import SpecStore, Speculation
from dspy_rlm_hooks.speculation.streaming import (
    ChainMeta,
    ChainPlan,
    Plan,
    Segment,
    plan_peeks_with_chains,
)
from dspy_rlm_hooks.speculation.tool import (
    NonSpeculated,
    ToolSpec,
    canonical_hash,
    spec_key,
)


def _picklable_helper() -> int:
    return 42


def _seg(source: str, index: int = 0) -> Segment:
    return Segment(block_id=0, index=index, source=source, has_call=True)


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


def _tool(name: str = "llm_query", latency: float = 1000.0, **kw) -> ToolSpec:
    def fn(prompt: str = "p") -> str:
        return f"r:{prompt}"

    return ToolSpec(name=name, fn=fn, latency_hint_ms=latency, **kw)


def _feed(turn, code: str) -> None:
    turn.feed("```repl\n" + code + "\n```\n")


def _registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(
        "fetch",
        lambda q: f"doc:{q}",
        speculatable=True,
        pure=True,
        deterministic=True,
        latency_hint_ms=50.0,
    )
    reg.register(
        "llm_query",
        lambda prompt: f"r:{prompt}",
        speculatable=True,
        pure=True,
        deterministic=False,
        latency_hint_ms=50.0,
    )
    return reg


# ---------------------------------------------------------------------------
# classify_ns / load_ns
# ---------------------------------------------------------------------------


def test_classify_ns_skips_dunders_and_pickles_values():
    out = classify_ns({"__builtins__": object(), "x": 1, "s": "hello"})
    assert "__builtins__" not in out
    assert load_ns(out)["s"] == "hello"


def test_classify_ns_nonspeculated_becomes_none():
    out = classify_ns({"m": NonSpeculated("t")})
    assert out["m"] is None
    loaded = load_ns(out)
    from dspy_rlm_hooks.speculation.shadow import Opaque

    assert isinstance(loaded["m"], Opaque)


def _reduce_boom(self):
    raise RuntimeError("nope")


class _BadDict(dict):
    def __reduce__(self):
        raise RuntimeError("nope")


class _BadList(list):
    def __reduce__(self):
        raise RuntimeError("nope")


def test_classify_ns_dict_subclass_plain_cast_fallback():
    out = classify_ns({"d": _BadDict({"x": 1})})
    assert load_ns(out)["d"] == {"x": 1}


def test_classify_ns_list_subclass_plain_cast_fallback():
    out = classify_ns({"l": _BadList([1, 2])})
    assert load_ns(out)["l"] == [1, 2]


def test_classify_ns_double_failure_becomes_none():
    lock = threading.Lock()

    class _BadLockDict(dict):
        def __reduce__(self):
            raise RuntimeError("nope")

    d = _BadLockDict({"k": lock})
    out = classify_ns({"d": d})
    assert out["d"] is None  # cast container still unpicklable -> Opaque

    out2 = classify_ns({"l": _BadList([lock])})
    assert out2["l"] is None


def test_classify_ns_unpicklable_value_becomes_none():
    out = classify_ns({"lock": threading.Lock()})
    assert out["lock"] is None


def test_worker_reset_rebinds_functions_and_end_turn_acks():
    seed = classify_ns({"helper": _picklable_helper, "x": 1})
    conn = _FakeConn([("reset",), ("end_turn",), None])
    parent = _FakeConn([])
    _shadow_worker(
        conn,
        parent,
        {"ns_seed": seed, "spec_names": set(), "taint_skip": True, "budget": 1.0},
    )
    assert ("turn_ended",) in conn.sent


def test_worker_fire_chain_message_in_loop():
    seg = _seg("doc = fetch('auth')")
    conn = _FakeConn(
        [
            seg,
            ("peek", "y = llm_query(doc)"),
            ("fire_chain", 1, {"doc": pickle.dumps("doc:auth")}),
            None,
        ]
    )
    parent = _FakeConn([])
    _shadow_worker(
        conn,
        parent,
        {
            "ns_seed": classify_ns({}),
            "spec_names": {"fetch", "llm_query"},
            "taint_skip": True,
            "budget": 1.0,
        },
    )
    tools = [m for m in conn.sent if isinstance(m, tuple) and m[0] == "tool"]
    assert any(len(m) == 5 and m[1] == "llm_query" for m in tools)


# ---------------------------------------------------------------------------
# _register_segment_productions
# ---------------------------------------------------------------------------


def test_register_segment_productions_syntax_error_is_ignored():
    seg = _seg("def broken(:")
    seg_productions: dict = {}
    _register_segment_productions(seg, {}, {"fetch"}, seg_productions)
    assert seg_productions == {}


def test_register_segment_productions_non_hooked_calls_skipped():
    seg = _seg("y = unknown_tool('q')")
    seg_productions: dict = {}
    _register_segment_productions(seg, {}, {"fetch"}, seg_productions)
    assert seg_productions == {}


def test_register_segment_productions_records_kwargs_and_args():
    seg = _seg("d = fetch('auth')\nr = llm_query(prompt=d)")
    ns = {"d": "doc:auth"}
    seg_productions: dict = {}
    _register_segment_productions(seg, ns, {"fetch", "llm_query"}, seg_productions)
    assert "d" in seg_productions
    assert seg_productions["d"][0] == "segkey"
    assert "r" in seg_productions


def test_register_segment_productions_unresolvable_args_skipped():
    seg = _seg("d = fetch(question)")
    seg_productions: dict = {}
    _register_segment_productions(seg, {}, {"fetch"}, seg_productions)
    assert seg_productions == {}


# ---------------------------------------------------------------------------
# _worker_peek / _worker_fire_chain (worker side, in-process)
# ---------------------------------------------------------------------------


def test_worker_peek_plans_chains_from_segment_productions():
    seg_productions = {
        "doc": ("segkey", ("fetch", canonical_hash("fetch", ("auth",), {})))
    }
    conn = _FakeConn([])
    chains: dict = {}
    _worker_peek(
        conn,
        "summary = llm_query('s: ' + doc)",
        {"fetch", "llm_query"},
        {},
        chains,
        seg_productions,
    )
    assert chains, "expected a chained continuation"
    sent_plans = [m for m in conn.sent if m[0] == "plans"]
    assert sent_plans and sent_plans[0][2], "expected chain metas on the plans message"


def test_worker_fire_chain_unknown_cont_is_noop():
    conn = _FakeConn([])
    _worker_fire_chain(conn, 99, {}, {}, {})
    assert conn.sent == []


def test_worker_fire_chain_unpicklable_dep_drops_chain():
    chain = ChainPlan(
        cont_id=1,
        tool="llm_query",
        arg_specs=[("expr", ast.parse("f(doc)", mode="eval").body)],
        kwarg_specs={},
        deps={},
    )
    conn = _FakeConn([])
    _worker_fire_chain(conn, 1, {"doc": b"not-a-pickle"}, {1: chain}, {})
    assert conn.sent == []


def test_worker_fire_chain_eval_failure_drops_chain():
    chain = ChainPlan(
        cont_id=1,
        tool="llm_query",
        arg_specs=[("expr", ast.parse("missing_name", mode="eval").body)],
        kwarg_specs={},
        deps={},
    )
    conn = _FakeConn([])
    _worker_fire_chain(conn, 1, {}, {1: chain}, {})
    assert conn.sent == []


def test_worker_fire_chain_success_sends_tool_with_cont_id():
    chain = ChainPlan(
        cont_id=1,
        tool="llm_query",
        arg_specs=[("const", "s: "), ("expr", ast.parse("doc", mode="eval").body)],
        kwarg_specs={"prompt": ("const", "kv")},
        deps={},
    )
    conn = _FakeConn([])
    _worker_fire_chain(conn, 1, {"doc": pickle.dumps("doc:auth")}, {1: chain}, {})
    assert len(conn.sent) == 1
    kind, name, args, kwargs, cont_id = conn.sent[0]
    assert (kind, name, args, kwargs, cont_id) == (
        "tool",
        "llm_query",
        ("s: ", "doc:auth"),
        {"prompt": "kv"},
        1,
    )


# ---------------------------------------------------------------------------
# parent-side chain machinery
# ---------------------------------------------------------------------------


def _make_runner(monkeypatch=None) -> ShadowRunner:
    return ShadowRunner(
        {},
        {},
        SpecStore(),
        {},
        launcher=Launcher(SpecStore(), EventBus(), budget=None),
        registry=None,
    )


def test_chain_dep_matches_key_and_cont_and_miss():
    runner = _make_runner()
    meta = ChainMeta(
        cont_id=1, tool="llm_query", deps={"doc": ("key", ("fetch", "h1"))}
    )
    assert runner._chain_dep_matches(meta, ("fetch", "h1"))
    assert not runner._chain_dep_matches(meta, ("fetch", "other"))

    runner._cont_keys[7] = ("fetch", "h1")
    meta2 = ChainMeta(cont_id=2, tool="llm_query", deps={"doc": ("cont", 7)})
    assert runner._chain_dep_matches(meta2, ("fetch", "h1"))
    assert not runner._chain_dep_matches(meta2, ("fetch", "zzz"))


def test_fire_chains_failed_producer_drops_dependent_chains():
    runner = _make_runner()
    key = ("fetch", "h1")
    runner._pending_chains = {
        1: ChainMeta(cont_id=1, tool="llm_query", deps={"doc": ("key", key)})
    }
    spec = Speculation(key=key, seq=1, args=("auth",), kwargs={}, source="peek")
    spec.error = RuntimeError("boom")
    spec.state = "failed"
    spec.done.set()
    runner._fire_chains_for_key(key, spec)
    assert runner._pending_chains == {}


def test_fire_chains_prior_dep_resolution_fires_chain():
    store = SpecStore()
    runner = ShadowRunner(
        {}, {}, store, {}, launcher=Launcher(store, EventBus()), registry=None
    )
    key_a = ("fetch", "ha")
    key_b = ("fetch", "hb")
    runner._pending_chains = {
        1: ChainMeta(
            cont_id=1,
            tool="llm_query",
            deps={"a": ("key", key_a), "b": ("key", key_b)},
        )
    }
    spec_a = Speculation(key=key_a, seq=1, args=(), kwargs={}, source="peek")
    spec_a._result = "A"
    spec_a.state = "ready"
    spec_a.done.set()
    store.put(spec_a)
    spec_b = Speculation(key=key_b, seq=2, args=(), kwargs={}, source="peek")
    spec_b._result = "B"
    spec_b.state = "ready"
    spec_b.done.set()
    runner._fire_chains_for_key(key_b, spec_b)
    # the fire goes to the (terminated-by-now) pipe; assert no crash and the
    # chain satisfied bookkeeping recorded both values
    assert runner._dep_values[1] == {"a": "A", "b": "B"}


def test_fire_chains_second_ready_event_skips_already_satisfied_dep():
    runner = _make_runner()
    key = ("fetch", "h1")
    runner._pending_chains = {
        1: ChainMeta(cont_id=1, tool="llm_query", deps={"doc": ("key", key)})
    }
    spec = Speculation(key=key, seq=1, args=(), kwargs={}, source="peek")
    spec._result = "V"
    spec.state = "ready"
    spec.done.set()
    runner._fire_chains_for_key(key, spec)
    runner._fire_chains_for_key(key, spec)  # second event: values already set
    assert runner._dep_values[1] == {"doc": "V"}


def test_on_bus_event_swallows_subscriber_errors(monkeypatch):
    runner = _make_runner()
    monkeypatch.setattr(
        runner,
        "_fire_chains_for_key",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")),
    )
    runner._on_bus_event("ready", {"key": ("fetch", "h"), "spec": None})
    runner._on_bus_event("other", {})  # non-ready kinds return early


def test_result_for_key_and_key_resolved():
    store = SpecStore()
    runner = ShadowRunner(
        {}, {}, store, {}, launcher=Launcher(store, EventBus()), registry=None
    )
    key = ("fetch", "h")
    assert runner._result_for_key(key) is None
    assert not runner._key_resolved(key)
    spec = Speculation(key=key, seq=1, args=(), kwargs={}, source="peek")
    spec._result = None  # a legitimate None result
    spec.state = "ready"
    spec.done.set()
    store.put(spec)
    assert runner._key_resolved(key)
    assert runner._result_for_key(key) is None
    spec2 = Speculation(key=("fetch", "h2"), seq=2, args=(), kwargs={}, source="peek")
    spec2._result = "V"
    spec2.state = "ready"
    spec2.done.set()
    store.put(spec2)
    assert runner._result_for_key(("fetch", "h2")) == "V"


def test_send_chain_fire_unpicklable_and_dead_pipe():
    runner = _make_runner()
    runner._send_chain_fire(1, {"lock": threading.Lock()})  # unpicklable: dropped
    runner.abort("test")
    runner._send_chain_fire(1, {"v": "x"})  # dead pipe: swallowed


def test_ensure_alive_respawns_after_abort():
    runner = ShadowRunner({}, {}, SpecStore(), {})
    runner.abort("test")
    assert not runner.is_alive
    runner.ensure_alive()
    assert runner.is_alive
    runner.shutdown()


def test_end_turn_early_returns_and_dead_pipe():
    runner = ShadowRunner({}, {}, SpecStore(), {}, persistent=True)
    assert runner.end_turn() is True  # drains the fresh worker
    assert runner.end_turn() is True  # not open -> immediate ack
    runner._turn_open = True
    runner.abort("test")  # proc dead -> False
    runner._turn_open = True
    assert runner.end_turn() is False
    runner2 = ShadowRunner({}, {}, SpecStore(), {}, persistent=True)
    runner2._turn_open = True
    runner2._conn.close()  # broken pipe -> False
    assert runner2.end_turn() is False
    runner.shutdown()
    runner2.shutdown()


def test_handle_plans_translates_and_drops_unwatchable_segkeys():
    reg = ToolRegistry()
    reg.register("fetch", lambda q: "doc", speculatable=True, pure=True)
    runner = ShadowRunner(
        {},
        {},
        SpecStore(),
        {},
        launcher=Launcher(SpecStore(), EventBus()),
        registry=reg,
    )
    tool = _tool("fetch")
    plan_key = ("fetch", canonical_hash("fetch", ("auth",), {}))
    real_key = spec_key(tool, ("auth",), {})
    plans = [
        Plan(tool="fetch", args=("auth",), kwargs={}, key=plan_key),
        Plan(tool="fetch", args=("q",), kwargs={}, key=None),  # key None: skipped
    ]
    metas = [
        ChainMeta(cont_id=1, tool="llm_query", deps={"doc": ("key", plan_key)}),
        ChainMeta(
            cont_id=2, tool="llm_query", deps={"doc": ("segkey", ("fetch", "missing"))}
        ),
        ChainMeta(cont_id=3, tool="llm_query", deps={"doc": ("cont", 9)}),
    ]
    runner._handle_plans(plans, metas)
    assert 1 in runner._pending_chains
    assert runner._pending_chains[1].deps["doc"] == ("key", real_key)
    assert 2 not in runner._pending_chains  # unwatchable segkey: chain dropped
    assert 3 in runner._pending_chains
    assert runner._pending_chains[3].deps["doc"] == ("cont", 9)


def test_immediate_fire_skips_non_key_refs():
    runner = _make_runner()
    meta = ChainMeta(cont_id=1, tool="llm_query", deps={"x": ("cont", 42)})
    runner._handle_plans([], [meta])
    assert 1 in runner._pending_chains  # no fire attempted for cont refs


# ---------------------------------------------------------------------------
# latency-aware claiming
# ---------------------------------------------------------------------------


def _hedge_setup(monkeypatch=None):
    bus = EventBus()
    store = SpecStore()
    launcher = Launcher(store, bus, budget=None, latency_aware=True)
    return bus, store, launcher


def test_claim_wait_budget_ceiling_without_launcher():
    tool = _tool()
    spec = Speculation(
        key=spec_key(tool, ("p",), {}), seq=1, args=("p",), kwargs={}, source="peek"
    )
    assert _claim_wait_budget(spec, tool, None) == _MAX_CLAIM_WAIT_S


def test_claim_wait_budget_running_and_queued():
    tool = _tool(latency=1000.0)
    key = spec_key(tool, ("p",), {})
    spec = Speculation(key=key, seq=1, args=("p",), kwargs={}, source="peek")
    spec.state = "running"
    spec.dispatched_at = time.monotonic() - 10.0  # way overdue
    launcher = Launcher(SpecStore(), EventBus(), latency_aware=True)
    launcher.latency.record("llm_query", 500.0)
    b = _claim_wait_budget(spec, tool, launcher)
    assert 0 < b <= 1.0  # capped at 2x duplicate cost
    spec2 = Speculation(key=key, seq=2, args=("p",), kwargs={}, source="peek")
    spec2.state = "pending"
    launcher._queued = 2
    assert _claim_wait_budget(spec2, tool, launcher) == 0.0  # queue too deep
    launcher._queued = 0
    assert _claim_wait_budget(spec2, tool, launcher) > 0.0


def test_hedge_marks_evicted_and_runs_real_tool():
    bus = EventBus()
    tool = _tool()
    spec = Speculation(
        key=spec_key(tool, ("p",), {}), seq=1, args=("p",), kwargs={}, source="peek"
    )
    calls: list = []
    tool2 = ToolSpec(
        name="llm_query",
        fn=lambda prompt: calls.append(prompt) or "real",
        latency_hint_ms=1.0,
    )
    spec.cancel = lambda: (_ for _ in ()).throw(RuntimeError("cancel failed"))
    out = _hedge(spec, tool2, ("p",), {}, bus, spec.key, time.perf_counter())
    assert out == "real"
    assert spec.state == "evicted"
    kinds = [e[0] for e in bus.history]
    assert "claim_hedge" in kinds and "claim_miss" in kinds


def test_claim_or_run_hedges_after_budget_timeout():
    bus = EventBus()
    store = SpecStore()
    launcher = Launcher(store, bus, latency_aware=True)
    started = threading.Event()

    def slow_tool(prompt: str) -> str:
        started.set()
        time.sleep(0.5)
        return "slow"

    tool = ToolSpec(name="llm_query", fn=slow_tool, latency_hint_ms=1000.0)
    spec = launcher.dispatch(tool, ("p",), {}, "peek")
    assert spec is not None
    started.wait(2)
    launcher.latency.record("llm_query", 100.0)  # tight ewma -> tight budget
    real_calls: list = []
    tool_real = ToolSpec(
        name="llm_query",
        fn=lambda prompt: real_calls.append(prompt) or "real",
        latency_hint_ms=1000.0,
    )
    out = _claim_or_run(tool_real, ("p",), {}, store, bus, launcher)
    assert out == "real"
    assert spec.state == "evicted"


def test_claim_or_run_hedges_pending_queued_spec_immediately():
    bus = EventBus()
    store = SpecStore()
    launcher = Launcher(
        store,
        bus,
        budget=Budget(max_inflight=1, max_dispatches_per_turn=100),
        latency_aware=True,
    )
    launcher.latency.record("llm_query", 1000.0)

    def slow_tool(prompt: str) -> str:
        time.sleep(0.4)
        return "slow"

    tool = ToolSpec(name="llm_query", fn=slow_tool, latency_hint_ms=1000.0)
    s1 = launcher.dispatch(tool, ("p1",), {}, "peek")
    s2 = launcher.dispatch(tool, ("p2",), {}, "peek")  # queued behind s1
    assert s1 is not None and s2 is not None and s2.state == "pending"
    real_calls: list = []
    tool_real = ToolSpec(
        name="llm_query",
        fn=lambda prompt: real_calls.append(prompt) or "real",
        latency_hint_ms=1000.0,
    )
    out = _claim_or_run(tool_real, ("p2",), {}, store, bus, launcher)
    assert out == "real"
    assert s2.state == "evicted"
    s1.result(timeout=5)
    launcher.shutdown()


async def test_async_real_hook_hedges_immediately_and_after_timeout():
    import asyncio

    reg = ToolRegistry()

    async def llm_query(prompt: str) -> str:
        await asyncio.sleep(0.5)
        return "slow"

    reg.register(
        "llm_query", llm_query, speculatable=True, pure=True, latency_hint_ms=1000.0
    )

    # immediate hedge: the spec is PENDING behind a full single-worker pool
    store = SpecStore()
    bus = EventBus()
    launcher = Launcher(
        store,
        bus,
        budget=Budget(max_inflight=1, max_dispatches_per_turn=100),
        latency_aware=True,
    )
    launcher.latency.record("llm_query", 1000.0)
    hooks = make_real_hooks(reg, store, launcher, bus)
    single = reg.get("llm_query")
    assert single is not None
    launcher.dispatch(single, ("p1",), {}, "peek")  # occupies the pool
    s2 = launcher.dispatch(single, ("p2",), {}, "peek")  # queued
    assert s2 is not None and s2.state == "pending"
    out = await hooks["llm_query"](prompt="p2")
    assert out == "slow"  # hedged: ran the real (slow) tool
    assert s2.state == "evicted"

    # timeout hedge: running spec, tight recorded ewma -> budget miss
    launcher2 = Launcher(store, bus, latency_aware=True)
    launcher2.latency.record("llm_query", 100.0)
    hooks2 = make_real_hooks(reg, store, launcher2, bus)
    single2 = reg.get("llm_query")
    assert single2 is not None
    spec2 = launcher2.dispatch(single2, ("q",), {}, "peek")
    assert spec2 is not None
    await asyncio.sleep(0.02)
    out2 = await hooks2["llm_query"](prompt="q")
    assert out2 == "slow"
    assert spec2.state == "evicted"


async def test_async_batched_hook_hedges_elements():
    """One element hedges immediately (queued), one after a budget miss."""
    import asyncio

    reg = ToolRegistry()

    async def llm_query(prompt: str) -> str:
        await asyncio.sleep(0.6)
        return f"slow:{prompt}"

    async def llm_query_batched(prompts: list) -> list:
        await asyncio.sleep(0.05)
        return [f"batch:{p}" for p in prompts]

    reg.register(
        "llm_query", llm_query, speculatable=True, pure=True, latency_hint_ms=1000.0
    )
    reg.register(
        "llm_query_batched",
        llm_query_batched,
        speculatable=True,
        pure=True,
        latency_hint_ms=1000.0,
    )
    store = SpecStore()
    bus = EventBus()
    launcher = Launcher(
        store,
        bus,
        budget=Budget(max_inflight=1, max_dispatches_per_turn=100),
        latency_aware=True,
    )
    launcher.latency.record("llm_query", 100.0)  # tight: running specs miss it
    hooks = make_real_hooks(reg, store, launcher, bus)
    single = reg.get("llm_query")
    assert single is not None
    s1 = launcher.dispatch(single, ("a",), {}, "peek")  # running
    s2 = launcher.dispatch(single, ("b",), {}, "peek")  # queued -> budget 0
    assert s1 is not None and s2 is not None
    await asyncio.sleep(0.02)

    out = await hooks["llm_query_batched"](["a", "b"])
    assert out == ["batch:a", "batch:b"]
    assert s1.state == "evicted" and s2.state == "evicted"


def test_sync_batched_hook_hedges_elements():
    reg = ToolRegistry()

    def llm_query(prompt: str) -> str:
        time.sleep(0.6)
        return f"slow:{prompt}"

    def llm_query_batched(prompts: list) -> list:
        return [f"batch:{p}" for p in prompts]

    reg.register(
        "llm_query", llm_query, speculatable=True, pure=True, latency_hint_ms=1000.0
    )
    reg.register(
        "llm_query_batched",
        llm_query_batched,
        speculatable=True,
        pure=True,
        latency_hint_ms=1000.0,
    )
    store = SpecStore()
    bus = EventBus()
    launcher = Launcher(
        store,
        bus,
        budget=Budget(max_inflight=1, max_dispatches_per_turn=100),
        latency_aware=True,
    )
    launcher.latency.record("llm_query", 100.0)  # tight: the running spec misses it
    hooks = make_real_hooks(reg, store, launcher, bus)
    single = reg.get("llm_query")
    assert single is not None
    s1 = launcher.dispatch(single, ("a",), {}, "peek")  # running
    s2 = launcher.dispatch(single, ("b",), {}, "peek")  # queued -> budget 0
    assert s1 is not None and s2 is not None
    out = hooks["llm_query_batched"](["a", "b"])
    assert out == ["batch:a", "batch:b"]
    assert s1.state == "evicted" and s2.state == "evicted"


# ---------------------------------------------------------------------------
# session / EventBus
# ---------------------------------------------------------------------------


def test_event_bus_subscriber_errors_swallowed():
    bus = EventBus()
    bus.subscribe(lambda kind, data: (_ for _ in ()).throw(RuntimeError("sub boom")))
    seen: list = []
    bus.subscribe(lambda kind, data: seen.append(kind))
    ev = bus.emit("ready", key="k")
    assert ev[0] == "ready"
    assert seen == ["ready"]


def test_latency_stats_samples_counter():
    stats = LatencyStats()
    stats.record("t", 100.0)
    stats.record("t", 200.0)
    assert stats.samples("t") == 2
    assert stats.samples("other") == 0
    assert 100 < stats.ewma_ms("t", 0) < 200


def test_stream_turn_acquire_without_shadow_or_factory_raises():
    from dspy_rlm_hooks.speculation.streaming import StreamSegmenter

    turn = StreamTurn(StreamSegmenter(), shadow=None, shadow_factory=None)
    with pytest.raises(RuntimeError):
        turn._acquire()


def test_session_close_tolerates_shutdown_failure(monkeypatch):
    session = SpecSession(_registry())
    try:
        turn = session.begin_stream_turn({}, {}, peek=True)
        _feed(turn, "x = llm_query('a')")
        turn.end(timeout=5.0)
        runner = session._warm_runner
        assert runner is not None
        monkeypatch.setattr(
            runner,
            "shutdown",
            lambda: (_ for _ in ()).throw(RuntimeError("shutdown boom")),
        )
        session.close()  # must not raise
    finally:
        session.close()


# ---------------------------------------------------------------------------
# streaming: direct planner coverage
# ---------------------------------------------------------------------------


def test_planner_same_tail_producer_consumer_chain():
    tail = "doc = fetch('auth')\nsummary = llm_query('s: ' + doc)"
    plans, chains, metas = plan_peeks_with_chains(tail, {"fetch", "llm_query"}, {})
    assert len(plans) == 1 and plans[0].tool == "fetch"
    assert len(chains) == 1
    cp = chains[0]
    assert cp.tool == "llm_query"
    assert len(cp.arg_specs) == 1
    assert cp.arg_specs[0][0] == "expr"  # the whole BinOp reads the dep
    assert cp.deps == {"doc": ("key", plans[0].key)}
    assert metas[0].cont_id == cp.cont_id


def test_planner_kwarg_dep_chain():
    tail = "doc = fetch('auth')\nres = llm_query(prompt=doc)"
    plans, chains, metas = plan_peeks_with_chains(tail, {"fetch", "llm_query"}, {})
    assert plans and plans[0].tool == "fetch"
    assert len(chains) == 1
    assert chains[0].kwarg_specs["prompt"][0] == "expr"
    assert "doc" in chains[0].deps


def test_planner_kwarg_unpack_rejected():
    tail = "doc = fetch('auth')\nres = llm_query(**doc)"
    plans, chains, metas = plan_peeks_with_chains(tail, {"fetch", "llm_query"}, {})
    assert plans and plans[0].tool == "fetch"
    assert chains == []


def test_planner_non_assign_target_records_no_production():
    tail = "data[0] = fetch('auth')\nsummary = llm_query('s: ' + data[0])"
    plans, chains, metas = plan_peeks_with_chains(tail, {"fetch", "llm_query"}, {})
    assert plans  # the fetch call itself is planned
    assert chains == []  # data[0] is not a chainable single-name target


def test_planner_loop_body_chain_per_item():
    tail = "for u in urls:\n    read_page(cached + u)"
    seg_productions = {
        "cached": ("segkey", ("fetch", canonical_hash("fetch", ("c",), {})))
    }
    plans, chains, metas = plan_peeks_with_chains(
        tail, {"read_page"}, {"urls": ["a", "b"]}, segment_productions=seg_productions
    )
    assert len(chains) == 2  # one continuation per loop item
    assert all(cp.deps == {"cached": seg_productions["cached"]} for cp in chains)
    assert len(metas) == 2


def test_planner_transitive_chain_in_tail():
    tail = (
        "doc = fetch('auth')\nsummary = llm_query('s: ' + doc)\nfinal = rank(summary)"
    )
    plans, chains, metas = plan_peeks_with_chains(
        tail, {"fetch", "llm_query", "rank"}, {}
    )
    assert len(plans) == 1
    assert len(chains) == 2
    by_tool = {cp.tool: cp for cp in chains}
    assert by_tool["llm_query"].deps["doc"] == ("key", plans[0].key)
    assert by_tool["rank"].deps["summary"] == ("cont", by_tool["llm_query"].cont_id)


def test_worker_loop_plans_tainted_segment_chains():
    """A taint-skipped segment's hooked calls become chained continuations."""
    seg1 = _seg("doc = fetch('a')")
    seg2 = _seg("y = llm_query('s: ' + doc)", index=1)
    conn = _FakeConn([seg1, seg2, None])
    parent = _FakeConn([])
    _shadow_worker(
        conn,
        parent,
        {
            "ns_seed": classify_ns({}),
            "spec_names": {"fetch", "llm_query"},
            "taint_skip": True,
            "budget": 1.0,
        },
    )
    plans_msgs = [m for m in conn.sent if isinstance(m, tuple) and m[0] == "plans"]
    assert plans_msgs, "expected the tainted segment to plan chains"
    tools = [m for m in conn.sent if isinstance(m, tuple) and m[0] == "tool"]
    assert any(m[1] == "fetch" for m in tools)  # the producer dispatched


def test_register_segment_productions_skips_non_assign_statements():
    seg = _seg("llm_query('x')")  # bare Expr: no assignment target
    seg_productions: dict = {}
    _register_segment_productions(seg, {}, {"llm_query"}, seg_productions)
    assert seg_productions == {}


def test_spec_for_key_returns_none_when_absent():
    runner = _make_runner()
    assert runner._spec_for_key(("fetch", "nope")) is None


def test_sync_batched_hook_claims_ready_and_inflight_elements():
    """Mixed batch: one element already ready (fast path), one in-flight that
    completes within its budget -> both claimed (no hedge)."""
    reg = ToolRegistry()

    def llm_query(prompt: str) -> str:
        time.sleep(0.3)
        return f"slow:{prompt}"

    def llm_query_batched(prompts: list) -> list:
        return [f"batch:{p}" for p in prompts]

    reg.register(
        "llm_query", llm_query, speculatable=True, pure=True, latency_hint_ms=1000.0
    )
    reg.register(
        "llm_query_batched",
        llm_query_batched,
        speculatable=True,
        pure=True,
        latency_hint_ms=1000.0,
    )
    store = SpecStore()
    bus = EventBus()
    launcher = Launcher(store, bus, latency_aware=True)
    launcher.latency.record("llm_query", 1000.0)  # generous budget
    hooks = make_real_hooks(reg, store, launcher, bus)
    single = reg.get("llm_query")
    assert single is not None
    s1 = launcher.dispatch(single, ("a",), {}, "peek")  # completes fast
    s2 = launcher.dispatch(single, ("b",), {}, "peek")
    assert s1 is not None and s2 is not None
    s1.result(timeout=5)  # ready before the claim
    out = hooks["llm_query_batched"](["a", "b"])
    assert out == ["slow:a", "slow:b"]
    assert s1.state == "claimed" and s2.state == "claimed"


def test_planner_stale_kwarg_value_rejected():
    """A kwarg reading a tail-assigned name with a STALE ns value is rejected
    (the rail), not dispatched with the stale value."""
    tail = "y = 5\nres = llm_query(prompt=y)"
    plans, chains, metas = plan_peeks_with_chains(tail, {"llm_query"}, {"y": 1})
    assert plans == [] and chains == [] and metas == []


class _Unrepr:
    """An object safe_eval passes through but whose repr explodes."""

    def __repr__(self) -> str:
        raise RuntimeError("no repr for you")


def test_plan_tainted_segment_survives_unrepr_args():
    """A concrete-path canonicalization failure inside a tainted segment is
    contained (the segment's plans are dropped, the worker stays alive)."""
    seg1 = _seg("m = llm_query('q')")
    seg2 = _seg("both = [fetch(m), fetch(weird)]", index=1)
    conn = _FakeConn([seg1, seg2, None])
    parent = _FakeConn([])
    _shadow_worker(
        conn,
        parent,
        {
            "ns_seed": classify_ns({"weird": _Unrepr()}),
            "spec_names": {"fetch", "llm_query"},
            "taint_skip": True,
            "budget": 1.0,
        },
    )
    tools = [m for m in conn.sent if isinstance(m, tuple) and m[0] == "tool"]
    assert any(m[1] == "llm_query" for m in tools)
    # no abort: the worker survived the failure
    assert not any(isinstance(m, tuple) and m[0] == "abort" for m in conn.sent)


def test_sync_batched_hook_waits_for_inflight_element():
    """One element ready before the claim, one in-flight that completes within
    its latency budget -> both claimed, no hedge."""
    reg = ToolRegistry()

    def llm_query(prompt: str) -> str:
        time.sleep(0.8 if prompt == "b" else 0.05)
        return f"slow:{prompt}"

    def llm_query_batched(prompts: list) -> list:
        return [f"batch:{p}" for p in prompts]

    reg.register(
        "llm_query", llm_query, speculatable=True, pure=True, latency_hint_ms=1000.0
    )
    reg.register(
        "llm_query_batched",
        llm_query_batched,
        speculatable=True,
        pure=True,
        latency_hint_ms=1000.0,
    )
    store = SpecStore()
    bus = EventBus()
    launcher = Launcher(store, bus, latency_aware=True)
    launcher.latency.record("llm_query", 1000.0)  # generous budget
    hooks = make_real_hooks(reg, store, launcher, bus)
    single = reg.get("llm_query")
    assert single is not None
    s1 = launcher.dispatch(single, ("a",), {}, "peek")
    s2 = launcher.dispatch(single, ("b",), {}, "peek")
    assert s1 is not None and s2 is not None
    s1.result(timeout=5)  # "a" ready; "b" still in flight (0.8s tool)
    out = hooks["llm_query_batched"](["a", "b"])
    assert out == ["slow:a", "slow:b"]
    assert s1.state == "claimed" and s2.state == "claimed"
    assert not any(e[0] == "claim_hedge" for e in bus.history)
