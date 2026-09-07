"""Targeted tests for hedge-racing, batch-call parsing, state-sync and
reseed branches that the broader scenario tests do not reach."""

from __future__ import annotations

import asyncio
import builtins
import functools
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import dspy_rlm_hooks.speculation.hooks as _hooks
import dspy_rlm_hooks.speculation.integration as SI
from dspy_rlm_hooks.speculation.guards import fully_raw
from dspy_rlm_hooks.speculation.hooks import _claim_wait_budget, _evict, _race_hedge
from dspy_rlm_hooks.speculation.integration import _live_state_seed, _sync_registry_fns
from dspy_rlm_hooks.speculation.session import EventBus, Launcher, ToolRegistry
from dspy_rlm_hooks.speculation.shadow import Segment, ShadowRunner
from dspy_rlm_hooks.speculation.store import SpecStore, Speculation
from dspy_rlm_hooks.speculation.streaming import plan_peeks_with_chains
from dspy_rlm_hooks.speculation.tool import ToolSpec, spec_key, split_batch_call

REAL_BUILTINS = dict(builtins.__dict__)


def _seg(src: str, index: int = 0) -> Segment:
    return Segment(block_id=0, index=index, source=src, has_call=True)


class _Bus:
    """Minimal event bus stand-in."""

    def __init__(self) -> None:
        self.history: list[tuple] = []

    def emit(self, kind: str, **data):
        self.history.append((kind, data))


_BUS = _Bus()


def _tool() -> ToolSpec:
    return ToolSpec(name="llm_query", fn=lambda prompt: "real")


def _fake_spec(
    *,
    done: bool = False,
    value: object = "v",
    error: BaseException | None = None,
    state: str = "ready",
) -> Speculation:
    spec = Speculation(
        key=("llm_query", "k"), seq=1, args=("p",), kwargs={}, source="peek"
    )
    spec.state = state
    spec.error = error
    if done:
        spec._result = value
        spec.done.set()
    return spec


class _FakeLauncher:
    """Duck-typed launcher: returns a pre-built dup or raises on dispatch."""

    def __init__(self, dup: Speculation | None = None, raises: bool = False):
        self.dup = dup
        self.raises = raises

    def dispatch(self, tool, args, kwargs, source):
        if self.raises:
            raise RuntimeError("dispatch exploded")
        return self.dup


# -- _evict --------------------------------------------------------------------


def test_evict_cancels_and_swallows_cancel_errors():
    # Given a spec with a working cancel and one with a failing cancel.
    calls: list[int] = []
    spec = _fake_spec()
    spec.cancel = lambda: calls.append(1)
    _evict(spec)
    assert calls == [1] and spec.state == "evicted"

    def boom() -> None:
        raise RuntimeError("cancel failed")

    spec2 = _fake_spec()
    spec2.cancel = boom
    _evict(spec2)  # must not raise
    assert spec2.state == "evicted"


# -- _race_hedge: racing unavailable -------------------------------------------


def test_race_hedge_without_launcher_runs_real_tool():
    tool = _tool()
    spec = _fake_spec()
    out = _race_hedge(spec, tool, ("p",), {}, _BUS, spec.key, time.perf_counter())
    assert out == "real"
    assert spec.state == "evicted"


def test_race_hedge_with_failing_dispatch_runs_real_tool():
    tool = _tool()
    spec = _fake_spec()
    out = _race_hedge(
        spec,
        tool,
        ("p",),
        {},
        _BUS,
        spec.key,
        time.perf_counter(),
        _FakeLauncher(raises=True),
    )
    assert out == "real"
    assert spec.state == "evicted"


# -- _race_hedge: in-loop winners -----------------------------------------------


def test_race_hedge_original_wins_in_loop():
    tool = _tool()
    spec = _fake_spec(done=True, value="original")
    dup = _fake_spec(done=False)
    out = _race_hedge(
        spec,
        tool,
        ("p",),
        {},
        _BUS,
        spec.key,
        time.perf_counter(),
        _FakeLauncher(dup=dup),
    )
    assert out == "original"
    assert spec.state == "claimed" and dup.state == "evicted"


def test_race_hedge_duplicate_wins_in_loop():
    tool = _tool()
    spec = _fake_spec(done=False)
    dup = _fake_spec(done=True, value="duplicate")
    out = _race_hedge(
        spec,
        tool,
        ("p",),
        {},
        _BUS,
        spec.key,
        time.perf_counter(),
        _FakeLauncher(dup=dup),
    )
    assert out == "duplicate"
    assert dup.state == "claimed" and spec.state == "evicted"


def test_race_hedge_original_evicted_falls_to_duplicate_failure():
    # Given the original completes as evicted and the duplicate fails.
    tool = _tool()
    spec = _fake_spec(done=True, state="evicted")
    dup = _fake_spec(done=True, error=RuntimeError("dup failed"))
    out = _race_hedge(
        spec,
        tool,
        ("p",),
        {},
        _BUS,
        spec.key,
        time.perf_counter(),
        _FakeLauncher(dup=dup),
    )
    # Then the real tool runs.
    assert out == "real"
    assert spec.state == "evicted" and dup.state == "evicted"


def test_race_hedge_sleeps_until_original_resolves():
    # Given the duplicate never resolves and the original lands mid-race.
    tool = _tool()
    spec = _fake_spec(done=False)
    dup = _fake_spec(done=False)

    def resolve_original():
        spec._result = "late"
        spec.done.set()

    t = threading.Timer(0.05, resolve_original)
    t.start()
    try:
        out = _race_hedge(
            spec,
            tool,
            ("p",),
            {},
            _BUS,
            spec.key,
            time.perf_counter(),
            _FakeLauncher(dup=dup),
        )
    finally:
        t.cancel()
    # Then the late original wins without a duplicate execution.
    assert out == "late"
    assert spec.state == "claimed" and dup.state == "evicted"


def test_race_hedge_duplicate_failed_keeps_waiting_then_real_tool():
    # Given the duplicate fails while the original never resolves.
    tool = _tool()
    spec = _fake_spec(done=False)
    dup = _fake_spec(done=True, error=RuntimeError("dup failed"))
    out = _race_hedge(
        spec,
        tool,
        ("p",),
        {},
        _BUS,
        spec.key,
        time.perf_counter(),
        _FakeLauncher(dup=dup),
    )
    assert out == "real"


# -- _race_hedge: post-loop branches (deadline already expired) -----------------


def _expire_deadline(monkeypatch):
    """Every monotonic() call advances 100s: the race deadline (now + 30s) is
    always already expired at the first `while` check, skipping the poll loop.
    The large per-call jump also makes asyncio.wait_for(10) unreliable, so the
    async tests await the hook directly."""
    state = {"v": time.monotonic()}

    def fake():
        out = state["v"]
        state["v"] += 100
        return out

    monkeypatch.setattr(time, "monotonic", fake)


def test_race_hedge_post_loop_original_win(monkeypatch):
    _expire_deadline(monkeypatch)
    tool = _tool()
    spec = _fake_spec(done=True, value="original")
    dup = _fake_spec(done=False)
    out = _race_hedge(
        spec,
        tool,
        ("p",),
        {},
        _BUS,
        spec.key,
        time.perf_counter(),
        _FakeLauncher(dup=dup),
    )
    assert out == "original" and spec.state == "claimed"


def test_race_hedge_post_loop_duplicate_win(monkeypatch):
    _expire_deadline(monkeypatch)
    tool = _tool()
    spec = _fake_spec(done=False)
    dup = _fake_spec(done=True, value="duplicate")
    out = _race_hedge(
        spec,
        tool,
        ("p",),
        {},
        _BUS,
        spec.key,
        time.perf_counter(),
        _FakeLauncher(dup=dup),
    )
    assert out == "duplicate" and dup.state == "claimed"


def test_race_hedge_post_loop_neither_resolves_runs_real_tool(monkeypatch):
    _expire_deadline(monkeypatch)
    tool = _tool()
    spec = _fake_spec(done=False)
    dup = _fake_spec(done=False)
    out = _race_hedge(
        spec,
        tool,
        ("p",),
        {},
        _BUS,
        spec.key,
        time.perf_counter(),
        _FakeLauncher(dup=dup),
    )
    assert out == "real"
    assert spec.state == "evicted" and dup.state == "evicted"


# -- async race paths -----------------------------------------------------------


class _AsyncRaceHarness:
    """Real store + launcher; the claimed spec is planted manually so its
    resolution timing is fully controlled by the test."""

    def __init__(self):
        self.reg = ToolRegistry()

        async def llm_query(prompt: str) -> str:
            return f"real:{prompt}"

        self.reg.register(
            "llm_query", llm_query, speculatable=True, pure=True, deterministic=True
        )
        self.store = SpecStore()
        self.launcher = Launcher(self.store, EventBus())
        self.launcher.latency.record("llm_query", 50.0)
        self.tool = self.reg.get("llm_query")
        assert self.tool is not None
        self.hooks = _hooks.make_real_hooks(
            self.reg, self.store, self.launcher, self.launcher.bus
        )

    def plant(self, *, done: bool = False, state: str = "running") -> Speculation:
        tool = self.tool
        assert tool is not None
        spec = Speculation(
            key=spec_key(tool, ("q",), {}),  # self.tool narrowed by assert
            seq=1,
            args=("q",),
            kwargs={},
            source="peek",
        )
        spec.state = state
        spec.dispatched_at = time.monotonic()
        if done:
            spec._result = "planted"
            spec.done.set()
        self.store.put(spec)
        return spec


@pytest.mark.asyncio()
async def test_async_race_dispatch_failing_runs_real_tool():
    h = _AsyncRaceHarness()
    spec = h.plant()
    with patch.object(h.launcher, "dispatch", side_effect=RuntimeError("boom")):
        out = await asyncio.wait_for(h.hooks["llm_query"]("q"), timeout=10)
    assert out == "real:q"
    assert spec.state == "evicted"


@pytest.mark.asyncio()
async def test_async_race_duplicate_wins_in_loop():
    h = _AsyncRaceHarness()
    spec = h.plant()
    dup = _fake_spec(done=True, value="duplicate")
    with patch.object(h.launcher, "dispatch", return_value=dup):
        out = await asyncio.wait_for(h.hooks["llm_query"]("q"), timeout=10)
    assert out == "duplicate"
    assert dup.state == "claimed" and spec.state == "evicted"


@pytest.mark.asyncio()
async def test_async_race_original_evicted_breaks_to_real_tool():
    h = _AsyncRaceHarness()
    spec = h.plant()  # claimable; completes as EVICTED mid-race below
    dup = _fake_spec(done=False)

    def complete_as_evicted():
        spec.state = "evicted"
        spec.done.set()

    asyncio.get_running_loop().call_later(0.05, complete_as_evicted)

    async def timeout_immediately(s, timeout=600.0):
        raise asyncio.TimeoutError()

    with (
        patch.object(_hooks, "_await_spec", timeout_immediately),
        patch.object(h.launcher, "dispatch", return_value=dup),
    ):
        out = await asyncio.wait_for(h.hooks["llm_query"]("q"), timeout=10)
    assert out == "real:q"
    assert dup.state == "evicted"


@pytest.mark.asyncio()
async def test_async_race_duplicate_fails_breaks_to_real_tool():
    h = _AsyncRaceHarness()
    h.plant()
    dup = _fake_spec(done=True, error=RuntimeError("dup failed"))
    with patch.object(h.launcher, "dispatch", return_value=dup):
        out = await asyncio.wait_for(h.hooks["llm_query"]("q"), timeout=10)
    assert out == "real:q"
    assert dup.state == "evicted"


async def test_async_race_post_loop_original_win(
    monkeypatch,
):
    h = _AsyncRaceHarness()
    spec = h.plant(done=True, state="ready")
    spec._result = "planted"
    dup = _fake_spec(done=False)
    _expire_deadline(monkeypatch)

    async def timeout_immediately(s, timeout=600.0):
        raise asyncio.TimeoutError()

    with (
        patch.object(_hooks, "_await_spec", timeout_immediately),
        patch.object(h.launcher, "dispatch", return_value=dup),
    ):
        out = await h.hooks["llm_query"]("q")
    assert out == "planted" and spec.state == "claimed" and dup.state == "evicted"


@pytest.mark.asyncio()
async def test_async_race_post_loop_original_win_async(monkeypatch):
    h = _AsyncRaceHarness()
    spec = h.plant(done=True, state="ready")
    spec._result = "planted"
    dup = _fake_spec(done=False)
    _expire_deadline(monkeypatch)

    async def timeout_immediately(s, timeout=600.0):
        raise asyncio.TimeoutError()

    with (
        patch.object(_hooks, "_await_spec", timeout_immediately),
        patch.object(h.launcher, "dispatch", return_value=dup),
    ):
        out = await h.hooks["llm_query"]("q")
    assert out == "planted" and spec.state == "claimed" and dup.state == "evicted"


async def test_async_race_post_loop_duplicate_win(
    monkeypatch,
):
    h = _AsyncRaceHarness()
    spec = h.plant()
    dup = _fake_spec(done=True, value="duplicate")
    _expire_deadline(monkeypatch)

    async def timeout_immediately(s, timeout=600.0):
        raise asyncio.TimeoutError()

    with (
        patch.object(_hooks, "_await_spec", timeout_immediately),
        patch.object(h.launcher, "dispatch", return_value=dup),
    ):
        out = await h.hooks["llm_query"]("q")
    assert out == "duplicate" and dup.state == "claimed" and spec.state == "evicted"


# -- claim budget plumbing touched indirectly above ------------------------------


def test_claim_wait_budget_running_uses_started_at():
    tool = _tool()
    key = spec_key(tool, ("p",), {})
    spec = Speculation(key=key, seq=1, args=("p",), kwargs={}, source="peek")
    spec.state = "running"
    spec.dispatched_at = time.monotonic() - 100.0  # huge queue time
    spec.started_at = time.monotonic()  # just started
    launcher = Launcher(SpecStore(), EventBus())
    launcher.latency.record("llm_query", 500.0)
    b = _claim_wait_budget(spec, tool, launcher)
    assert 0 < b <= 1.0  # NOT capped at zero despite ancient dispatched_at


# -- split_batch_call branch coverage --------------------------------------------


def test_split_batch_call_no_args_no_kwargs_is_none():
    assert split_batch_call((), {}) is None


def test_split_batch_call_known_kwarg_name():
    split = split_batch_call((), {"prompts": ["a", "b"]})
    assert split is not None
    prompts, rest, clean, kwname = split
    assert prompts == ["a", "b"] and rest == () and clean == {} and kwname == "prompts"


def test_split_batch_call_unknown_single_list_kwarg():
    split = split_batch_call((), {"queries": ["a"], "n": 2})
    assert split is not None
    prompts, rest, clean, kwname = split
    assert prompts == ["a"] and clean == {"n": 2} and kwname == "queries"


def test_split_batch_call_unknown_kwarg_name_used_as_key():
    split = split_batch_call((), {"qs": ["a"], "n": 1})
    assert split is not None
    prompts, rest, clean, kwname = split
    assert prompts == ["a"] and clean == {"n": 1} and kwname == "qs"


def test_split_batch_call_ambiguous_kwargs_is_none():
    assert split_batch_call((), {"a": [1], "b": [2]}) is None


def test_split_batch_call_named_kwarg_with_non_list_value_is_none():
    # "prompts" matches a known name but is not a list: falls to the candidate
    # scan, finds no list-shaped kwarg, and gives up.
    assert split_batch_call((), {"prompts": "nope"}) is None


# -- fully_raw: wraps chain -------------------------------------------------------


def test_fully_raw_unwraps_wraps_chain():
    def base(prompt: str) -> str:
        return prompt

    @functools.wraps(base)
    def wrapper(*args, **kwargs):
        return base(*args, **kwargs)

    assert fully_raw(wrapper) is base


# -- _sync_registry_fns fallbacks --------------------------------------------------


def _sync_fixture():
    reg = ToolRegistry()
    reg.register("llm_query", lambda prompt: "raw", speculatable=True, pure=True)
    spec = SimpleNamespace(registry=reg, _raw_fns={})
    return reg, spec


def test_sync_registry_fns_none_candidate_falls_back_to_cached_raw():
    reg, spec = _sync_fixture()
    repl = SimpleNamespace(tools={"llm_query": lambda prompt: "fresh"})
    _sync_registry_fns(spec, repl)
    cached = reg.get("llm_query").fn
    assert cached is not None
    # a later tools dict entry of None must keep the cached raw fn
    repl2 = SimpleNamespace(tools={"llm_query": None})
    _sync_registry_fns(spec, repl2)
    assert reg.get("llm_query").fn is cached


def test_sync_registry_fns_none_candidate_with_no_cache_skips():
    reg, spec = _sync_fixture()
    repl = SimpleNamespace(tools={"llm_query": None})
    _sync_registry_fns(spec, repl)
    assert reg.get("llm_query").fn is not None  # untouched raw registration


# -- _live_state_seed error/skip paths ---------------------------------------------


def _seed_fixture():
    reg = ToolRegistry()
    reg.register("llm_query", lambda prompt: "raw", speculatable=True, pure=True)
    spec = SimpleNamespace(registry=reg)
    return spec


def test_live_state_seed_unparsable_code_returns_input_args():
    spec = _seed_fixture()
    seed = _live_state_seed(None, "def broken(:", {"q": 1}, spec)
    assert seed == {"q": 1}


def test_live_state_seed_probe_failure_returns_input_args():
    spec = _seed_fixture()

    class BoomRepl:
        def execute(self, code):
            raise RuntimeError("sandbox died")

    seed = _live_state_seed(BoomRepl(), "print(llm_query(v))", {}, spec)
    assert seed == {}


def test_live_state_seed_non_dict_snapshot_returns_input_args():
    spec = _seed_fixture()

    class Repl:
        def execute(self, code):
            return "null"

    seed = _live_state_seed(Repl(), "print(llm_query(v))", {}, spec)
    assert seed == {}


def test_live_state_seed_skips_non_string_values():
    spec = _seed_fixture()

    class Repl:
        def execute(self, code):
            return '{"v": 5}'  # JSON number, not a repr string

    seed = _live_state_seed(Repl(), "print(llm_query(v))", {}, spec)
    assert seed == {}


# -- speculation wrapper: streaming snapshot feed failure is contained -------------


def test_streaming_snapshot_feed_failure_is_contained():
    class BoomTurn:
        def feed(self, text):
            raise RuntimeError("pipe broken")

        def end(self, timeout=None):
            raise RuntimeError("pipe broken")

    spec = SimpleNamespace(registry=None, end_turn=lambda: None)
    config = SimpleNamespace(enabled=True, timeout_s=1.0)
    repl = SimpleNamespace(execute=lambda code: '{"previous": "\'R:first\'"}')

    class FakeSub:
        def __init__(self, turn):
            self.turn = turn

        def __call__(self, r, code, ia):
            return "ran"

    rlm = SimpleNamespace(
        _speculator=spec,
        _speculation_config=config,
        _speculation_original_execute_code=lambda repl, code, input_args: "ran",
        _active_stream_turn=BoomTurn(),
        _streaming_fed_any=False,
    )
    # turn.feed raising inside the snapshot-assign path must NOT propagate
    assert (
        SI._speculation_execute_code(rlm, repl, "print(llm_query(previous))", {})
        == "ran"
    )
    assert rlm._active_stream_turn is None


# -- shadow worker reseed + batch key guards ----------------------------------------


def test_worker_reset_with_reseed_rebuilds_namespace():

    from dspy_rlm_hooks.speculation.shadow import _shadow_worker, classify_ns

    class FakeConn:
        def __init__(self, msgs):
            self.msgs = list(msgs)
            self.sent = []

        def recv(self):
            if self.msgs:
                return self.msgs.pop(0)
            raise EOFError

        def send(self, msg):
            self.sent.append(msg)

        def close(self):
            pass

    payload = {
        "ns_seed": classify_ns({"x": 1}),
        "spec_names": {"llm_query"},
        "taint_skip": True,
        "budget": 1.0,
    }
    conn = FakeConn(
        [
            ("reset", classify_ns({"x": 2})),  # reseed: worker rebuilds from it
            ("end_turn",),
        ]
    )
    _shadow_worker(conn, FakeConn([]), payload)
    assert ("turn_ended",) in conn.sent


def test_begin_turn_reseeds_persistent_worker():
    runner = ShadowRunner(
        {"x": 1},
        {name: None for name in ("llm_query",)},
        SpecStore(),
        REAL_BUILTINS,
        persistent=True,
    )
    try:
        runner.begin_turn({"x": 1})  # identical seed: no reseed message
        runner.feed(_seg("llm_query(str(x))"))
        assert runner.end_turn(10)
        first = runner.predicted[-1]

        runner.begin_turn({"x": 2})  # differs: reseed path
        runner.feed(_seg("llm_query(str(x))"))
        assert runner.end_turn(10)
        second = runner.predicted[-1]
        assert first[1] == ("1",) and second[1] == ("2",)
    finally:
        runner.finish()
        runner.join(10)


def test_batch_key_for_missing_single_tool_is_none():
    from dspy_rlm_hooks.speculation.session import ToolRegistry as Reg

    reg = Reg()
    reg.register("llm_query_batched", lambda prompts: prompts)
    runner = ShadowRunner(
        {}, {"llm_query_batched": None}, SpecStore(), REAL_BUILTINS, registry=reg
    )
    assert runner._batch_key_for("llm_query_batched", (["a"],), {}) is None
    runner.finish()
    runner.join(10)


def test_batch_key_for_split_failure_is_none():
    from dspy_rlm_hooks.speculation.session import ToolRegistry as Reg

    reg = Reg()
    reg.register("llm_query", lambda prompt: prompt)
    reg.register("llm_query_batched", lambda prompts: prompts)
    runner = ShadowRunner(
        {},
        {"llm_query_batched": None, "llm_query": None},
        SpecStore(),
        REAL_BUILTINS,
        registry=reg,
    )
    # batched name but no list-shaped argument anywhere: split guard
    assert runner._batch_key_for("llm_query_batched", (), {}) is None
    runner.finish()
    runner.join(10)


def test_decompose_batched_none_split_guard():
    from dspy_rlm_hooks.speculation.session import ToolRegistry as Reg

    reg = Reg()
    reg.register("llm_query", lambda prompt: prompt)
    reg.register("llm_query_batched", lambda prompts: prompts)
    runner = ShadowRunner(
        {},
        {"llm_query_batched": None, "llm_query": None},
        SpecStore(),
        REAL_BUILTINS,
        registry=reg,
        launcher=Launcher(SpecStore(), EventBus()),
    )
    runner._batch_key_for = lambda *a, **k: ("__batch__", "forced")  # type: ignore[method-assign]  # ty: ignore[invalid-assignment]
    # forced batch key but args carry no list: split guard returns None
    assert runner._decompose_batched("llm_query_batched", (), {}, 1, False) is None
    runner.finish()
    runner.join(10)


# -- streaming: chained continuation inside an unrolled loop ------------------------


def test_unrolled_loop_binds_chain_productions():
    # Given a loop whose second call reads the first call's target.
    tail = "for s in sections:\n    r = llm_query(s)\n    out = llm_query(r + '!')\n"
    plans, chains, metas = plan_peeks_with_chains(
        tail, {"llm_query"}, {"sections": ["a", "b"]}
    )
    # Then stage-1 dispatches per item and stage-2 chains off its key.
    assert sorted(p.args for p in plans) == [("a",), ("b",)]
    assert len(chains) == 2
    assert metas
