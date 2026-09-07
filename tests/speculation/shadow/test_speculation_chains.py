"""Chained continuations: calls whose args depend on a speculated
predecessor's result fire automatically when the producer resolves, pipelining
whole dataflow chains under the model's still-streaming output."""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from dspy_rlm_hooks.speculation.session import (
    EventBus,
    Launcher,
    SpecSession,
    ToolRegistry,
)
from dspy_rlm_hooks.speculation.shadow import ShadowRunner
from dspy_rlm_hooks.speculation.store import SpecStore
from dspy_rlm_hooks.speculation.streaming import ChainMeta, Plan

CODE_CHAIN = """```repl
doc = fetch("auth")
summary = llm_query("summarize: " + doc)
```
"""


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


def _feed(turn, code: str) -> None:
    for i in range(0, len(code), 6):
        turn.feed(code[i : i + 6])
        time.sleep(0.001)


def _wait_for(predicate, timeout: float = 10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        out = predicate()
        if out:
            return out
        time.sleep(0.02)
    return predicate()


def test_chain_fires_when_producer_resolves():
    session = SpecSession(_registry())
    try:
        turn = session.begin_stream_turn({}, {}, peek=True)
        _feed(turn, CODE_CHAIN)
        turn.end(timeout=5.0)

        hooks = session.real_hooks()
        # the producer resolves when the real run claims it
        out_doc = hooks["fetch"]("auth")
        assert out_doc == "doc:auth"

        # the chained llm_query("summarize: doc:auth") must have been
        # dispatched automatically; the real run claims it without re-calling
        summary_spec = _wait_for(
            lambda: next(
                (
                    s
                    for s in session.store.all
                    if s.key[0] == "llm_query"
                    and s.args == ("summarize: doc:auth",)
                    and s.state in ("pending", "running", "ready", "claimed")
                ),
                None,
            )
        )
        assert summary_spec is not None, (
            f"chained call never dispatched: {[(s.key[0], s.args) for s in session.store.all]}"
        )
        out_summary = hooks["llm_query"](prompt="summarize: doc:auth")
        assert out_summary == "r:summarize: doc:auth"

        llm_executions = [s for s in session.store.all if s.key[0] == "llm_query"]
        # the chained call dispatched exactly once (no duplicate from re-peeks)
        assert len(llm_executions) == 1
    finally:
        session.close()


def test_three_level_chain():
    reg = _registry()
    reg.register(
        "rank",
        lambda text: f"ranked:{text}",
        speculatable=True,
        pure=True,
        deterministic=True,
        latency_hint_ms=50.0,
    )
    session = SpecSession(reg)
    try:
        code = """```repl
doc = fetch("auth")
summary = llm_query("summarize: " + doc)
final = rank(summary)
```
"""
        turn = session.begin_stream_turn({}, {}, peek=True)
        _feed(turn, code)
        turn.end(timeout=5.0)

        hooks = session.real_hooks()
        _wait_for(lambda: session.store.all)
        assert hooks["fetch"]("auth") == "doc:auth"
        _wait_for(
            lambda: any(
                s.key[0] == "llm_query" and s.args == ("summarize: doc:auth",)
                for s in session.store.all
            )
        )
        assert (
            hooks["llm_query"](prompt="summarize: doc:auth") == "r:summarize: doc:auth"
        )
        _wait_for(
            lambda: any(
                s.key[0] == "rank" and s.args == ("ranked:r:summarize: doc:auth",)
                for s in session.store.all
            ),
        )
        out = hooks["rank"]("r:summarize: doc:auth")
        assert out == "ranked:r:summarize: doc:auth"
    finally:
        session.close()


def test_unrelated_stale_arg_still_skipped():
    """A call whose arg reads a tail-assigned name with NO producer is still
    skipped (the staleness rail is not loosened by chain support)."""
    from dspy_rlm_hooks.speculation.streaming import plan_peeks, plan_peeks_with_chains

    tail = "x = unknown_tool()" + chr(10) + "y = llm_query(x)"
    plans, chains, metas = plan_peeks_with_chains(tail, {"llm_query"}, {})
    # unknown_tool is not a spec name -> no producer -> no plan, no chain
    assert plans == []
    assert chains == [] and metas == []
    assert plan_peeks(tail, {"llm_query"}, {}) == []


def test_dispatch_decomposes_batched_segment_records():
    """A segment-recorded llm_query_batched call dispatches one peek PER
    ELEMENT under the single tool's claim keys (adopted), not one whole-batch
    peek under the batched key."""
    reg = ToolRegistry()
    reg.register(
        "llm_query", lambda prompt: f"r:{prompt}", speculatable=True, pure=True
    )
    reg.register(
        "llm_query_batched",
        lambda prompts: [f"r:{p}" for p in prompts],
        speculatable=True,
        pure=True,
    )
    store = SpecStore()
    runner = ShadowRunner(
        {},
        {},
        store,
        {},
        launcher=Launcher(store, EventBus()),
        registry=reg,
        persistent=True,
    )
    try:
        key = runner._dispatch("llm_query_batched", (["a", "b"],), {})
        assert key is not None and key[0] == "__batch__"
        elem_specs = [s for s in store.all if s.key[0] == "llm_query"]
        assert [s.args for s in elem_specs] == [("a",), ("b",)]
        assert all(s.adopted for s in elem_specs)  # shielded from retraction
        assert not [s for s in store.all if s.key[0] == "llm_query_batched"]
    finally:
        runner.shutdown()


def test_batched_chain_fires_on_assembled_list():
    """A chain consuming a batched call's result fires with the ASSEMBLED
    ordered list once every element speculation resolves."""
    import time as _time

    reg = ToolRegistry()
    reg.register(
        "llm_query_batched",
        lambda prompts: [f"s:{p}" for p in prompts],
        speculatable=True,
        pure=True,
        deterministic=True,
        latency_hint_ms=50.0,
    )
    reg.register(
        "rank",
        lambda items: f"ranked:{items}",
        speculatable=True,
        pure=True,
        deterministic=True,
        latency_hint_ms=50.0,
    )
    session = SpecSession(reg)
    try:
        turn = session.begin_stream_turn({}, {}, peek=True)
        code = (
            "```repl\n"
            'reviews = llm_query_batched(["a", "b"])\n'
            "final = rank(reviews)\n"
            "```\n"
        )
        for i in range(0, len(code), 6):
            turn.feed(code[i : i + 6])
            _time.sleep(0.001)
        turn.end(timeout=5.0)

        hooks = session.real_hooks()
        reviews = hooks["llm_query_batched"](prompts=["a", "b"])
        assert reviews == ["s:a", "s:b"]
        deadline = _time.monotonic() + 5.0
        rank_spec = None
        while _time.monotonic() < deadline:
            rank_spec = next((s for s in session.store.all if s.key[0] == "rank"), None)
            if rank_spec is not None:
                break
            _time.sleep(0.02)
        assert rank_spec is not None, (
            f"chain off the batched result never fired; "
            f"bus={[(k, v.get('tool') or '') for k, v in session.bus.history]}; "
            f"chains={list(session._warm_runner._pending_chains.values()) if session._warm_runner else None}; "
            f"store={[(s.key[0], s.args, s.state) for s in session.store.all]}"
        )
        assert rank_spec.args == (["s:a", "s:b"],)
        assert hooks["rank"](reviews) == "ranked:['s:a', 's:b']"
    finally:
        session.close()


def test_llm_spec_fns_counter_free_and_registered():
    """The llm tools get COUNTER-FREE speculative executors: they call the
    sub-LM directly (speculative runs must not consume the model's logical
    max_llm_calls budget) and reuse dspy's output extraction."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, MagicMock

    import dspy_rlm_hooks.speculation.integration as SI

    calls: list = []

    class FakeSubLM:
        def __call__(self, prompt):
            calls.append(prompt)
            return [{"text": f"T:{prompt}"}]

    rlm = SimpleNamespace(sub_lm=FakeSubLM())
    fns = SI._make_llm_spec_fns(rlm)
    assert fns["llm_query"]("hello") == "T:hello"
    assert fns["llm_query_batched"](["a", "b"]) == ["T:a", "T:b"]
    assert len(calls) == 3

    # registered through enable_rlm_speculation with the tuple policy form

    rlm2 = MagicMock()
    rlm2._execute_code = MagicMock(return_value="ok")
    rlm2._execute_iteration = MagicMock(return_value=MagicMock())
    rlm2._aexecute_iteration = MagicMock(return_value=MagicMock())
    rlm2._process_execution_result = MagicMock(return_value=MagicMock())
    rlm2.generate_action = MagicMock()
    rlm2.generate_action.acall = AsyncMock(return_value=MagicMock())
    rlm2.max_llm_calls = 50
    rlm2.sub_lm = FakeSubLM()
    SI.enable_rlm_speculation(
        rlm2, speculate_llm_query=True, speculate_llm_query_batched=True
    )
    spec = rlm2._speculator
    assert spec.registry.get("llm_query").spec_fn is not None
    assert spec.registry.get("llm_query_batched").spec_fn is not None
    # the spec fn runs the sub-LM directly (counter-free path)
    out = spec.registry.get("llm_query").spec_fn(prompt="x")
    assert out == "T:x"
    SI.disable_rlm_speculation(rlm2)


def test_extract_sub_lm_text_shapes():
    from dspy_rlm_hooks.speculation.integration import _extract_sub_lm_text

    assert _extract_sub_lm_text([{"text": "a"}]) == "a"
    assert _extract_sub_lm_text(["b"]) == "b"
    # best-effort fallback for non-conforming responses (the REAL call
    # enforces the strict contract; speculative results are predictions)
    assert _extract_sub_lm_text([]) == "[]"
    assert _extract_sub_lm_text([{"text": 5}]) == "5"
    assert _extract_sub_lm_text("raw") == "raw"


def test_batched_peek_decomposition_and_chain_via_handle_plans():
    """A batched call in an OPEN tail (peek path) decomposes into element
    peeks, and a chained call on the batched result fires with the ASSEMBLED
    list once all elements resolve."""
    import time as _time

    reg = ToolRegistry()
    reg.register(
        "llm_query_batched",
        lambda prompts: [f"s:{p}" for p in prompts],
        speculatable=True,
        pure=True,
        latency_hint_ms=50.0,
    )
    reg.register(
        "rank",
        lambda items: f"ranked:{items}",
        speculatable=True,
        pure=True,
        latency_hint_ms=50.0,
    )
    session = SpecSession(reg)
    try:
        turn = session.begin_stream_turn({}, {}, peek=True)
        # the for-loop keeps the batched call in the OPEN tail (peek path)
        code = (
            "```repl\n"
            "out = {}\n"
            'for kind in ["x"]:\n'
            '    reviews = llm_query_batched(["a", "b"])\n'
            "    out[kind] = rank(reviews)\n"
            "print(out)\n"
            "```\n"
        )
        for i in range(0, len(code), 6):
            turn.feed(code[i : i + 6])
            _time.sleep(0.001)
        turn.end(timeout=5.0)

        # the rank chain fired with the ASSEMBLED list once both elements
        # resolved (the last peek generation clears the tally, so assert the
        # OUTCOME, not the tally)
        deadline = _time.monotonic() + 5.0
        rank_specs = []
        while _time.monotonic() < deadline:
            rank_specs = [s for s in session.store.all if s.key[0] == "rank"]
            if rank_specs:
                break
            _time.sleep(0.02)
        assert rank_specs, "chained rank call never dispatched"
        assert rank_specs[0].args == (["s:a", "s:b"],)
    finally:
        session.close()


def test_decompose_batched_guards():
    """Launcher/registry-less runners and non-batched calls return None."""
    reg = ToolRegistry()
    reg.register("llm_query", lambda prompt: "x", speculatable=True, pure=True)
    store = SpecStore()
    runner = ShadowRunner({}, {}, store, {}, launcher=None, registry=reg)
    try:
        # launcher None -> None
        assert (
            runner._decompose_batched("llm_query_batched", (["a"],), {}, 1, True)
            is None
        )
        # non-batched name -> None
        runner2 = ShadowRunner(
            {},
            {},
            SpecStore(),
            {},
            launcher=Launcher(SpecStore(), EventBus()),
            registry=reg,
        )
        try:
            assert runner2._batch_key_for("llm_query", (["a"],), {}) is None
            assert (
                runner2._decompose_batched("llm_query", (["a"],), {}, 1, True) is None
            )
            # batched name but no single-tool registry entry -> None
            reg3 = ToolRegistry()
            reg3.register("rank", lambda items: "r", speculatable=True, pure=True)
            runner3 = ShadowRunner(
                {},
                {},
                SpecStore(),
                {},
                launcher=Launcher(SpecStore(), EventBus()),
                registry=reg3,
            )
            try:
                assert runner3._batch_key_for("llm_query_batched", (["a"],), {}) is None
            finally:
                runner3.shutdown()
        finally:
            runner2.shutdown()
    finally:
        pass


def test_record_batch_element_failure_never_assembles():
    """A failed element poisons the batch: the assembled result is never
    published to dependent chains."""
    from types import SimpleNamespace

    reg = ToolRegistry()
    reg.register("llm_query", lambda prompt: "x", speculatable=True, pure=True)
    store = SpecStore()
    runner = ShadowRunner(
        {}, {}, store, {}, launcher=Launcher(store, EventBus()), registry=reg
    )
    try:
        batch_key = ("__batch__", "llm_query_batched|abc")
        runner._batch_keys[batch_key] = [("llm_query", "k1"), ("llm_query", "k2")]
        runner._batch_values[batch_key] = {}
        runner._key_to_batch[("llm_query", "k1")] = (batch_key, 0)
        runner._key_to_batch[("llm_query", "k2")] = (batch_key, 1)
        # element 0 succeeds, element 1 FAILS -> batch never assembles
        runner._fire_chains_for_key(
            ("llm_query", "k1"), SimpleNamespace(_result="v0", error=None)
        )
        runner._fire_chains_for_key(
            ("llm_query", "k2"),
            SimpleNamespace(_result=None, error=RuntimeError("boom")),
        )
        assert batch_key not in runner._batch_keys  # batch dropped
    finally:
        runner.shutdown()


def test_llm_spec_fns_error_paths(monkeypatch):
    """Empty prompts, empty batch, and the no-LM error path."""
    import dspy

    from dspy_rlm_hooks.speculation.integration import _make_llm_spec_fns

    rlm = SimpleNamespace(sub_lm=None)
    fns = _make_llm_spec_fns(rlm)
    with pytest.raises(ValueError):
        fns["llm_query"]("")
    assert fns["llm_query_batched"]([]) == []

    import dspy_rlm_hooks.speculation.integration as SI

    monkeypatch.setattr(dspy.settings, "lm", None, raising=False)
    with pytest.raises(Exception):
        fns["llm_query"]("x")

    # LMResponse-shaped response extraction
    class FakeLMResponse:
        def __init__(self, text):
            self.text = text

    monkeypatch.setattr(dspy, "LMResponse", FakeLMResponse, raising=False)
    assert SI._extract_sub_lm_text(FakeLMResponse("resp")) == "resp"


def test_handle_plans_dedup_and_batched_decomposition():
    """Duplicate plans dedupe (needed=2), and a batched plan decomposes into
    per-element tally entries under the single tool's claim keys."""
    reg = ToolRegistry()
    reg.register("llm_query", lambda prompt: "x", speculatable=True, pure=True)
    reg.register(
        "llm_query_batched",
        lambda prompts: ["x"],
        speculatable=True,
        pure=True,
    )
    store = SpecStore()
    runner = ShadowRunner(
        {}, {}, store, {}, launcher=Launcher(store, EventBus()), registry=reg
    )
    try:
        plan = Plan(tool="llm_query_batched", args=(["a", "b"],), kwargs={})
        runner._handle_plans([plan, plan], [])
        # both elements dispatched exactly once per needed slot
        elem = [s for s in store.all if s.key[0] == "llm_query"]
        assert [(s.args) for s in elem] == [("a",), ("a",), ("b",), ("b",)]
        # the tally tracks ELEMENT keys
        assert any(k[0] == "llm_query" for k in runner._last_peek_tally)
    finally:
        runner.shutdown()


def test_plan_key_to_real_batched_plan():
    reg = ToolRegistry()
    reg.register("llm_query", lambda prompt: "x", speculatable=True, pure=True)
    reg.register(
        "llm_query_batched",
        lambda prompts: ["x"],
        speculatable=True,
        pure=True,
    )
    store = SpecStore()
    runner = ShadowRunner(
        {}, {}, store, {}, launcher=Launcher(store, EventBus()), registry=reg
    )
    try:
        from dspy_rlm_hooks.speculation.tool import canonical_hash

        plan = Plan(
            tool="llm_query_batched",
            args=(["a"],),
            kwargs={},
            key=(
                "llm_query_batched",
                canonical_hash("llm_query_batched", (["a"],), {}),
            ),
        )
        meta = ChainMeta(cont_id=1, tool="rank", deps={"reviews": ("key", plan.key)})
        runner._handle_plans([plan], [meta])
        # the chain dep was translated to the BATCH key, not a raw plan key
        (meta,) = runner._pending_chains.values()
        assert meta.deps["reviews"][0] == "key"
        assert meta.deps["reviews"][1][0] == "__batch__"
    finally:
        runner.shutdown()


def test_decompose_batched_no_launcher():
    reg = ToolRegistry()
    reg.register("llm_query", lambda prompt: "x", speculatable=True, pure=True)
    reg.register("llm_query_batched", lambda prompts: "x", speculatable=True, pure=True)
    store = SpecStore()
    runner = ShadowRunner({}, {}, store, {}, launcher=None, registry=reg)
    try:
        assert (
            runner._decompose_batched("llm_query_batched", (["a"],), {}, 1, True)
            is None
        )
    finally:
        runner.shutdown()


def test_unroll_binds_productions_for_loop_body_chains():
    """A loop-body producer binds its target so a LATER loop-body call chains
    off it (productions binding inside _unroll_for)."""
    from dspy_rlm_hooks.speculation.streaming import plan_peeks_with_chains

    tail = (
        'for kind in ["x"]:\n'
        '    reviews = llm_query_batched(["a", "b"])\n'
        "    out[kind] = rank(reviews)\n"
    )
    plans, chains, metas = plan_peeks_with_chains(
        tail, {"llm_query_batched", "rank"}, {}
    )
    # the batched call is planned; rank chains off the batched result
    assert any(p.tool == "llm_query_batched" for p in plans)
    assert any(cp.tool == "rank" for cp in chains)
    assert any(m.deps.get("reviews", ("",))[0] in ("key", "cont") for m in metas)


def test_unroll_binds_cont_productions_for_chained_loop_bodies():
    """A loop-body call chained on a SEGMENT production, followed by another
    loop-body call reading the first's target: the cont binding fires."""
    from dspy_rlm_hooks.speculation.streaming import plan_peeks_with_chains

    tail = 'reviews = llm_query("s: " + doc)\nout = rank(reviews)\n'
    seg_productions = {"doc": ("segkey", ("llm_query", "h"))}
    plans, chains, metas = plan_peeks_with_chains(
        tail, {"llm_query", "rank"}, {}, segment_productions=seg_productions
    )
    # llm_query chains on the seg-produced doc; rank chains on the llm cont
    assert not plans
    assert len(chains) == 2
    by_tool = {c.tool: c for c in chains}
    assert by_tool["llm_query"].deps["doc"][0] == "segkey"
    assert by_tool["rank"].deps["reviews"] == ("cont", by_tool["llm_query"].cont_id)
    assert any(m.cont_id == by_tool["rank"].cont_id for m in metas)
