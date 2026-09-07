"""Coverage-completion tests for speculation_integration.py.

Targets the error/fallback branches and streaming-wrapper paths not exercised by
the existing test_speculation_integration.py, so the module reaches 100% line
coverage. All tests use mocks / fake RLMs — never a real LLM.
"""

from __future__ import annotations

import gc
import inspect
from unittest.mock import MagicMock

import pytest

from dspy_rlm_hooks.speculation.integration import (
    _close_speculators,
    _install_claim_hooks,
    _maybe_begin_streaming_turn,
    _register_speculator,
    _sync_registry_fns,
    disable_rlm_speculation,
    enable_rlm_speculation,
)


def _real_execute_code(repl, code, input_args):
    """A real _execute_code: run the code through the repl."""
    return repl.execute(code, variables=dict(input_args))


def _make_execute(tools):
    """A repl.execute that actually runs the code with the tools in scope."""

    def execute(code, variables=None):
        ns = dict(tools)
        ns.update(variables or {})
        exec(code, ns, ns)
        return "ok"

    return execute


def _setup_real(mock_rlm, mock_repl, tools):
    """Wire a mock RLM/repl so real execution actually runs code + claim hooks."""
    mock_rlm.max_llm_calls = 50
    mock_repl.tools = tools
    mock_repl.execute = _make_execute(tools)
    mock_rlm._execute_code = _real_execute_code
    return mock_rlm, mock_repl


# ---------------------------------------------------------------------------
# _close_speculators (lines 66-73)
# ---------------------------------------------------------------------------


def test_close_speculators_handles_live_dead_and_erroring_specs():
    """_close_speculators drains live specs, swallows close() errors, skips
    GC'd (dead) refs, and clears the registry."""
    from dspy_rlm_hooks.speculation.integration import _active_speculators

    saved = list(_active_speculators)
    _active_speculators.clear()
    try:
        # live spec whose close() raises -> except branch (lines 70-72)
        boom = MagicMock()
        boom.close.side_effect = RuntimeError("close boom")
        _register_speculator(boom)
        # live spec that closes cleanly (line 70 happy path)
        ok = MagicMock()
        _register_speculator(ok)
        # spec that gets GC'd -> ref() returns None (line 68 false branch)
        dead = MagicMock()
        _register_speculator(dead)
        del dead
        gc.collect()

        _close_speculators()

        assert _active_speculators == []
        boom.close.assert_called_once()
        ok.close.assert_called_once()
    finally:
        _active_speculators.clear()
        _active_speculators.extend(saved)


# ---------------------------------------------------------------------------
# _sync_registry_fns: claim hook with no raw fn (lines 171, 173)
# ---------------------------------------------------------------------------


def test_sync_registry_fns_claim_hook_without_raw_falls_back_to_empty(
    mock_rlm, mock_repl
):
    """A claim hook with no recorded raw fn and no cached raw fn leaves the
    registry entry untouched (lines 171, 173)."""
    from dspy_rlm_hooks.speculation.guards import tag_claim_hook

    def lookup_price(x):
        return x * 2

    tools = {"llm_query": lambda p: p, "lookup_price": lookup_price}
    _setup_real(mock_rlm, mock_repl, tools)
    enable_rlm_speculation(
        mock_rlm, tools={"lookup_price": lookup_price}, speculate_user_tools=True
    )
    spec = mock_rlm._speculator

    # Replace the tool with a claim hook that carries NO raw_fn tag.
    mock_repl.tools["lookup_price"] = tag_claim_hook(lambda x: x)
    # spec._raw_fns is empty -> candidate stays None -> continue (no crash).
    _sync_registry_fns(spec, mock_repl)

    # The registry entry was left untouched (still the placeholder from enable).
    assert spec.registry.get("lookup_price") is not None


# ---------------------------------------------------------------------------
# _install_claim_hooks: user tool with __signature__ (line 269)
# ---------------------------------------------------------------------------


def test_install_claim_hooks_user_tool_with_signature(mock_rlm, mock_repl):
    """A speculatable user tool whose raw fn exposes __signature__ gets that
    signature copied onto the claim hook (line 269)."""
    from dspy_rlm_hooks.speculation.guards import is_claim_hook

    def lookup_price(x):
        return x * 2

    setattr(lookup_price, "__signature__", inspect.signature(lookup_price))

    tools = {"llm_query": lambda p: p, "lookup_price": lookup_price}
    mock_rlm.max_llm_calls = 50
    mock_repl.tools = tools
    mock_repl.execute = _make_execute(tools)
    mock_rlm._execute_code = _real_execute_code

    enable_rlm_speculation(
        mock_rlm, tools={"lookup_price": lookup_price}, speculate_user_tools=True
    )
    spec = mock_rlm._speculator
    config = mock_rlm._speculation_config

    _install_claim_hooks(mock_repl, spec, config, mock_rlm)

    wrapped = mock_repl.tools["lookup_price"]
    assert is_claim_hook(wrapped)
    assert getattr(wrapped, "__signature__", None) is not None


# ---------------------------------------------------------------------------
# _maybe_begin_streaming_turn (lines 292, 300, 303-304)
# ---------------------------------------------------------------------------


def test_maybe_begin_streaming_turn_disabled(mock_rlm, mock_repl):
    """config.enabled=False -> early return, no turn begun (line 290)."""
    mock_rlm.max_llm_calls = 50
    mock_repl.tools = {"llm_query": lambda p: p}
    enable_rlm_speculation(mock_rlm, streaming=True)
    mock_rlm._speculation_config.enabled = False
    _maybe_begin_streaming_turn(mock_rlm, mock_repl, {})
    assert mock_rlm._active_stream_turn is None


def test_maybe_begin_streaming_turn_no_speculatable(mock_rlm, mock_repl):
    """No speculatable tools -> early return, no turn begun (line 292)."""
    mock_rlm.max_llm_calls = 50
    mock_repl.tools = {"llm_query": lambda p: p}
    enable_rlm_speculation(
        mock_rlm,
        streaming=True,
        speculate_llm_query=False,
        speculate_llm_query_batched=False,
    )
    _maybe_begin_streaming_turn(mock_rlm, mock_repl, {})
    assert mock_rlm._active_stream_turn is None


def test_maybe_begin_streaming_turn_feeds_prelude(mock_rlm, mock_repl):
    """A non-empty repl_globals prelude is fed into the streaming turn (line 300)."""
    mock_rlm.max_llm_calls = 50
    mock_repl.tools = {"llm_query": lambda p: p}
    mock_repl.repl_globals = "print('hi')"
    enable_rlm_speculation(mock_rlm, streaming=True)
    _maybe_begin_streaming_turn(mock_rlm, mock_repl, {})
    turn = mock_rlm._active_stream_turn
    assert turn is not None
    assert mock_rlm._streaming_fed_any is False
    # Clean up the shadow subprocess.
    turn.end(timeout=5)


def test_maybe_begin_streaming_turn_begin_raises(mock_rlm, mock_repl, monkeypatch):
    """begin_stream_turn failure is swallowed; no turn is stashed (lines 303-304)."""
    mock_rlm.max_llm_calls = 50
    mock_repl.tools = {"llm_query": lambda p: p}
    enable_rlm_speculation(mock_rlm, streaming=True)
    spec = mock_rlm._speculator

    def boom(*a, **k):
        raise RuntimeError("begin boom")

    monkeypatch.setattr(spec.session, "begin_stream_turn", boom)
    _maybe_begin_streaming_turn(mock_rlm, mock_repl, {})
    assert mock_rlm._active_stream_turn is None


# ---------------------------------------------------------------------------
# _StreamingGenerateAction (lines 329, 365, 370, 381-383, 388, 391-392, 399-401)
# ---------------------------------------------------------------------------


def _streaming_wrapper(mock_rlm):
    enable_rlm_speculation(mock_rlm, streaming=True)
    return mock_rlm._speculation_generate_action_wrapper


def test_streaming_generate_ensure_cached(mock_rlm):
    """_ensure returns the cached (sync, async) pair on repeat calls (line 329)."""
    wrapper = _streaming_wrapper(mock_rlm)
    first = wrapper._ensure()
    second = wrapper._ensure()
    assert second is not None
    # The cached sync/async callables are reused (line 329).
    assert second[0] is first[0]
    assert second[1] is first[1]


def test_streaming_generate_feed_item_non_stream(mock_rlm):
    """_feed_item returns False for a non-StreamResponse item (line 365)."""
    wrapper = _streaming_wrapper(mock_rlm)
    assert wrapper._feed_item("not a stream response") is False


def test_streaming_generate_feed_item_stream_response(mock_rlm):
    """_feed_item feeds a code delta into the active turn (lines 359-363)."""
    from dspy.streaming import StreamResponse

    wrapper = _streaming_wrapper(mock_rlm)
    turn = MagicMock()
    mock_rlm._active_stream_turn = turn

    item = StreamResponse(
        predict_name="p",
        signature_field_name="code",
        chunk="delta",
        is_last_chunk=False,
    )
    assert wrapper._feed_item(item) is True
    turn.feed.assert_called_once_with("delta")
    assert mock_rlm._streaming_fed_any is True

    # A non-code field still counts as a streamed item (line 364).
    item2 = StreamResponse(
        predict_name="p",
        signature_field_name="reasoning",
        chunk="x",
        is_last_chunk=False,
    )
    assert wrapper._feed_item(item2) is True


def test_streaming_generate_call_no_turn(mock_rlm):
    """__call__ with no active turn delegates straight to the original (line 370)."""
    wrapper = _streaming_wrapper(mock_rlm)
    mock_rlm._active_stream_turn = None
    result = wrapper("x")
    assert result is mock_rlm._speculation_original_generate_action.return_value
    mock_rlm._speculation_original_generate_action.assert_called_once_with("x")


def test_streaming_generate_call_sync_stream_raises(mock_rlm, monkeypatch):
    """A sync stream that raises clears the turn and falls back to the original
    predict (lines 381-383)."""
    wrapper = _streaming_wrapper(mock_rlm)
    mock_rlm._active_stream_turn = MagicMock()

    def fake_streamify(*a, **k):
        def boom(*a, **k):
            raise RuntimeError("stream boom")

        return boom

    monkeypatch.setattr("dspy.streaming.streamify", fake_streamify)
    result = wrapper("x")
    assert result is mock_rlm._speculation_original_generate_action.return_value
    assert mock_rlm._active_stream_turn is None


def test_streaming_generate_call_streams_unavailable(mock_rlm, monkeypatch):
    """__call__ when streaming is unavailable clears the turn and falls back
    (lines 373-374)."""
    wrapper = _streaming_wrapper(mock_rlm)
    mock_rlm._active_stream_turn = MagicMock()

    def raising_streamify(*a, **k):
        raise RuntimeError("no stream")

    monkeypatch.setattr("dspy.streaming.streamify", raising_streamify)
    result = wrapper("x")
    assert result is mock_rlm._speculation_original_generate_action.return_value
    assert mock_rlm._active_stream_turn is None


def test_streaming_generate_call_prediction_item(mock_rlm, monkeypatch):
    """__call__ returns a Prediction yielded by the sync stream (lines 378-380)."""
    from dspy.primitives.prediction import Prediction

    wrapper = _streaming_wrapper(mock_rlm)
    mock_rlm._active_stream_turn = MagicMock()
    pred = Prediction()

    def fake_streamify(*a, **k):
        def gen(*a, **k):
            yield "not a prediction"
            yield pred

        return gen

    monkeypatch.setattr("dspy.streaming.streamify", fake_streamify)
    result = wrapper("x")
    assert result is pred


@pytest.mark.asyncio
async def test_streaming_generate_acall_no_turn(mock_rlm):
    """acall with no active turn delegates to the original acall (line 388)."""
    wrapper = _streaming_wrapper(mock_rlm)
    mock_rlm._active_stream_turn = None
    await wrapper.acall("x")
    mock_rlm._speculation_original_generate_action.acall.assert_awaited_once_with("x")


@pytest.mark.asyncio
async def test_streaming_generate_acall_streams_unavailable(mock_rlm, monkeypatch):
    """acall when streaming is unavailable clears the turn and falls back (lines 391-392)."""
    wrapper = _streaming_wrapper(mock_rlm)
    mock_rlm._active_stream_turn = MagicMock()

    def raising_streamify(*a, **k):
        raise RuntimeError("no stream")

    monkeypatch.setattr("dspy.streaming.streamify", raising_streamify)
    await wrapper.acall("x")
    assert mock_rlm._active_stream_turn is None
    mock_rlm._speculation_original_generate_action.acall.assert_awaited_once_with("x")


@pytest.mark.asyncio
async def test_streaming_generate_acall_async_stream_raises(mock_rlm, monkeypatch):
    """An async stream that raises mid-turn clears the turn and falls back (lines 399-401)."""
    wrapper = _streaming_wrapper(mock_rlm)
    mock_rlm._active_stream_turn = MagicMock()

    def fake_streamify(*a, **k):
        def boom(*a, **k):
            raise RuntimeError("stream boom")

        return boom

    monkeypatch.setattr("dspy.streaming.streamify", fake_streamify)
    await wrapper.acall("x")
    assert mock_rlm._active_stream_turn is None
    mock_rlm._speculation_original_generate_action.acall.assert_awaited_once_with("x")


@pytest.mark.asyncio
async def test_streaming_generate_acall_prediction_item(mock_rlm, monkeypatch):
    """acall returns a Prediction yielded by the async stream (lines 396-398)."""
    from dspy.primitives.prediction import Prediction

    wrapper = _streaming_wrapper(mock_rlm)
    mock_rlm._active_stream_turn = MagicMock()
    pred = Prediction()

    def fake_streamify(*a, **k):
        async def agen(*a, **k):
            yield "not a prediction"
            yield pred

        return agen

    monkeypatch.setattr("dspy.streaming.streamify", fake_streamify)
    result = await wrapper.acall("x")
    assert result is pred


# ---------------------------------------------------------------------------
# _speculation_execute_iteration / _speculation_aexecute_iteration (414-416, 429-431)
# ---------------------------------------------------------------------------


def test_speculation_execute_iteration(
    mock_rlm, mock_repl, mock_variables, mock_history
):
    """The sync iteration wrapper begins the streaming turn then delegates to the
    original iteration (lines 414-416)."""
    mock_rlm.max_llm_calls = 50
    mock_repl.tools = {"llm_query": lambda p: p}
    enable_rlm_speculation(mock_rlm, streaming=True)
    mock_rlm._speculation_original_execute_iteration.return_value = "iter_done"

    result = mock_rlm._speculation_execute_iteration_wrapper(
        mock_repl, mock_variables, mock_history, 0, {"q": "x"}, ["answer"]
    )
    assert result == "iter_done"
    mock_rlm._speculation_original_execute_iteration.assert_called_once()
    # Clean up the streaming turn's shadow subprocess.
    if mock_rlm._active_stream_turn is not None:
        mock_rlm._active_stream_turn.end(timeout=5)


@pytest.mark.asyncio
async def test_speculation_aexecute_iteration(
    mock_rlm, mock_repl, mock_variables, mock_history
):
    """The async iteration wrapper begins the streaming turn then delegates to the
    original async iteration (lines 429-431)."""
    from unittest.mock import AsyncMock

    mock_rlm.max_llm_calls = 50
    mock_repl.tools = {"llm_query": lambda p: p}
    enable_rlm_speculation(mock_rlm, streaming=True)
    mock_rlm._speculation_original_aexecute_iteration = AsyncMock(return_value="done")

    result = await mock_rlm._speculation_aexecute_iteration_wrapper(
        mock_repl, mock_variables, mock_history, 0, {"q": "x"}, ["answer"]
    )
    assert result == "done"
    mock_rlm._speculation_original_aexecute_iteration.assert_awaited_once()
    if mock_rlm._active_stream_turn is not None:
        mock_rlm._active_stream_turn.end(timeout=5)


# ---------------------------------------------------------------------------
# _speculation_execute_code streaming branch (lines 485-486, 493-494)
# ---------------------------------------------------------------------------


def test_execute_code_streaming_turn_feed_and_end_errors(mock_rlm, mock_repl):
    """A streaming turn whose feed() and end() both raise is swallowed; real
    execution still proceeds (lines 485-486, 493-494)."""
    tools = {"llm_query": lambda p: f"r:{p}"}
    _setup_real(mock_rlm, mock_repl, tools)
    enable_rlm_speculation(mock_rlm, streaming=True)

    turn = MagicMock()
    turn.feed.side_effect = RuntimeError("feed boom")
    turn.end.side_effect = RuntimeError("end boom")
    mock_rlm._active_stream_turn = turn
    mock_rlm._streaming_fed_any = False

    result = mock_rlm._execute_code(mock_repl, "x = llm_query('hi')\n", {})
    assert result == "ok"
    assert mock_rlm._active_stream_turn is None


# ---------------------------------------------------------------------------
# disable_rlm_speculation: close() error path (line 660-661)
# ---------------------------------------------------------------------------


def test_disable_close_exception_swallowed(mock_rlm, mock_repl, monkeypatch):
    """A speculator.close() failure is swallowed during disable."""
    mock_rlm.max_llm_calls = 50
    mock_repl.tools = {"llm_query": lambda p: p}

    enable_rlm_speculation(mock_rlm)
    spec = mock_rlm._speculator

    def boom():
        raise RuntimeError("close boom")

    monkeypatch.setattr(spec, "close", boom)
    disable_rlm_speculation(mock_rlm)
    assert not hasattr(mock_rlm, "_speculator")


# ---------------------------------------------------------------------------
# _install_claim_hooks: signature hash error (lines 372-373)
# ---------------------------------------------------------------------------


def test_install_claim_hooks_sig_hash_error_keeps_registration(mock_rlm, mock_repl):
    """A tool whose signature cannot be stringified leaves tool registration
    untouched (sig_hash falls back to None)."""

    class _EvilSig:
        def __str__(self):
            raise RuntimeError("sig boom")

    def lookup_price(x):
        return x * 2

    setattr(lookup_price, "__signature__", _EvilSig())

    tools = {"llm_query": lambda p: p, "lookup_price": lookup_price}
    mock_rlm.max_llm_calls = 50
    mock_repl.tools = tools
    mock_repl._tools_registered = True
    mock_rlm._execute_code = _real_execute_code
    enable_rlm_speculation(
        mock_rlm, tools={"lookup_price": lookup_price}, speculate_user_tools=True
    )
    spec = mock_rlm._speculator
    config = mock_rlm._speculation_config

    _install_claim_hooks(mock_repl, spec, config, mock_rlm)

    assert mock_repl._tools_registered is True
    # MagicMock auto-creates attributes, so check __dict__ explicitly.
    assert "_spec_last_tool_sig_hash" not in mock_rlm.__dict__


# ---------------------------------------------------------------------------
# _snapshot_reads / _pure_assigned_names (lines 429-430, 457, 459-460)
# ---------------------------------------------------------------------------


def test_snapshot_reads_import_bound_names_are_not_required():
    """A name bound by an import statement is not a snapshot-required read."""
    import ast

    from dspy_rlm_hooks.speculation.integration import _snapshot_reads

    tree = ast.parse("import time\nstamp = time.perf_counter()\n")
    assert _snapshot_reads(tree) == set()


def test_pure_assigned_names_covers_assign_and_imports():
    """Assign targets not read in their own value and import aliases are pure."""
    import ast

    from dspy_rlm_hooks.speculation.integration import _pure_assigned_names

    tree = ast.parse("a = 1\nb = a + 1\nimport os\nfrom json import dumps as jd\n")
    assert _pure_assigned_names(tree) == {"a", "b", "os", "jd"}


# ---------------------------------------------------------------------------
# _live_state_seed early returns (lines 489, 504-506, 517, 528)
# ---------------------------------------------------------------------------


def _seed_setup(mock_rlm, mock_repl):
    mock_rlm.max_llm_calls = 50
    mock_repl.tools = {"llm_query": lambda p: p}
    mock_rlm._execute_code = _real_execute_code
    enable_rlm_speculation(mock_rlm, streaming=False)
    return mock_rlm._speculator


def test_live_state_seed_skips_pure_assigned_reads(mock_rlm, mock_repl):
    """A name read before it is purely assigned later needs no snapshot."""
    from dspy_rlm_hooks.speculation.integration import _live_state_seed

    spec = _seed_setup(mock_rlm, mock_repl)
    mock_repl.execute = MagicMock(return_value="None")

    seed = _live_state_seed(mock_repl, "z = w\nw = 1\n", {}, spec)

    assert seed == {}
    assert mock_repl.execute.call_count == 0


def test_live_state_seed_skips_loop_targets(mock_rlm, mock_repl):
    """Loop targets over a non-empty literal need no snapshot."""
    from dspy_rlm_hooks.speculation.integration import _live_state_seed

    spec = _seed_setup(mock_rlm, mock_repl)
    mock_repl.execute = MagicMock(return_value="None")

    code = "for i in [1, 2]:\n    pass\ny = llm_query(i)\n"
    seed = _live_state_seed(mock_repl, code, {}, spec)

    assert seed == {}
    assert mock_repl.execute.call_count == 0


def test_live_state_seed_non_dict_probe_result(mock_rlm, mock_repl):
    """A probe whose output does not literal_eval to a dict is discarded."""
    from dspy_rlm_hooks.speculation.integration import _live_state_seed

    spec = _seed_setup(mock_rlm, mock_repl)
    mock_repl.execute = MagicMock(return_value="None")

    seed = _live_state_seed(mock_repl, "y = llm_query(q)\n", {}, spec)

    assert seed == {}
    assert mock_repl.execute.call_count == 1


# ---------------------------------------------------------------------------
# _speculation_execute_code: streaming top-up feeds the live snapshot (783-790)
# ---------------------------------------------------------------------------


def test_execute_code_streaming_top_up_feeds_live_snapshot(mock_rlm, mock_repl):
    """A streaming turn with no streamed deltas on a non-first execution feeds
    the live-state snapshot assigns, then the assembled block, into the turn."""
    import builtins as _builtins

    from dspy_rlm_hooks.speculation.shadow import shadow_builtins

    tools = {"llm_query": lambda p: f"r:{p}"}
    _setup_real(mock_rlm, mock_repl, tools)
    enable_rlm_speculation(mock_rlm, streaming=True)
    spec = mock_rlm._speculator

    mock_repl.execute = MagicMock(return_value="{'val': \"'hello'\"}")

    turn = spec.session.begin_stream_turn({}, shadow_builtins(dict(_builtins.__dict__)))
    mock_rlm._active_stream_turn = turn
    mock_rlm._streaming_fed_any = False
    mock_rlm._spec_exec_count = 1
    mock_rlm._spec_last_repl = mock_repl
    mock_rlm._spec_synced_this_iter = True

    try:
        result = mock_rlm._execute_code(mock_repl, "x = llm_query(val)\n", {})
        assert result == "{'val': \"'hello'\"}"
        assert mock_rlm._active_stream_turn is None
        # the snapshot assign fed the shadow: llm_query(val) was dispatched
        assert any(
            s.key[0] == "llm_query" for s in mock_rlm._speculator.session.store.all
        )
    finally:
        turn.end(timeout=5)
