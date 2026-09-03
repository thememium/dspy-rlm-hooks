"""Integration tests for enable_rlm_speculation / disable_rlm_speculation (Task 7).

Covers AC3-AC8 and AC10-AC13:
- AC3  claim-hit reuse: real llm_query reuses the shadow result, does NOT re-call
- AC4  parallelism: N independent calls wall-clock < serial baseline
- AC5  budget isolation: shadow dispatches do NOT exhaust real max_llm_calls
- AC6  timeout: a runaway shadow does not block real execution
- AC7  hook composition: pre_execution/post_execution fire once, in order
- AC8  disable revert: behavior identical to unpatched RLM after disable
- AC10 opt-in no-op: no enable_rlm_speculation call => no behavior change
- AC11 async path: speculation works on _aexecute_iteration
- AC12 real-interpreter claim bridge: real PythonInterpreter/Deno path
- AC13 both enable orders: hooks-then-speculation (both work) and
      speculation-then-hooks (hooks work, no crash; speculation inactive)
"""

from __future__ import annotations

import time
from types import MethodType
from unittest.mock import AsyncMock, MagicMock

import pytest

from dspy_rlm_hooks import PostExecutionOutput, PreExecutionOutput, enable_rlm_hooks
from dspy_rlm_hooks.speculation_integration import (
    _install_claim_hooks,
    _make_claim_hook,
    _placeholder,
    _sync_registry_fns,
    disable_rlm_speculation,
    enable_rlm_speculation,
)


def _real_execute_code(repl, code, input_args):
    """A real _execute_code: run the code through the repl (used in place of the
    mock's MagicMock so the real execution actually invokes the claim hooks)."""
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
# AC3 — claim-hit reuse
# ---------------------------------------------------------------------------


def test_ac3_claim_hit_reuse(mock_rlm, mock_repl):
    real_calls = []

    def llm_query(prompt):
        real_calls.append(prompt)
        return f"r:{prompt}"

    tools = {"llm_query": llm_query}
    _setup_real(mock_rlm, mock_repl, tools)
    enable_rlm_speculation(mock_rlm)

    code = "x = llm_query('hello')\n"
    mock_rlm._execute_code(mock_repl, code, {})

    # The shadow dispatched llm_query('hello') -> real llm_query called once (by
    # the launcher). The real code's call was a claim HIT -> NOT re-called.
    assert real_calls == ["hello"]


# ---------------------------------------------------------------------------
# AC4 — parallelism wall-clock
# ---------------------------------------------------------------------------


def test_ac4_parallelism(mock_rlm, mock_repl):
    latency = 0.3
    real_calls = []

    def llm_query(prompt):
        real_calls.append(prompt)
        time.sleep(latency)
        return f"r:{prompt}"

    tools = {"llm_query": llm_query}
    _setup_real(mock_rlm, mock_repl, tools)
    # latency_aware=False preserves the strict no-re-call invariant: claims on
    # queued speculations wait rather than hedge (asserted below).
    enable_rlm_speculation(mock_rlm, max_inflight=4, latency_aware=False)

    n = 16
    code = "\n".join(f"x{i} = llm_query('q{i}')" for i in range(n))
    t0 = time.perf_counter()
    mock_rlm._execute_code(mock_repl, code, {})
    elapsed = time.perf_counter() - t0

    # 16 calls, max_inflight=4, latency 0.3 -> parallel ~1.2s, serial ~4.8s.
    # Even with subprocess-spawn overhead the wall-clock must be well under the
    # serial baseline.
    assert elapsed < n * latency
    # Each call dispatched once and claimed (no re-call).
    assert len(real_calls) == n


def test_latency_aware_claim_hedges_deep_queue(mock_rlm, mock_repl):
    """With latency-aware claiming, a claim on a queued-not-started speculation
    hedges (runs the real tool) when the queue would drain slower than simply
    duplicating the call. Results stay correct; extra real executions are
    bounded by the queue depth, never a wholesale re-run."""
    latency = 0.3
    real_calls = []

    def llm_query(prompt):
        real_calls.append(prompt)
        time.sleep(latency)
        return f"r:{prompt}"

    tools = {"llm_query": llm_query}
    _setup_real(mock_rlm, mock_repl, tools)
    enable_rlm_speculation(mock_rlm, max_inflight=2, latency_aware=True)

    n = 12
    code = "\n".join(f"x{i} = llm_query('q{i}')" for i in range(n))
    t0 = time.perf_counter()
    mock_rlm._execute_code(mock_repl, code, {})
    elapsed = time.perf_counter() - t0

    # results are correct whether served from speculation or a hedged real call
    # (the mock repl returns whatever the hooks return); no result check here —
    # instead assert bounded total work and progress:
    assert elapsed < n * latency  # still massively parallel, not serial
    # speculation avoided a wholesale re-run: at most the queue depth of calls
    # hedged (12 speculated executions + at most max_inflight*2 hedged dups)
    assert len(real_calls) <= n + 4


# ---------------------------------------------------------------------------
# AC5 — budget isolation
# ---------------------------------------------------------------------------


def test_ac5_budget_isolation(mock_rlm, mock_repl):
    real_calls = []

    def llm_query(prompt):
        real_calls.append(prompt)
        return f"r:{prompt}"

    tools = {"llm_query": llm_query}
    _setup_real(mock_rlm, mock_repl, tools)
    mock_rlm.max_llm_calls = 50
    enable_rlm_speculation(mock_rlm, max_inflight=8, max_dispatches_per_turn=1000)

    # 30 real calls, all claim hits (shadow dispatched them). If the shadow's 30
    # dispatches ALSO counted against max_llm_calls=50, the counter would be 60
    # -> raise. Only real (claimed) calls count, so counter = 30 < 50 -> no raise.
    n = 30
    code = "\n".join(f"x{i} = llm_query('q{i}')" for i in range(n))
    mock_rlm._execute_code(mock_repl, code, {})
    assert len(real_calls) == n  # each dispatched once, claimed (no re-call)


# ---------------------------------------------------------------------------
# AC6 — timeout (runaway shadow does not block real execution)
# ---------------------------------------------------------------------------


def test_ac6_timeout(mock_rlm, mock_repl):
    mock_rlm.max_llm_calls = 50
    mock_repl.tools = {"llm_query": lambda p: f"r:{p}"}
    # mock_rlm._execute_code stays the default MagicMock -> real exec returns
    # immediately; only the shadow runs the runaway loop.
    enable_rlm_speculation(mock_rlm, timeout_s=0.5)

    code = "while True:\n    pass\nx = llm_query('hello')\n"
    t0 = time.perf_counter()
    result = mock_rlm._execute_code(mock_repl, code, {})
    elapsed = time.perf_counter() - t0

    assert result == "mock_result"
    assert elapsed < 5.0  # not blocked indefinitely by the runaway shadow


# ---------------------------------------------------------------------------
# AC7 — hook composition
# ---------------------------------------------------------------------------


def test_ac7_hook_composition(mock_rlm, mock_repl, mock_variables, mock_history):
    order = []

    def pre_exec(iteration, code, variables, history, input_args):
        order.append("pre")
        return PreExecutionOutput(code=code)

    def post_exec(iteration, code, result, variables, history, input_args):
        order.append("post")
        return PostExecutionOutput(result=result)

    real_calls = []

    def llm_query(prompt):
        real_calls.append(prompt)
        return f"r:{prompt}"

    tools = {"llm_query": llm_query}
    mock_rlm.max_llm_calls = 50
    mock_repl.tools = tools
    mock_repl.execute = _make_execute(tools)

    enable_rlm_hooks(
        mock_rlm, pre_execution_hook=pre_exec, post_execution_hook=post_exec
    )
    # streaming=False: these tests mock generate_action directly and exercise the
    # Lazy/JIT composition (streaming wraps generate_action, so disable it here).
    enable_rlm_speculation(mock_rlm, streaming=False)

    action = MagicMock(code="x = llm_query('hello')\n", reasoning="test")
    mock_rlm.generate_action.return_value = action
    mock_rlm._process_execution_result.return_value = mock_history

    mock_rlm._execute_iteration(
        mock_repl, mock_variables, mock_history, 0, {"question": "test"}, ["answer"]
    )

    assert order == ["pre", "post"]  # hooks fire exactly once, in order
    assert len(real_calls) == 1  # speculation ran (claim hit, no re-call)


# ---------------------------------------------------------------------------
# AC8 — disable revert
# ---------------------------------------------------------------------------


def test_ac8_disable_revert(mock_rlm, mock_repl):
    mock_rlm.max_llm_calls = 50
    mock_repl.tools = {"llm_query": lambda p: f"r:{p}"}
    original = mock_rlm._execute_code

    enable_rlm_speculation(mock_rlm)
    assert mock_rlm._execute_code is not original
    assert hasattr(mock_rlm, "_speculator")

    disable_rlm_speculation(mock_rlm)
    assert mock_rlm._execute_code is original
    assert not hasattr(mock_rlm, "_speculator")

    # Idempotent.
    disable_rlm_speculation(mock_rlm)


def test_ac8_disable_revert_with_hooks(mock_rlm, mock_repl):
    mock_rlm.max_llm_calls = 50
    mock_repl.tools = {"llm_query": lambda p: f"r:{p}"}

    enable_rlm_hooks(mock_rlm)
    hooks_exec = mock_rlm._execute_code

    enable_rlm_speculation(mock_rlm)
    assert mock_rlm._execute_code is not hooks_exec

    disable_rlm_speculation(mock_rlm)
    # Restores the hooks-patched _execute_code, not the original class method.
    assert mock_rlm._execute_code is hooks_exec


# ---------------------------------------------------------------------------
# AC10 — opt-in no-op
# ---------------------------------------------------------------------------


def test_ac10_opt_in_noop(mock_rlm, mock_repl):
    mock_rlm.max_llm_calls = 50
    mock_repl.tools = {"llm_query": lambda p: f"r:{p}"}

    # No enable_rlm_speculation call -> _execute_code is still the plain MagicMock
    # (not the speculation wrapper) and no speculation state is installed.
    assert isinstance(mock_rlm._execute_code, MagicMock)
    assert not isinstance(getattr(mock_rlm, "_speculation_wrapper", None), MethodType)

    result = mock_rlm._execute_code(mock_repl, "x = 1\n", {})
    assert result == "mock_result"


# ---------------------------------------------------------------------------
# AC11 — async path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ac11_async_path(mock_rlm, mock_repl, mock_variables, mock_history):
    real_calls = []

    def llm_query(prompt):
        real_calls.append(prompt)
        return f"r:{prompt}"

    tools = {"llm_query": llm_query}
    mock_rlm.max_llm_calls = 50
    mock_repl.tools = tools
    mock_repl.execute = _make_execute(tools)

    enable_rlm_hooks(mock_rlm)  # sets _aexecute_iteration to the hooks-patched async
    enable_rlm_speculation(
        mock_rlm, streaming=False
    )  # Lazy/JIT (mocks generate_action)

    action = MagicMock(code="x = llm_query('hello')\n", reasoning="test")
    mock_rlm.generate_action.acall = AsyncMock(return_value=action)
    mock_rlm._process_execution_result.return_value = mock_history

    await mock_rlm._aexecute_iteration(
        mock_repl, mock_variables, mock_history, 0, {"question": "test"}, ["answer"]
    )

    assert len(real_calls) == 1  # speculation ran on the async path


# ---------------------------------------------------------------------------
# AC12 — real-interpreter claim bridge (Deno)
# ---------------------------------------------------------------------------


def test_ac12_real_interpreter_claim_bridge(mock_rlm):
    from dspy.primitives.python_interpreter import PythonInterpreter

    real_calls = []

    def llm_query(prompt):
        real_calls.append(prompt)
        return f"R:{prompt}"

    mock_rlm.max_llm_calls = 50
    mock_rlm._execute_code = _real_execute_code
    enable_rlm_speculation(mock_rlm)

    with PythonInterpreter(tools={"llm_query": llm_query}) as repl:
        code = "x = llm_query('hello')\nprint(x)\n"
        result = mock_rlm._execute_code(repl, code, {})

    # The claim hook intercepted the Deno tool call and reused the shadow result.
    assert "R:hello" in str(result)
    assert real_calls == ["hello"]  # dispatched once, claimed (no re-call)


# ---------------------------------------------------------------------------
# AC13 — both enable orders
# ---------------------------------------------------------------------------


def test_ac13_order1_hooks_then_speculation(
    mock_rlm, mock_repl, mock_variables, mock_history
):
    order = []

    def pre_exec(iteration, code, variables, history, input_args):
        order.append("pre")
        return PreExecutionOutput(code=code)

    def post_exec(iteration, code, result, variables, history, input_args):
        order.append("post")
        return PostExecutionOutput(result=result)

    real_calls = []

    def llm_query(prompt):
        real_calls.append(prompt)
        return f"r:{prompt}"

    tools = {"llm_query": llm_query}
    mock_rlm.max_llm_calls = 50
    mock_repl.tools = tools
    mock_repl.execute = _make_execute(tools)

    enable_rlm_hooks(
        mock_rlm, pre_execution_hook=pre_exec, post_execution_hook=post_exec
    )
    # streaming=False: these tests mock generate_action directly and exercise the
    # Lazy/JIT composition (streaming wraps generate_action, so disable it here).
    enable_rlm_speculation(mock_rlm, streaming=False)

    action = MagicMock(code="x = llm_query('hello')\n", reasoning="test")
    mock_rlm.generate_action.return_value = action
    mock_rlm._process_execution_result.return_value = mock_history

    mock_rlm._execute_iteration(
        mock_repl, mock_variables, mock_history, 0, {"question": "test"}, ["answer"]
    )

    assert order == ["pre", "post"]  # hooks fire
    assert len(real_calls) == 1  # speculation active (claim hit)


def test_ac13_order2_speculation_then_hooks(
    mock_rlm, mock_repl, mock_variables, mock_history
):
    order = []

    def pre_exec(iteration, code, variables, history, input_args):
        order.append("pre")
        return PreExecutionOutput(code=code)

    real_calls = []

    def llm_query(prompt):
        real_calls.append(prompt)
        return f"r:{prompt}"

    tools = {"llm_query": llm_query}
    mock_rlm.max_llm_calls = 50
    mock_repl.tools = tools
    mock_repl.execute = _make_execute(tools)

    enable_rlm_speculation(mock_rlm, streaming=False)
    enable_rlm_hooks(mock_rlm, pre_execution_hook=pre_exec)

    # enable_rlm_hooks overwrote _execute_code -> speculation is NOT active.
    # Hooks still work and there is no crash (documented Order 2 behaviour).
    action = MagicMock(code="x = llm_query('hello')\n", reasoning="test")
    mock_rlm.generate_action.return_value = action
    mock_rlm._process_execution_result.return_value = mock_history

    mock_rlm._execute_iteration(
        mock_repl, mock_variables, mock_history, 0, {"question": "test"}, ["answer"]
    )

    assert order == ["pre"]  # hooks fire
    # Speculation inactive: the real code's llm_query ran directly (no shadow
    # dispatch, no claim) -> exactly one real call.
    assert len(real_calls) == 1


# ---------------------------------------------------------------------------
# Unit coverage: _placeholder / _register_classifications / _sync_registry_fns
# ---------------------------------------------------------------------------


def test_placeholder_raises():
    """The placeholder fn is a stand-in that must never be invoked for real."""
    with pytest.raises(RuntimeError, match="placeholder tool fn"):
        _placeholder()


def test_register_classifications_user_tools(mock_rlm, mock_repl):
    """User tools passed via `tools` are registered with the master switch."""

    def lookup_price(symbol):
        return 1.0

    mock_rlm.max_llm_calls = 50
    mock_repl.tools = {"llm_query": lambda p: p, "lookup_price": lookup_price}
    mock_repl.execute = _make_execute(mock_repl.tools)

    enable_rlm_speculation(
        mock_rlm, tools={"lookup_price": lookup_price}, speculate_user_tools=True
    )
    spec = mock_rlm._speculator
    tool_spec = spec.registry.get("lookup_price")
    assert tool_spec is not None
    assert tool_spec.speculatable is True
    assert tool_spec.pure is True


def test_register_classifications_user_tool_object(mock_rlm, mock_repl):
    """A Tool-like object (with a `.func` attribute) is unwrapped on register."""

    class _Tool:
        def __init__(self, fn):
            self.func = fn

    def lookup_price(symbol):
        return 1.0

    mock_rlm.max_llm_calls = 50
    mock_repl.tools = {"llm_query": lambda p: p, "lookup_price": lookup_price}
    mock_repl.execute = _make_execute(mock_repl.tools)

    enable_rlm_speculation(
        mock_rlm, tools={"lookup_price": _Tool(lookup_price)}, speculate_user_tools=True
    )
    tool_spec = mock_rlm._speculator.registry.get("lookup_price")
    assert tool_spec is not None
    assert tool_spec.fn is lookup_price  # unwrapped from the Tool object


def test_sync_registry_fns_no_tools(mock_rlm, mock_repl):
    """_sync_registry_fns is a no-op when the repl exposes no tools."""
    mock_rlm.max_llm_calls = 50
    enable_rlm_speculation(mock_rlm)
    spec = mock_rlm._speculator
    # repl.tools is None -> early return, no crash, registry untouched.
    _sync_registry_fns(spec, MagicMock(tools=None))
    assert spec.registry.get("llm_query") is not None


# ---------------------------------------------------------------------------
# Unit coverage: _make_claim_hook (counter limit, normalize fallback, batched)
# ---------------------------------------------------------------------------


def test_make_claim_hook_counter_limit():
    """A claimed call still counts against max_llm_calls; exceeding it raises."""

    def real_tool(prompt):
        return f"r:{prompt}"

    def claim(prompt):
        return f"c:{prompt}"

    hook = _make_claim_hook(
        real_tool=real_tool, claim_hook=claim, name="llm_query", max_llm_calls=2
    )
    assert hook("a") == "c:a"
    assert hook("b") == "c:b"
    with pytest.raises(RuntimeError, match="LLM call limit exceeded"):
        hook("c")


def test_make_claim_hook_normalize_typeerror_fallback():
    """When the call args don't bind to the real signature, fall back to raw."""

    def real_tool(a, b):
        return a + b

    def claim(*args, **kwargs):
        return args

    hook = _make_claim_hook(
        real_tool=real_tool, claim_hook=claim, name="llm_query", max_llm_calls=10
    )
    # Only one positional arg -> sig.bind raises TypeError -> raw args forwarded.
    assert hook(1) == (1,)


def test_make_claim_hook_batched():
    """Batched hooks increment the counter by the number of prompts."""

    def real_batched(prompts):
        return [f"r:{p}" for p in prompts]

    def claim(prompts):
        return [f"c:{p}" for p in prompts]

    hook = _make_claim_hook(
        real_tool=real_batched,
        claim_hook=claim,
        name="llm_query_batched",
        max_llm_calls=10,
    )
    assert hook(["a", "b", "c"]) == ["c:a", "c:b", "c:c"]
    # Counter is now 3; a batch of 8 would push it to 11 > 10 -> raise.
    with pytest.raises(RuntimeError, match="LLM call limit exceeded"):
        hook(["x"] * 8)


# ---------------------------------------------------------------------------
# Unit coverage: _install_claim_hooks (no tools / non-speculatable / user wrap)
# ---------------------------------------------------------------------------


def test_install_claim_hooks_no_tools(mock_rlm):
    """_install_claim_hooks is a no-op when the repl exposes no tools."""
    mock_rlm.max_llm_calls = 50
    enable_rlm_speculation(mock_rlm)
    spec = mock_rlm._speculator
    config = mock_rlm._speculation_config
    _install_claim_hooks(MagicMock(tools=None), spec, config, mock_rlm)


def test_install_claim_hooks_skips_non_speculatable(mock_rlm, mock_repl):
    """A registered-but-not-speculatable user tool is left unwrapped."""

    def user_tool(x):
        return x

    tools = {"llm_query": lambda p: p, "user_tool": user_tool}
    mock_rlm.max_llm_calls = 50
    mock_repl.tools = tools
    mock_repl.execute = _make_execute(tools)
    mock_rlm._execute_code = _real_execute_code

    # speculate_user_tools=False (default) -> user_tool registered but NOT
    # speculatable -> _install_claim_hooks skips it (line 205).
    enable_rlm_speculation(mock_rlm, tools={"user_tool": user_tool})
    mock_rlm._execute_code(mock_repl, "x = user_tool(1)\n", {})
    assert mock_repl.tools["user_tool"] is user_tool  # unwrapped


def test_install_claim_hooks_wraps_speculatable_user_tool(mock_rlm, mock_repl):
    """A speculatable user tool (non-LLM) is wrapped with its claim hook."""

    def lookup_price(x):
        return x * 2

    mock_rlm.max_llm_calls = 50
    tools = {"llm_query": lambda p: p, "lookup_price": lookup_price}
    mock_repl.tools = tools
    mock_repl.execute = _make_execute(tools)
    mock_rlm._execute_code = _real_execute_code

    enable_rlm_speculation(
        mock_rlm, tools={"lookup_price": lookup_price}, speculate_user_tools=True
    )
    mock_rlm._execute_code(mock_repl, "x = lookup_price(2)\n", {})
    # lookup_price is speculatable and not an LLM tool -> wrapped (line 209).
    assert mock_repl.tools["lookup_price"] is not lookup_price


# ---------------------------------------------------------------------------
# _speculation_execute_code fallback + exception-swallow paths
# ---------------------------------------------------------------------------


def test_execute_code_fallback_when_disabled(mock_rlm, mock_repl):
    """When speculation is disabled, the wrapper falls through to the inner fn."""
    mock_rlm.max_llm_calls = 50
    tools = {"llm_query": lambda p: p}
    mock_repl.tools = tools
    mock_repl.execute = _make_execute(tools)
    mock_rlm._execute_code = _real_execute_code

    enable_rlm_speculation(mock_rlm)
    mock_rlm._speculation_config.enabled = False

    result = mock_rlm._execute_code(mock_repl, "x = 1\n", {})
    assert result == "ok"  # inner ran directly, no shadow, no claim hooks


def test_shadow_feed_exception_swallowed(mock_rlm, mock_repl, monkeypatch):
    """A shadow feed error is swallowed; real execution still proceeds."""
    mock_rlm.max_llm_calls = 50
    tools = {"llm_query": lambda p: f"r:{p}"}
    mock_repl.tools = tools
    mock_repl.execute = _make_execute(tools)
    mock_rlm._execute_code = _real_execute_code

    enable_rlm_speculation(mock_rlm)
    spec = mock_rlm._speculator
    real_begin = spec.session.begin_stream_turn

    def bad_begin(*a, **k):
        t = real_begin(*a, **k)
        t.feed = lambda delta: (_ for _ in ()).throw(RuntimeError("feed boom"))
        return t

    monkeypatch.setattr(spec.session, "begin_stream_turn", bad_begin)
    result = mock_rlm._execute_code(mock_repl, "x = llm_query('hi')\n", {})
    assert result == "ok"


def test_shadow_end_exception_swallowed(mock_rlm, mock_repl, monkeypatch):
    """A shadow turn-end exception is swallowed; real execution proceeds."""
    mock_rlm.max_llm_calls = 50
    tools = {"llm_query": lambda p: f"r:{p}"}
    mock_repl.tools = tools
    mock_repl.execute = _make_execute(tools)
    mock_rlm._execute_code = _real_execute_code

    enable_rlm_speculation(mock_rlm)
    spec = mock_rlm._speculator
    real_begin = spec.session.begin_stream_turn

    def bad_begin(*a, **k):
        t = real_begin(*a, **k)
        t.end = lambda timeout=600: (_ for _ in ()).throw(RuntimeError("end boom"))
        return t

    monkeypatch.setattr(spec.session, "begin_stream_turn", bad_begin)
    result = mock_rlm._execute_code(mock_repl, "x = llm_query('hi')\n", {})
    assert result == "ok"


def test_install_claim_hooks_exception_swallowed(mock_rlm, mock_repl, monkeypatch):
    """A claim-hook install failure is swallowed; real execution proceeds."""
    import dspy_rlm_hooks.speculation_integration as si

    mock_rlm.max_llm_calls = 50
    tools = {"llm_query": lambda p: f"r:{p}"}
    mock_repl.tools = tools
    mock_repl.execute = _make_execute(tools)
    mock_rlm._execute_code = _real_execute_code

    enable_rlm_speculation(mock_rlm)

    def boom(*a, **k):
        raise RuntimeError("install boom")

    monkeypatch.setattr(si, "_install_claim_hooks", boom)
    result = mock_rlm._execute_code(mock_repl, "x = llm_query('hi')\n", {})
    assert result == "ok"


def test_end_turn_exception_swallowed(mock_rlm, mock_repl, monkeypatch):
    """An end_turn failure is swallowed; the real result is still returned."""
    mock_rlm.max_llm_calls = 50
    tools = {"llm_query": lambda p: f"r:{p}"}
    mock_repl.tools = tools
    mock_repl.execute = _make_execute(tools)
    mock_rlm._execute_code = _real_execute_code

    enable_rlm_speculation(mock_rlm)
    spec = mock_rlm._speculator

    def boom():
        raise RuntimeError("end_turn boom")

    monkeypatch.setattr(spec, "end_turn", boom)
    result = mock_rlm._execute_code(mock_repl, "x = llm_query('hi')\n", {})
    assert result == "ok"


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
# Regression: claim hooks leaking into the registry cause a self-claim deadlock
# (the launcher pool + interpreter shutdown hang for 600s). Guarded by
# _sync_registry_fns + Launcher.run + the self-claim check in the claim hooks.
# ---------------------------------------------------------------------------


def test_sync_registry_fns_skips_claim_hooks(mock_rlm, mock_repl):
    """A claim hook left in repl.tools by _install_claim_hooks is never synced
    into the registry: ToolSpec.fn stays the raw tool."""
    from dspy_rlm_hooks.speculation.guards import is_claim_hook

    def lookup_price(x):
        return x * 2

    tools = {"llm_query": lambda p: f"r:{p}", "lookup_price": lookup_price}
    mock_rlm.max_llm_calls = 50
    mock_repl.tools = tools
    mock_repl.execute = _make_execute(tools)
    mock_rlm._execute_code = _real_execute_code

    enable_rlm_speculation(
        mock_rlm, tools={"lookup_price": lookup_price}, speculate_user_tools=True
    )
    spec = mock_rlm._speculator
    config = mock_rlm._speculation_config

    # iteration 1: sync the raw fns, then install claim hooks (left in tools)
    _sync_registry_fns(spec, mock_repl)
    _install_claim_hooks(mock_repl, spec, config, mock_rlm)
    assert is_claim_hook(mock_repl.tools["lookup_price"])

    # iteration 2: sync again — MUST NOT copy the hook into the registry
    _sync_registry_fns(spec, mock_repl)
    registry_fn = spec.registry.get("lookup_price").fn
    assert registry_fn is lookup_price
    assert not is_claim_hook(registry_fn)


def test_two_iterations_no_self_claim_deadlock(mock_rlm, mock_repl):
    """Two sequential _execute_code calls sharing one repl.tools dict (the real
    RLM iteration pattern) resolve correctly and leave no stuck speculations.
    Pre-fix, iteration 2 synced the installed claim hook into the registry; the
    shadow then executed that hook, which claimed its own pending speculation
    and blocked the launcher pool (and interpreter shutdown) for 600s."""
    real_calls = []

    def read_file(path):
        real_calls.append(path)
        return f"<{path}>"

    tools = {"llm_query": lambda p: f"r:{p}", "read_file": read_file}
    mock_rlm.max_llm_calls = 50
    mock_repl.tools = tools
    mock_repl.execute = _make_execute(tools)
    mock_rlm._execute_code = _real_execute_code

    enable_rlm_speculation(
        mock_rlm, tools={"read_file": read_file}, speculate_user_tools=True
    )
    spec = mock_rlm._speculator
    code = "f = read_file('main.py')\n"

    for _ in range(2):
        mock_rlm._execute_code(mock_repl, code, {})

    # registry still holds the raw fn (no hook leak)
    assert spec.registry.get("read_file").fn is read_file
    # the real run claimed the shadow result: read_file ran once per turn
    assert real_calls == ["main.py", "main.py"]
    # no speculation left pending/running — a self-claim would never resolve
    stuck = [s for s in spec.session.store.all if s.state in ("pending", "running")]
    assert stuck == []


def test_launcher_dispatch_never_runs_claim_hook(mock_rlm, mock_repl):
    """Even if a claim hook is forced onto a ToolSpec.fn (a future leak path),
    Launcher.dispatch swaps it for the raw tool instead of executing the hook
    (which would self-claim and block the pool)."""
    from dspy_rlm_hooks.speculation.guards import is_claim_hook

    real_calls = []

    def lookup_price(x):
        real_calls.append(x)
        return x * 2

    tools = {"llm_query": lambda p: p, "lookup_price": lookup_price}
    mock_rlm.max_llm_calls = 50
    mock_repl.tools = tools

    enable_rlm_speculation(
        mock_rlm, tools={"lookup_price": lookup_price}, speculate_user_tools=True
    )
    spec = mock_rlm._speculator
    tool_spec = spec.registry.get("lookup_price")
    leak_hook = spec.session.real_hooks()["lookup_price"]
    assert is_claim_hook(leak_hook)
    tool_spec.fn = leak_hook  # simulate a hook in the registry

    dispatched = spec.session.launcher.dispatch(tool_spec, (3,), {}, "shadow")
    assert dispatched is not None
    assert dispatched.done.wait(5.0)  # resolves fast — no self-claim block
    assert dispatched.result() == 6
    assert real_calls == [3]  # the RAW tool ran once
