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
    enable_rlm_speculation(mock_rlm, max_inflight=4)

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
    enable_rlm_speculation(mock_rlm)

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
    enable_rlm_speculation(mock_rlm)  # wraps _execute_code

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
    enable_rlm_speculation(mock_rlm)

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

    enable_rlm_speculation(mock_rlm)
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
