"""Live interpreter state must be usable without replaying prior code."""

from __future__ import annotations

import threading
from types import SimpleNamespace

from dspy.primitives.python_interpreter import PythonInterpreter

from dspy_rlm_hooks.speculation_integration import (
    disable_rlm_speculation,
    enable_rlm_speculation,
)


def test_prior_iteration_result_is_claimed_and_updated():
    # Given a live REPL and a tool whose execution thread identifies speculation.
    calls: list[tuple[str, str]] = []

    def query(prompt: str) -> str:
        calls.append((prompt, threading.current_thread().name))
        return f"R:{prompt}"

    rlm = SimpleNamespace(
        _execute_code=lambda repl, code, inputs: repl.execute(code, variables=inputs),
        _execute_iteration=lambda *args: None,
        _aexecute_iteration=lambda *args: None,
        _process_execution_result=lambda *args: None,
        generate_action=lambda *args: None,
        max_iters=5,
        verbose=False,
        max_llm_calls=500,
        sub_lm=lambda prompt: [query(prompt)],
    )
    enable_rlm_speculation(rlm, streaming=False)
    try:
        with PythonInterpreter(tools={"llm_query": query}) as repl:
            # When later iterations consume and mutate a previous tool result.
            rlm._execute_code(repl, "previous = llm_query('first')", {})
            result = rlm._execute_code(repl, "print(llm_query(previous))", {})
            repl.execute("previous = 'changed'")
            changed = rlm._execute_code(repl, "print(llm_query(previous))", {})
        # Then every request was speculated once with the current real value.
        assert "R:R:first" in result
        assert "R:changed" in changed
        assert [prompt for prompt, _ in calls] == ["first", "R:first", "changed"]
        assert all(name != threading.current_thread().name for _, name in calls)
    finally:
        disable_rlm_speculation(rlm)
