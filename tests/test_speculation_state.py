"""Live interpreter state must be usable without replaying prior code."""

from __future__ import annotations

import json
import threading
from types import SimpleNamespace

import pytest
from dspy.primitives.python_interpreter import PythonInterpreter

from dspy_rlm_hooks.speculation_integration import (
    _SNAPSHOT_MAX_VALUE_CHARS,
    _live_state_seed,
    disable_rlm_speculation,
    enable_rlm_speculation,
)
from dspy_rlm_hooks.speculator import Speculator


@pytest.mark.parametrize(
    "invalid",
    [5, "x" * (_SNAPSHOT_MAX_VALUE_CHARS + 1), "not_a_literal", "["],
    ids=["non-string", "oversized", "non-literal", "invalid-syntax"],
)
def test_snapshot_keeps_valid_entries_when_an_entry_is_invalid(invalid: str | int):
    def execute(code: str) -> str:
        return json.dumps(
            {"invalid": invalid, "valid": "'kept'", "unrequested": "'ignored'"}
        )

    spec = Speculator()
    try:
        seed = _live_state_seed(
            SimpleNamespace(execute=execute), "print(invalid, valid)", {}, spec
        )
        assert seed == {"valid": "kept"}
    finally:
        spec.close()


def test_snapshot_only_represents_requested_globals():
    # Given an unrelated value whose repr has an observable cost.
    spec = Speculator()
    try:
        with PythonInterpreter() as repl:
            repl.execute(
                "repr_calls = []\n"
                "class Expensive:\n"
                "    def __repr__(self):\n"
                "        repr_calls.append(1)\n"
                "        return 'expensive'\n"
                "unrelated = Expensive()\n"
                "previous = 'needed'\n"
            )
            # When only one prior value is read.
            seed = _live_state_seed(repl, "print(previous)", {}, spec)
            # Then unrelated globals are neither represented nor returned.
            assert seed == {"previous": "needed"}
            assert repl.execute("print(len(repr_calls))").strip() == "0"
    finally:
        spec.close()


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("value = 'new'\nprint(value)", {}),
        ("value = value + 'new'\nprint(value)", {"value": "old"}),
        ("value += 'new'", {"value": "old"}),
        ("if False:\n    value = 'new'\nprint(value)", {"value": "old"}),
        ("for value in []:\n    pass\nprint(value)", {"value": "old"}),
    ],
)
def test_snapshot_respects_read_before_write(code: str, expected: dict[str, str]):
    spec = Speculator()
    try:
        with PythonInterpreter() as repl:
            repl.execute("value = 'old'")
            seed = _live_state_seed(repl, code, {}, spec)
            assert seed == expected
    finally:
        spec.close()


def test_self_contained_block_avoids_snapshot_round_trip():
    calls: list[str] = []

    def execute(code: str) -> str:
        calls.append(code)
        return "{}"

    spec = Speculator()
    try:
        seed = _live_state_seed(
            SimpleNamespace(execute=execute), "value = 'new'\nprint(value)", {}, spec
        )
        assert seed == {}
        assert calls == []
    finally:
        spec.close()


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
