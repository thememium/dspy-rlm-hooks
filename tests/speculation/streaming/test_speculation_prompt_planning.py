"""Early prompt construction must match the program's eventual arguments."""

import ast

import pytest

from dspy_rlm_hooks.speculation.streaming import plan_peeks, safe_eval


def test_loop_local_prompt_is_planned_before_loop_closes() -> None:
    # Given a prompt constructed separately from the tool call.
    code = 'for section in sections:\n    prompt = "score: " + section\n    llm_query(prompt)\n'
    # When the model has streamed the call but not closed the loop.
    plans = plan_peeks(code, {"llm_query"}, {"sections": ["a", "b"]})
    # Then every independent call can already start.
    assert [plan.args for plan in plans] == [("score: a",), ("score: b",)]


def test_loop_local_prompt_overrides_stale_value() -> None:
    # Given a prior iteration's prompt.
    code = 'for section in sections:\n    prompt = "new: " + section\n    llm_query(prompt)\n'
    # When planning a new loop.
    plans = plan_peeks(code, {"llm_query"}, {"sections": ["a"], "prompt": "old"})
    # Then only the newly constructed value is eligible.
    assert [plan.args for plan in plans] == [("new: a",)]


def test_mutated_loop_local_is_not_planned_from_old_value() -> None:
    # Given a local whose mutation is not evaluated by the planner.
    code = 'for section in sections:\n    prompt = [section]\n    prompt.append("extra")\n    llm_query(prompt)\n'
    # When planning the following call.
    plans = plan_peeks(code, {"llm_query"}, {"sections": ["a"]})
    # Then leave it to real execution instead of using incomplete arguments.
    assert plans == []


@pytest.mark.parametrize(
    "expression", ["f'{value!r:>8}'", "f'{value!a}'", "f'{3.14159:.2f}'"]
)
def test_formatted_prompt_matches_python(expression: str) -> None:
    # Given a formatted prompt built from a builtin value.
    namespace = {"value": "é"}
    # When evaluating arguments for early dispatch.
    actual = safe_eval(ast.parse(expression, mode="eval").body, namespace)
    # Then the key matches the actual interpreter's arguments.
    assert actual == eval(expression, {"__builtins__": {}}, namespace)
