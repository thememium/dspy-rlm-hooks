"""Coverage tests for the DSPy >=3.5 iteration contract.

DSPy 3.5 passes ``interpreter_factory`` **and** ``sub_agent_rules`` to
``_execute_iteration`` and returns ``(REPLEntry, final_outputs)`` from it.
``_process_execution_result`` lost its ``history`` parameter and now returns
the same tuple, ``generate_action``'s history input is ``history`` (not
``repl_history``), and ``_extract_fallback`` takes the turn's entries.

These fakes mirror that contract so the compat branches are covered without a
real LM or interpreter.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from dspy.primitives.repl_types import REPLEntry, REPLHistory

from dspy_rlm_hooks.core import patcher
from dspy_rlm_hooks.core.types import (
    PostExecutionOutput,
    PostIterationOutput,
    PreExecutionOutput,
)

_FACTORY = lambda: "interpreter"  # noqa: E731


def _entry(code: str) -> REPLEntry:
    return REPLEntry(reasoning="plan", code=code, output="out")


class _FakeGenerateAction:
    """DSPy 3.5-style predictor: history input, ``signature=`` kwarg."""

    def __init__(self) -> None:
        self.signature = SimpleNamespace(input_fields={"history": object()})
        self.sync_kwargs: dict[str, object] | None = None
        self.async_kwargs: dict[str, object] | None = None

    def __call__(self, **kwargs):
        self.sync_kwargs = kwargs
        return SimpleNamespace(reasoning="plan", code="x = 1")

    async def acall(self, **kwargs):
        self.async_kwargs = kwargs
        return SimpleNamespace(reasoning="plan", code="x = 1")


class _RLM35:
    """Duck-typed DSPy 3.5 RLM (entry-tuple contract)."""

    max_iters = 3
    verbose = False
    _hook_pre_iteration = None
    _hook_pre_execution = None
    _hook_post_execution = None
    _hook_post_iteration = None
    _initial_sub_agent_rules = "initial rules"

    def __init__(self) -> None:
        self.generate_action = _FakeGenerateAction()
        self.processed_calls: list[tuple[str, object]] = []
        self.fallback_calls: list[tuple[object, list[object], list[str]]] = []
        self.entries: list[REPLHistory] = []
        self._hook_turn_entries: list[REPLEntry] = []
        self.final_outputs: dict[str, Any] | None = None

    def _action_signature_for_current_factory(self, factory, sub_agent_rules):
        return f"sig:{factory()}:{sub_agent_rules}"

    def _process_execution_result(self, action, code, result, output_field_names):
        self.processed_calls.append((code, result))
        return _entry(code), self.final_outputs

    def _execute_code(self, repl, code, input_args):
        return "out"

    def _extract_fallback(self, variables, history, turn_entries, output_field_names):
        self.fallback_calls.append((variables, turn_entries, output_field_names))
        return SimpleNamespace(**{name: "fallback" for name in output_field_names})


def _call_args(iteration: int = 0):
    return ("repl", [], MagicMock(), iteration, {}, ["answer"], _FACTORY, "RULES")


class TestEntryTupleContract:
    def test_detection(self):
        assert patcher._uses_entry_tuple_contract(_RLM35())
        assert patcher._action_history_kwarg(_RLM35()) == "history"

    def test_sync_iteration_returns_entry_tuple(self):
        rlm = _RLM35()
        out = patcher._execute_iteration(rlm, *_call_args())

        entry, final_outputs = out
        assert isinstance(entry, REPLEntry) and final_outputs is None
        assert rlm.processed_calls == [("x = 1", "out")]
        kwargs = rlm.generate_action.sync_kwargs
        assert kwargs is not None
        assert kwargs["signature"] == "sig:interpreter:RULES"
        assert "history" in kwargs and "repl_history" not in kwargs

    @pytest.mark.asyncio
    async def test_async_iteration_returns_entry_tuple(self):
        rlm = _RLM35()
        out = await patcher._aexecute_iteration(rlm, *_call_args())

        entry, final_outputs = out
        assert isinstance(entry, REPLEntry) and final_outputs is None
        kwargs = rlm.generate_action.async_kwargs
        assert kwargs is not None and "history" in kwargs

    def test_missing_rules_fall_back_to_initial_rules(self):
        rlm = _RLM35()
        kwargs = patcher._generate_action_signature_kwargs(rlm, _FACTORY)
        assert kwargs["signature"] == "sig:interpreter:initial rules"


class TestPostIterationEntryTuple:
    def test_hook_sees_accumulated_history(self):
        rlm = _RLM35()
        seen: list[int] = []

        def hook(iteration, pred, code, result, history):
            assert isinstance(history, REPLHistory)
            seen.append(len(history.entries))
            return PostIterationOutput(history=history)

        rlm._hook_post_iteration = hook
        patcher._execute_iteration(rlm, *_call_args(0))
        patcher._execute_iteration(rlm, *_call_args(1))

        assert seen == [1, 2]
        assert len(rlm._hook_turn_entries) == 2

    def test_hook_can_replace_entry(self):
        rlm = _RLM35()

        def hook(iteration, pred, code, result, history):
            replacement = _entry("replaced")
            return PostIterationOutput(
                history=REPLHistory(entries=[*history.entries, replacement])
            )

        rlm._hook_post_iteration = hook
        entry, final_outputs = patcher._execute_iteration(rlm, *_call_args())
        assert entry.code == "replaced" and final_outputs is None

    def test_stop_returns_extract_fallback_outputs(self):
        rlm = _RLM35()

        def hook(iteration, pred, code, result, history):
            return PostIterationOutput(history=history, stop=True)

        rlm._hook_post_iteration = hook
        entry, final_outputs = patcher._execute_iteration(rlm, *_call_args())

        assert isinstance(entry, REPLEntry)
        assert final_outputs == {"answer": "fallback"}
        assert len(rlm.fallback_calls) == 1
        _, turn_entries, output_field_names = rlm.fallback_calls[0]
        assert len(turn_entries) == 1 and output_field_names == ["answer"]

    def test_final_outputs_skip_hook(self):
        rlm = _RLM35()
        rlm._hook_post_iteration = MagicMock()
        rlm.final_outputs = {"answer": "done"}

        entry, final_outputs = patcher._execute_iteration(rlm, *_call_args())

        assert isinstance(entry, REPLEntry)
        assert final_outputs == {"answer": "done"}
        rlm._hook_post_iteration.assert_not_called()


class TestExecutionHookHistory:
    """Hooks keep the REPLHistory contract even though 3.5 passes dspy.History."""

    def test_execution_hooks_get_repl_history(self):
        rlm = _RLM35()
        seen: list[tuple[str, int]] = []

        def pre_execution(iteration, code, variables, history, input_args):
            assert isinstance(history, REPLHistory)
            seen.append(("pre", len(history.entries)))
            return PreExecutionOutput(code=code)

        def post_execution(iteration, code, result, variables, history, input_args):
            assert isinstance(history, REPLHistory)
            transcript = history.append(code=code, output=str(result)).format()
            assert code in transcript
            seen.append(("post", len(history.entries)))
            return PostExecutionOutput(result=result)

        rlm._hook_pre_execution = pre_execution
        rlm._hook_post_execution = post_execution

        patcher._execute_iteration(rlm, *_call_args(0))
        patcher._execute_iteration(rlm, *_call_args(1))

        assert seen == [("pre", 0), ("post", 0), ("pre", 1), ("post", 1)]

    @pytest.mark.asyncio
    async def test_async_execution_hooks_get_repl_history(self):
        rlm = _RLM35()
        sizes: list[int] = []

        def pre_execution(iteration, code, variables, history, input_args):
            assert isinstance(history, REPLHistory)
            sizes.append(len(history.entries))
            return PreExecutionOutput(code=code)

        rlm._hook_pre_execution = pre_execution
        await patcher._aexecute_iteration(rlm, *_call_args(0))
        await patcher._aexecute_iteration(rlm, *_call_args(1))

        assert sizes == [0, 1]


class TestLegacyProcessSignature:
    """DSPy 3.1's ``_process_execution_result`` has no separate ``code`` arg."""

    def test_sync_iteration_uses_31_signature(self):
        class _RLM31(_RLM35):
            def _process_execution_result(  # ty: ignore[invalid-method-override]
                self, pred, result, history, output_field_names
            ):
                self.processed_calls.append((pred.code, result))
                return REPLHistory(entries=[_entry(pred.code)])

        rlm = _RLM31()
        out = patcher._execute_iteration(rlm, *_call_args())

        assert isinstance(out, REPLHistory)
        assert out.entries[-1].code == "x = 1"
        assert rlm.processed_calls == [("x = 1", "out")]

    @pytest.mark.asyncio
    async def test_async_iteration_uses_31_signature(self):
        class _RLM31(_RLM35):
            def _process_execution_result(  # ty: ignore[invalid-method-override]
                self, pred, result, history, output_field_names
            ):
                return REPLHistory(entries=[_entry(pred.code)])

        out = await patcher._aexecute_iteration(_RLM31(), *_call_args())
        assert isinstance(out, REPLHistory)
