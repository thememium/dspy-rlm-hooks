"""Tests for async hook support."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest
from dspy.primitives.repl_types import REPLHistory

from dspy_rlm_hooks import (
    PostExecutionOutput,
    PostIterationOutput,
    PreExecutionOutput,
    PreIterationOutput,
    enable_rlm_hooks,
)


class TestAsyncHooks:
    """Tests for async hook detection and execution."""

    def test_async_pre_iteration_hook(
        self, mock_rlm, mock_repl, mock_history, mock_variables
    ):
        """Test that async pre_iteration hooks are awaited correctly."""

        async def async_hook(iteration, variables, history, input_args):
            await asyncio.sleep(0)  # Simulate async work
            return PreIterationOutput(extra_vars={"async": True})

        enable_rlm_hooks(mock_rlm, pre_iteration_hook=async_hook)

        action = MagicMock()
        action.code = "print('hello')"
        action.reasoning = "test"
        mock_rlm.generate_action.return_value = action
        mock_rlm._process_execution_result.return_value = mock_history

        mock_rlm._execute_iteration(
            mock_repl, mock_variables, mock_history, 0, {"question": "test"}, ["answer"]
        )

        # Verify that the hook ran and injected variables
        call_args = mock_repl.execute.call_args
        assert call_args.kwargs["variables"]["async"] is True

    def test_async_pre_execution_hook(
        self, mock_rlm, mock_repl, mock_history, mock_variables
    ):
        """Test that async pre_execution hooks are awaited correctly."""

        async def async_hook(iteration, code, variables, history, input_args):
            await asyncio.sleep(0)
            return PreExecutionOutput(code=f"# async\n{code}")

        enable_rlm_hooks(mock_rlm, pre_execution_hook=async_hook)

        action = MagicMock()
        action.code = "print('hello')"
        action.reasoning = "test"
        mock_rlm.generate_action.return_value = action
        mock_rlm._process_execution_result.return_value = mock_history

        mock_rlm._execute_iteration(
            mock_repl, mock_variables, mock_history, 0, {"question": "test"}, ["answer"]
        )

        call_args = mock_repl.execute.call_args
        assert "# async" in call_args.args[0]

    def test_async_post_execution_hook(
        self, mock_rlm, mock_repl, mock_history, mock_variables
    ):
        """Test that async post_execution hooks are awaited correctly."""

        async def async_hook(iteration, code, result, variables, history, input_args):
            await asyncio.sleep(0)
            return PostExecutionOutput(result="async_result")

        enable_rlm_hooks(mock_rlm, post_execution_hook=async_hook)

        action = MagicMock()
        action.code = "print('hello')"
        action.reasoning = "test"
        mock_rlm.generate_action.return_value = action
        mock_rlm._process_execution_result.return_value = mock_history

        mock_rlm._execute_iteration(
            mock_rlm, mock_variables, mock_history, 0, {"question": "test"}, ["answer"]
        )

        call_args = mock_rlm._process_execution_result.call_args
        assert call_args.args[2] == "async_result"

    def test_async_post_iteration_hook(
        self, mock_rlm, mock_repl, mock_history, mock_variables
    ):
        """Test that async post_iteration hooks are awaited correctly."""
        modified_history = MagicMock(spec=REPLHistory)

        async def async_hook(iteration, pred, code, result, history):
            await asyncio.sleep(0)
            return PostIterationOutput(history=modified_history)

        enable_rlm_hooks(mock_rlm, post_iteration_hook=async_hook)

        action = MagicMock()
        action.code = "print('hello')"
        action.reasoning = "test"
        mock_rlm.generate_action.return_value = action
        mock_rlm._process_execution_result.return_value = mock_history

        result = mock_rlm._execute_iteration(
            mock_rlm, mock_variables, mock_history, 0, {"question": "test"}, ["answer"]
        )

        assert result is modified_history

    @pytest.mark.asyncio
    async def test_aexecute_iteration_with_async_hooks(
        self, mock_rlm, mock_repl, mock_history, mock_variables
    ):
        """Test that _aexecute_iteration properly awaits async hooks."""
        call_order = []

        async def async_pre_iter(iteration, variables, history, input_args):
            call_order.append("pre_iteration")
            await asyncio.sleep(0)
            return PreIterationOutput()

        async def async_pre_exec(iteration, code, variables, history, input_args):
            call_order.append("pre_execution")
            await asyncio.sleep(0)
            return PreExecutionOutput(code=code)

        async def async_post_exec(
            iteration, code, result, variables, history, input_args
        ):
            call_order.append("post_execution")
            await asyncio.sleep(0)
            return PostExecutionOutput(result=result)

        async def async_post_iter(iteration, pred, code, result, history):
            call_order.append("post_iteration")
            await asyncio.sleep(0)
            return PostIterationOutput(history=history)

        enable_rlm_hooks(
            mock_rlm,
            pre_iteration_hook=async_pre_iter,
            pre_execution_hook=async_pre_exec,
            post_execution_hook=async_post_exec,
            post_iteration_hook=async_post_iter,
        )

        action = MagicMock()
        action.code = "print('hello')"
        action.reasoning = "test"
        from unittest.mock import AsyncMock

        mock_rlm.generate_action.acall = AsyncMock(return_value=action)
        mock_rlm._process_execution_result.return_value = mock_history

        await mock_rlm._aexecute_iteration(
            mock_repl, mock_variables, mock_history, 0, {"question": "test"}, ["answer"]
        )

        assert call_order == [
            "pre_iteration",
            "pre_execution",
            "post_execution",
            "post_iteration",
        ]

    def test_mixed_sync_async_hooks(
        self, mock_rlm, mock_repl, mock_history, mock_variables
    ):
        """Test mixing sync and async hooks on the same RLM."""

        def sync_pre_iter(iteration, variables, history, input_args):
            return PreIterationOutput(extra_vars={"sync": True})

        async def async_pre_exec(iteration, code, variables, history, input_args):
            await asyncio.sleep(0)
            return PreExecutionOutput(code=f"# async\n{code}")

        enable_rlm_hooks(
            mock_rlm,
            pre_iteration_hook=sync_pre_iter,
            pre_execution_hook=async_pre_exec,
        )

        action = MagicMock()
        action.code = "print('hello')"
        action.reasoning = "test"
        mock_rlm.generate_action.return_value = action
        mock_rlm._process_execution_result.return_value = mock_history

        mock_rlm._execute_iteration(
            mock_repl, mock_variables, mock_history, 0, {"question": "test"}, ["answer"]
        )

        # Both hooks should have run
        call_args = mock_repl.execute.call_args
        assert call_args.kwargs["variables"]["sync"] is True
        assert "# async" in call_args.args[0]

    def test_async_hook_in_sync_context(
        self, mock_rlm, mock_repl, mock_history, mock_variables
    ):
        """Test that async hooks work in sync _execute_iteration."""

        async def async_hook(iteration, variables, history, input_args):
            return PreIterationOutput(extra_vars={"in_sync_context": True})

        enable_rlm_hooks(mock_rlm, pre_iteration_hook=async_hook)

        action = MagicMock()
        action.code = "print('hello')"
        action.reasoning = "test"
        mock_rlm.generate_action.return_value = action
        mock_rlm._process_execution_result.return_value = mock_history

        # Should not raise — asyncio.run() handles the coroutine
        mock_rlm._execute_iteration(
            mock_repl, mock_variables, mock_history, 0, {"question": "test"}, ["answer"]
        )

        call_args = mock_repl.execute.call_args
        assert call_args.kwargs["variables"]["in_sync_context"] is True
