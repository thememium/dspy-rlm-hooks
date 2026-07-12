"""Extended tests for patcher.py to cover remaining gaps.

Covers:
- _run_async else branch (running event loop)
- Verbose logging paths
- Async SyntaxError handling
- Async pre_execution hook
- PredictRLM branches in enable/disable
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dspy_rlm_hooks import (PreExecutionOutput, PreIterationOutput,
                            disable_rlm_hooks, enable_rlm_hooks)
from dspy_rlm_hooks.patcher import _run_async

# ---------------------------------------------------------------------------
# _run_async
# ---------------------------------------------------------------------------


class TestRunAsyncPatcher:
    """Tests for the _run_async function in patcher.py."""

    def test_run_async_no_running_loop(self):
        """Test _run_async when no event loop is running."""
        async def coro():
            return "test_value"

        result = _run_async(coro())
        assert result == "test_value"

    def test_run_async_with_running_loop(self):
        """Test _run_async when an event loop is already running.

        Mocks asyncio to simulate a running loop.
        """

        async def coro():
            return "new_loop_result"

        with patch("dspy_rlm_hooks.patcher.asyncio") as mock_asyncio:
            mock_asyncio.get_running_loop = MagicMock()  # no RuntimeError
            mock_asyncio.new_event_loop = MagicMock()
            mock_loop = MagicMock()
            mock_asyncio.new_event_loop.return_value = mock_loop
            mock_loop.run_until_complete.return_value = "new_loop_result"

            result = _run_async(coro())

            assert result == "new_loop_result"
            mock_loop.run_until_complete.assert_called_once()
            mock_loop.close.assert_called_once()


# ---------------------------------------------------------------------------
# Verbose logging
# ---------------------------------------------------------------------------


class TestVerboseLogging:
    """Tests for verbose logging paths in _execute_iteration."""

    def test_verbose_logging_sync(self, mock_rlm, mock_repl, mock_history, mock_variables):
        """Test that verbose=True triggers logging in sync path."""
        mock_rlm.verbose = True

        enable_rlm_hooks(mock_rlm)

        action = MagicMock()
        action.code = "print('hello')"
        action.reasoning = "test reasoning"
        mock_rlm.generate_action.return_value = action
        mock_rlm._process_execution_result.return_value = mock_history

        with patch("dspy_rlm_hooks.patcher.logger") as mock_logger:
            mock_rlm._execute_iteration(
                mock_repl, mock_variables, mock_history, 0, {"question": "test"}, ["answer"]
            )

            mock_logger.info.assert_called_once()
            call_args = mock_logger.info.call_args
            assert "RLM iteration" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_verbose_logging_async(self, mock_rlm, mock_repl, mock_history, mock_variables):
        """Test that verbose=True triggers logging in async path."""
        mock_rlm.verbose = True

        enable_rlm_hooks(mock_rlm)

        action = MagicMock()
        action.code = "print('hello')"
        action.reasoning = "test reasoning"
        mock_rlm.generate_action.acall = AsyncMock(return_value=action)
        mock_rlm._process_execution_result.return_value = mock_history

        with patch("dspy_rlm_hooks.patcher.logger") as mock_logger:
            await mock_rlm._aexecute_iteration(
                mock_repl, mock_variables, mock_history, 0, {"question": "test"}, ["answer"]
            )

            mock_logger.info.assert_called_once()
            call_args = mock_logger.info.call_args
            assert "RLM iteration" in call_args[0][0]


# ---------------------------------------------------------------------------
# Async SyntaxError handling
# ---------------------------------------------------------------------------


class TestAsyncSyntaxError:
    """Tests for SyntaxError handling in _aexecute_iteration."""

    @pytest.mark.asyncio
    async def test_async_syntax_error_handling(self, mock_rlm, mock_repl, mock_history, mock_variables):
        """Test that SyntaxError from code fences is handled in async path."""
        enable_rlm_hooks(mock_rlm)

        action = MagicMock()
        action.code = "```json\nnot python\n```"
        action.reasoning = "test"
        mock_rlm.generate_action.acall = AsyncMock(return_value=action)
        mock_rlm._process_execution_result.return_value = mock_history

        await mock_rlm._aexecute_iteration(
            mock_repl, mock_variables, mock_history, 0, {"question": "test"}, ["answer"]
        )

        # Should have caught syntax error and passed it to _process_execution_result
        mock_rlm._process_execution_result.assert_called_once()
        call_args = mock_rlm._process_execution_result.call_args
        assert "[Error]" in str(call_args[0][2])


# ---------------------------------------------------------------------------
# Async pre_execution hook
# ---------------------------------------------------------------------------


class TestAsyncPreExecutionHook:
    """Tests for async pre_execution hook in _aexecute_iteration."""

    @pytest.mark.asyncio
    async def test_async_pre_execution_hook_awaited(self, mock_rlm, mock_repl, mock_history, mock_variables):
        """Test that async pre_execution hook is awaited in _aexecute_iteration."""
        captured_code = []

        async def async_hook(iteration, code, variables, history, input_args):
            captured_code.append(code)
            await asyncio.sleep(0)
            return PreExecutionOutput(code=f"# modified\n{code}")

        enable_rlm_hooks(mock_rlm, pre_execution_hook=async_hook)

        action = MagicMock()
        action.code = "print('hello')"
        action.reasoning = "test"
        mock_rlm.generate_action.acall = AsyncMock(return_value=action)
        mock_rlm._process_execution_result.return_value = mock_history

        await mock_rlm._aexecute_iteration(
            mock_repl, mock_variables, mock_history, 0, {"question": "test"}, ["answer"]
        )

        assert len(captured_code) == 1
        assert "print('hello')" in captured_code[0]

    @pytest.mark.asyncio
    async def test_sync_pre_execution_hook_in_async(self, mock_rlm, mock_repl, mock_history, mock_variables):
        """Test that sync pre_execution hook works in async path."""
        def sync_hook(iteration, code, variables, history, input_args):
            return PreExecutionOutput(code=f"# sync_modified\n{code}")

        enable_rlm_hooks(mock_rlm, pre_execution_hook=sync_hook)

        action = MagicMock()
        action.code = "print('hello')"
        action.reasoning = "test"
        mock_rlm.generate_action.acall = AsyncMock(return_value=action)
        mock_rlm._process_execution_result.return_value = mock_history

        await mock_rlm._aexecute_iteration(
            mock_repl, mock_variables, mock_history, 0, {"question": "test"}, ["answer"]
        )

        # Check that the code was modified
        call_args = mock_repl.execute.call_args
        assert "# sync_modified" in call_args[0][0]


# ---------------------------------------------------------------------------
# PredictRLM branches in enable/disable
# ---------------------------------------------------------------------------


class TestPredictRLMBranches:
    """Tests for PredictRLM detection branches in enable_rlm_hooks and disable_rlm_hooks."""

    def test_enable_rlm_hooks_detects_predict_rlm(self, mock_rlm):
        """Test that enable_rlm_hooks delegates to enable_predict_rlm_hooks for PredictRLM."""
        # _is_predict_rlm is imported locally in patcher.py functions,
        # so we need to patch it in the predict_rlm_compat module
        with patch("dspy_rlm_hooks.predict_rlm_compat._is_predict_rlm", return_value=True):
            with patch("dspy_rlm_hooks.predict_rlm_compat.enable_predict_rlm_hooks") as mock_enable:
                enable_rlm_hooks(
                    mock_rlm,
                    pre_iteration_hook=MagicMock(),
                    pre_execution_hook=MagicMock(),
                    post_execution_hook=MagicMock(),
                    post_iteration_hook=MagicMock(),
                )

                mock_enable.assert_called_once()
                # Verify the hooks were passed through
                call_kwargs = mock_enable.call_args[1]
                assert call_kwargs["pre_iteration_hook"] is not None
                assert call_kwargs["pre_execution_hook"] is not None
                assert call_kwargs["post_execution_hook"] is not None
                assert call_kwargs["post_iteration_hook"] is not None

    def test_disable_rlm_hooks_detects_predict_rlm(self, mock_rlm):
        """Test that disable_rlm_hooks delegates to disable_predict_rlm_hooks for PredictRLM."""
        with patch("dspy_rlm_hooks.predict_rlm_compat._is_predict_rlm", return_value=True):
            with patch("dspy_rlm_hooks.predict_rlm_compat.disable_predict_rlm_hooks") as mock_disable:
                disable_rlm_hooks(mock_rlm)

                mock_disable.assert_called_once_with(mock_rlm)

    def test_enable_rlm_hooks_normal_path(self, mock_rlm):
        """Test that enable_rlm_hooks uses normal path for non-PredictRLM."""
        with patch("dspy_rlm_hooks.predict_rlm_compat._is_predict_rlm", return_value=False):
            enable_rlm_hooks(mock_rlm, pre_iteration_hook=MagicMock())

            # Should have set hook attributes directly
            assert hasattr(mock_rlm, "_hook_pre_iteration")

    def test_disable_rlm_hooks_normal_path(self, mock_rlm):
        """Test that disable_rlm_hooks uses normal path for non-PredictRLM."""
        with patch("dspy_rlm_hooks.predict_rlm_compat._is_predict_rlm", return_value=False):
            enable_rlm_hooks(mock_rlm, pre_iteration_hook=MagicMock())
            assert hasattr(mock_rlm, "_hook_pre_iteration")

            disable_rlm_hooks(mock_rlm)
            assert not hasattr(mock_rlm, "_hook_pre_iteration")


# ---------------------------------------------------------------------------
# Async pre-iteration hook with variables
# ---------------------------------------------------------------------------


class TestAsyncPreIterationVariables:
    """Tests for async pre-iteration hook variable injection."""

    @pytest.mark.asyncio
    async def test_async_pre_iteration_injects_vars(self, mock_rlm, mock_repl, mock_history, mock_variables):
        """Test that async pre_iteration hook injects extra_vars in async path."""
        captured = {}

        async def inject_hook(iteration, variables, history, input_args):
            await asyncio.sleep(0)
            return PreIterationOutput(extra_vars={"async_injected": True})

        enable_rlm_hooks(mock_rlm, pre_iteration_hook=inject_hook)

        action = MagicMock()
        action.code = "print('hello')"
        action.reasoning = "test"
        mock_rlm.generate_action.acall = AsyncMock(return_value=action)
        mock_rlm._process_execution_result.return_value = mock_history

        await mock_rlm._aexecute_iteration(
            mock_repl, mock_variables, mock_history, 0, {"question": "test"}, ["answer"]
        )

        # Verify the hook ran and injected variables
        call_args = mock_repl.execute.call_args
        assert call_args[1]["variables"]["async_injected"] is True

    @pytest.mark.asyncio
    async def test_async_pre_iteration_python_code(self, mock_rlm, mock_repl, mock_history, mock_variables):
        """Test that async pre_iteration hook python_code is stored in repl_globals."""

        async def code_hook(iteration, variables, history, input_args):
            await asyncio.sleep(0)
            return PreIterationOutput(python_code="import numpy")

        enable_rlm_hooks(mock_rlm, pre_iteration_hook=code_hook)

        action = MagicMock()
        action.code = "print('hello')"
        action.reasoning = "test"
        mock_rlm.generate_action.acall = AsyncMock(return_value=action)
        mock_rlm._process_execution_result.return_value = mock_history

        await mock_rlm._aexecute_iteration(
            mock_repl, mock_variables, mock_history, 0, {"question": "test"}, ["answer"]
        )

        assert "import numpy" in mock_repl.repl_globals
