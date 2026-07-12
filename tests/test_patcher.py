"""Tests for patcher-specific functionality."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from dspy_rlm_hooks.patcher import (_REQUIRED_METHODS, _execute_code,
                                    _validate_rlm, disable_rlm_hooks,
                                    enable_rlm_hooks)


class TestValidateRLM:
    """Tests for the _validate_rlm function."""

    def test_valid_rlm_passes(self, mock_rlm):
        """Test that a valid RLM passes validation."""
        _validate_rlm(mock_rlm)  # Should not raise

    def test_missing_methods_raises(self):
        """Test that missing methods raise AttributeError."""
        rlm = MagicMock()
        delattr(rlm, "_execute_iteration")

        with pytest.raises(AttributeError, match="missing required attributes"):
            _validate_rlm(rlm)

    def test_missing_multiple_methods(self):
        """Test that multiple missing methods are all reported."""
        rlm = MagicMock()
        for method in _REQUIRED_METHODS:
            if hasattr(rlm, method):
                delattr(rlm, method)

        with pytest.raises(AttributeError, match="missing required attributes"):
            _validate_rlm(rlm)

    def test_error_message_includes_dspy_version_hint(self):
        """Test that error message suggests checking DSPy version."""
        rlm = MagicMock()
        delattr(rlm, "_execute_iteration")

        with pytest.raises(AttributeError, match="dspy>=3.1"):
            _validate_rlm(rlm)


class TestExecuteCode:
    """Tests for the _execute_code helper."""

    def test_execute_code_without_globals(self, mock_repl):
        """Test executing code without persisted globals."""
        result = _execute_code(MagicMock(), mock_repl, "print('hello')", {})

        mock_repl.execute.assert_called_once_with("print('hello')", variables={})
        assert result == "mock_result"

    def test_execute_code_with_globals(self, mock_repl):
        """Test executing code prepends repl_globals."""
        mock_repl.repl_globals = "import math"
        _execute_code(MagicMock(), mock_repl, "print(math.pi)", {})

        mock_repl.execute.assert_called_once()
        call_args = mock_repl.execute.call_args
        expected_code = "import math\nprint(math.pi)"
        assert call_args.args[0] == expected_code

    def test_execute_code_error_handling(self, mock_repl):
        """Test that execute errors are caught and returned as strings."""
        mock_repl.execute.side_effect = RuntimeError("Test error")

        result = _execute_code(MagicMock(), mock_repl, "bad_code", {})

        assert isinstance(result, str)
        assert "[Error]" in result
        assert "Test error" in result


class TestDisableCleanup:
    """Tests for disable_rlm_hooks cleanup behaviour."""

    def test_disable_is_idempotent(self, mock_rlm):
        """Test that disable_rlm_hooks is safe to call multiple times."""

        def dummy_hook(iteration, variables, history, input_args):
            return MagicMock()

        enable_rlm_hooks(mock_rlm, pre_iteration_hook=dummy_hook)
        disable_rlm_hooks(mock_rlm)
        disable_rlm_hooks(mock_rlm)  # Should not raise
        disable_rlm_hooks(mock_rlm)  # Should not raise

    def test_disable_leaves_other_attributes(self, mock_rlm, pre_iteration_hook):
        """Test that disable only removes hook-related attributes."""
        mock_rlm.custom_attribute = "should_remain"
        enable_rlm_hooks(mock_rlm, pre_iteration_hook=pre_iteration_hook)
        disable_rlm_hooks(mock_rlm)

        assert mock_rlm.custom_attribute == "should_remain"

    def test_disable_partially_patched_rlm(self, mock_rlm):
        """Test disabling an RLM that was only partially patched."""
        # Manually add only some attributes
        mock_rlm._hook_pre_iteration = MagicMock()
        mock_rlm._execute_iteration = MagicMock()
        # Don't add other hook attributes

        # Should not raise
        disable_rlm_hooks(mock_rlm)

        assert not hasattr(mock_rlm, "_hook_pre_iteration")
        assert not hasattr(mock_rlm, "_execute_iteration")


class TestEnableIdempotency:
    """Tests for enable_rlm_hooks idempotency."""

    def test_enable_overwrites_previous_hooks(self, mock_rlm, pre_iteration_hook):
        """Test that enabling hooks twice replaces previous hooks."""
        hook1 = pre_iteration_hook

        def hook2(iteration, variables, history, input_args):
            return MagicMock()

        enable_rlm_hooks(mock_rlm, pre_iteration_hook=hook1)
        assert mock_rlm._hook_pre_iteration is hook1

        enable_rlm_hooks(mock_rlm, pre_iteration_hook=hook2)
        assert mock_rlm._hook_pre_iteration is hook2

    def test_enable_preserves_unset_hooks(
        self, mock_rlm, pre_iteration_hook, post_execution_hook
    ):
        """Test that enabling with only some hooks preserves others."""
        enable_rlm_hooks(
            mock_rlm,
            pre_iteration_hook=pre_iteration_hook,
            post_execution_hook=post_execution_hook,
        )

        assert mock_rlm._hook_pre_iteration is pre_iteration_hook
        assert mock_rlm._hook_post_execution is post_execution_hook
        assert mock_rlm._hook_pre_execution is None
        assert mock_rlm._hook_post_iteration is None
