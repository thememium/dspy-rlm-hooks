"""Basic tests for the DSPy RLM Hooks package."""

from __future__ import annotations

from unittest.mock import MagicMock

import dspy
import pytest

from dspy_rlm_hooks import (PostExecutionOutput, PostIterationOutput,
                            PreExecutionOutput, PreIterationOutput,
                            disable_rlm_hooks, enable_rlm_hooks)


class TestEnableDisableHooks:
    """Tests for the core enable/disable functionality."""

    def test_enable_rlm_hooks_assigns_hooks(self, mock_rlm, pre_iteration_hook):
        """Test that enable_rlm_hooks stores hook references on the RLM instance."""
        enable_rlm_hooks(mock_rlm, pre_iteration_hook=pre_iteration_hook)

        assert mock_rlm._hook_pre_iteration is pre_iteration_hook
        assert mock_rlm._hook_pre_execution is None
        assert mock_rlm._hook_post_execution is None
        assert mock_rlm._hook_post_iteration is None

    def test_enable_rlm_hooks_patches_methods(self, mock_rlm, pre_iteration_hook):
        """Test that enable_rlm_hooks replaces iteration methods."""
        original_execute = mock_rlm._execute_iteration
        original_aexecute = mock_rlm._aexecute_iteration
        getattr(mock_rlm, "_execute_code", None)

        enable_rlm_hooks(mock_rlm, pre_iteration_hook=pre_iteration_hook)

        assert mock_rlm._execute_iteration is not original_execute
        assert mock_rlm._aexecute_iteration is not original_aexecute
        assert hasattr(mock_rlm, "_execute_code")

    def test_enable_all_hooks(self, mock_rlm, pre_iteration_hook, pre_execution_hook,
                               post_execution_hook, post_iteration_hook):
        """Test enabling all four hook types simultaneously."""
        enable_rlm_hooks(
            mock_rlm,
            pre_iteration_hook=pre_iteration_hook,
            pre_execution_hook=pre_execution_hook,
            post_execution_hook=post_execution_hook,
            post_iteration_hook=post_iteration_hook,
        )

        assert mock_rlm._hook_pre_iteration is pre_iteration_hook
        assert mock_rlm._hook_pre_execution is pre_execution_hook
        assert mock_rlm._hook_post_execution is post_execution_hook
        assert mock_rlm._hook_post_iteration is post_iteration_hook

    def test_disable_rlm_hooks_removes_hooks(self, mock_rlm, pre_iteration_hook):
        """Test that disable_rlm_hooks cleans up all hook attributes."""
        enable_rlm_hooks(mock_rlm, pre_iteration_hook=pre_iteration_hook)
        disable_rlm_hooks(mock_rlm)

        assert not hasattr(mock_rlm, "_hook_pre_iteration")
        assert not hasattr(mock_rlm, "_hook_pre_execution")
        assert not hasattr(mock_rlm, "_hook_post_execution")
        assert not hasattr(mock_rlm, "_hook_post_iteration")

    def test_disable_rlm_hooks_removes_patched_methods(self, mock_rlm, pre_iteration_hook):
        """Test that disable_rlm_hooks removes patched methods."""
        enable_rlm_hooks(mock_rlm, pre_iteration_hook=pre_iteration_hook)
        disable_rlm_hooks(mock_rlm)

        assert not hasattr(mock_rlm, "_execute_iteration")
        assert not hasattr(mock_rlm, "_aexecute_iteration")
        assert not hasattr(mock_rlm, "_execute_code")

    def test_disable_on_unpatched_rlm(self, mock_rlm):
        """Test that disable_rlm_hooks is safe on an unpatched RLM."""
        # Should not raise
        disable_rlm_hooks(mock_rlm)

    def test_enable_rlm_hooks_validates_rlm(self):
        """Test that enable_rlm_hooks validates the RLM instance."""
        invalid_rlm = MagicMock()
        # Missing required attributes
        delattr(invalid_rlm, "_execute_iteration")

        with pytest.raises(AttributeError, match="missing required attributes"):
            enable_rlm_hooks(invalid_rlm)

    def test_enable_rlm_hooks_validation_missing_max_iterations(self):
        """Test validation catches missing max_iterations."""
        invalid_rlm = MagicMock()
        invalid_rlm._execute_iteration = MagicMock()
        invalid_rlm._aexecute_iteration = MagicMock()
        invalid_rlm._process_execution_result = MagicMock()
        invalid_rlm.generate_action = MagicMock()
        delattr(invalid_rlm, "max_iterations")

        with pytest.raises(AttributeError, match="missing required attributes"):
            enable_rlm_hooks(invalid_rlm)


class TestHookOutputTypes:
    """Tests for hook output dataclasses."""

    def test_pre_iteration_output_defaults(self):
        """Test PreIterationOutput default values."""
        output = PreIterationOutput()
        assert output.extra_vars == {}
        assert output.python_code == ""

    def test_pre_iteration_output_with_values(self):
        """Test PreIterationOutput with explicit values."""
        output = PreIterationOutput(
            extra_vars={"key": "value"},
            python_code="import os",
        )
        assert output.extra_vars == {"key": "value"}
        assert output.python_code == "import os"

    def test_pre_execution_output(self):
        """Test PreExecutionOutput."""
        output = PreExecutionOutput(code="print('hello')")
        assert output.code == "print('hello')"

    def test_post_execution_output(self):
        """Test PostExecutionOutput."""
        output = PostExecutionOutput(result="transformed")
        assert output.result == "transformed"

    def test_post_iteration_output(self):
        """Test PostIterationOutput."""
        from dspy.primitives.repl_types import REPLHistory
        history = REPLHistory(entries=[])
        output = PostIterationOutput(history=history)
        assert output.history is history


class TestRealRLMValidation:
    """Tests with real DSPy RLM objects (requires DSPy installation)."""

    def test_validate_rlm_with_real_rlm(self):
        """Test that _validate_rlm passes with a real dspy.RLM instance."""
        from dspy_rlm_hooks.patcher import _validate_rlm

        rlm = dspy.RLM(
            signature="question -> answer",
            tools=[],
        )
        # Should not raise
        _validate_rlm(rlm)

    def test_enable_disable_with_real_rlm(self):
        """Test enable/disable with a real dspy.RLM instance."""
        rlm = dspy.RLM(
            signature="question -> answer",
            tools=[],
        )

        def hook(iteration, variables, history, input_args):
            return PreIterationOutput()

        enable_rlm_hooks(rlm, pre_iteration_hook=hook)
        assert hasattr(rlm, "_hook_pre_iteration")
        assert rlm._hook_pre_iteration is hook

        disable_rlm_hooks(rlm)
        assert not hasattr(rlm, "_hook_pre_iteration")
