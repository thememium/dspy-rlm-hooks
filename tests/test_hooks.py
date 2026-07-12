"""Tests for lifecycle hook invocation and ordering."""

from __future__ import annotations

from unittest.mock import MagicMock

from dspy.primitives.repl_types import REPLHistory

from dspy_rlm_hooks import (
    PostExecutionOutput,
    PostIterationOutput,
    PreExecutionOutput,
    PreIterationOutput,
    enable_rlm_hooks,
)


class TestHookInvocationOrder:
    """Tests that hooks fire in the correct order during iteration."""

    def test_pre_iteration_hook_receives_correct_args(
        self, mock_rlm, mock_repl, mock_history, mock_variables, pre_iteration_hook
    ):
        """Test that pre_iteration hook receives expected arguments."""
        calls = []

        def tracking_hook(iteration, variables, history, input_args):
            calls.append(
                {
                    "iteration": iteration,
                    "variables": variables,
                    "history": history,
                    "input_args": input_args,
                }
            )
            return PreIterationOutput()

        enable_rlm_hooks(mock_rlm, pre_iteration_hook=tracking_hook)

        # Set up the generate_action to return something
        action = MagicMock()
        action.code = "print('hello')"
        action.reasoning = "test"
        mock_rlm.generate_action.return_value = action

        # Mock _process_execution_result to return history
        mock_rlm._process_execution_result.return_value = mock_history

        # Call the patched method
        mock_rlm._execute_iteration(
            mock_repl, mock_variables, mock_history, 0, {"question": "test"}, ["answer"]
        )

        assert len(calls) == 1
        assert calls[0]["iteration"] == 0
        assert calls[0]["variables"] == mock_variables
        assert calls[0]["history"] == mock_history
        assert "question" in calls[0]["input_args"]

    def test_pre_iteration_injects_variables(
        self, mock_rlm, mock_repl, mock_history, mock_variables
    ):
        """Test that pre_iteration hook injects extra_vars into input_args."""

        def inject_hook(iteration, variables, history, input_args):
            return PreIterationOutput(extra_vars={"debug": True, "count": 42})

        enable_rlm_hooks(mock_rlm, pre_iteration_hook=inject_hook)

        action = MagicMock()
        action.code = "print('hello')"
        action.reasoning = "test"
        mock_rlm.generate_action.return_value = action
        mock_rlm._process_execution_result.return_value = mock_history

        mock_rlm._execute_iteration(
            mock_repl, mock_variables, mock_history, 0, {"question": "test"}, ["answer"]
        )

        # The generate_action should have been called with updated input_args
        # We verify by checking that the repl was called with the right input_args
        mock_repl.execute.assert_called_once()
        call_args = mock_repl.execute.call_args
        assert call_args.kwargs["variables"]["debug"] is True
        assert call_args.kwargs["variables"]["count"] == 42

    def test_pre_execution_rewrites_code(
        self, mock_rlm, mock_repl, mock_history, mock_variables
    ):
        """Test that pre_execution hook can rewrite generated code."""

        def rewrite_hook(
            iteration, code, variables, history, input_args, *, raw_code=""
        ):
            return PreExecutionOutput(code=f"# rewritten\n{code}")

        enable_rlm_hooks(mock_rlm, pre_execution_hook=rewrite_hook)

        action = MagicMock()
        action.code = "print('hello')"
        action.reasoning = "test"
        mock_rlm.generate_action.return_value = action
        mock_rlm._process_execution_result.return_value = mock_history

        mock_rlm._execute_iteration(
            mock_repl, mock_variables, mock_history, 0, {"question": "test"}, ["answer"]
        )

        # Check that repl.execute was called with rewritten code
        call_args = mock_repl.execute.call_args
        assert "# rewritten" in call_args.args[0]

    def test_post_execution_transforms_result(
        self, mock_rlm, mock_repl, mock_history, mock_variables
    ):
        """Test that post_execution hook transforms the execution result."""

        def transform_hook(
            iteration, code, result, variables, history, input_args, *, raw_code=""
        ):
            return PostExecutionOutput(result=f"transformed: {result}")

        enable_rlm_hooks(mock_rlm, post_execution_hook=transform_hook)

        action = MagicMock()
        action.code = "print('hello')"
        action.reasoning = "test"
        mock_rlm.generate_action.return_value = action
        mock_rlm._process_execution_result.return_value = mock_history

        mock_rlm._execute_iteration(
            mock_repl, mock_variables, mock_history, 0, {"question": "test"}, ["answer"]
        )

        # Check that _process_execution_result was called with transformed result
        call_args = mock_rlm._process_execution_result.call_args
        assert "transformed:" in str(call_args.args[2])

    def test_post_iteration_modifies_history(
        self, mock_rlm, mock_repl, mock_history, mock_variables
    ):
        """Test that post_iteration hook can modify history."""
        modified_history = MagicMock(spec=REPLHistory)

        def modify_hook(iteration, pred, code, result, history):
            return PostIterationOutput(history=modified_history)

        enable_rlm_hooks(mock_rlm, post_iteration_hook=modify_hook)

        action = MagicMock()
        action.code = "print('hello')"
        action.reasoning = "test"
        mock_rlm.generate_action.return_value = action
        mock_rlm._process_execution_result.return_value = mock_history

        result = mock_rlm._execute_iteration(
            mock_repl, mock_variables, mock_history, 0, {"question": "test"}, ["answer"]
        )

        # Result should be the modified history
        assert result is modified_history

    def test_all_hooks_fire_in_order(
        self, mock_rlm, mock_repl, mock_history, mock_variables
    ):
        """Test that all four hooks fire in the correct lifecycle order."""
        order = []

        def pre_iter(iteration, variables, history, input_args):
            order.append("pre_iteration")
            return PreIterationOutput()

        def pre_exec(iteration, code, variables, history, input_args, *, raw_code=""):
            order.append("pre_execution")
            return PreExecutionOutput(code=code)

        def post_exec(
            iteration, code, result, variables, history, input_args, *, raw_code=""
        ):
            order.append("post_execution")
            return PostExecutionOutput(result=result)

        def post_iter(iteration, pred, code, result, history):
            order.append("post_iteration")
            return PostIterationOutput(history=history)

        enable_rlm_hooks(
            mock_rlm,
            pre_iteration_hook=pre_iter,
            pre_execution_hook=pre_exec,
            post_execution_hook=post_exec,
            post_iteration_hook=post_iter,
        )

        action = MagicMock()
        action.code = "print('hello')"
        action.reasoning = "test"
        mock_rlm.generate_action.return_value = action
        mock_rlm._process_execution_result.return_value = mock_history

        mock_rlm._execute_iteration(
            mock_repl, mock_variables, mock_history, 0, {"question": "test"}, ["answer"]
        )

        assert order == [
            "pre_iteration",
            "pre_execution",
            "post_execution",
            "post_iteration",
        ]


class TestHookEdgeCases:
    """Edge cases for hook behaviour."""

    def test_no_hooks_enabled(self, mock_rlm, mock_repl, mock_history, mock_variables):
        """Test that iteration works normally when no hooks are enabled."""
        enable_rlm_hooks(mock_rlm)  # No hooks

        action = MagicMock()
        action.code = "print('hello')"
        action.reasoning = "test"
        mock_rlm.generate_action.return_value = action
        mock_rlm._process_execution_result.return_value = mock_history

        mock_rlm._execute_iteration(
            mock_repl, mock_variables, mock_history, 0, {"question": "test"}, ["answer"]
        )

        mock_rlm.generate_action.assert_called_once()
        mock_repl.execute.assert_called_once()

    def test_syntax_error_in_code(
        self, mock_rlm, mock_repl, mock_history, mock_variables
    ):
        """Test that syntax errors in generated code are handled."""
        enable_rlm_hooks(mock_rlm)

        action = MagicMock()
        action.code = "```json\nnot python\n```"
        action.reasoning = "test"
        mock_rlm.generate_action.return_value = action
        mock_rlm._process_execution_result.return_value = mock_history

        mock_rlm._execute_iteration(
            mock_repl, mock_variables, mock_history, 0, {"question": "test"}, ["answer"]
        )

        # Should have caught syntax error and returned it as result
        mock_rlm._process_execution_result.assert_called_once()
        call_args = mock_rlm._process_execution_result.call_args
        assert "[Error]" in str(call_args.args[2])

    def test_pre_iteration_code_globals(
        self, mock_rlm, mock_repl, mock_history, mock_variables
    ):
        """Test that pre_iteration python_code is stored in repl_globals."""

        def code_hook(iteration, variables, history, input_args):
            return PreIterationOutput(python_code="import math")

        enable_rlm_hooks(mock_rlm, pre_iteration_hook=code_hook)

        action = MagicMock()
        action.code = "print('hello')"
        action.reasoning = "test"
        mock_rlm.generate_action.return_value = action
        mock_rlm._process_execution_result.return_value = mock_history

        mock_rlm._execute_iteration(
            mock_repl, mock_variables, mock_history, 0, {"question": "test"}, ["answer"]
        )

        # Check that repl_globals was updated
        assert "import math" in mock_repl.repl_globals

    def test_post_iteration_stop_flag_calls_extract_fallback(
        self, mock_rlm, mock_repl, mock_history, mock_variables
    ):
        """Test that post_iteration hook with stop=True triggers extract fallback."""
        from dspy.primitives.prediction import Prediction

        expected_prediction = Prediction(answer="extracted")
        mock_rlm._extract_fallback.return_value = expected_prediction

        def stop_hook(iteration, pred, code, result, history):
            return PostIterationOutput(history=history, stop=True)

        enable_rlm_hooks(mock_rlm, post_iteration_hook=stop_hook)

        action = MagicMock()
        action.code = "print('hello')"
        action.reasoning = "test"
        mock_rlm.generate_action.return_value = action
        mock_rlm._process_execution_result.return_value = mock_history

        result = mock_rlm._execute_iteration(
            mock_repl, mock_variables, mock_history, 0, {"question": "test"}, ["answer"]
        )

        mock_rlm._extract_fallback.assert_called_once_with(
            mock_variables, mock_history, ["answer"]
        )
        assert result is expected_prediction

    def test_post_iteration_stop_flag_false_continues(
        self, mock_rlm, mock_repl, mock_history, mock_variables
    ):
        """Test that post_iteration hook with stop=False returns history normally."""
        modified_history = MagicMock(spec=REPLHistory)

        def continue_hook(iteration, pred, code, result, history):
            return PostIterationOutput(history=modified_history, stop=False)

        enable_rlm_hooks(mock_rlm, post_iteration_hook=continue_hook)

        action = MagicMock()
        action.code = "print('hello')"
        action.reasoning = "test"
        mock_rlm.generate_action.return_value = action
        mock_rlm._process_execution_result.return_value = mock_history

        result = mock_rlm._execute_iteration(
            mock_repl, mock_variables, mock_history, 0, {"question": "test"}, ["answer"]
        )

        mock_rlm._extract_fallback.assert_not_called()
        assert result is modified_history

    def test_post_iteration_not_called_for_prediction(
        self, mock_rlm, mock_repl, mock_history, mock_variables
    ):
        """Test that post_iteration is only called when result is REPLHistory."""
        post_called = [False]

        def post_iter(iteration, pred, code, result, history):
            post_called[0] = True
            return PostIterationOutput(history=history)

        enable_rlm_hooks(mock_rlm, post_iteration_hook=post_iter)

        action = MagicMock()
        action.code = "print('hello')"
        action.reasoning = "test"
        mock_rlm.generate_action.return_value = action
        # Return a Prediction instead of REPLHistory
        mock_rlm._process_execution_result.return_value = MagicMock()

        mock_rlm._execute_iteration(
            mock_repl, mock_variables, mock_history, 0, {"question": "test"}, ["answer"]
        )

        # post_iteration should not be called when result is not REPLHistory
        assert post_called[0] is False


class TestRawCodeParameter:
    """Tests for the raw_code parameter in pre_execution and post_execution hooks."""

    def test_pre_execution_receives_raw_code(
        self, mock_rlm, mock_repl, mock_history, mock_variables
    ):
        """Test that pre_execution hook receives raw_code before fence stripping."""
        raw_codes = []

        def tracking_hook(
            iteration, code, variables, history, input_args, *, raw_code=""
        ):
            raw_codes.append(raw_code)
            return PreExecutionOutput(code=code)

        enable_rlm_hooks(mock_rlm, pre_execution_hook=tracking_hook)

        # Code with fences that will be stripped
        action = MagicMock()
        action.code = "```python\nprint('hello')\n```"
        action.reasoning = "test"
        mock_rlm.generate_action.return_value = action
        mock_rlm._process_execution_result.return_value = mock_history

        mock_rlm._execute_iteration(
            mock_repl, mock_variables, mock_history, 0, {"question": "test"}, ["answer"]
        )

        assert len(raw_codes) == 1
        assert raw_codes[0] == "```python\nprint('hello')\n```"

    def test_post_execution_receives_raw_code(
        self, mock_rlm, mock_repl, mock_history, mock_variables
    ):
        """Test that post_execution hook receives raw_code before fence stripping."""
        raw_codes = []

        def tracking_hook(
            iteration, code, result, variables, history, input_args, *, raw_code=""
        ):
            raw_codes.append(raw_code)
            return PostExecutionOutput(result=result)

        enable_rlm_hooks(mock_rlm, post_execution_hook=tracking_hook)

        # Code with fences that will be stripped
        action = MagicMock()
        action.code = "```python\nprint('hello')\n```"
        action.reasoning = "test"
        mock_rlm.generate_action.return_value = action
        mock_rlm._process_execution_result.return_value = mock_history

        mock_rlm._execute_iteration(
            mock_repl, mock_variables, mock_history, 0, {"question": "test"}, ["answer"]
        )

        assert len(raw_codes) == 1
        assert raw_codes[0] == "```python\nprint('hello')\n```"

    def test_raw_code_preserves_multiple_blocks(
        self, mock_rlm, mock_repl, mock_history, mock_variables
    ):
        """Test that raw_code preserves multiple code blocks that would be lost."""
        raw_codes = []
        stripped_codes = []

        def tracking_hook(
            iteration, code, variables, history, input_args, *, raw_code=""
        ):
            raw_codes.append(raw_code)
            stripped_codes.append(code)
            return PreExecutionOutput(code=code)

        enable_rlm_hooks(mock_rlm, pre_execution_hook=tracking_hook)

        # Code with multiple blocks - only first block survives stripping
        multi_block_code = """```python
x = 1
```
Some explanation
```python
y = 2
```"""
        action = MagicMock()
        action.code = multi_block_code
        action.reasoning = "test"
        mock_rlm.generate_action.return_value = action
        mock_rlm._process_execution_result.return_value = mock_history

        mock_rlm._execute_iteration(
            mock_repl, mock_variables, mock_history, 0, {"question": "test"}, ["answer"]
        )

        assert len(raw_codes) == 1
        # raw_code should have the full multi-block code
        assert raw_codes[0] == multi_block_code
        # stripped code should only have the first block
        assert "x = 1" in stripped_codes[0]
        assert "y = 2" not in stripped_codes[0]

    def test_raw_code_empty_when_no_fences(
        self, mock_rlm, mock_repl, mock_history, mock_variables
    ):
        """Test that raw_code is still provided even when no fences exist."""
        raw_codes = []

        def tracking_hook(
            iteration, code, variables, history, input_args, *, raw_code=""
        ):
            raw_codes.append(raw_code)
            return PreExecutionOutput(code=code)

        enable_rlm_hooks(mock_rlm, pre_execution_hook=tracking_hook)

        # Code without fences
        action = MagicMock()
        action.code = "print('hello')"
        action.reasoning = "test"
        mock_rlm.generate_action.return_value = action
        mock_rlm._process_execution_result.return_value = mock_history

        mock_rlm._execute_iteration(
            mock_repl, mock_variables, mock_history, 0, {"question": "test"}, ["answer"]
        )

        assert len(raw_codes) == 1
        assert raw_codes[0] == "print('hello')"
