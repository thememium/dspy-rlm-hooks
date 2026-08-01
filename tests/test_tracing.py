"""Tests for MLflow tracing wrapper."""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from dspy_rlm_hooks import (
    PostExecutionOutput,
    PostIterationOutput,
    PreExecutionOutput,
    PreIterationOutput,
    enable_rlm_hooks_with_tracing,
)


@pytest.fixture
def mock_mlflow():
    """Mock the mlflow module with a working start_span context manager."""
    mock_span = MagicMock()
    mock_span.set_inputs = MagicMock()
    mock_span.set_outputs = MagicMock()

    @contextmanager
    def mock_start_span(name):
        mock_span.name = name
        yield mock_span

    mock_mlflow_module = MagicMock()
    mock_mlflow_module.start_span = mock_start_span

    with patch(
        "dspy_rlm_hooks.tracing._import_mlflow", return_value=mock_mlflow_module
    ):
        yield mock_mlflow_module, mock_span


class TestTracingPreIteration:
    """Tests for traced pre_iteration hooks."""

    def test_pre_iteration_creates_span(
        self, mock_rlm, mock_repl, mock_history, mock_variables, mock_mlflow
    ):
        """Test that pre_iteration hook creates an MLflow span."""
        mlflow_mod, mock_span = mock_mlflow

        def inject_hook(iteration, variables, history, input_args):
            return PreIterationOutput(extra_vars={"context": "test"})

        enable_rlm_hooks_with_tracing(mock_rlm, pre_iteration_hook=inject_hook)

        action = MagicMock()
        action.code = "print('hello')"
        action.reasoning = "test"
        mock_rlm.generate_action.return_value = action
        mock_rlm._process_execution_result.return_value = mock_history

        mock_rlm._execute_iteration(
            mock_repl, mock_variables, mock_history, 0, {"question": "test"}, ["answer"]
        )

        # Verify span was created with correct name
        assert mock_span.name == "rlm_hook/pre_iteration/0"

    def test_pre_iteration_records_inputs_and_outputs(
        self, mock_rlm, mock_repl, mock_history, mock_variables, mock_mlflow
    ):
        """Test that pre_iteration hook records inputs and outputs on the span."""
        mlflow_mod, mock_span = mock_mlflow

        def inject_hook(iteration, variables, history, input_args):
            return PreIterationOutput(
                extra_vars={"context": "test"},
                python_code="import math",
            )

        enable_rlm_hooks_with_tracing(mock_rlm, pre_iteration_hook=inject_hook)

        action = MagicMock()
        action.code = "print('hello')"
        action.reasoning = "test"
        mock_rlm.generate_action.return_value = action
        mock_rlm._process_execution_result.return_value = mock_history

        mock_rlm._execute_iteration(
            mock_repl, mock_variables, mock_history, 0, {"question": "test"}, ["answer"]
        )

        # Verify inputs were recorded
        mock_span.set_inputs.assert_called_once()
        inputs = mock_span.set_inputs.call_args[0][0]
        assert inputs["iteration"] == 0
        assert "question" in inputs["input_args"]

        # Verify outputs were recorded
        mock_span.set_outputs.assert_called_once()
        outputs = mock_span.set_outputs.call_args[0][0]
        assert outputs["extra_vars"]["context"] == "test"
        assert outputs["python_code"] == "import math"

    def test_pre_iteration_still_injects_variables(
        self, mock_rlm, mock_repl, mock_history, mock_variables, mock_mlflow
    ):
        """Test that tracing wrapper preserves variable injection behavior."""
        mlflow_mod, mock_span = mock_mlflow

        def inject_hook(iteration, variables, history, input_args):
            return PreIterationOutput(extra_vars={"debug": True, "count": 42})

        enable_rlm_hooks_with_tracing(mock_rlm, pre_iteration_hook=inject_hook)

        action = MagicMock()
        action.code = "print('hello')"
        action.reasoning = "test"
        mock_rlm.generate_action.return_value = action
        mock_rlm._process_execution_result.return_value = mock_history

        mock_rlm._execute_iteration(
            mock_repl, mock_variables, mock_history, 0, {"question": "test"}, ["answer"]
        )

        # Verify variables were actually injected into execution
        mock_repl.execute.assert_called_once()
        call_args = mock_repl.execute.call_args
        assert call_args.kwargs["variables"]["debug"] is True
        assert call_args.kwargs["variables"]["count"] == 42


class TestTracingPreExecution:
    """Tests for traced pre_execution hooks."""

    def test_pre_execution_creates_span(
        self, mock_rlm, mock_repl, mock_history, mock_variables, mock_mlflow
    ):
        """Test that pre_execution hook creates an MLflow span."""
        mlflow_mod, mock_span = mock_mlflow

        def rewrite_hook(iteration, code, variables, history, input_args):
            return PreExecutionOutput(code=f"# modified\n{code}")

        enable_rlm_hooks_with_tracing(mock_rlm, pre_execution_hook=rewrite_hook)

        action = MagicMock()
        action.code = "print('hello')"
        action.reasoning = "test"
        mock_rlm.generate_action.return_value = action
        mock_rlm._process_execution_result.return_value = mock_history

        mock_rlm._execute_iteration(
            mock_repl, mock_variables, mock_history, 0, {"question": "test"}, ["answer"]
        )

        assert mock_span.name == "rlm_hook/pre_execution/0"

    def test_pre_execution_records_inputs_and_outputs(
        self, mock_rlm, mock_repl, mock_history, mock_variables, mock_mlflow
    ):
        """Test that pre_execution hook records original and modified code."""
        mlflow_mod, mock_span = mock_mlflow

        def rewrite_hook(iteration, code, variables, history, input_args):
            return PreExecutionOutput(code=f"# modified\n{code}")

        enable_rlm_hooks_with_tracing(mock_rlm, pre_execution_hook=rewrite_hook)

        action = MagicMock()
        action.code = "print('hello')"
        action.reasoning = "test"
        mock_rlm.generate_action.return_value = action
        mock_rlm._process_execution_result.return_value = mock_history

        mock_rlm._execute_iteration(
            mock_repl, mock_variables, mock_history, 0, {"question": "test"}, ["answer"]
        )

        inputs = mock_span.set_inputs.call_args[0][0]
        assert inputs["original_code"] == "print('hello')"

        outputs = mock_span.set_outputs.call_args[0][0]
        assert "# modified" in outputs["modified_code"]

    def test_pre_execution_still_rewrites_code(
        self, mock_rlm, mock_repl, mock_history, mock_variables, mock_mlflow
    ):
        """Test that tracing wrapper preserves code rewriting behavior."""
        mlflow_mod, mock_span = mock_mlflow

        def rewrite_hook(iteration, code, variables, history, input_args):
            return PreExecutionOutput(code=f"# rewritten\n{code}")

        enable_rlm_hooks_with_tracing(mock_rlm, pre_execution_hook=rewrite_hook)

        action = MagicMock()
        action.code = "print('hello')"
        action.reasoning = "test"
        mock_rlm.generate_action.return_value = action
        mock_rlm._process_execution_result.return_value = mock_history

        mock_rlm._execute_iteration(
            mock_repl, mock_variables, mock_history, 0, {"question": "test"}, ["answer"]
        )

        call_args = mock_repl.execute.call_args
        assert "# rewritten" in call_args.args[0]


class TestTracingPostExecution:
    """Tests for traced post_execution hooks."""

    def test_post_execution_creates_span(
        self, mock_rlm, mock_repl, mock_history, mock_variables, mock_mlflow
    ):
        """Test that post_execution hook creates an MLflow span."""
        mlflow_mod, mock_span = mock_mlflow

        def transform_hook(iteration, code, result, variables, history, input_args):
            return PostExecutionOutput(result=f"transformed: {result}")

        enable_rlm_hooks_with_tracing(mock_rlm, post_execution_hook=transform_hook)

        action = MagicMock()
        action.code = "print('hello')"
        action.reasoning = "test"
        mock_rlm.generate_action.return_value = action
        mock_rlm._process_execution_result.return_value = mock_history

        mock_rlm._execute_iteration(
            mock_repl, mock_variables, mock_history, 0, {"question": "test"}, ["answer"]
        )

        assert mock_span.name == "rlm_hook/post_execution/0"

    def test_post_execution_records_inputs_and_outputs(
        self, mock_rlm, mock_repl, mock_history, mock_variables, mock_mlflow
    ):
        """Test that post_execution hook records original and final result."""
        mlflow_mod, mock_span = mock_mlflow

        def transform_hook(iteration, code, result, variables, history, input_args):
            return PostExecutionOutput(result=f"transformed: {result}")

        enable_rlm_hooks_with_tracing(mock_rlm, post_execution_hook=transform_hook)

        action = MagicMock()
        action.code = "print('hello')"
        action.reasoning = "test"
        mock_rlm.generate_action.return_value = action
        mock_rlm._process_execution_result.return_value = mock_history

        mock_rlm._execute_iteration(
            mock_repl, mock_variables, mock_history, 0, {"question": "test"}, ["answer"]
        )

        inputs = mock_span.set_inputs.call_args[0][0]
        assert inputs["code"] == "print('hello')"

        outputs = mock_span.set_outputs.call_args[0][0]
        assert "transformed:" in outputs["final_result"]


class TestTracingPostIteration:
    """Tests for traced post_iteration hooks."""

    def test_post_iteration_creates_span(
        self, mock_rlm, mock_repl, mock_history, mock_variables, mock_mlflow
    ):
        """Test that post_iteration hook creates an MLflow span."""
        mlflow_mod, mock_span = mock_mlflow

        def post_iter_hook(iteration, pred, code, result, history):
            return PostIterationOutput(history=history)

        enable_rlm_hooks_with_tracing(mock_rlm, post_iteration_hook=post_iter_hook)

        action = MagicMock()
        action.code = "print('hello')"
        action.reasoning = "test"
        mock_rlm.generate_action.return_value = action
        mock_rlm._process_execution_result.return_value = mock_history

        mock_rlm._execute_iteration(
            mock_repl, mock_variables, mock_history, 0, {"question": "test"}, ["answer"]
        )

        assert mock_span.name == "rlm_hook/post_iteration/0"

    def test_post_iteration_records_stop_flag(
        self, mock_rlm, mock_repl, mock_history, mock_variables, mock_mlflow
    ):
        """Test that post_iteration hook records the stop flag."""
        mlflow_mod, mock_span = mock_mlflow

        def post_iter_hook(iteration, pred, code, result, history):
            return PostIterationOutput(history=history, stop=True)

        enable_rlm_hooks_with_tracing(mock_rlm, post_iteration_hook=post_iter_hook)

        action = MagicMock()
        action.code = "print('hello')"
        action.reasoning = "test"
        mock_rlm.generate_action.return_value = action
        mock_rlm._process_execution_result.return_value = mock_history
        mock_rlm._extract_fallback.return_value = MagicMock()

        mock_rlm._execute_iteration(
            mock_repl, mock_variables, mock_history, 0, {"question": "test"}, ["answer"]
        )

        outputs = mock_span.set_outputs.call_args[0][0]
        assert outputs["stop"] is True


class TestTracingIterationNumbering:
    """Tests that span names include correct iteration numbers."""

    def test_iteration_number_in_span_name(
        self, mock_rlm, mock_repl, mock_history, mock_variables, mock_mlflow
    ):
        """Test that iteration number appears in the span name."""
        mlflow_mod, mock_span = mock_mlflow
        span_names = []

        original_start_span = mlflow_mod.start_span

        @contextmanager
        def tracking_start_span(name):
            span_names.append(name)
            with original_start_span(name) as span:
                yield span

        mlflow_mod.start_span = tracking_start_span

        def hook(iteration, variables, history, input_args):
            return PreIterationOutput()

        enable_rlm_hooks_with_tracing(mock_rlm, pre_iteration_hook=hook)

        action = MagicMock()
        action.code = "print('hello')"
        action.reasoning = "test"
        mock_rlm.generate_action.return_value = action
        mock_rlm._process_execution_result.return_value = mock_history

        # Run iteration 0
        mock_rlm._execute_iteration(
            mock_repl, mock_variables, mock_history, 0, {"question": "test"}, ["answer"]
        )
        # Run iteration 3
        mock_rlm._execute_iteration(
            mock_repl, mock_variables, mock_history, 3, {"question": "test"}, ["answer"]
        )

        assert "rlm_hook/pre_iteration/0" in span_names
        assert "rlm_hook/pre_iteration/3" in span_names


class TestTracingAllHooks:
    """Tests for all hooks enabled simultaneously with tracing."""

    def test_all_hooks_create_spans(
        self, mock_rlm, mock_repl, mock_history, mock_variables, mock_mlflow
    ):
        """Test that all four hooks create their respective spans."""
        mlflow_mod, mock_span = mock_mlflow
        span_names = []

        original_start_span = mlflow_mod.start_span

        @contextmanager
        def tracking_start_span(name):
            span_names.append(name)
            with original_start_span(name) as span:
                yield span

        mlflow_mod.start_span = tracking_start_span

        enable_rlm_hooks_with_tracing(
            mock_rlm,
            pre_iteration_hook=lambda iteration, variables, history, input_args: (
                PreIterationOutput()
            ),
            pre_execution_hook=lambda iteration, code, variables, history, input_args: (
                PreExecutionOutput(code=code)
            ),
            post_execution_hook=lambda iteration, code, result, variables, history, input_args: (
                PostExecutionOutput(result=result)
            ),
            post_iteration_hook=lambda iteration, pred, code, result, history: (
                PostIterationOutput(history=history)
            ),
        )

        action = MagicMock()
        action.code = "print('hello')"
        action.reasoning = "test"
        mock_rlm.generate_action.return_value = action
        mock_rlm._process_execution_result.return_value = mock_history

        mock_rlm._execute_iteration(
            mock_repl, mock_variables, mock_history, 0, {"question": "test"}, ["answer"]
        )

        assert "rlm_hook/pre_iteration/0" in span_names
        assert "rlm_hook/pre_execution/0" in span_names
        assert "rlm_hook/post_execution/0" in span_names
        assert "rlm_hook/post_iteration/0" in span_names


class TestTracingImportError:
    """Tests for MLflow import error handling."""

    def test_raises_import_error_when_mlflow_missing(self, mock_rlm):
        """Test that enable_rlm_hooks_with_tracing raises ImportError when mlflow is not installed."""
        with patch(
            "dspy_rlm_hooks.tracing._import_mlflow",
            side_effect=ImportError("mlflow is required"),
        ):
            with pytest.raises(ImportError, match="mlflow is required"):
                enable_rlm_hooks_with_tracing(
                    mock_rlm,
                    pre_iteration_hook=lambda iteration, variables, history, input_args: (
                        PreIterationOutput()
                    ),
                )

    def test_no_hooks_still_requires_mlflow(self, mock_rlm):
        """Test that even with no hooks, mlflow import is validated."""
        with patch(
            "dspy_rlm_hooks.tracing._import_mlflow",
            side_effect=ImportError("mlflow is required"),
        ):
            with pytest.raises(ImportError, match="mlflow is required"):
                enable_rlm_hooks_with_tracing(mock_rlm)
