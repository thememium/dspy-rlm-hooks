"""Tests for MLflow tracing wrapper."""

from __future__ import annotations

import inspect
import sys
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from dspy_rlm_hooks import (
    PostExecutionOutput,
    PostIterationOutput,
    PreExecutionOutput,
    PreIterationOutput,
    enable_rlm_hooks,
    enable_rlm_hooks_with_tracing,
)
from dspy_rlm_hooks.tracing import (
    _ensure_type,
    _import_mlflow,
    _is_mlflow_tracing_available,
    _load_mlflow,
    _safe_serialize,
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
        assert mock_span.name == "rlm_hook/pre_iteration"

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

        assert mock_span.name == "rlm_hook/pre_execution"

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

        assert mock_span.name == "rlm_hook/post_execution"

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

        assert mock_span.name == "rlm_hook/post_iteration"

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


class TestTracingIterationNaming:
    """Tests that span names remain stable across iterations."""

    def test_iteration_number_omitted_from_span_name(
        self, mock_rlm, mock_repl, mock_history, mock_variables, mock_mlflow
    ):
        """Test that iteration numbers do not appear in the span name."""
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

        assert span_names == [
            "rlm_hook/pre_iteration",
            "rlm_hook/pre_iteration",
        ]
        recorded_iterations = [
            call.args[0]["iteration"] for call in mock_span.set_inputs.call_args_list
        ]
        assert recorded_iterations == [0, 3]


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

        assert "rlm_hook/pre_iteration" in span_names
        assert "rlm_hook/pre_execution" in span_names
        assert "rlm_hook/post_execution" in span_names
        assert "rlm_hook/post_iteration" in span_names


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


class TestSafeSerialize:
    """Tests for _safe_serialize helper."""

    def test_none_passthrough(self):
        """Test that None passes through unchanged."""
        assert _safe_serialize(None) is None

    def test_bool_passthrough(self):
        """Test that booleans pass through unchanged."""
        assert _safe_serialize(True) is True
        assert _safe_serialize(False) is False

    def test_int_passthrough(self):
        """Test that integers pass through unchanged."""
        assert _safe_serialize(42) == 42
        assert _safe_serialize(0) == 0
        assert _safe_serialize(-1) == -1

    def test_float_passthrough(self):
        """Test that floats pass through unchanged."""
        assert _safe_serialize(3.14) == 3.14
        assert _safe_serialize(0.0) == 0.0

    def test_string_passthrough(self):
        """Test that strings pass through unchanged."""
        assert _safe_serialize("hello") == "hello"
        assert _safe_serialize("") == ""

    def test_dict_recursive(self):
        """Test that dicts are serialized recursively."""
        result = _safe_serialize({"key": "value", "num": 42})
        assert result == {"key": "value", "num": 42}

    def test_dict_with_nested_complex(self):
        """Test that dicts with complex values use repr."""
        result = _safe_serialize({"obj": object()})
        assert "object" in result["obj"]

    def test_list_recursive(self):
        """Test that lists are serialized recursively."""
        result = _safe_serialize([1, "two", None])
        assert result == [1, "two", None]

    def test_tuple_converted_to_list(self):
        """Test that tuples are converted to lists."""
        result = _safe_serialize((1, 2, 3))
        assert result == [1, 2, 3]
        assert isinstance(result, list)

    def test_complex_object_uses_repr(self):
        """Test that complex objects fall back to repr."""

        class CustomObj:
            def __repr__(self):
                return "CustomObj()"

        result = _safe_serialize(CustomObj())
        assert result == "CustomObj()"

    def test_nested_dict_and_list(self):
        """Test deeply nested structures."""
        data = {"a": [1, {"b": (2, 3)}]}
        result = _safe_serialize(data)
        assert result == {"a": [1, {"b": [2, 3]}]}


class TestEnsureType:
    """Tests for _ensure_type helper."""

    def test_matching_type_returns_value(self):
        """Test that a correctly-typed value passes through."""
        value = PreIterationOutput(extra_vars={"x": 1})
        result = _ensure_type(value, PreIterationOutput)
        assert result is value

    def test_mismatched_type_logs_warning_and_returns_value(self, caplog):
        """Test that a mismatched type logs a warning and returns as-is."""
        import logging

        with caplog.at_level(logging.WARNING, logger="dspy_rlm_hooks.tracing"):
            result = _ensure_type("not an output", PreIterationOutput)

        assert result == "not an output"
        assert "expected PreIterationOutput" in caplog.text

    def test_none_with_non_none_type_logs_warning(self, caplog):
        """Test that None passed to a non-None type logs warning."""
        import logging

        with caplog.at_level(logging.WARNING, logger="dspy_rlm_hooks.tracing"):
            result = _ensure_type(None, PreExecutionOutput)

        assert result is None
        assert "expected PreExecutionOutput" in caplog.text


class TestTracingAsyncPreIteration:
    """Tests for async traced pre_iteration hooks."""

    def test_async_pre_iteration_hook_returns_async_traced(
        self, mock_rlm, mock_repl, mock_history, mock_variables, mock_mlflow
    ):
        """Test that an async pre_iteration hook is wrapped with async_traced."""
        mlflow_mod, mock_span = mock_mlflow

        async def async_hook(iteration, variables, history, input_args):
            return PreIterationOutput(extra_vars={"async": True})

        enable_rlm_hooks_with_tracing(mock_rlm, pre_iteration_hook=async_hook)

        action = MagicMock()
        action.code = "print('hello')"
        action.reasoning = "test"
        mock_rlm.generate_action.return_value = action
        mock_rlm._process_execution_result.return_value = mock_history

        mock_rlm._execute_iteration(
            mock_repl, mock_variables, mock_history, 0, {"question": "test"}, ["answer"]
        )

        assert mock_span.name == "rlm_hook/pre_iteration"
        outputs = mock_span.set_outputs.call_args[0][0]
        assert outputs["extra_vars"]["async"] is True


class TestTracingAsyncPreExecution:
    """Tests for async traced pre_execution hooks."""

    def test_async_pre_execution_hook_returns_async_traced(
        self, mock_rlm, mock_repl, mock_history, mock_variables, mock_mlflow
    ):
        """Test that an async pre_execution hook is wrapped with async_traced."""
        mlflow_mod, mock_span = mock_mlflow

        async def async_hook(iteration, code, variables, history, input_args):
            return PreExecutionOutput(code=f"# async modified\n{code}")

        enable_rlm_hooks_with_tracing(mock_rlm, pre_execution_hook=async_hook)

        action = MagicMock()
        action.code = "print('hello')"
        action.reasoning = "test"
        mock_rlm.generate_action.return_value = action
        mock_rlm._process_execution_result.return_value = mock_history

        mock_rlm._execute_iteration(
            mock_repl, mock_variables, mock_history, 0, {"question": "test"}, ["answer"]
        )

        assert mock_span.name == "rlm_hook/pre_execution"
        outputs = mock_span.set_outputs.call_args[0][0]
        assert "# async modified" in outputs["modified_code"]


class TestTracingAsyncPostExecution:
    """Tests for async traced post_execution hooks."""

    def test_async_post_execution_hook_returns_async_traced(
        self, mock_rlm, mock_repl, mock_history, mock_variables, mock_mlflow
    ):
        """Test that an async post_execution hook is wrapped with async_traced."""
        mlflow_mod, mock_span = mock_mlflow

        async def async_hook(iteration, code, result, variables, history, input_args):
            return PostExecutionOutput(result=f"async transformed: {result}")

        enable_rlm_hooks_with_tracing(mock_rlm, post_execution_hook=async_hook)

        action = MagicMock()
        action.code = "print('hello')"
        action.reasoning = "test"
        mock_rlm.generate_action.return_value = action
        mock_rlm._process_execution_result.return_value = mock_history

        mock_rlm._execute_iteration(
            mock_repl, mock_variables, mock_history, 0, {"question": "test"}, ["answer"]
        )

        assert mock_span.name == "rlm_hook/post_execution"
        outputs = mock_span.set_outputs.call_args[0][0]
        assert "async transformed:" in outputs["final_result"]


class TestTracingAsyncPostIteration:
    """Tests for async traced post_iteration hooks."""

    def test_async_post_iteration_hook_returns_async_traced(
        self, mock_rlm, mock_repl, mock_history, mock_variables, mock_mlflow
    ):
        """Test that an async post_iteration hook is wrapped with async_traced."""
        mlflow_mod, mock_span = mock_mlflow

        async def async_hook(iteration, pred, code, result, history):
            return PostIterationOutput(history=history, stop=False)

        enable_rlm_hooks_with_tracing(mock_rlm, post_iteration_hook=async_hook)

        action = MagicMock()
        action.code = "print('hello')"
        action.reasoning = "test"
        mock_rlm.generate_action.return_value = action
        mock_rlm._process_execution_result.return_value = mock_history

        mock_rlm._execute_iteration(
            mock_repl, mock_variables, mock_history, 0, {"question": "test"}, ["answer"]
        )

        assert mock_span.name == "rlm_hook/post_iteration"
        outputs = mock_span.set_outputs.call_args[0][0]
        assert outputs["stop"] is False

    def test_async_post_iteration_with_stop(
        self, mock_rlm, mock_repl, mock_history, mock_variables, mock_mlflow
    ):
        """Test that async post_iteration hook with stop=True is recorded."""
        mlflow_mod, mock_span = mock_mlflow

        async def async_hook(iteration, pred, code, result, history):
            return PostIterationOutput(history=history, stop=True)

        enable_rlm_hooks_with_tracing(mock_rlm, post_iteration_hook=async_hook)

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


class TestAutomaticTracing:
    """Tests for runtime MLflow detection in the main public API."""

    def test_enable_rlm_hooks_uses_tracing_when_mlflow_is_available(
        self, mock_rlm, mock_repl, mock_history, mock_variables, mock_mlflow
    ):
        _, mock_span = mock_mlflow

        def hook(iteration, variables, history, input_args):
            return PreIterationOutput(extra_vars={"traced": True})

        with patch(
            "dspy_rlm_hooks._is_mlflow_tracing_available",
            return_value=True,
        ):
            enable_rlm_hooks(mock_rlm, pre_iteration_hook=hook)

        action = MagicMock(code="print('hello')", reasoning="test")
        mock_rlm.generate_action.return_value = action
        mock_rlm._process_execution_result.return_value = mock_history

        mock_rlm._execute_iteration(
            mock_repl,
            mock_variables,
            mock_history,
            0,
            {"question": "test"},
            ["answer"],
        )

        assert mock_span.name == "rlm_hook/pre_iteration"
        assert mock_repl.execute.call_args.kwargs["variables"]["traced"] is True

    def test_enable_rlm_hooks_falls_back_when_mlflow_is_missing(self, mock_rlm):
        def hook(iteration, variables, history, input_args):
            return PreIterationOutput()

        with patch(
            "dspy_rlm_hooks._is_mlflow_tracing_available",
            return_value=False,
        ):
            enable_rlm_hooks(mock_rlm, pre_iteration_hook=hook)

        assert mock_rlm._hook_pre_iteration is hook

    def test_detection_requires_mlflow_span_api(self):
        mlflow_without_tracing = MagicMock(spec=[])
        with patch(
            "dspy_rlm_hooks.tracing._load_mlflow",
            return_value=mlflow_without_tracing,
        ):
            assert _is_mlflow_tracing_available() is False

    def test_detection_accepts_mlflow_span_api(self):
        mlflow_with_tracing = MagicMock()
        mlflow_with_tracing.start_span = MagicMock()
        with patch(
            "dspy_rlm_hooks.tracing._load_mlflow",
            return_value=mlflow_with_tracing,
        ):
            assert _is_mlflow_tracing_available() is True

    def test_explicit_import_rejects_mlflow_without_span_api(self):
        mlflow_without_tracing = MagicMock(spec=[])
        with (
            patch(
                "dspy_rlm_hooks.tracing._load_mlflow",
                return_value=mlflow_without_tracing,
            ),
            pytest.raises(ImportError, match="mlflow>=2.14.0"),
        ):
            _import_mlflow()

    def test_load_mlflow_returns_none_when_package_is_missing(self):
        with patch.dict(sys.modules, {"mlflow": None}):
            assert _load_mlflow() is None

    def test_load_mlflow_preserves_transitive_import_errors(self):
        error = ModuleNotFoundError("missing dependency", name="mlflow_dependency")
        with (
            patch("builtins.__import__", side_effect=error),
            pytest.raises(ModuleNotFoundError, match="missing dependency"),
        ):
            _load_mlflow()

    def test_public_enable_signature_matches_non_tracing_implementation(self):
        from dspy_rlm_hooks.patcher import enable_rlm_hooks as non_tracing_enable

        assert inspect.signature(enable_rlm_hooks) == inspect.signature(
            non_tracing_enable
        )
