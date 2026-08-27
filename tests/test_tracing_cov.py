"""Coverage tests for tracing.py edge/fallback branches.

Targets the remaining uncovered lines:

- ``_load_mlflow`` success path (line 70)
- ``_import_mlflow`` missing-module ImportError (line 77) and success path (line 84)
- the sync ``traced`` wrappers' ``run_until_complete`` branch for each hook type
  (lines 154, 206, 260, 315) — reached when a *sync* hook returns a coroutine.
"""

from __future__ import annotations

import asyncio
import sys
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
from dspy_rlm_hooks.tracing import _import_mlflow, _load_mlflow


@pytest.fixture
def mock_mlflow():
    """Mock MLflow while keeping hook and execution spans independently inspectable."""
    mock_span = MagicMock()
    mock_span.set_inputs = MagicMock()
    mock_span.set_outputs = MagicMock()
    created_spans = []

    @contextmanager
    def mock_start_span(name):
        if name.startswith("rlm_hook/"):
            span = mock_span
        else:
            span = MagicMock()
            span.set_inputs = MagicMock()
            span.set_outputs = MagicMock()
        span.name = name
        created_spans.append(span)
        yield span

    mock_mlflow_module = MagicMock()
    mock_mlflow_module.start_span = mock_start_span
    mock_mlflow_module.created_spans = created_spans

    with patch(
        "dspy_rlm_hooks.tracing._import_mlflow", return_value=mock_mlflow_module
    ):
        yield mock_mlflow_module, mock_span


class TestLoadMlflowSuccess:
    def test_load_mlflow_returns_imported_module(self):
        """The success path returns the imported ``mlflow`` module."""
        fake = MagicMock()
        with patch.dict(sys.modules, {"mlflow": fake}):
            assert _load_mlflow() is fake


class TestImportMlflow:
    def test_import_mlflow_raises_when_missing(self):
        """``_import_mlflow`` raises a clear ImportError when mlflow is absent."""
        with patch("dspy_rlm_hooks.tracing._load_mlflow", return_value=None):
            with pytest.raises(ImportError, match="mlflow is required"):
                _import_mlflow()

    def test_import_mlflow_returns_module_with_span_api(self):
        """``_import_mlflow`` returns the module when it exposes ``start_span``."""
        mlflow_mod = MagicMock()
        mlflow_mod.start_span = MagicMock()
        with patch("dspy_rlm_hooks.tracing._load_mlflow", return_value=mlflow_mod):
            assert _import_mlflow() is mlflow_mod


@pytest.fixture
def current_loop():
    """Provide a current (non-running) event loop so the sync traced wrappers'
    ``asyncio.get_event_loop().run_until_complete(...)`` branch can run."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()
    asyncio.set_event_loop(None)


class TestSyncHookReturningCoroutine:
    """Sync hooks that return a coroutine hit the ``run_until_complete`` branch."""

    def test_pre_iteration_sync_hook_returning_coroutine(
        self,
        current_loop,
        mock_rlm,
        mock_repl,
        mock_history,
        mock_variables,
        mock_mlflow,
    ):
        mlflow_mod, mock_span = mock_mlflow

        def hook(iteration, variables, history, input_args):
            async def inner():
                return PreIterationOutput(extra_vars={"x": 1})

            return inner()

        enable_rlm_hooks_with_tracing(mock_rlm, pre_iteration_hook=hook)
        action = MagicMock(code="print('hello')", reasoning="test")
        mock_rlm.generate_action.return_value = action
        mock_rlm._process_execution_result.return_value = mock_history
        mock_rlm._execute_iteration(
            mock_repl, mock_variables, mock_history, 0, {"question": "test"}, ["answer"]
        )
        outputs = mock_span.set_outputs.call_args[0][0]
        assert outputs == {"extra_vars": {"x": 1}}

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_pre_execution_sync_hook_returning_coroutine(
        self,
        current_loop,
        mock_rlm,
        mock_repl,
        mock_history,
        mock_variables,
        mock_mlflow,
    ):
        mlflow_mod, mock_span = mock_mlflow

        def hook(iteration, code, variables, history, input_args):
            async def inner():
                return PreExecutionOutput(code=f"# mod\n{code}")

            return inner()

        enable_rlm_hooks_with_tracing(mock_rlm, pre_execution_hook=hook)
        action = MagicMock(code="print('hello')", reasoning="test")
        mock_rlm.generate_action.return_value = action
        mock_rlm._process_execution_result.return_value = mock_history
        mock_rlm._execute_iteration(
            mock_repl, mock_variables, mock_history, 0, {"question": "test"}, ["answer"]
        )
        outputs = mock_span.set_outputs.call_args[0][0]
        assert outputs["modified_code"] == "```python\n# mod\nprint('hello')\n```"

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_post_execution_sync_hook_returning_coroutine(
        self,
        current_loop,
        mock_rlm,
        mock_repl,
        mock_history,
        mock_variables,
        mock_mlflow,
    ):
        mlflow_mod, mock_span = mock_mlflow

        def hook(iteration, code, result, variables, history, input_args):
            async def inner():
                return PostExecutionOutput(result=f"transformed: {result}")

            return inner()

        enable_rlm_hooks_with_tracing(mock_rlm, post_execution_hook=hook)
        action = MagicMock(code="print('hello')", reasoning="test")
        mock_rlm.generate_action.return_value = action
        mock_rlm._process_execution_result.return_value = mock_history
        mock_rlm._execute_iteration(
            mock_repl, mock_variables, mock_history, 0, {"question": "test"}, ["answer"]
        )
        outputs = mock_span.set_outputs.call_args[0][0]
        assert "transformed:" in outputs["final_result"]

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_post_iteration_sync_hook_returning_coroutine(
        self,
        current_loop,
        mock_rlm,
        mock_repl,
        mock_history,
        mock_variables,
        mock_mlflow,
    ):
        mlflow_mod, mock_span = mock_mlflow

        def hook(iteration, pred, code, result, history):
            async def inner():
                return PostIterationOutput(history=history, stop=True)

            return inner()

        enable_rlm_hooks_with_tracing(mock_rlm, post_iteration_hook=hook)
        action = MagicMock(code="print('hello')", reasoning="test")
        mock_rlm.generate_action.return_value = action
        mock_rlm._process_execution_result.return_value = mock_history
        mock_rlm._extract_fallback.return_value = MagicMock()
        mock_rlm._execute_iteration(
            mock_repl, mock_variables, mock_history, 0, {"question": "test"}, ["answer"]
        )
        outputs = mock_span.set_outputs.call_args[0][0]
        assert outputs["stop"] is True
