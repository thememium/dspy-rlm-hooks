"""Coverage tests for patcher.py and predict_rlm_compat.py fallback branches.

Targets the remaining uncovered lines:

- ``patcher._max_iterations`` raising ``AttributeError`` when neither
  ``max_iters`` nor ``max_iterations`` is present (line 75)
- ``predict_rlm_compat``'s async ``_wrapped_aexecute_iteration`` persistent
  ``repl_globals`` replacement (lines 238-239)
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from dspy_rlm_hooks.core.patcher import _max_iterations
from dspy_rlm_hooks.core.predict_rlm_compat import enable_predict_rlm_hooks
from dspy_rlm_hooks.core.types import PreIterationOutput


class TestMaxIterations:
    def test_max_iterations_raises_when_missing(self):
        """``_max_iterations`` raises when neither iteration-cap attribute exists."""

        class NoMax:
            pass

        with pytest.raises(AttributeError, match="max_iters"):
            _max_iterations(NoMax())


class TestAsyncPersistentPythonCode:
    @pytest.mark.asyncio
    async def test_async_persistent_python_code_sets_repl_globals(self):
        """The async pre-iteration wrapper replaces ``repl.repl_globals`` when the
        hook supplies explicit persistent code."""
        mock = MagicMock(name="PredictRLM_instance")
        mock._execute_iteration = MagicMock(return_value="result")
        mock._aexecute_iteration = AsyncMock(return_value="async_result")
        mock._process_execution_result = MagicMock(return_value="processed_result")
        mock.generate_action = MagicMock()
        mock.generate_action.forward = MagicMock(
            return_value=MagicMock(code="generated code", reasoning="thinking"),
        )
        mock.generate_action.aforward = AsyncMock(
            return_value=MagicMock(code="generated async code", reasoning="thinking"),
        )
        mock.max_iterations = 10
        mock.verbose = False

        def code_hook(iteration, variables, history, input_args):
            return PreIterationOutput(persistent_python_code="import os")

        async def mock_aexecute(
            self,
            repl,
            variables,
            history,
            iteration,
            input_args,
            output_field_names,
            **kw,
        ):
            return "result"

        mock._aexecute_iteration = mock_aexecute
        enable_predict_rlm_hooks(mock, pre_iteration_hook=code_hook)

        repl = MagicMock()
        repl.repl_globals = None
        await mock._aexecute_iteration(  # ty: ignore[missing-argument]
            repl, [], MagicMock(), 0, {}, ["answer"]
        )

        assert repl.repl_globals == "import os"
