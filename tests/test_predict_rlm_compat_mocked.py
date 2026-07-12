"""Tests for predict_rlm_compat module that work without predict-rlm installed.

Uses mocking to simulate the predict_rlm module and test all code paths.
"""

from __future__ import annotations

import asyncio
import sys
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from dspy.primitives.repl_types import REPLHistory

from dspy_rlm_hooks.predict_rlm_compat import (
    _is_predict_rlm,
    _run_async,
    _StopIteration,
    disable_predict_rlm_hooks,
    enable_predict_rlm_hooks,
)
from dspy_rlm_hooks.types import (
    PostExecutionOutput,
    PostIterationOutput,
    PreExecutionOutput,
    PreIterationOutput,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_predict_rlm_instance():
    """Create a mock that looks like a PredictRLM instance."""
    mock = MagicMock(name="PredictRLM_instance")
    mock._execute_iteration = MagicMock(return_value="result")
    mock._aexecute_iteration = AsyncMock(return_value="async_result")
    mock._process_execution_result = MagicMock(return_value="processed_result")
    mock.generate_action = MagicMock()
    mock.generate_action.forward = MagicMock(
        return_value=MagicMock(code="generated code", reasoning="thinking"),
    )
    mock.max_iterations = 10
    mock.verbose = False
    return mock


@pytest.fixture(autouse=True)
def patch_is_predict_rlm(monkeypatch):
    """Auto-patch _is_predict_rlm to return True for all tests."""
    monkeypatch.setattr(
        "dspy_rlm_hooks.predict_rlm_compat._is_predict_rlm",
        lambda _obj: True,
    )


# ---------------------------------------------------------------------------
# _run_async
# ---------------------------------------------------------------------------


class TestRunAsync:
    """Tests for the _run_async helper."""

    def test_run_async_no_running_loop(self):
        """Test _run_async when no event loop is running (uses asyncio.run)."""
        result = _run_async(async_return("test_value"))
        assert result == "test_value"

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_run_async_with_running_loop_creates_new(self):
        """Test _run_async when an event loop is already running.

        The function detects a running loop and creates a new one.
        We can't easily test this from inside an async context because
        pytest-asyncio uses its own loop. Instead we mock get_running_loop.
        """

        async def coro():
            return "new_loop_result"

        # Simulate a running loop by making get_running_loop not raise
        with patch("dspy_rlm_hooks.predict_rlm_compat.asyncio") as mock_asyncio:
            mock_asyncio.get_running_loop = MagicMock()  # no RuntimeError
            mock_asyncio.new_event_loop = MagicMock()
            mock_loop = MagicMock()
            mock_asyncio.new_event_loop.return_value = mock_loop
            mock_loop.run_until_complete.return_value = "new_loop_result"

            result = _run_async(coro())

            assert result == "new_loop_result"
            mock_loop.run_until_complete.assert_called_once()
            mock_loop.close.assert_called_once()


async def async_return(value):
    """Simple async helper that returns a value."""
    return value


# ---------------------------------------------------------------------------
# _is_predict_rlm
# ---------------------------------------------------------------------------


class TestIsPredictRLM:
    """Tests for _is_predict_rlm detection."""

    def test_is_predict_rlm_false_for_regular_mock(self):
        """Test returns False for a regular MagicMock (no predict_rlm installed)."""
        with patch.dict(sys.modules, {"predict_rlm": None}):
            result = _is_predict_rlm(MagicMock())
            assert result is False

    def test_is_predict_rlm_false_for_none(self):
        """Test returns False for None."""
        with patch.dict(sys.modules, {"predict_rlm": None}):
            result = _is_predict_rlm(None)
            assert result is False

    def test_is_predict_rlm_true_for_predict_rlm_instance(self):
        """Test returns True for a PredictRLM instance when predict_rlm is available."""
        # Create a fake predict_rlm module with a PredictRLM class
        fake_module = ModuleType("predict_rlm")
        fake_PredictRLM = type("PredictRLM", (), {})
        setattr(fake_module, "PredictRLM", fake_PredictRLM)

        with patch.dict(sys.modules, {"predict_rlm": fake_module}):
            # Create an instance of the fake PredictRLM
            instance = fake_PredictRLM()
            result = _is_predict_rlm(instance)
            assert result is True


# ---------------------------------------------------------------------------
# enable_predict_rlm_hooks
# ---------------------------------------------------------------------------


class TestEnablePredictRLMHooks:
    """Tests for enable_predict_rlm_hooks."""

    def test_enable_stores_hook_references(self, mock_predict_rlm_instance):
        """Test that enable stores all four hook references."""
        pre_iter = MagicMock()
        pre_exec = MagicMock()
        post_exec = MagicMock()
        post_iter = MagicMock()

        enable_predict_rlm_hooks(
            mock_predict_rlm_instance,
            pre_iteration_hook=pre_iter,
            pre_execution_hook=pre_exec,
            post_execution_hook=post_exec,
            post_iteration_hook=post_iter,
        )

        assert mock_predict_rlm_instance._hook_pre_iteration is pre_iter
        assert mock_predict_rlm_instance._hook_pre_execution is pre_exec
        assert mock_predict_rlm_instance._hook_post_execution is post_exec
        assert mock_predict_rlm_instance._hook_post_iteration is post_iter

    def test_enable_stores_originals(self, mock_predict_rlm_instance):
        """Test that enable stores original method references."""
        orig_exec = mock_predict_rlm_instance._execute_iteration
        orig_aexec = mock_predict_rlm_instance._aexecute_iteration
        orig_forward = mock_predict_rlm_instance.generate_action.forward
        orig_process = mock_predict_rlm_instance._process_execution_result

        enable_predict_rlm_hooks(mock_predict_rlm_instance)

        originals = mock_predict_rlm_instance._hook_originals
        assert originals["_execute_iteration"] is orig_exec
        assert originals["_aexecute_iteration"] is orig_aexec
        assert originals["generate_action_forward"] is orig_forward
        assert originals["_process_execution_result"] is orig_process

    def test_enable_wraps_methods(self, mock_predict_rlm_instance):
        """Test that enable wraps _execute_iteration and _aexecute_iteration."""
        orig_exec = mock_predict_rlm_instance._execute_iteration
        orig_aexec = mock_predict_rlm_instance._aexecute_iteration
        orig_forward = mock_predict_rlm_instance.generate_action.forward

        enable_predict_rlm_hooks(mock_predict_rlm_instance)

        assert mock_predict_rlm_instance._execute_iteration is not orig_exec
        assert mock_predict_rlm_instance._aexecute_iteration is not orig_aexec
        assert mock_predict_rlm_instance.generate_action.forward is not orig_forward

    def test_enable_with_no_hooks(self, mock_predict_rlm_instance):
        """Test enable with all hooks set to None."""
        enable_predict_rlm_hooks(mock_predict_rlm_instance)

        assert mock_predict_rlm_instance._hook_pre_iteration is None
        assert mock_predict_rlm_instance._hook_pre_execution is None
        assert mock_predict_rlm_instance._hook_post_execution is None
        assert mock_predict_rlm_instance._hook_post_iteration is None

    def test_enable_is_idempotent(self, mock_predict_rlm_instance):
        """Test calling enable twice doesn't double-wrap.

        The second enable should disable first (restoring originals),
        then create new wrappers. So the wrappers will be different objects
        but the originals should be the same.
        """
        # Store the original method before any enable
        _original_exec = mock_predict_rlm_instance._execute_iteration

        enable_predict_rlm_hooks(mock_predict_rlm_instance)
        _first_exec = mock_predict_rlm_instance._execute_iteration
        first_originals = mock_predict_rlm_instance._hook_originals.copy()

        enable_predict_rlm_hooks(mock_predict_rlm_instance)
        _second_exec = mock_predict_rlm_instance._execute_iteration
        second_originals = mock_predict_rlm_instance._hook_originals

        # Both should wrap the same original
        assert (
            first_originals["_execute_iteration"]
            is second_originals["_execute_iteration"]
        )
        assert "_execute_iteration" in second_originals


# ---------------------------------------------------------------------------
# disable_predict_rlm_hooks
# ---------------------------------------------------------------------------


class TestDisablePredictRLMHooks:
    """Tests for disable_predict_rlm_hooks."""

    def test_disable_restores_originals(self, mock_predict_rlm_instance):
        """Test that disable restores original methods."""
        orig_exec = mock_predict_rlm_instance._execute_iteration
        orig_aexec = mock_predict_rlm_instance._aexecute_iteration
        orig_forward = mock_predict_rlm_instance.generate_action.forward
        orig_process = mock_predict_rlm_instance._process_execution_result

        enable_predict_rlm_hooks(mock_predict_rlm_instance)
        disable_predict_rlm_hooks(mock_predict_rlm_instance)

        assert mock_predict_rlm_instance._execute_iteration is orig_exec
        assert mock_predict_rlm_instance._aexecute_iteration is orig_aexec
        assert mock_predict_rlm_instance.generate_action.forward is orig_forward
        assert mock_predict_rlm_instance._process_execution_result is orig_process

    def test_disable_cleans_hook_attrs(self, mock_predict_rlm_instance):
        """Test that disable removes hook attributes."""
        enable_predict_rlm_hooks(
            mock_predict_rlm_instance,
            pre_iteration_hook=MagicMock(),
        )
        disable_predict_rlm_hooks(mock_predict_rlm_instance)

        assert not hasattr(mock_predict_rlm_instance, "_hook_originals")
        assert not hasattr(mock_predict_rlm_instance, "_hook_pre_iteration")
        assert not hasattr(mock_predict_rlm_instance, "_hook_pre_execution")
        assert not hasattr(mock_predict_rlm_instance, "_hook_post_execution")
        assert not hasattr(mock_predict_rlm_instance, "_hook_post_iteration")
        assert not hasattr(mock_predict_rlm_instance, "_hook_current_context")

    def test_disable_on_unpatched_is_noop(self, mock_predict_rlm_instance):
        """Test that disable on an unpatched instance is a no-op."""
        # MagicMock auto-creates attributes, so we need to explicitly
        # make _hook_originals not exist or be empty
        # Use a spec to prevent auto-creation of _hook_originals

        # Create a mock that doesn't have _hook_originals
        class NoOriginals:
            def __init__(self) -> None:
                self._execute_iteration = MagicMock()
                self._aexecute_iteration = MagicMock()
                self.generate_action = MagicMock()
                self._process_execution_result = MagicMock()

        no_orig_mock = NoOriginals()

        # This should hit the early return path (line 364)
        disable_predict_rlm_hooks(no_orig_mock)


# ---------------------------------------------------------------------------
# Wrapped _execute_iteration (sync)
# ---------------------------------------------------------------------------


class TestWrappedExecuteIteration:
    """Tests for the wrapped _execute_iteration method."""

    def test_pre_iteration_hook_fires(self, mock_predict_rlm_instance):
        """Test that pre_iteration_hook is called with correct args."""
        hook = MagicMock(return_value=PreIterationOutput())
        enable_predict_rlm_hooks(
            mock_predict_rlm_instance,
            pre_iteration_hook=hook,
        )

        mock_vars = [MagicMock()]
        mock_history = MagicMock()
        input_args = {"question": "test"}

        mock_predict_rlm_instance._execute_iteration(
            MagicMock(),
            mock_vars,
            mock_history,
            0,
            input_args,
            ["answer"],
        )

        hook.assert_called_once()
        call_args = hook.call_args[0]
        assert call_args[0] == 0
        assert call_args[1] is mock_vars
        assert call_args[2] is mock_history
        assert call_args[3] == input_args

    def test_pre_iteration_hook_injects_vars(self, mock_predict_rlm_instance):
        """Test that pre_iteration_hook extra_vars are merged into input_args."""
        captured_args = {}

        # Set up the original mock to capture what it receives
        def capture_fn(
            self,
            repl,
            variables,
            history,
            iteration,
            input_args,
            output_field_names,
            **kw,
        ):
            captured_args.update(input_args)
            return "result"

        mock_predict_rlm_instance._execute_iteration = capture_fn

        def inject_hook(iteration, variables, history, input_args):
            return PreIterationOutput(extra_vars={"debug": True, "count": 42})

        enable_predict_rlm_hooks(
            mock_predict_rlm_instance,
            pre_iteration_hook=inject_hook,
        )

        mock_predict_rlm_instance._execute_iteration(
            repl=MagicMock(),
            variables=[],
            history=MagicMock(),
            iteration=0,
            input_args={"question": "test"},
            output_field_names=["answer"],
        )

        assert captured_args.get("debug") is True
        assert captured_args.get("count") == 42

    def test_pre_iteration_hook_python_code(self, mock_predict_rlm_instance):
        """Test that pre_iteration_hook python_code is stored in repl_globals."""

        def code_hook(iteration, variables, history, input_args):
            return PreIterationOutput(python_code="import math")

        enable_predict_rlm_hooks(
            mock_predict_rlm_instance,
            pre_iteration_hook=code_hook,
        )

        repl = MagicMock()
        repl.repl_globals = ""

        mock_predict_rlm_instance._execute_iteration(
            repl, [], MagicMock(), 0, {}, ["answer"]
        )

        assert "import math" in repl.repl_globals

    def test_pre_iteration_hook_async(self, mock_predict_rlm_instance):
        """Test that async pre_iteration_hook is awaited."""
        call_order = []

        async def async_hook(iteration, variables, history, input_args):
            call_order.append("pre_iteration")
            await asyncio.sleep(0)
            return PreIterationOutput()

        enable_predict_rlm_hooks(
            mock_predict_rlm_instance,
            pre_iteration_hook=async_hook,
        )

        mock_predict_rlm_instance._execute_iteration(
            MagicMock(), [], MagicMock(), 0, {}, ["answer"]
        )

        assert "pre_iteration" in call_order

    def test_context_published(self, mock_predict_rlm_instance):
        """Test that _hook_current_context is set during execution."""
        enable_predict_rlm_hooks(mock_predict_rlm_instance)

        mock_vars = [MagicMock()]
        mock_history = MagicMock()

        mock_predict_rlm_instance._execute_iteration(
            MagicMock(), mock_vars, mock_history, 5, {"q": "test"}, ["answer"]
        )

        ctx = mock_predict_rlm_instance._hook_current_context
        assert ctx["iteration"] == 5
        assert ctx["variables"] is mock_vars
        assert ctx["history"] is mock_history
        assert ctx["input_args"] == {"q": "test"}

    def test_stop_iteration_catch(self, mock_predict_rlm_instance):
        """Test that _StopIteration from post_iteration_hook is caught."""
        mock_predict_rlm_instance._execute_iteration = MagicMock(
            side_effect=_StopIteration("stopped_result")
        )

        enable_predict_rlm_hooks(mock_predict_rlm_instance)

        result = mock_predict_rlm_instance._execute_iteration(
            MagicMock(), [], MagicMock(), 0, {}, ["answer"]
        )

        assert result == "stopped_result"

    def test_pre_iteration_hook_with_repl_globals_none(self, mock_predict_rlm_instance):
        """Test pre_iteration_hook when repl.repl_globals is None."""

        def code_hook(iteration, variables, history, input_args):
            return PreIterationOutput(python_code="import os")

        enable_predict_rlm_hooks(
            mock_predict_rlm_instance,
            pre_iteration_hook=code_hook,
        )

        repl = MagicMock()
        repl.repl_globals = None  # Not empty string, but None
        # getattr(repl, "repl_globals", "") or "" should handle None
        if hasattr(repl, "repl_globals"):
            repl.repl_globals = None

        mock_predict_rlm_instance._execute_iteration(
            repl=repl,
            variables=[],
            history=MagicMock(),
            iteration=0,
            input_args={},
            output_field_names=["answer"],
        )

        assert repl.repl_globals is not None and "import os" in repl.repl_globals


# ---------------------------------------------------------------------------
# Wrapped _aexecute_iteration (async)
# ---------------------------------------------------------------------------


class TestWrappedAexecuteIteration:
    """Tests for the wrapped _aexecute_iteration method."""

    @pytest.mark.asyncio
    async def test_async_pre_iteration_hook(self, mock_predict_rlm_instance):
        """Test that pre_iteration_hook fires in async iteration."""
        hook = MagicMock(return_value=PreIterationOutput())

        # Set up the original mock to be awaitable
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

        mock_predict_rlm_instance._aexecute_iteration = mock_aexecute

        enable_predict_rlm_hooks(
            mock_predict_rlm_instance,
            pre_iteration_hook=hook,
        )

        await mock_predict_rlm_instance._aexecute_iteration(
            MagicMock(), [], MagicMock(), 0, {}, ["answer"]
        )

        hook.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_pre_iteration_async_hook(self, mock_predict_rlm_instance):
        """Test that async pre_iteration_hook is awaited in _aexecute_iteration."""
        call_order = []

        async def async_hook(iteration, variables, history, input_args):
            call_order.append("pre_iteration")
            await asyncio.sleep(0)
            return PreIterationOutput()

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

        mock_predict_rlm_instance._aexecute_iteration = mock_aexecute

        enable_predict_rlm_hooks(
            mock_predict_rlm_instance,
            pre_iteration_hook=async_hook,
        )

        await mock_predict_rlm_instance._aexecute_iteration(
            MagicMock(), [], MagicMock(), 0, {}, ["answer"]
        )

        assert "pre_iteration" in call_order

    @pytest.mark.asyncio
    async def test_async_pre_iteration_injects_vars(self, mock_predict_rlm_instance):
        """Test that pre_iteration_hook injects vars in async path."""
        captured = {}

        async def capture_aexecute(
            self,
            repl,
            variables,
            history,
            iteration,
            input_args,
            output_field_names,
            **kw,
        ):
            captured.update(input_args)
            return "result"

        mock_predict_rlm_instance._aexecute_iteration = capture_aexecute

        def inject_hook(iteration, variables, history, input_args):
            return PreIterationOutput(extra_vars={"async_var": True})

        enable_predict_rlm_hooks(
            mock_predict_rlm_instance,
            pre_iteration_hook=inject_hook,
        )

        await mock_predict_rlm_instance._aexecute_iteration(
            MagicMock(), [], MagicMock(), 0, {}, ["answer"]
        )

        assert captured.get("async_var") is True

    @pytest.mark.asyncio
    async def test_async_pre_iteration_python_code(self, mock_predict_rlm_instance):
        """Test that python_code is injected in async path."""

        def code_hook(iteration, variables, history, input_args):
            return PreIterationOutput(python_code="import os")

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

        mock_predict_rlm_instance._aexecute_iteration = mock_aexecute

        enable_predict_rlm_hooks(
            mock_predict_rlm_instance,
            pre_iteration_hook=code_hook,
        )

        repl = MagicMock()
        repl.repl_globals = ""

        await mock_predict_rlm_instance._aexecute_iteration(
            repl, [], MagicMock(), 0, {}, ["answer"]
        )

        assert repl.repl_globals is not None and "import os" in repl.repl_globals

    @pytest.mark.asyncio
    async def test_async_stop_iteration(self, mock_predict_rlm_instance):
        """Test that _StopIteration is caught in async path."""

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
            raise _StopIteration("async_stopped")

        mock_predict_rlm_instance._aexecute_iteration = mock_aexecute

        enable_predict_rlm_hooks(mock_predict_rlm_instance)

        result = await mock_predict_rlm_instance._aexecute_iteration(
            MagicMock(), [], MagicMock(), 0, {}, ["answer"]
        )

        assert result == "async_stopped"

    @pytest.mark.asyncio
    async def test_async_context_published(self, mock_predict_rlm_instance):
        """Test that _hook_current_context is set in async path."""

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

        mock_predict_rlm_instance._aexecute_iteration = mock_aexecute

        enable_predict_rlm_hooks(mock_predict_rlm_instance)

        mock_vars = [MagicMock()]
        mock_history = MagicMock()

        await mock_predict_rlm_instance._aexecute_iteration(
            MagicMock(), mock_vars, mock_history, 3, {"q": "test"}, ["answer"]
        )

        ctx = mock_predict_rlm_instance._hook_current_context
        assert ctx["iteration"] == 3
        assert ctx["variables"] is mock_vars


# ---------------------------------------------------------------------------
# Wrapped generate_action.forward (pre_execution_hook)
# ---------------------------------------------------------------------------


class TestWrappedForward:
    """Tests for the wrapped generate_action.forward method."""

    def test_pre_execution_hook_mutates_code(self, mock_predict_rlm_instance):
        """Test that pre_execution_hook can modify generated code."""

        def rewrite_hook(iteration, code, variables, history, input_args):
            return PreExecutionOutput(code=f"# modified\n{code}")

        enable_predict_rlm_hooks(
            mock_predict_rlm_instance,
            pre_execution_hook=rewrite_hook,
        )

        result = mock_predict_rlm_instance.generate_action.forward()
        assert "# modified" in result.code

    def test_pre_execution_hook_not_called_when_none(self, mock_predict_rlm_instance):
        """Test that forward works normally when no pre_execution_hook is set."""
        enable_predict_rlm_hooks(mock_predict_rlm_instance)

        result = mock_predict_rlm_instance.generate_action.forward()
        assert result.code == "generated code"

    def test_pre_execution_hook_strips_code_fences(self, mock_predict_rlm_instance):
        """Test that code fences are stripped before passing to hook."""
        captured_code = []

        def capture_hook(iteration, code, variables, history, input_args):
            captured_code.append(code)
            return PreExecutionOutput(code=code)

        mock_predict_rlm_instance.generate_action.forward = MagicMock(
            return_value=MagicMock(code="```python\nprint('hi')\n```", reasoning="test")
        )

        enable_predict_rlm_hooks(
            mock_predict_rlm_instance,
            pre_execution_hook=capture_hook,
        )

        mock_predict_rlm_instance.generate_action.forward()

        assert len(captured_code) == 1
        assert "print('hi')" in captured_code[0]

    def test_pre_execution_hook_async(self, mock_predict_rlm_instance):
        """Test that async pre_execution_hook is awaited in forward."""
        captured = []

        async def async_hook(iteration, code, variables, history, input_args):
            captured.append(code)
            await asyncio.sleep(0)
            return PreExecutionOutput(code=f"async_{code}")

        enable_predict_rlm_hooks(
            mock_predict_rlm_instance,
            pre_execution_hook=async_hook,
        )

        result = mock_predict_rlm_instance.generate_action.forward()
        assert "async_" in result.code

    def test_forward_with_context(self, mock_predict_rlm_instance):
        """Test that forward reads _hook_current_context for hook args."""
        captured_args = []

        def capture_hook(iteration, code, variables, history, input_args):
            captured_args.append(
                {
                    "iteration": iteration,
                    "variables": variables,
                    "history": history,
                    "input_args": input_args,
                }
            )
            return PreExecutionOutput(code=code)

        enable_predict_rlm_hooks(
            mock_predict_rlm_instance,
            pre_execution_hook=capture_hook,
        )

        # Set context before calling forward
        mock_predict_rlm_instance._hook_current_context = {
            "iteration": 5,
            "variables": ["var1"],
            "history": ["hist"],
            "input_args": {"q": "test"},
        }

        mock_predict_rlm_instance.generate_action.forward()

        assert len(captured_args) == 1
        assert captured_args[0]["iteration"] == 5

    def test_forward_without_context(self, mock_predict_rlm_instance):
        """Test that forward works when _hook_current_context doesn't exist."""
        captured_args = []

        def capture_hook(iteration, code, variables, history, input_args):
            captured_args.append(True)
            return PreExecutionOutput(code=code)

        enable_predict_rlm_hooks(
            mock_predict_rlm_instance,
            pre_execution_hook=capture_hook,
        )

        # Make sure no context exists
        if hasattr(mock_predict_rlm_instance, "_hook_current_context"):
            delattr(mock_predict_rlm_instance, "_hook_current_context")

        mock_predict_rlm_instance.generate_action.forward()

        assert len(captured_args) == 1


# ---------------------------------------------------------------------------
# Wrapped _process_execution_result (post hooks)
# ---------------------------------------------------------------------------


class TestWrappedProcessResult:
    """Tests for the wrapped _process_execution_result method."""

    def test_post_execution_hook_transforms_result_with_code_arg(
        self, mock_predict_rlm_instance
    ):
        """Test post_execution_hook when first arg is a string (code)."""
        captured = {}

        def original_process(self, pred, *args, **kwargs):
            # args should be (code, transformed_result, *rest)
            captured["args"] = args
            return "processed"

        mock_predict_rlm_instance._process_execution_result = original_process

        def transform_hook(iteration, code, result, variables, history, input_args):
            return PostExecutionOutput(result="TRANSFORMED")

        enable_predict_rlm_hooks(
            mock_predict_rlm_instance,
            post_execution_hook=transform_hook,
        )

        mock_predict_rlm_instance._process_execution_result(
            MagicMock(), "test_code", "raw_result", MagicMock(), ["answer"]
        )

        # When code is a string, args are rebuilt as (code, transformed_result, *rest)
        assert captured["args"][0] == "test_code"  # code stays in position 0
        assert captured["args"][1] == "TRANSFORMED"  # transformed result in position 1

    def test_post_execution_hook_transforms_result_without_code_arg(
        self, mock_predict_rlm_instance
    ):
        """Test post_execution_hook when first arg is NOT a string."""
        captured = {}

        def original_process(self, pred, *args, **kwargs):
            captured["args"] = args
            return "processed"

        mock_predict_rlm_instance._process_execution_result = original_process

        def transform_hook(iteration, code, result, variables, history, input_args):
            return PostExecutionOutput(result="TRANSFORMED")

        enable_predict_rlm_hooks(
            mock_predict_rlm_instance,
            post_execution_hook=transform_hook,
        )

        # Set context for code
        mock_predict_rlm_instance._hook_current_context = {"code": "ctx_code"}

        # First arg is not a string (it's a MagicMock = result)
        mock_predict_rlm_instance._process_execution_result(
            MagicMock(), MagicMock(), ["answer"]
        )

        assert captured["args"][0] == "TRANSFORMED"

    def test_post_execution_hook_async(self, mock_predict_rlm_instance):
        """Test that async post_execution_hook is awaited."""
        captured = {}

        def original_process(self, pred, *args, **kwargs):
            captured["args"] = args
            return "processed"

        mock_predict_rlm_instance._process_execution_result = original_process

        async def async_hook(iteration, code, result, variables, history, input_args):
            await asyncio.sleep(0)
            return PostExecutionOutput(result="ASYNC_TRANSFORMED")

        enable_predict_rlm_hooks(
            mock_predict_rlm_instance,
            post_execution_hook=async_hook,
        )

        mock_predict_rlm_instance._process_execution_result(
            MagicMock(), "code", "raw_result", MagicMock(), ["answer"]
        )

        # args are rebuilt as (code, transformed_result, *rest)
        assert captured["args"][0] == "code"
        assert captured["args"][1] == "ASYNC_TRANSFORMED"

    def test_post_iteration_hook(self, mock_predict_rlm_instance):
        """Test that post_iteration_hook receives processed result."""
        modified_history = MagicMock(spec=REPLHistory)

        def original_process(self, pred, *args, **kwargs):
            return MagicMock(spec=REPLHistory)

        mock_predict_rlm_instance._process_execution_result = original_process

        def post_hook(iteration, pred, code, result, history):
            return PostIterationOutput(history=modified_history)

        enable_predict_rlm_hooks(
            mock_predict_rlm_instance,
            post_iteration_hook=post_hook,
        )

        result = mock_predict_rlm_instance._process_execution_result(
            MagicMock(), "code", "result", MagicMock(), ["answer"]
        )

        assert result is modified_history

    def test_post_iteration_hook_async(self, mock_predict_rlm_instance):
        """Test that async post_iteration_hook is awaited."""
        modified_history = MagicMock(spec=REPLHistory)

        def original_process(self, pred, *args, **kwargs):
            return MagicMock(spec=REPLHistory)

        mock_predict_rlm_instance._process_execution_result = original_process

        async def async_hook(iteration, pred, code, result, history):
            await asyncio.sleep(0)
            return PostIterationOutput(history=modified_history)

        enable_predict_rlm_hooks(
            mock_predict_rlm_instance,
            post_iteration_hook=async_hook,
        )

        result = mock_predict_rlm_instance._process_execution_result(
            MagicMock(), "code", "result", MagicMock(), ["answer"]
        )

        assert result is modified_history

    def test_post_iteration_stop_raises_stop_iteration(self, mock_predict_rlm_instance):
        """Test that post_iteration_hook with stop=True raises _StopIteration."""

        def original_process(self, pred, *args, **kwargs):
            return MagicMock(spec=REPLHistory)

        mock_predict_rlm_instance._process_execution_result = original_process

        def stop_hook(iteration, pred, code, result, history):
            return PostIterationOutput(history=history, stop=True)

        enable_predict_rlm_hooks(
            mock_predict_rlm_instance,
            post_iteration_hook=stop_hook,
        )

        with pytest.raises(_StopIteration):
            mock_predict_rlm_instance._process_execution_result(
                MagicMock(), "code", "result", MagicMock(), ["answer"]
            )

    def test_process_result_no_hooks(self, mock_predict_rlm_instance):
        """Test _process_execution_result with no hooks set."""

        def original_process(self, pred, *args, **kwargs):
            return "no_hooks_result"

        mock_predict_rlm_instance._process_execution_result = original_process

        enable_predict_rlm_hooks(mock_predict_rlm_instance)

        result = mock_predict_rlm_instance._process_execution_result(
            MagicMock(), "code", "result", MagicMock(), ["answer"]
        )

        assert result == "no_hooks_result"

    def test_process_result_post_iter_always_called(self, mock_predict_rlm_instance):
        """Test post_iteration_hook IS called even when result is not REPLHistory.

        Unlike patcher.py, predict_rlm_compat always calls post_iteration_hook.
        """
        modified_history = MagicMock(spec=REPLHistory)

        def original_process(self, pred, *args, **kwargs):
            return "not_repl_history"  # String, not REPLHistory

        mock_predict_rlm_instance._process_execution_result = original_process

        def post_hook(iteration, pred, code, result, history):
            return PostIterationOutput(history=modified_history)

        enable_predict_rlm_hooks(
            mock_predict_rlm_instance,
            post_iteration_hook=post_hook,
        )

        result = mock_predict_rlm_instance._process_execution_result(
            MagicMock(), "code", "result", MagicMock(), ["answer"]
        )

        # post_iteration_hook IS called and its return value is used
        assert result is modified_history


# ---------------------------------------------------------------------------
# Integration: full lifecycle
# ---------------------------------------------------------------------------


class TestFullLifecycle:
    """Integration tests for the full hook lifecycle."""

    def test_all_hooks_fire_in_order_via_forward_and_process(
        self, mock_predict_rlm_instance
    ):
        """Test that all hooks fire in lifecycle order."""
        order = []

        def pre_iter(iteration, variables, history, input_args):
            order.append("pre_iteration")
            return PreIterationOutput()

        def pre_exec(iteration, code, variables, history, input_args):
            order.append("pre_execution")
            return PreExecutionOutput(code=code)

        def post_exec(iteration, code, result, variables, history, input_args):
            order.append("post_execution")
            return PostExecutionOutput(result=result)

        def post_iter(iteration, pred, code, result, history):
            order.append("post_iteration")
            return PostIterationOutput(history=history)

        def original_process(self, pred, *args, **kwargs):
            return MagicMock(spec=REPLHistory)

        mock_predict_rlm_instance._process_execution_result = original_process

        enable_predict_rlm_hooks(
            mock_predict_rlm_instance,
            pre_iteration_hook=pre_iter,
            pre_execution_hook=pre_exec,
            post_execution_hook=post_exec,
            post_iteration_hook=post_iter,
        )

        # Call execute_iteration which triggers pre_iteration
        mock_predict_rlm_instance._execute_iteration(
            MagicMock(), [], MagicMock(), 0, {}, ["answer"]
        )

        # Call forward which triggers pre_execution
        mock_predict_rlm_instance.generate_action.forward()

        # Call process which triggers post_execution + post_iteration
        mock_predict_rlm_instance._process_execution_result(
            MagicMock(), "code", "result", MagicMock(), ["answer"]
        )

        assert order == [
            "pre_iteration",
            "pre_execution",
            "post_execution",
            "post_iteration",
        ]

    def test_enable_disable_roundtrip(self, mock_predict_rlm_instance):
        """Test full enable/disable roundtrip restores everything."""
        original_exec = mock_predict_rlm_instance._execute_iteration

        enable_predict_rlm_hooks(
            mock_predict_rlm_instance,
            pre_iteration_hook=MagicMock(),
        )

        assert mock_predict_rlm_instance._execute_iteration is not original_exec
        assert hasattr(mock_predict_rlm_instance, "_hook_originals")

        disable_predict_rlm_hooks(mock_predict_rlm_instance)

        assert mock_predict_rlm_instance._execute_iteration is original_exec
        assert not hasattr(mock_predict_rlm_instance, "_hook_originals")
