"""Coverage tests for the DSPy >=3.4 ``interpreter_factory`` argument paths.

DSPy 3.4 calls the rebound ``_execute_iteration``/``_aexecute_iteration`` with
a trailing positional ``interpreter_factory`` and derives a per-call action
signature from it. Covers the branches that only run when the factory is
present:

- ``patcher._generate_action_signature_kwargs`` returning the per-call
  ``signature=`` kwarg (patcher.py)
- ``predict_rlm_compat``'s sync/async iteration wrappers forwarding the
  seventh positional argument to the wrapped original
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from dspy_rlm_hooks.core import patcher
from dspy_rlm_hooks.core.predict_rlm_compat import enable_predict_rlm_hooks

_FACTORY = lambda: "interpreter"  # noqa: E731


class _FakeGenerateAction:
    """Records kwargs from both the sync call and the async ``acall``."""

    def __init__(self) -> None:
        self.sync_kwargs: dict[str, object] | None = None
        self.async_kwargs: dict[str, object] | None = None

    def __call__(self, **kwargs):
        self.sync_kwargs = kwargs
        return SimpleNamespace(reasoning="thinking", code="x = 1")

    async def acall(self, **kwargs):
        self.async_kwargs = kwargs
        return SimpleNamespace(reasoning="thinking", code="x = 1")


class _FakeRLM:
    """Duck-typed RLM without the DSPy >=3.4 signature-helper attribute."""

    max_iters = 3
    verbose = False
    _hook_pre_iteration = None
    _hook_pre_execution = None
    _hook_post_execution = None
    _hook_post_iteration = None

    def __init__(self) -> None:
        self.generate_action = _FakeGenerateAction()

    def _execute_code(self, repl, code, input_args):
        return "result"

    def _process_execution_result(
        self, action, code, result, history, output_field_names
    ):
        return "processed"


class _ModernRLM(_FakeRLM):
    """DSPy >=3.4-style RLM: exposes ``_action_signature_for_current_factory``."""

    def _action_signature_for_current_factory(self, factory):
        return f"sig-for-{factory()}"


class TestPatcherSignatureKwargs:
    def test_sync_iteration_passes_signature_kwarg(self):
        """With a factory present, ``generate_action`` receives the per-call
        ``signature=`` derived from it."""
        rlm = _ModernRLM()
        out = patcher._execute_iteration(
            rlm, "repl", [], MagicMock(), 0, {}, ["answer"], _FACTORY
        )
        assert out == "processed"
        kwargs = rlm.generate_action.sync_kwargs
        assert kwargs is not None and kwargs["signature"] == "sig-for-interpreter"

    @pytest.mark.asyncio
    async def test_async_iteration_passes_signature_kwarg(self):
        """Same for the async path via ``generate_action.acall``."""
        rlm = _ModernRLM()
        out = await patcher._aexecute_iteration(
            rlm, "repl", [], MagicMock(), 0, {}, ["answer"], _FACTORY
        )
        assert out == "processed"
        kwargs = rlm.generate_action.async_kwargs
        assert kwargs is not None and kwargs["signature"] == "sig-for-interpreter"

    def test_sync_iteration_omits_signature_without_helper(self):
        """DSPy <=3.3 RLMs have no ``_action_signature_for_current_factory``;
        the call must stay keyword-compatible with them."""
        rlm = _FakeRLM()
        patcher._execute_iteration(
            rlm, "repl", [], MagicMock(), 0, {}, ["answer"], _FACTORY
        )
        kwargs = rlm.generate_action.sync_kwargs
        assert kwargs is not None and "signature" not in kwargs


class TestCompatForwardsFactory:
    def _mocked_rlm(self):
        """PredictRLM-like mock whose originals accept the DSPy 3.4 arity."""
        seen = {"sync": None, "async": None}

        def orig_execute(self, *args):
            seen["sync"] = args
            return "result"

        async def orig_aexecute(self, *args):
            seen["async"] = args
            return "async_result"

        mock = MagicMock(name="PredictRLM_instance")
        mock._execute_iteration = orig_execute
        mock._aexecute_iteration = orig_aexecute
        mock._process_execution_result = MagicMock(return_value="processed")
        mock.generate_action = MagicMock()
        mock.generate_action.forward = MagicMock(
            return_value=MagicMock(code="generated code", reasoning="thinking"),
        )
        mock.generate_action.aforward = AsyncMock(
            return_value=MagicMock(code="generated async code", reasoning="thinking"),
        )
        mock.max_iterations = 10
        mock.verbose = False
        return mock, seen

    def test_sync_wrapper_forwards_seventh_positional(self):
        mock, seen = self._mocked_rlm()
        enable_predict_rlm_hooks(mock)
        out = mock._execute_iteration(
            "repl", [], MagicMock(), 0, {}, ["answer"], _FACTORY
        )
        assert out == "result"
        assert len(seen["sync"]) == 7 and seen["sync"][-1] is _FACTORY

    @pytest.mark.asyncio
    async def test_async_wrapper_forwards_seventh_positional(self):
        mock, seen = self._mocked_rlm()
        enable_predict_rlm_hooks(mock)
        out = await mock._aexecute_iteration(
            "repl", [], MagicMock(), 0, {}, ["answer"], _FACTORY
        )
        assert out == "async_result"
        assert len(seen["async"]) == 7 and seen["async"][-1] is _FACTORY
