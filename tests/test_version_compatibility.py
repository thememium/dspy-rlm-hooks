"""Tests for DSPy version compatibility (3.1+)."""

from __future__ import annotations


import dspy
import pytest

from dspy_rlm_hooks import enable_rlm_hooks
from dspy_rlm_hooks.patcher import _validate_rlm


class TestDSPyVersionCompatibility:
    """Tests to ensure compatibility with DSPy 3.1 and newer."""

    def test_dspy_version_meets_requirement(self):
        """Test that installed DSPy version is >= 3.1."""
        import importlib.metadata

        version = importlib.metadata.version("dspy")
        major, minor = version.split(".")[:2]
        assert int(major) >= 3, f"DSPy major version {major} < 3"
        if int(major) == 3:
            assert int(minor) >= 1, f"DSPy minor version {minor} < 1"

    def test_rlm_has_required_methods(self):
        """Test that real dspy.RLM has all methods we need to patch."""
        rlm = dspy.RLM(
            signature="question -> answer",
            tools=[],
        )

        required = [
            "_execute_iteration",
            "_aexecute_iteration",
            "_process_execution_result",
            "generate_action",
            "max_iterations",
            "verbose",
        ]

        for method in required:
            assert hasattr(rlm, method), f"RLM missing required attribute: {method}"

    def test_rlm_generate_action_has_acall(self):
        """Test that generate_action has acall for async support."""
        rlm = dspy.RLM(
            signature="question -> answer",
            tools=[],
        )

        assert hasattr(rlm.generate_action, "acall")
        assert callable(rlm.generate_action.acall)

    def test_repl_types_available(self):
        """Test that REPLHistory is importable from dspy.primitives.repl_types."""
        from dspy.primitives.repl_types import REPLHistory

        history = REPLHistory(entries=[])
        assert hasattr(history, "entries")

    def test_validation_passes_with_real_rlm(self):
        """Test that validation passes with a real RLM from current DSPy."""
        rlm = dspy.RLM(
            signature="question -> answer",
            tools=[],
        )
        _validate_rlm(rlm)  # Should not raise

    def test_enable_hooks_on_real_rlm(self):
        """Test enabling hooks on a real RLM instance."""
        rlm = dspy.RLM(
            signature="question -> answer",
            tools=[],
        )

        from dspy_rlm_hooks import PreIterationOutput

        def hook(iteration, variables, history, input_args):
            return PreIterationOutput()

        enable_rlm_hooks(rlm, pre_iteration_hook=hook)

        assert hasattr(rlm, "_hook_pre_iteration")
        assert rlm._hook_pre_iteration is hook
        assert hasattr(rlm, "_execute_iteration")
        assert hasattr(rlm, "_aexecute_iteration")

    def test_rlm_signature_attribute(self):
        """Test that RLM exposes signature-related attributes."""
        rlm = dspy.RLM(
            signature="question -> answer",
            tools=[],
        )

        # These are used internally by DSPy and should be present
        assert hasattr(rlm, "signature")

    def test_rlm_tools_attribute(self):
        """Test that RLM has tools attribute."""
        rlm = dspy.RLM(
            signature="question -> answer",
            tools=[],
        )
        assert hasattr(rlm, "tools")
        assert rlm.tools == [] or rlm.tools == {}

    def test_repl_history_model_structure(self):
        """Test that REPLHistory has the expected structure."""
        from dspy.primitives.repl_types import REPLHistory

        history = REPLHistory(entries=[])
        assert hasattr(history, "entries")
        assert len(history.entries) == 0

    def test_backward_compatibility_3_1_api(self):
        """Test that our code works with the 3.1+ API structure."""
        # In DSPy 3.1+, RLM uses signature and tools parameters
        rlm = dspy.RLM(
            signature="input -> output",
            tools=[],
        )

        # Should be able to create and patch without issues
        _validate_rlm(rlm)

    @pytest.mark.skipif(
        not hasattr(dspy, "__version__"),
        reason="DSPy version not exposed"
    )
    def test_specific_version_3_2(self):
        """Test compatibility with DSPy 3.2.x specifically."""
        version = dspy.__version__
        if version.startswith("3.2"):
            rlm = dspy.RLM(
                signature="question -> answer",
                tools=[],
            )
            _validate_rlm(rlm)

            from dspy_rlm_hooks import PreIterationOutput
            enable_rlm_hooks(rlm, pre_iteration_hook=lambda **kwargs: PreIterationOutput())

            assert hasattr(rlm, "_execute_iteration")
            assert hasattr(rlm, "_aexecute_iteration")
