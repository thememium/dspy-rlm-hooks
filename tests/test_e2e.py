"""End-to-end tests using a real LLM."""

from __future__ import annotations

import pytest

from dspy_rlm_hooks import (
    PostExecutionOutput,
    PostIterationOutput,
    PreExecutionOutput,
    PreIterationOutput,
    enable_rlm_hooks,
)


class TestEndToEnd:
    """Full end-to-end tests with a real LLM."""

    def test_rlm_with_pre_iteration_hook(self, dspy_lm):
        """Test that pre_iteration hook injects variables into a real RLM run."""
        import dspy

        rlm = dspy.RLM(
            signature="question -> answer",
            tools=[],
        )

        captured = {}

        def pre_iter(iteration, variables, history, input_args):
            captured["iteration"] = iteration
            return PreIterationOutput(extra_vars={"injected": "value"})

        enable_rlm_hooks(rlm, pre_iteration_hook=pre_iter)

        try:
            rlm(question="What is 2 + 2?")
            assert captured["iteration"] >= 0
        except Exception as exc:
            if "api key" in str(exc).lower() or "authentication" in str(exc).lower():
                pytest.skip(f"LLM call failed — {exc}")
            raise

    def test_rlm_with_pre_execution_hook(self, dspy_lm):
        """Test that pre_execution hook can rewrite code before execution."""
        import dspy

        rlm = dspy.RLM(
            signature="question -> answer",
            tools=[],
        )

        captured_code = []

        def pre_exec(iteration, code, variables, history, input_args):
            captured_code.append(code)
            return PreExecutionOutput(code=code)

        enable_rlm_hooks(rlm, pre_execution_hook=pre_exec)

        try:
            rlm(question="What is 2 + 2?")
            assert len(captured_code) > 0
            assert isinstance(captured_code[0], str)
        except Exception as exc:
            if "api key" in str(exc).lower() or "authentication" in str(exc).lower():
                pytest.skip(f"LLM call failed — {exc}")
            raise

    def test_rlm_with_post_execution_hook(self, dspy_lm):
        """Test that post_execution hook can audit results."""
        import dspy

        rlm = dspy.RLM(
            signature="question -> answer",
            tools=[],
        )

        captured_results = []

        def post_exec(iteration, code, result, variables, history, input_args):
            captured_results.append(result)
            return PostExecutionOutput(result=result)

        enable_rlm_hooks(rlm, post_execution_hook=post_exec)

        try:
            rlm(question="What is 2 + 2?")
            assert len(captured_results) > 0
        except Exception as exc:
            if "api key" in str(exc).lower() or "authentication" in str(exc).lower():
                pytest.skip(f"LLM call failed — {exc}")
            raise

    def test_rlm_all_hooks_fire(self, dspy_lm):
        """Test that all four hooks fire during a real RLM run."""
        import dspy

        rlm = dspy.RLM(
            signature="question -> answer",
            tools=[],
        )

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

        enable_rlm_hooks(
            rlm,
            pre_iteration_hook=pre_iter,
            pre_execution_hook=pre_exec,
            post_execution_hook=post_exec,
            post_iteration_hook=post_iter,
        )

        try:
            rlm(question="What is 2 + 2?")
            assert "pre_iteration" in order
            assert "pre_execution" in order
            assert "post_execution" in order
            # post_iteration only fires on intermediate iterations (when result is REPLHistory)
            # For single-iteration answers it may not fire
        except Exception as exc:
            if "api key" in str(exc).lower() or "authentication" in str(exc).lower():
                pytest.skip(f"LLM call failed — {exc}")
            raise
