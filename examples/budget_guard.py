"""Example: Stop RLM iteration when a cost budget is exhausted.

This shows how to use a post_iteration_hook with closure state to
accumulate an estimated cost and halt the loop once a threshold is
reached.  The LLM's extract-fallback path fires automatically, so
the caller still gets a Prediction result.

Usage:
    uv run python examples/budget_guard.py
"""

from __future__ import annotations

import dspy
from dspy.primitives.repl_types import REPLHistory

from dspy_rlm_hooks import PostIterationOutput, enable_rlm_hooks

MAX_COST_USD = 0.50


def _estimate_cost(pred: object) -> float:
    """Stub cost estimator — replace with real usage tracking."""
    return 0.015


def make_budget_hook(max_cost: float = MAX_COST_USD):
    """Return a post_iteration hook that stops after *max_cost* dollars.

    Each call creates isolated state so budgets don't leak across
    concurrent requests on a multi-threaded or async server.
    """
    accumulated = 0.0

    def post_iteration(
        iteration: int,
        pred: object,
        code: str,
        result: object,
        history: REPLHistory,
    ) -> PostIterationOutput:
        nonlocal accumulated
        accumulated += _estimate_cost(pred)
        if accumulated >= max_cost:
            print(
                f"[budget_guard] Budget exhausted (${accumulated:.3f} >= ${max_cost:.3f}), stopping."
            )
            return PostIterationOutput(history=history, stop=True)
        return PostIterationOutput(history=history)

    return post_iteration


def main() -> None:
    lm = dspy.LM("openai/gpt-4o-mini", cache=False)
    dspy.configure(lm=lm)

    rlm = dspy.RLM(signature="question -> answer", tools=[])
    enable_rlm_hooks(rlm, post_iteration_hook=make_budget_hook())

    result = rlm(question="What is the capital of France?")
    print(result)


if __name__ == "__main__":
    main()
