"""Example: Add retry hints when code execution fails.

This shows how to use a post_execution_hook to detect errors in the
execution result and append a hint so the LLM can self-correct on the
next iteration.

Usage:
    uv run python examples/error_retry.py
"""

from __future__ import annotations

from typing import Any

import dspy

from dspy_rlm_hooks import PostExecutionOutput, enable_rlm_hooks


def retry_with_hint(
    iteration: int,
    code: str,
    result: Any,
    variables: list,
    history: list,
    input_args: dict,
    *,
    raw_code: str = "",
) -> PostExecutionOutput:
    """If execution raised an error, append a hint for the next iteration."""
    if isinstance(result, str) and result.startswith("[Error]"):
        hint = (
            f"{result}\n"
            "# Hint: check variable names and ensure imports are present. "
            "The following variables are in scope: "
            + ", ".join(str(v) for v in input_args)
        )
        print(f"[error_retry] Error in iteration {iteration}, adding hint.")
        return PostExecutionOutput(result=hint)
    return PostExecutionOutput(result=result)


def main() -> None:
    lm = dspy.LM("openai/gpt-4o-mini", cache=False)
    dspy.configure(lm=lm)

    rlm = dspy.RLM(signature="question -> answer", tools=[])
    enable_rlm_hooks(rlm, post_execution_hook=retry_with_hint)

    result = rlm(question="What is the square root of 1764?")
    print(result)


if __name__ == "__main__":
    main()
