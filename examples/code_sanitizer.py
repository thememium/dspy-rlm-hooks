"""Example: Block dangerous code patterns before execution.

This shows how to use a pre_execution_hook to inspect and rewrite
LLM-generated code, preventing calls to os.system, eval, exec, etc.

Usage:
    uv run python examples/code_sanitizer.py
"""

from __future__ import annotations

import re

import dspy

from dspy_rlm_hooks import PreExecutionOutput, enable_rlm_hooks

FORBIDDEN = re.compile(r"\b(os\.system|eval|exec|compile|__import__|subprocess)\b")


def sanitize_code(
    iteration: int,
    code: str,
    variables: list,
    history: list,
    input_args: dict,
) -> PreExecutionOutput:
    """Block forbidden patterns and log when something is caught."""
    if FORBIDDEN.search(code):
        safe = FORBIDDEN.sub("# BLOCKED", code)
        print(f"[code_sanitizer] Blocked forbidden pattern in iteration {iteration}")
        return PreExecutionOutput(code=safe)
    return PreExecutionOutput(code=code)


def main() -> None:
    lm = dspy.LM("openai/gpt-4o-mini", cache=False)
    dspy.configure(lm=lm)

    rlm = dspy.RLM(signature="question -> answer", tools=[])
    enable_rlm_hooks(rlm, pre_execution_hook=sanitize_code)

    result = rlm(question="What is 2 + 2?")
    print(result)


if __name__ == "__main__":
    main()
