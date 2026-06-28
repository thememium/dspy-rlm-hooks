"""Example: Inject context from an async source before each iteration.

This shows how to use an async pre_iteration_hook to fetch data from a
remote service and seed it into the interpreter namespace.  The hook
system auto-detects coroutines — no special configuration needed.

Usage:
    uv run python examples/async_context_injection.py
"""

from __future__ import annotations

import asyncio
from typing import Any

import dspy

from dspy_rlm_hooks import PreIterationOutput, enable_rlm_hooks


async def fetch_context(
    iteration: int,
    variables: list[Any],
    history: list[Any],
    input_args: dict[str, Any],
) -> PreIterationOutput:
    """Simulate fetching context from a remote service."""
    # Replace with a real async call (database, API, cache, etc.)
    await asyncio.sleep(0.01)
    context = "The user prefers metric units and concise answers."
    print(f"[async_context] Injected context at iteration {iteration}")
    return PreIterationOutput(extra_vars={"user_context": context})


def main() -> None:
    lm = dspy.LM("openai/gpt-4o-mini", cache=False)
    dspy.configure(lm=lm)

    rlm = dspy.RLM(signature="question -> answer", tools=[])
    enable_rlm_hooks(rlm, pre_iteration_hook=fetch_context)

    result = rlm(question="How far is the Moon from the Earth?")
    print(result)


if __name__ == "__main__":
    main()
