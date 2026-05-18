"""DSPy RLM Hooks — lifecycle instrumentation for :class:`~dspy.RLM`.

This package monkey-patches the internal iteration loop of DSPy's
:class:`~dspy.RLM` (Recursive Language Model) to expose *lifecycle hooks* at
every stage of an iteration:

1. **pre_iteration** – before the LLM generates the next action
2. **pre_execution** – after code generation, before running it
3. **post_execution** – after code runs, before the result is recorded
4. **post_iteration** – after the result is folded into history

Hooks may be synchronous **or** asynchronous.  The system auto-detects
coroutine return values and handles both paths transparently.

Compatibility
-------------
Tested against **DSPy 3.1+**.  Because the package instruments *private*
DSPy internals, future DSPy releases may require updates.  A runtime
validation check raises :class:`AttributeError` if the RLM instance does
not expose the expected API.

Quick start
-----------
::

    import dspy
    from dspy_rlm_hooks import enable_rlm_hooks, PreIterationOutput

    rlm = dspy.RLM(...)

    def inject_debug(iteration, variables, history, input_args):
        return PreIterationOutput(extra_vars={"debug": True})

    enable_rlm_hooks(rlm, pre_iteration_hook=inject_debug)

    # Use rlm normally — hooks fire automatically.
    result = rlm(question="What is 2 + 2?")

Public API
----------
"""

from __future__ import annotations

from dspy_rlm_hooks.patcher import disable_rlm_hooks, enable_rlm_hooks
from dspy_rlm_hooks.types import (
    PostExecutionHook,
    PostExecutionOutput,
    PostIterationHook,
    PostIterationOutput,
    PreExecutionHook,
    PreExecutionOutput,
    PreIterationHook,
    PreIterationOutput,
    RLMHook,
)

try:
    from importlib.metadata import version

    __version__ = version("dspy-rlm-hooks")
except ImportError:
    __version__ = "unknown"

__all__ = [
    "PreIterationHook",
    "PreExecutionHook",
    "PostExecutionHook",
    "PostIterationHook",
    "PreIterationOutput",
    "PreExecutionOutput",
    "PostExecutionOutput",
    "PostIterationOutput",
    "RLMHook",
    "enable_rlm_hooks",
    "disable_rlm_hooks",
]
