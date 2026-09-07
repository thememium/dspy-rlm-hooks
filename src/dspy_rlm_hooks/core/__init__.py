"""Hook lifecycle core: types, utils, patcher, tracing, and PredictRLM compatibility.

Re-exports the public API of the core submodules::

    from dspy_rlm_hooks.core import enable_rlm_hooks, PreIterationOutput

Private helpers are intentionally not re-exported: tests and internal callers
patch and import them at their defining modules (``core.patcher``,
``core.tracing``, ...), where the consuming code reads them.
"""

from __future__ import annotations

from dspy_rlm_hooks.core.patcher import disable_rlm_hooks, enable_rlm_hooks
from dspy_rlm_hooks.core.predict_rlm_compat import (
    disable_predict_rlm_hooks,
    enable_predict_rlm_hooks,
)
from dspy_rlm_hooks.core.tracing import enable_rlm_hooks_with_tracing
from dspy_rlm_hooks.core.types import (
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
    "enable_rlm_hooks_with_tracing",
    "enable_predict_rlm_hooks",
    "disable_predict_rlm_hooks",
]
