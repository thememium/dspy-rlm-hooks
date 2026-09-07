"""Integration subpackage: wire the speculation engine into DSPy's RLM path.

Split from the former single-module ``integration.py`` into focused modules;
every name that lived on the original module (including private helpers) is
re-exported here unchanged, so ``dspy_rlm_hooks.speculation.integration`` keeps
its historical namespace.

Layout (import DAG, acyclic, arrows point downward):

- ``registry``       — live-speculator tracking (the ONE home of the
  ``_active_speculators`` global) + tool classifications/registry sync
- ``claim_hooks``    — claiming hooks installed into ``repl.tools``
- ``live_state``     — cross-iteration shadow-seed snapshot helpers
- ``streaming_turn`` — streaming turn begin + streamified ``generate_action``
- ``execute``        — wrapped ``_execute_code`` / iteration methods
- ``api``            — ``enable_rlm_speculation`` / ``disable_rlm_speculation``
"""

from __future__ import annotations

import ast
import atexit
import builtins
import inspect
import weakref
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
from types import MethodType
from typing import TYPE_CHECKING, Any

from dspy_rlm_hooks.core.patcher import _validate_rlm
from dspy_rlm_hooks.core.utils import _assemble_execution_code
from dspy_rlm_hooks.speculation.config import SpeculationConfig
from dspy_rlm_hooks.speculation.guards import fully_raw, tag_claim_hook
from dspy_rlm_hooks.speculation.integration.api import (
    disable_rlm_speculation,
    enable_rlm_speculation,
)
from dspy_rlm_hooks.speculation.integration.claim_hooks import (
    _LLM_TOOLS,
    _install_claim_hooks,
    _make_claim_hook,
)
from dspy_rlm_hooks.speculation.integration.execute import (
    _speculation_aexecute_iteration,
    _speculation_execute_code,
    _speculation_execute_iteration,
)
from dspy_rlm_hooks.speculation.integration.live_state import (
    _SNAPSHOT_MAX_VALUE_CHARS,
    _SNAPSHOT_PROBE,
    _live_state_seed,
    _pure_assigned_names,
    _snapshot_reads,
)
from dspy_rlm_hooks.speculation.integration.registry import (
    _active_speculators,
    _close_speculators,
    _extract_sub_lm_text,
    _has_speculatable,
    _make_llm_spec_fns,
    _placeholder,
    _prediction_type,
    _register_classifications,
    _register_speculator,
    _sync_registry_fns,
    _unregister_speculator,
)
from dspy_rlm_hooks.speculation.integration.streaming_turn import (
    _maybe_begin_streaming_turn,
    _StreamingGenerateAction,
)
from dspy_rlm_hooks.speculation.shadow import shadow_builtins
from dspy_rlm_hooks.speculation.speculator import Speculator
from dspy_rlm_hooks.speculation.streaming import _free_names

__all__ = [
    # public API
    "enable_rlm_speculation",
    "disable_rlm_speculation",
    # registry
    "_prediction_type",
    "_active_speculators",
    "_close_speculators",
    "_register_speculator",
    "_unregister_speculator",
    "_placeholder",
    "_extract_sub_lm_text",
    "_make_llm_spec_fns",
    "_register_classifications",
    "_has_speculatable",
    "_sync_registry_fns",
    # claim hooks
    "_LLM_TOOLS",
    "_make_claim_hook",
    "_install_claim_hooks",
    # live state
    "_SNAPSHOT_MAX_VALUE_CHARS",
    "_SNAPSHOT_PROBE",
    "_snapshot_reads",
    "_pure_assigned_names",
    "_live_state_seed",
    # streaming turn
    "_maybe_begin_streaming_turn",
    "_StreamingGenerateAction",
    # execute
    "_speculation_execute_iteration",
    "_speculation_aexecute_iteration",
    "_speculation_execute_code",
    # names that were module-level imports of the original integration.py and
    # therefore part of its importable surface
    "Any",
    "Callable",
    "MethodType",
    "SpeculationConfig",
    "Speculator",
    "TYPE_CHECKING",
    "ThreadPoolExecutor",
    "_assemble_execution_code",
    "_free_names",
    "_validate_rlm",
    "ast",
    "atexit",
    "builtins",
    "fully_raw",
    "inspect",
    "shadow_builtins",
    "tag_claim_hook",
    "weakref",
    "wraps",
]
