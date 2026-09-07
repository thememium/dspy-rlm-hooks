"""Speculative execution engine for DSPy RLM.

Task 1 (contracts): this package freezes the shared configuration and tool
contracts that the store/budget (Task 2) and streaming (Task 3) engines import
and build against. No store/budget/shadow implementation lives here yet.
"""

from __future__ import annotations

from dspy_rlm_hooks.speculation.budget import Budget
from dspy_rlm_hooks.speculation.config import SpeculationConfig, SpeculationPolicy
from dspy_rlm_hooks.speculation.hooks import (
    ShadowBudgetDenied,
    make_baseline_hooks,
    make_real_hooks,
    make_shadow_hooks,
)
from dspy_rlm_hooks.speculation.session import (
    EventBus,
    Launcher,
    SpecSession,
    StreamTurn,
    ToolRegistry,
)
from dspy_rlm_hooks.speculation.shadow import (
    Opaque,
    ShadowAborted,
    ShadowRunner,
    shadow_builtins,
    snapshot_ns,
)
from dspy_rlm_hooks.speculation.speculator import Speculator
from dspy_rlm_hooks.speculation.store import SpecStore, Speculation
from dspy_rlm_hooks.speculation.streaming import (
    MAX_UNROLL,
    Plan,
    Segment,
    StreamSegmenter,
    Unresolvable,
    plan_peeks,
    repair_tail,
    safe_eval,
)
from dspy_rlm_hooks.speculation.tool import (
    NonSpeculated,
    Spec,
    SpecKey,
    SpecTool,
    SpeculativeTool,
    SpeculativeToolRequest,
    SpecValue,
    ToolSpec,
    canonical_hash,
    contains_nonspec,
    spec_key,
    speculate,
    speculative,
)

__all__ = [
    "SpeculationConfig",
    "SpeculationPolicy",
    "ToolSpec",
    "SpeculativeTool",
    "SpeculativeToolRequest",
    "SpecKey",
    "SpecValue",
    "NonSpeculated",
    "Speculation",
    "SpecStore",
    "Budget",
    "canonical_hash",
    "spec_key",
    "contains_nonspec",
    "speculate",
    "speculative",
    "Spec",
    "SpecTool",
    "Segment",
    "StreamSegmenter",
    "Plan",
    "Unresolvable",
    "MAX_UNROLL",
    "repair_tail",
    "safe_eval",
    "plan_peeks",
    "ShadowRunner",
    "Opaque",
    "ShadowAborted",
    "shadow_builtins",
    "snapshot_ns",
    "SpecSession",
    "Launcher",
    "StreamTurn",
    "EventBus",
    "ToolRegistry",
    "make_real_hooks",
    "make_shadow_hooks",
    "make_baseline_hooks",
    "ShadowBudgetDenied",
    "Speculator",
]
