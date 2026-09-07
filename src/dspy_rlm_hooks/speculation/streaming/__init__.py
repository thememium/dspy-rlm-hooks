"""Streaming side: statement segmentation of the token stream + tail peeking
(SafeEval over live state, pre-close loop unrolling).

Task 3 — builds against the frozen contracts from Task 1 (``tool.py`` /
``config.py``). This package is pure parsing/planning: it never executes code.
The shadow (Task 4) consumes the emitted :class:`Segment` objects and the
planned :class:`Plan` objects; the session/hooks (Task 5) build against this
surface.

The Lazy/JIT stage uses :meth:`StreamSegmenter.feed_complete` (wrap the whole
code block as one block, then feed). The streaming follow-up uses
:meth:`StreamSegmenter.feed` with raw model-output deltas.

Submodules:
- ``segmenter``: statement segmentation (:class:`StreamSegmenter`) and tail
  repair (:func:`repair_tail`).
- ``evaluator``: the pure, bounded expression evaluator (:func:`safe_eval`).
- ``planning``: peek planning (:func:`plan_peeks`,
  :func:`plan_peeks_with_chains`, :class:`Plan`, chained continuations).
"""

from __future__ import annotations

import ast
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from dspy_rlm_hooks.speculation.streaming.evaluator import (
    _PURE_BUILTINS,
    _PURE_STR_METHODS,
    Unresolvable,
    _bind,
    _reject_weird,
    safe_eval,
)
from dspy_rlm_hooks.speculation.streaming.planning import (
    _MUTATING_METHODS,
    MAX_UNROLL,
    ChainMeta,
    ChainPlan,
    DepRef,
    Plan,
    _assigned_names,
    _assigned_names_set,
    _base_names,
    _call_closed_in,
    _ContCounter,
    _free_names,
    _hooked_calls,
    _no_calls_after,
    _parse_repaired,
    _plan_body,
    _resolve_call_or_chain,
    _single_assign_target,
    _target_names,
    _unroll_for,
    plan_peeks,
    plan_peeks_with_chains,
)
from dspy_rlm_hooks.speculation.streaming.segmenter import (
    _COMPOUND,
    _CONTINUATION,
    _PYTHON_FENCE_LANGS,
    _TAIL_LIMIT,
    Segment,
    StreamSegmenter,
    _BlockState,
    _bracket_closers,
    _is_repl_open,
    _last_line_indented,
    _scan_line_state,
    repair_tail,
)
from dspy_rlm_hooks.speculation.tool import SpecKey, canonical_hash

__all__ = [
    "MAX_UNROLL",
    "Plan",
    "Segment",
    "SpecKey",
    "StreamSegmenter",
    "Unresolvable",
    "_BlockState",
    "_COMPOUND",
    "_CONTINUATION",
    "_ContCounter",
    "_MUTATING_METHODS",
    "_PURE_BUILTINS",
    "_PURE_STR_METHODS",
    "_PYTHON_FENCE_LANGS",
    "_TAIL_LIMIT",
    "_assigned_names",
    "_assigned_names_set",
    "_base_names",
    "_bind",
    "_bracket_closers",
    "_call_closed_in",
    "_free_names",
    "_hooked_calls",
    "_is_repl_open",
    "_last_line_indented",
    "_no_calls_after",
    "_parse_repaired",
    "_plan_body",
    "_reject_weird",
    "_resolve_call_or_chain",
    "_scan_line_state",
    "_single_assign_target",
    "_target_names",
    "_unroll_for",
    "Any",
    "Callable",
    "ChainMeta",
    "ChainPlan",
    "DepRef",
    "annotations",
    "ast",
    "canonical_hash",
    "dataclass",
    "field",
    "plan_peeks",
    "plan_peeks_with_chains",
    "repair_tail",
    "safe_eval",
]
