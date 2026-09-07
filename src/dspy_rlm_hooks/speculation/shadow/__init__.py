"""ShadowRunner: subprocess-isolated speculative execution of generated code.

Task 4 — builds against the frozen contracts from Task 1 (``tool.py`` /
``config.py``), the store (Task 2 ``store.py``) and streaming (Task 3
``streaming.py``).

Per the SPIKE (Task 0) decision, the shadow runs in a SEPARATE SUBPROCESS, not
in the host process: the in-process jail blocks ``__import__``/``open``/``eval``
but cannot block the object-introspection escape
(``().__class__.__mro__[1].__subclasses__()``), which reaches real host objects.
In a subprocess that escape can only touch the subprocess's own memory, never
the host's. The real Deno interpreter already runs in a subprocess, so a
subprocess shadow is architecturally consistent.

The worker process is seeded with the assembled namespace (deepcopy-forked via
``snapshot_ns``) + jailed builtins + recording shadow hooks. It executes each
:class:`Segment`, records the predicted ``(tool, args)`` calls, and streams them
back to the parent over a ``multiprocessing`` pipe for claim-hook dispatch. A
runaway watchdog (SIGALRM in the worker) bounds each statement's wall-clock
time, and the parent's ``join(timeout)`` bounds the whole shadow so a hang never
blocks real execution.

The original single module is split into focused submodules, all re-exported
here so every historical import path keeps working: :mod:`snapshot` (namespace
snapshotting and opaque values), :mod:`builtins` (jailed builtins),
:mod:`worker` (subprocess side), :mod:`runner` (parent-side driver), and
:mod:`analysis` (AST name analysis and the per-statement wall-clock budget).
"""

from __future__ import annotations

import ast
import builtins as _builtins
import copy
import multiprocessing
import pickle
import signal
import threading
from types import FunctionType, SimpleNamespace
from typing import Any

from dspy_rlm_hooks.speculation.shadow.analysis import (
    STMT_WALL_BUDGET_S,
    _bound_names,
    _comp_local_names,
    _read_names,
)
from dspy_rlm_hooks.speculation.shadow.builtins import (
    _SHADOW_BLOCKED,
    _SHADOW_IMPORT_WHITELIST,
    _blocked,
    _shadow_import,
    _shadow_print,
    shadow_builtins,
)
from dspy_rlm_hooks.speculation.shadow.runner import ShadowRunner
from dspy_rlm_hooks.speculation.shadow.snapshot import (
    Opaque,
    _picklable_ns,
    _rebind,
    classify_ns,
    load_ns,
    snapshot_ns,
)
from dspy_rlm_hooks.speculation.shadow.worker import (
    ShadowAborted,
    _make_record_hook,
    _mp_context,
    _plan_tainted_segment,
    _register_segment_productions,
    _shadow_worker,
    _worker_exec,
    _worker_fire_chain,
    _worker_peek,
)
from dspy_rlm_hooks.speculation.store import SpecStore
from dspy_rlm_hooks.speculation.streaming import (
    ChainMeta,
    ChainPlan,
    Segment,
    _ContCounter,
    _plan_body,
    plan_peeks_with_chains,
    safe_eval,
)
from dspy_rlm_hooks.speculation.tool import (
    NonSpeculated,
    canonical_hash,
    contains_nonspec,
    spec_key,
    split_batch_call,
)

__all__ = [
    "Any",
    "ChainMeta",
    "ChainPlan",
    "FunctionType",
    "NonSpeculated",
    "Opaque",
    "STMT_WALL_BUDGET_S",
    "Segment",
    "ShadowAborted",
    "ShadowRunner",
    "SimpleNamespace",
    "SpecStore",
    "_ContCounter",
    "_SHADOW_BLOCKED",
    "_SHADOW_IMPORT_WHITELIST",
    "_blocked",
    "_bound_names",
    "_builtins",
    "_comp_local_names",
    "_make_record_hook",
    "_mp_context",
    "_picklable_ns",
    "_plan_body",
    "_plan_tainted_segment",
    "_read_names",
    "_rebind",
    "_register_segment_productions",
    "_shadow_import",
    "_shadow_print",
    "_shadow_worker",
    "_worker_exec",
    "_worker_fire_chain",
    "_worker_peek",
    "annotations",
    "ast",
    "canonical_hash",
    "classify_ns",
    "contains_nonspec",
    "copy",
    "load_ns",
    "multiprocessing",
    "pickle",
    "plan_peeks_with_chains",
    "safe_eval",
    "shadow_builtins",
    "signal",
    "snapshot_ns",
    "spec_key",
    "split_batch_call",
    "threading",
]
