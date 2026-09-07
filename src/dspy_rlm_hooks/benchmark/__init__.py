"""Deterministic A/B benchmark for DSPy RLM execution variants.

The problem with benchmarking RLM against a real LM is variance: the main-LM
latency dominates and swamps the effect under test. This module removes the LM
from the equation entirely:

- a SCRIPTED ``generate_action`` returns the same ``(reasoning, code)`` for
  every iteration in every variant (and feeds the code through the production
  streaming path when speculation is on);
- fake tools sleep for FIXED latencies and return deterministic strings;
- a fake ``sub_lm`` gives ``llm_query``/``llm_query_batched`` fixed latency and
  deterministic outputs.

Identical scripted steps + identical tool outputs mean the only thing that can
differ between variants is execution scheduling — exactly what speculative
tool calling changes. Every iteration's code and result are recorded and the
report asserts they are IDENTICAL across variants (timing lines stripped).

Metrics per variant (median over repeats):
- wall_seconds           end-to-end ``forward()``
- generate_ms            time in ``generate_action`` (scripted; measures
                         speculation's generation-path overhead)
- execute_ms             time in ``_execute_code`` (interpreter work)
- critical_path_tool_ms  interpreter-blocking tool time, measured INSIDE the
                         scripted code with ``time.perf_counter`` — identical
                         instrumentation in every variant
- serial_tool_ms         sum of ACTUAL tool-fn durations; speculative
                         executions are flagged via ``guards.current_spec()``
- spec_stats             {speculated, claimed, evicted} (spec variants)

Usage (in a venv with dspy + dspy-rlm-hooks installed)::

    python -m dspy_rlm_hooks.benchmark --variants default spec --repeats 5
    python -m dspy_rlm_hooks.benchmark --variant-fn mymod:my_variant --repeats 3

Custom variants (for benchmarking non-speculation changes) take the built RLM
plus the tool mapping and may reconfigure it however they like::

    def my_variant(rlm, tools):
        enable_rlm_hooks(rlm, pre_iteration_hook=my_hook)
"""

from __future__ import annotations

import argparse  # noqa: F401
import inspect  # noqa: F401
import json  # noqa: F401
import statistics  # noqa: F401
import time  # noqa: F401
from dataclasses import dataclass, field  # noqa: F401
from pathlib import Path  # noqa: F401
from typing import Any, Callable  # noqa: F401

from dspy_rlm_hooks.benchmark.fakes import (
    _FakeSubLM,
    _fence,
    _make_tool_fn,
    _scripted_generate_action,
)
from dspy_rlm_hooks.benchmark.report import _aggregate, _print_report, main
from dspy_rlm_hooks.benchmark.runner import (
    _TIMINGS_RE,
    TrialRecord,
    _build_program,
    _instrument,
    _parse_timings,
    _strip_timings,
    run_ab,
    run_variant,
)
from dspy_rlm_hooks.benchmark.scenario import (
    Scenario,
    ScriptedIteration,
    ToolPlan,
    default_scenario,
)
from dspy_rlm_hooks.benchmark.variants import (
    NAMED_VARIANTS,
    _apply_variant,
    _teardown,
    _variant_default,
    _variant_hooks,
    _variant_spec,
)

__all__ = [
    "ScriptedIteration",
    "Scenario",
    "default_scenario",
    "run_variant",
    "run_ab",
    "main",
    # Explicit re-exports of the complete pre-split module surface so that
    # `import dspy_rlm_hooks.benchmark as B` keeps working for every name
    # that worked before, including private names (tests patch and call
    # several of them, and the CLI resolves `benchmark:_variant_default`).
    "NAMED_VARIANTS",
    "ToolPlan",
    "TrialRecord",
    "_FakeSubLM",
    "_TIMINGS_RE",
    "_aggregate",
    "_apply_variant",
    "_build_program",
    "_fence",
    "_instrument",
    "_make_tool_fn",
    "_parse_timings",
    "_print_report",
    "_scripted_generate_action",
    "_strip_timings",
    "_teardown",
    "_variant_default",
    "_variant_hooks",
    "_variant_spec",
]
