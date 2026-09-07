"""Trial runner for the deterministic A/B benchmark: builds the instrumented
program for each repeat, applies the variant, and runs the scripted scenario."""

from __future__ import annotations

import inspect
import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from dspy_rlm_hooks.benchmark.fakes import (
    _FakeSubLM,
    _make_tool_fn,
    _scripted_generate_action,
)
from dspy_rlm_hooks.benchmark.scenario import Scenario, default_scenario
from dspy_rlm_hooks.benchmark.variants import _apply_variant, _teardown

# ---------------------------------------------------------------------------
# instrumentation (uniform across variants)
# ---------------------------------------------------------------------------


@dataclass
class TrialRecord:
    iterations: list[dict] = field(
        default_factory=list
    )  # {code, result, exec_ms, gen_ms}
    tool_events: list[dict] = field(default_factory=list)
    critical_path_ms: dict = field(default_factory=dict)
    wall_s: float = 0.0
    spec_stats: dict = field(default_factory=dict)
    bus_events: dict = field(default_factory=dict)
    iter_ms: list[float] = field(default_factory=list)
    exec: list[dict] = field(default_factory=list)


def _instrument(program: Any, record: TrialRecord) -> None:
    """Wrap ``_execute_iteration``/``_execute_code`` — the same choke points the
    speculation integration uses — so generation vs execution time is split
    identically in every variant."""
    orig_iter = program._execute_iteration
    orig_code = program._execute_code

    def timed_iteration(
        repl, variables, history, iteration, input_args, output_field_names
    ):
        t0 = time.perf_counter()
        out = orig_iter(
            repl, variables, history, iteration, input_args, output_field_names
        )
        record.iter_ms.append((time.perf_counter() - t0) * 1000)
        return out

    def timed_code(repl, code, input_args):
        t0 = time.perf_counter()
        result = orig_code(repl, code, input_args)
        record.exec.append(
            {
                "code": code,
                "result": str(result),
                "ms": (time.perf_counter() - t0) * 1000,
            }
        )
        return result

    program._execute_iteration = timed_iteration
    program._execute_code = timed_code


_TIMINGS_RE = None


def _strip_timings(result_text: str) -> str:
    out = []
    for line in result_text.splitlines():
        if line.startswith("TIMINGS "):
            continue
        out.append(line)
    return "\n".join(out)


def _parse_timings(result_text: str) -> dict:
    for line in result_text.splitlines():
        if line.startswith("TIMINGS "):
            try:
                return json.loads(line[len("TIMINGS ") :])
            except Exception:
                return {}
    return {}


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------


def _build_program(
    scenario: Scenario, record: TrialRecord
) -> tuple[Any, dict[str, Callable]]:
    import dspy

    tool_fns = {
        plan.name: _make_tool_fn(plan, record.tool_events) for plan in scenario.tools
    }
    sub_lm = _FakeSubLM(scenario.llm_latency_ms, record.tool_events)

    kwargs: dict[str, Any] = {
        "tools": list(tool_fns.values()),
        "sub_lm": sub_lm,
        "max_llm_calls": scenario.max_llm_calls,
    }
    params = inspect.signature(dspy.RLM.__init__).parameters
    iter_kw = "max_iters" if "max_iters" in params else "max_iterations"
    kwargs[iter_kw] = len(scenario.iterations)
    program = dspy.RLM("question -> answer", **kwargs)
    return program, tool_fns


def run_variant(
    scenario: Scenario,
    variant: Any = "default",
    repeats: int = 3,
    pace_ms: float = 1.0,
    question: str = "benchmark: scripted scenario",
) -> dict:
    """Run the scripted scenario ``repeats`` times in one variant; aggregate."""
    repeats_out: list[TrialRecord] = []
    for _ in range(repeats):
        rec = TrialRecord()
        program, tool_fns = _build_program(scenario, rec)
        variant_name = _apply_variant(program, variant, tool_fns)
        program.generate_action = _scripted_generate_action(program, scenario, pace_ms)
        _instrument(program, rec)

        t0 = time.perf_counter()
        try:
            program(question=question)
        finally:
            rec.wall_s = time.perf_counter() - t0

        # stitch iteration records: generate_ms = iteration_ms - code exec ms
        for i, ex in enumerate(rec.exec):
            iter_ms = rec.iter_ms[i] if i < len(rec.iter_ms) else 0.0
            rec.iterations.append(
                {
                    "index": i,
                    "code": ex["code"],
                    "result": ex["result"],
                    "exec_ms": round(ex["ms"], 2),
                    "iteration_ms": round(iter_ms, 2),
                    "generate_ms": round(max(iter_ms - ex["ms"], 0.0), 2),
                }
            )
        merged: dict = {}
        for ex in rec.exec:
            for label, elapsed_ms in _parse_timings(ex["result"]).items():
                merged[label] = merged.get(label, 0.0) + elapsed_ms
        rec.critical_path_ms = merged

        speculator = getattr(program, "_speculator", None)
        if speculator is not None:
            try:
                rec.spec_stats = speculator.stats() or {}
                from collections import Counter

                rec.bus_events = dict(
                    Counter(kind for kind, _ in speculator.session.bus.history)
                )
            except Exception:
                rec.spec_stats = {"error": "stats unavailable"}
        _teardown(program, variant_name)
        repeats_out.append(rec)

    from dspy_rlm_hooks.benchmark.report import _aggregate

    return _aggregate(variant_name, repeats_out)


def run_ab(
    scenario: Scenario | None = None,
    variants: list[Any] | None = None,
    repeats: int = 3,
    pace_ms: float = 1.0,
    question: str = "benchmark: scripted scenario",
) -> dict:
    """Run every variant and produce the comparison report.

    The report asserts STEP EQUIVALENCE: every iteration's code and
    (timing-stripped) result must be identical across variants, so a speedup
    only counts if all variants executed the exact same steps with the exact
    same outputs.
    """
    scenario = scenario or default_scenario()
    variants = variants or ["default", "spec"]
    results: dict[str, dict] = {}
    for v in variants:
        results[v if isinstance(v, str) else getattr(v, "__name__", "custom")] = (
            run_variant(
                scenario, v, repeats=repeats, pace_ms=pace_ms, question=question
            )
        )

    names = list(results)
    baseline = names[0]
    report: dict = {"scenario": scenario.name, "repeats": repeats, "variants": results}
    report["equivalent_steps"] = True
    report["mismatches"] = []
    for other in names[1:]:
        a, b = results[baseline]["steps"], results[other]["steps"]
        if a != b:
            report["equivalent_steps"] = False
            report["mismatches"].append({"baseline": baseline, "variant": other})
    med = lambda v, k: results[v][k]["median"]  # noqa: E731
    report["comparison"] = {}
    for other in names[1:]:
        wall = (
            med(baseline, "wall_s") / med(other, "wall_s")
            if med(other, "wall_s")
            else 0.0
        )
        tc = (
            med(baseline, "tool_critical_ms") / med(other, "tool_critical_ms")
            if med(other, "tool_critical_ms")
            else 0.0
        )
        report["comparison"][other] = {
            "wall_speedup_vs_baseline": round(wall, 3),
            "tool_critical_speedup_vs_baseline": round(tc, 3),
        }
    return report
