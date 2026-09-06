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

import argparse
import inspect
import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from dspy_rlm_hooks.speculation.guards import current_spec as _current_spec

__all__ = [
    "ScriptedIteration",
    "Scenario",
    "default_scenario",
    "run_variant",
    "run_ab",
    "main",
]


# ---------------------------------------------------------------------------
# scenario
# ---------------------------------------------------------------------------


@dataclass
class ScriptedIteration:
    """One iteration's scripted (reasoning, code) — identical in every variant."""

    reasoning: str
    code: str


@dataclass
class ToolPlan:
    """A fake tool: fixed latency, deterministic result."""

    name: str
    latency_ms: float
    result: str


@dataclass
class Scenario:
    """The full scripted run: iterations + tool plan + sub-LM latency."""

    name: str
    iterations: list[ScriptedIteration]
    tools: list[ToolPlan]
    llm_latency_ms: float = 150.0
    max_llm_calls: int = 50


def default_scenario(tool_ms: float = 60.0, llm_ms: float = 150.0) -> Scenario:
    """A 4-iteration scripted task exercising all speculation surfaces:

    1. one fast call + two parallel reads (constant args -> peeked + claimed);
    2. a dependent chain (args from prior repl state -> honest miss path);
    3. two llm_query calls inside a loop (constant args -> claimed);
    4. SUBMIT.

    Every tool call's interpreter-blocking time is captured in a TIMINGS line
    the benchmark parses back out (identical instrumentation in all variants).
    """
    tms = float(tool_ms)
    # Iterations use COMPOUND statements (for-loops over literal lists) on
    # purpose: while the model is still streaming the loop, the unclosed tail
    # contains the tool calls, exactly like real model code, so the peek
    # planner can pre-dispatch them. Flat one-statement-per-line code closes
    # every statement at its newline and would never expose a peekable tail.
    iterations = [
        ScriptedIteration(
            reasoning="Read both sources inside a loop (peekable tail).",
            code=(
                "import time\n"
                "import json\n"
                "now = time.perf_counter\n"
                "timings = {}\n"
                "sources = {}\n"
                'for name in ["alpha", "beta"]:\n'
                "    t0 = now()\n"
                "    sources[name] = read_source(name)\n"
                '    timings["read:" + name] = round((now() - t0) * 1000, 1)\n'
                "t0 = now()\n"
                "index = list_reports()\n"
                'timings["list_reports"] = round((now() - t0) * 1000, 1)\n'
                'print("TIMINGS " + json.dumps(timings))\n'
                'print("SOURCES:" + sources["alpha"][:14] + "|" + sources["beta"][:14])\n'
                'print("INDEX:" + index)\n'
            ),
        ),
        ScriptedIteration(
            reasoning="Analyze both sources (depends on prior repl state).",
            code=(
                "import time\n"
                "import json\n"
                "now = time.perf_counter\n"
                "timings = {}\n"
                "t0 = now()\n"
                'an_a = analyze(sources["alpha"])\n'
                'timings["analyze:alpha"] = round((now() - t0) * 1000, 1)\n'
                "t0 = now()\n"
                'an_b = analyze(sources["beta"])\n'
                'timings["analyze:beta"] = round((now() - t0) * 1000, 1)\n'
                'print("TIMINGS " + json.dumps(timings))\n'
                'print("OUT:" + an_a + "|" + an_b)\n'
            ),
        ),
        ScriptedIteration(
            reasoning="Sub-LLM summary inside a loop, then a batched review.",
            code=(
                "import time\n"
                "import json\n"
                "now = time.perf_counter\n"
                "timings = {}\n"
                "out = {}\n"
                'for kind in ["summary", "review"]:\n'
                "    t0 = now()\n"
                '    out[kind] = llm_query(kind + ": alpha")\n'
                '    timings["llm_query:" + kind] = round((now() - t0) * 1000, 1)\n'
                "t0 = now()\n"
                'reviews = llm_query_batched(["review-a", "review-b", "review-c"])\n'
                'timings["llm_query_batched"] = round((now() - t0) * 1000, 1)\n'
                'print("TIMINGS " + json.dumps(timings))\n'
                'print("SUB:" + out["summary"])\n'
                'print("REV:" + out["review"])\n'
            ),
        ),
        ScriptedIteration(
            reasoning="Submit.",
            code='SUBMIT("SCRIPTED|" + out["summary"][:24] + "|" + str(len(reviews)))\n',
        ),
    ]
    tools = [
        ToolPlan("list_reports", tms / 2, "alpha\nbeta"),
        ToolPlan("read_source", tms, "SOURCE-{}-" + "x" * 64),
        ToolPlan("analyze", tms * 2, "ANALYSIS-OK"),
    ]
    return Scenario(
        name="default", iterations=iterations, tools=tools, llm_latency_ms=llm_ms
    )


# ---------------------------------------------------------------------------
# fake sub-LM + fake tools
# ---------------------------------------------------------------------------


class _FakeSubLM:
    """Deterministic sub-LM: fixed latency, fixed output shape.

    ``llm_query``/``llm_query_batched`` call it; the speculation launcher calls
    it from worker threads (``current_spec()`` set) and the real interpreter
    calls it on its own thread (``current_spec()`` None), so every execution is
    attributable.
    """

    def __init__(self, latency_ms: float, record: list) -> None:
        self.latency_ms = latency_ms
        self.record = record

    def __call__(self, prompt: str):
        spec = _current_spec()
        t0 = time.perf_counter()
        time.sleep(self.latency_ms / 1000.0)
        ms = (time.perf_counter() - t0) * 1000
        self.record.append(
            {
                "tool": "sub_lm",
                "ms": round(ms, 2),
                "speculative": spec is not None,
                "prompt": prompt[:48],
            }
        )
        return [{"text": f"SCRIPTED[{prompt[:40]}]"}]


def _make_tool_fn(plan: ToolPlan, record: list) -> Callable[..., Any]:
    """A fake tool with OPTIONAL parameters.

    The parameters must have defaults: DSPy registers the tool's parameters
    with the sandbox, and a ``*args``/``**kwargs`` signature registers
    ``args``/``kwargs`` as REQUIRED positionals, breaking zero-arg calls.
    """

    def fn(a=None, b=None, c=None, d=None) -> str:
        spec = _current_spec()
        t0 = time.perf_counter()
        time.sleep(plan.latency_ms / 1000.0)
        ms = (time.perf_counter() - t0) * 1000
        record.append(
            {
                "tool": plan.name,
                "ms": round(ms, 2),
                "speculative": spec is not None,
                "args": repr(a)[:64],
            }
        )
        if "{}" in plan.result and a is not None:
            return plan.result.format(a)
        return plan.result

    fn.__name__ = plan.name  # DSPy names tools from __name__
    fn.__qualname__ = plan.name
    fn.__doc__ = (
        f"Benchmark tool {plan.name}(a, b, c, d) — sleeps {plan.latency_ms} ms "
        "and returns a fixed deterministic string."
    )
    return fn


# ---------------------------------------------------------------------------
# variants
# ---------------------------------------------------------------------------


def _variant_default(rlm: Any, tools: dict[str, Callable]) -> None:
    """Plain RLM — no hooks at all."""


def _variant_spec(rlm: Any, tools: dict[str, Callable]) -> None:
    """Speculative programmatic tool calling (the engine under test)."""
    from dspy_rlm_hooks import enable_rlm_speculation

    enable_rlm_speculation(
        rlm,
        tools={name: (fn, {"deterministic": True}) for name, fn in tools.items()},
        speculate_user_tools=True,
        streaming=True,
        timeout_s=5.0,
    )


def _variant_hooks(rlm: Any, tools: dict[str, Callable]) -> None:
    """Lifecycle hooks enabled with a no-op hook (for timing hook overhead)."""
    from dspy_rlm_hooks import enable_rlm_hooks

    enable_rlm_hooks(rlm)


NAMED_VARIANTS: dict[str, Callable[[Any, dict[str, Callable]], None]] = {
    "default": _variant_default,
    "spec": _variant_spec,
    "hooks": _variant_hooks,
}


# ---------------------------------------------------------------------------
# scripted generate_action
# ---------------------------------------------------------------------------


def _fence(body: str) -> str:
    return f"```python\n{body}\n```"


def _scripted_generate_action(
    rlm: Any, scenario: Scenario, pace_ms: float
) -> Callable[..., Any]:
    """Replace ``generate_action`` with a deterministic scripted one.

    When a streaming turn is active (speculation enabled), the code is fed
    through the SAME production path (``StreamTurn.feed`` on chunk deltas) so
    peek/dispatch overlap with generation is exercised honestly. Every variant
    receives the same chunk-arrival delays, including the non-streaming baseline.
    """

    def generate_action(variables_info: Any, repl_history: Any, iteration: Any) -> Any:
        idx = int(str(iteration).split("/")[0]) - 1
        it = scenario.iterations[idx]
        fenced = _fence(it.code)
        turn = getattr(rlm, "_active_stream_turn", None)
        if turn is not None:
            rlm._streaming_fed_any = True
        for i in range(0, len(fenced), 8):
            if turn is not None:
                turn.feed(fenced[i : i + 8])
            if pace_ms > 0:
                time.sleep(pace_ms / 1000.0)
        from dspy import Prediction

        return Prediction(reasoning=it.reasoning, code=fenced)

    return generate_action


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


def _apply_variant(program: Any, variant: Any, tool_fns: dict[str, Callable]) -> str:
    if isinstance(variant, str):
        fn = NAMED_VARIANTS.get(variant)
        if fn is None:
            raise ValueError(
                f"unknown variant {variant!r}; known: {sorted(NAMED_VARIANTS)}"
            )
        name = variant
    elif callable(variant):
        fn, name = variant, getattr(variant, "__name__", "custom")
    else:
        raise TypeError("variant must be a name or a callable(rlm, tools)")
    fn(program, tool_fns)
    return name


def _teardown(program: Any, variant_name: str) -> None:
    from dspy_rlm_hooks import disable_rlm_speculation

    try:
        disable_rlm_speculation(program)
    except Exception:
        pass
    try:
        from dspy_rlm_hooks import disable_rlm_hooks

        disable_rlm_hooks(program)
    except Exception:
        pass


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

    return _aggregate(variant_name, repeats_out)


def _aggregate(name: str, recs: list[TrialRecord]) -> dict:
    def med(xs):
        return round(statistics.median(xs), 3) if xs else 0.0

    def stats(xs):
        if not xs:
            return {"median": 0.0, "min": 0.0, "max": 0.0, "stdev": 0.0}
        return {
            "median": round(statistics.median(xs), 3),
            "min": round(min(xs), 3),
            "max": round(max(xs), 3),
            "stdev": round(statistics.stdev(xs), 3) if len(xs) > 1 else 0.0,
        }

    walls = [r.wall_s for r in recs]
    gens = [sum(i["generate_ms"] for i in r.iterations) for r in recs]
    execs = [sum(i["exec_ms"] for i in r.iterations) for r in recs]
    crit = [sum(r.critical_path_ms.values()) for r in recs]
    serial = [
        round(sum(e["ms"] for e in r.tool_events if not e["speculative"]), 2)
        for r in recs
    ]
    claimed = [r.spec_stats.get("claimed", 0) for r in recs]
    speculated = [r.spec_stats.get("speculated", 0) for r in recs]
    evicted = [r.spec_stats.get("evicted", 0) for r in recs]
    steps_sig = (
        [[i["code"], _strip_timings(i["result"])] for i in recs[0].iterations]
        if recs
        else []
    )
    return {
        "variant": name,
        "wall_s": stats(walls),
        "generate_ms_total": stats(gens),
        "execute_ms_total": stats(execs),
        "tool_critical_ms": stats(crit),
        "tool_serial_ms": stats(serial),
        "spec_stats_median": {
            "speculated": med(speculated),
            "claimed": med(claimed),
            "evicted": med(evicted),
        },
        "iterations": [
            {
                "index": i["index"],
                "code_head": i["code"][:60],
                "result_head": _strip_timings(i["result"])[:60],
                "exec_ms": i["exec_ms"],
                "generate_ms": i["generate_ms"],
            }
            for i in recs[0].iterations
        ]
        if recs
        else [],
        "steps_signature": steps_sig,
        "steps": steps_sig,
    }


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


def _print_report(report: dict) -> None:
    print(f"scenario: {report['scenario']}  repeats: {report['repeats']}")
    print(
        f"{'variant':<10} {'wall(s) med':>12} {'stdev':>8} {'gen(ms)':>10} "
        f"{'exec(ms)':>10} {'tool_crit(ms)':>14} {'tool_serial(ms)':>16} "
        f"{'spec/disp':>10} {'claimed':>8}"
    )
    for name, res in report["variants"].items():
        w = res["wall_s"]
        print(
            f"{name:<10} {w['median']:>12.3f} {w['stdev']:>8.3f} "
            f"{res['generate_ms_total']['median']:>10.1f} {res['execute_ms_total']['median']:>10.1f} "
            f"{res['tool_critical_ms']['median']:>14.1f} {res['tool_serial_ms']['median']:>16.1f} "
            f"{res['spec_stats_median']['speculated']:>10.0f} {res['spec_stats_median']['claimed']:>8.0f}"
        )
    for other, cmp in report["comparison"].items():
        print(
            f"speedup[{other} vs baseline]: wall x{cmp['wall_speedup_vs_baseline']}, "
            f"tool-critical x{cmp['tool_critical_speedup_vs_baseline']}"
        )
    print(f"steps identical across variants: {report['equivalent_steps']}")
    if report["mismatches"]:
        print(f"MISMATCHES: {report['mismatches']}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--variants",
        nargs="+",
        default=["default", "spec"],
        help="variant names or module:fn callables\n",
    )
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--tool-ms", type=float, default=60.0)
    ap.add_argument("--llm-ms", type=float, default=150.0)
    ap.add_argument(
        "--pace-ms",
        type=float,
        default=1.0,
        help="simulated token-arrival pacing per 8-char chunk",
    )
    ap.add_argument("--json-out", default=None, help="write the full report JSON here")
    args = ap.parse_args(argv)

    raw_variants = (
        args.variants
        if isinstance(args.variants, list)
        else str(args.variants).split(",")
    )
    variant_names = [v.strip() for v in raw_variants if v.strip()]
    variants: list[Any] = []
    for name in variant_names:
        if ":" in name:
            mod_name, fn_name = name.split(":", 1)
            import importlib

            variants.append(getattr(importlib.import_module(mod_name), fn_name))
        else:
            variants.append(name)

    scenario = default_scenario(tool_ms=args.tool_ms, llm_ms=args.llm_ms)
    report = run_ab(scenario, variants, repeats=args.repeats, pace_ms=args.pace_ms)
    _print_report(report)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=1, default=str))
        print(f"report written: {args.json_out}")
    return 0 if report["equivalent_steps"] else 1


if __name__ == "__main__":
    raise SystemExit(main())  # pragma: no cover
