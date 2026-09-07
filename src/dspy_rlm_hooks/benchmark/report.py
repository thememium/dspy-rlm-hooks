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
import json
import statistics
from pathlib import Path
from typing import Any

from dspy_rlm_hooks.benchmark.runner import TrialRecord, _strip_timings, run_ab
from dspy_rlm_hooks.benchmark.scenario import default_scenario


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
