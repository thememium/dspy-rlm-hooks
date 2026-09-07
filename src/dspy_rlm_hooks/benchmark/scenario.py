"""Scripted scenarios for the deterministic A/B benchmark.

A :class:`Scenario` pins the iterations, fake-tool latencies, and sub-LM
latency every variant executes, so the only free variable left is
scheduling (see the package docstring for the full rationale).
"""

from __future__ import annotations

from dataclasses import dataclass

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
