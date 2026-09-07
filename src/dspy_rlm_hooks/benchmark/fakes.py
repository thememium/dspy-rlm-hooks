"""Fake sub-LM, fake tools, and the scripted ``generate_action``.

These stand-ins give fixed latencies and deterministic outputs, so identical
scripted steps produce identical results in every variant.
"""

from __future__ import annotations

import time
from typing import Any, Callable

from dspy_rlm_hooks.benchmark.scenario import Scenario, ToolPlan
from dspy_rlm_hooks.speculation.guards import current_spec as _current_spec

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
