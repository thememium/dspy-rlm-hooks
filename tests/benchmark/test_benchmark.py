"""End-to-end smoke + coverage tests for the deterministic A/B benchmark."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import dspy_rlm_hooks.benchmark as B
import dspy_rlm_hooks.benchmark as BM


def test_ab_benchmark_equivalent_steps_and_spec_claims():
    scenario = B.default_scenario(tool_ms=10.0, llm_ms=20.0)
    report = B.run_ab(scenario, variants=["default", "spec"], repeats=1, pace_ms=0.0)

    assert report["equivalent_steps"] is True, report["mismatches"]
    default_res = report["variants"]["default"]
    spec_res = report["variants"]["spec"]

    assert default_res["spec_stats_median"]["claimed"] == 0
    assert spec_res["spec_stats_median"]["speculated"] > 0
    assert spec_res["spec_stats_median"]["claimed"] > 0
    assert len(default_res["iterations"]) == len(spec_res["iterations"]) == 4
    assert (
        spec_res["tool_critical_ms"]["median"]
        < default_res["tool_critical_ms"]["median"]
    )
    assert report["comparison"]["spec"]["wall_speedup_vs_baseline"] > 0


def test_benchmark_variants_hooks_preset_and_callable_and_errors():
    scenario = B.default_scenario(tool_ms=5.0, llm_ms=10.0)
    res = B.run_variant(scenario, "hooks", repeats=1, pace_ms=0.0)
    assert res["spec_stats_median"]["speculated"] == 0
    assert res["tool_critical_ms"]["median"] > 0

    def my_variant(rlm, tools):
        return None

    res2 = B.run_variant(scenario, my_variant, repeats=1, pace_ms=0.0)
    assert res2["variant"] == "my_variant"

    program, tool_fns = B._build_program(scenario, B.TrialRecord())
    with pytest.raises(ValueError):
        B._apply_variant(program, "nope", tool_fns)
    with pytest.raises(TypeError):
        B._apply_variant(program, 12345, tool_fns)


def test_benchmark_fake_sublm_and_tool_recording():
    record: list = []
    sub = B._FakeSubLM(1.0, record)
    out = sub("hello world")
    assert out[0]["text"].startswith("SCRIPTED[hello")
    assert record[-1]["speculative"] is False

    plan = B.ToolPlan("probe", 1.0, "R-{}")
    fn = B._make_tool_fn(plan, record)
    assert fn("zz") == "R-zz"
    assert fn() == plan.result
    assert any(e["tool"] == "probe" for e in record)


def test_benchmark_aggregate_empty_and_timings_parse():
    agg = B._aggregate("x", [])
    assert agg["steps"] == []
    assert B._parse_timings("no timings here") == {}
    assert B._parse_timings("TIMINGS {not json") == {}
    assert B._strip_timings("TIMINGS {}\nkeep") == "keep"


def test_benchmark_cli_runs_and_writes_json(tmp_path):
    rc = BM.main(
        [
            "--variants",
            "default",
            "spec",
            "--repeats",
            "1",
            "--tool-ms",
            "5",
            "--llm-ms",
            "10",
            "--json-out",
            str(tmp_path / "report.json"),
        ]
    )
    assert rc == 0
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["equivalent_steps"] is True


def test_benchmark_cli_detects_mismatch(tmp_path, monkeypatch):
    orig = BM._scripted_generate_action

    def broken(rlm, scenario, pace_ms):
        ga = orig(rlm, scenario, pace_ms)

        def gen(variables_info, repl_history, iteration):
            pred = ga(variables_info, repl_history, iteration)
            from dspy import Prediction

            code = pred.code
            # inject an extra statement into the SPEC variant only (it has an
            # active streaming turn; default does not)
            if getattr(rlm, "_active_stream_turn", None) is not None:
                assert code.endswith("```")
                code = code[:-3] + "\nprint('EXTRA')\n```"
            return Prediction(reasoning=pred.reasoning, code=code)

        return gen

    monkeypatch.setattr(
        "dspy_rlm_hooks.benchmark.runner._scripted_generate_action", broken
    )
    rc = BM.main(
        [
            "--variants",
            "default",
            "spec",
            "--repeats",
            "1",
            "--tool-ms",
            "5",
            "--llm-ms",
            "10",
        ]
    )
    assert rc == 1


def test_benchmark_teardown_and_stats_error_paths(monkeypatch):
    """_teardown swallows disable failures; run_variant tolerates stats errors."""
    import dspy_rlm_hooks as pkg

    scenario = B.default_scenario(tool_ms=5.0, llm_ms=10.0)

    def boom(*a, **k):
        raise RuntimeError("disable boom")

    monkeypatch.setattr(pkg, "disable_rlm_speculation", boom)
    res = B.run_variant(scenario, "spec", repeats=1, pace_ms=0.0)
    assert res["wall_s"]["median"] > 0

    def stats_boom():
        raise RuntimeError("stats boom")

    monkeypatch.setattr(pkg, "disable_rlm_speculation", lambda *a, **k: None)
    monkeypatch.setattr(
        pkg.speculation.speculator.Speculator, "stats", lambda self: stats_boom()
    )
    res2 = B.run_variant(scenario, "spec", repeats=1, pace_ms=0.0)
    assert res2["spec_stats_median"]["speculated"] == 0  # stats failed -> zeros


def test_benchmark_cli_module_fn_variant():
    rc = BM.main(
        [
            "--variants",
            "dspy_rlm_hooks.benchmark:_variant_default",
            "--repeats",
            "1",
            "--tool-ms",
            "5",
            "--llm-ms",
            "10",
        ]
    )
    assert rc == 0


@pytest.mark.parametrize("streaming", [False, True])
def test_scripted_generation_paces_every_variant(monkeypatch, streaming):
    # Given identical code and a clock recording generation delays.
    sleeps: list[float] = []
    chunks: list[str] = []
    monkeypatch.setattr(B.time, "sleep", sleeps.append)
    turn = SimpleNamespace(feed=chunks.append) if streaming else None
    program = SimpleNamespace(_active_stream_turn=turn)
    scenario = B.Scenario("pacing", [B.ScriptedIteration("", "print(1)")], [])

    # When the same generation runs with or without a speculative stream.
    prediction = B._scripted_generate_action(program, scenario, 2.0)(None, None, "1/1")

    # Then every chunk incurs the same simulated arrival delay.
    assert sleeps == [0.002] * ((len(prediction.code) + 7) // 8)
    assert "".join(chunks) == (prediction.code if streaming else "")


def test_benchmark_honors_scenario_llm_budget():
    # Given a scenario with a non-default sub-LM budget.
    scenario = B.Scenario("budget", [], [], max_llm_calls=2)

    # When its real DSPy program is constructed.
    program, _ = B._build_program(scenario, B.TrialRecord())

    # Then the configured limit reaches the program.
    assert program.max_llm_calls == 2


def test_benchmark_sums_repeated_timing_keys_across_iterations():
    # Given two iterations reporting the same timing label.
    scenario = B.Scenario(
        "repeated-timings",
        [
            B.ScriptedIteration("", "print('TIMINGS {\"tool\": 7}')"),
            B.ScriptedIteration("", "print('TIMINGS {\"tool\": 11}')"),
            B.ScriptedIteration("", 'SUBMIT("ok")'),
        ],
        [],
    )

    # When both iterations run in the real interpreter.
    result = B.run_variant(scenario, repeats=1, pace_ms=0)

    # Then their critical path contributions are added, not overwritten.
    assert result["tool_critical_ms"]["median"] == 18
