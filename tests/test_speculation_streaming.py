"""Tests for the streaming segmenter + peek planner (Task 3)."""

from __future__ import annotations

import ast

import pytest

from dspy_rlm_hooks.speculation import (
    MAX_UNROLL,
    Plan,
    StreamSegmenter,
    Unresolvable,
    plan_peeks,
    repair_tail,
    safe_eval,
)

SPEC = {"llm_query", "llm_query_batched"}


# -- segmentation of closed statements ---------------------------------------


def test_feed_emits_closed_simple_statements():
    seg = StreamSegmenter()
    out = seg.feed("```repl\nx = 1\ny = 2\n```\n")
    assert [s.source for s in out] == ["x = 1", "y = 2"]
    assert all(s.block_id == 0 for s in out)
    assert [s.index for s in out] == [0, 1]
    assert all(s.has_call is False for s in out)


def test_feed_marks_has_call():
    seg = StreamSegmenter()
    out = seg.feed("```repl\nx = 1\nllm_query('q')\n```\n")
    assert [s.source for s in out] == ["x = 1", "llm_query('q')"]
    assert out[0].has_call is False
    assert out[1].has_call is True


def test_feed_emits_simple_statement_at_its_newline():
    seg = StreamSegmenter()
    out = seg.feed("```repl\nx = 1\nllm_query('q')\n")
    # simple statements close at their newline and are emitted immediately
    assert [s.source for s in out] == ["x = 1", "llm_query('q')"]
    assert seg.pending_tail() == ""


def test_finish_emits_remaining_compound_statements():
    seg = StreamSegmenter()
    seg.feed("```repl\nfor i in range(3):\n    llm_query(str(i))\n")
    # compound not yet closed -> nothing emitted during feed
    assert seg.pending_tail() == "for i in range(3):\n    llm_query(str(i))\n"
    out = seg.finish()
    assert [s.source for s in out] == ["for i in range(3):\n    llm_query(str(i))"]


# -- compound-statement continuation -------------------------------------------


def test_compound_continues_until_dedent():
    seg = StreamSegmenter()
    out = seg.feed(
        "```repl\nfor i in range(3):\n    llm_query(str(i))\nnext = 1\n```\n"
    )
    assert [s.source for s in out] == [
        "for i in range(3):\n    llm_query(str(i))",
        "next = 1",
    ]
    assert out[0].has_call is True


def test_compound_not_emitted_until_closed():
    seg = StreamSegmenter()
    out = seg.feed("```repl\nfor i in range(3):\n    llm_query(str(i))\n")
    assert out == []
    assert seg.pending_tail() == "for i in range(3):\n    llm_query(str(i))\n"


def test_continuation_keywords_keep_compound_open():
    seg = StreamSegmenter()
    out = seg.feed("```repl\nif x:\n    a = 1\nelse:\n    a = 2\ny = 3\n```\n")
    assert [s.source for s in out] == [
        "if x:\n    a = 1\nelse:\n    a = 2",
        "y = 3",
    ]


def test_decorated_function_is_one_statement():
    seg = StreamSegmenter()
    out = seg.feed("```repl\n@deco\ndef f():\n    return 1\nx = 2\n```\n")
    assert [s.source for s in out] == ["@deco\ndef f():\n    return 1", "x = 2"]


# -- triple-string handling ----------------------------------------------------


def test_triple_string_spans_lines():
    seg = StreamSegmenter()
    out = seg.feed('```repl\ns = """line1\nline2\nline3"""\nx = 1\n```\n')
    assert [s.source for s in out] == ['s = """line1\nline2\nline3"""', "x = 1"]


def test_open_triple_string_blocks_emission():
    seg = StreamSegmenter()
    out = seg.feed('```repl\ns = """line1\nline2\n')
    assert out == []
    assert seg.pending_tail() == 's = """line1\nline2\n'


# -- fence handling ------------------------------------------------------------


def test_pending_tail_empty_outside_block():
    seg = StreamSegmenter()
    assert seg.pending_tail() == ""
    seg.feed("some prose\n")
    assert seg.pending_tail() == ""


def test_multiple_blocks_get_distinct_block_ids():
    seg = StreamSegmenter()
    out = seg.feed("```repl\nx = 1\n```\n```repl\ny = 2\n```\n")
    assert [s.source for s in out] == ["x = 1", "y = 2"]
    assert [s.block_id for s in out] == [0, 1]


def test_feed_complete_wraps_whole_code_as_one_block():
    seg = StreamSegmenter()
    out = seg.feed_complete("x = 1\nllm_query('q')\n")
    assert [s.source for s in out] == ["x = 1", "llm_query('q')"]
    assert all(s.block_id == 0 for s in out)


# -- repair_tail ----------------------------------------------------------------


def test_repair_tail_closes_open_bracket():
    assert repair_tail("llm_query('a', [1, 2") == "llm_query('a', [1, 2])"


def test_repair_tail_closes_open_string():
    assert repair_tail("llm_query('abc") == "llm_query('abc')"


def test_repair_tail_adds_pass_body():
    assert repair_tail("for i in range(3):") == "for i in range(3):\n    pass"


def test_repair_tail_returns_none_for_empty():
    assert repair_tail("") is None
    assert repair_tail("   ") is None


# -- safe_eval ------------------------------------------------------------------


def test_safe_eval_pure_ops_ok():
    ns = {"x": 3, "s": "hello"}
    assert safe_eval(ast.parse("x + 1", mode="eval").body, ns) == 4
    assert safe_eval(ast.parse("s.upper()", mode="eval").body, ns) == "HELLO"
    assert safe_eval(ast.parse("len(s)", mode="eval").body, ns) == 5
    assert safe_eval(ast.parse("[i * 2 for i in range(3)]", mode="eval").body, ns) == [
        0,
        2,
        4,
    ]
    assert safe_eval(ast.parse("x > 2", mode="eval").body, ns) is True
    assert safe_eval(ast.parse("'a' + 'b'", mode="eval").body, ns) == "ab"


def test_safe_eval_unsafe_raises():
    ns = {"x": 3}
    with pytest.raises(Unresolvable):
        safe_eval(ast.parse("__import__('os')", mode="eval").body, ns)
    with pytest.raises(Unresolvable):
        safe_eval(ast.parse("unknown_name", mode="eval").body, ns)
    with pytest.raises(Unresolvable):
        safe_eval(ast.parse("x.attr", mode="eval").body, ns)


# -- plan_peeks -----------------------------------------------------------------


def test_plan_peeks_literal_calls():
    plans = plan_peeks("llm_query('q')\n", SPEC, {})
    assert len(plans) == 1
    p = plans[0]
    assert isinstance(p, Plan)
    assert p.tool == "llm_query"
    assert p.args == ("q",)
    assert p.key is not None
    assert p.key[0] == "llm_query"
    assert len(p.key[1]) == 16


def test_plan_peeks_resolves_args_from_ns():
    plans = plan_peeks("llm_query(question)\n", SPEC, {"question": "hi"})
    assert len(plans) == 1
    assert plans[0].args == ("hi",)


def test_plan_peeks_for_loop_unroll():
    plans = plan_peeks(
        "for q in questions:\n    llm_query(q)\n", SPEC, {"questions": ["a", "b", "c"]}
    )
    assert [p.args for p in plans] == [("a",), ("b",), ("c",)]


def test_plan_peeks_for_loop_unroll_cap():
    plans = plan_peeks(
        "for q in questions:\n    llm_query(q)\n",
        SPEC,
        {"questions": list(range(200))},
    )
    assert len(plans) == MAX_UNROLL


def test_plan_peeks_skips_calls_under_if():
    plans = plan_peeks("if x:\n    llm_query('a')\n", SPEC, {"x": True})
    assert plans == []


def test_plan_peeks_skips_calls_under_def():
    plans = plan_peeks("def f():\n    llm_query('a')\n", SPEC, {})
    assert plans == []


def test_plan_peeks_skips_args_reading_earlier_assigned_names():
    plans = plan_peeks("y = 5\nllm_query(y)\n", SPEC, {})
    assert plans == []


def test_plan_peeks_loop_var_ok_despite_assignment():
    plans = plan_peeks(
        "for q in questions:\n    llm_query(q)\n", SPEC, {"questions": ["a"]}
    )
    assert [p.args for p in plans] == [("a",)]


def test_plan_peeks_skips_unclosed_call():
    # closing paren not yet streamed -> not planned
    plans = plan_peeks("llm_query('a'", SPEC, {})
    assert plans == []
