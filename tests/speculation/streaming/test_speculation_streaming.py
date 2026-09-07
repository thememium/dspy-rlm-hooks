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
from dspy_rlm_hooks.speculation.streaming import _call_closed_in

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


def test_plan_peeks_resolves_keyword_args():
    plans = plan_peeks("llm_query('/a', offset=2)\n", SPEC, {})
    assert len(plans) == 1
    assert plans[0].args == ("/a",)
    assert plans[0].kwargs == {"offset": 2}
    kw_plans = plan_peeks("llm_query(prompt='/a', offset=2)\n", SPEC, {})
    assert len(kw_plans) == 1
    assert kw_plans[0].args == ()
    assert kw_plans[0].kwargs == {"prompt": "/a", "offset": 2}


def test_plan_peeks_skips_keyword_unpack_and_unresolvable():
    assert plan_peeks("llm_query(**kw)\n", SPEC, {}) == []
    assert plan_peeks("llm_query('a', flag=missing)\n", SPEC, {}) == []


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


# -- StreamSegmenter edge cases ------------------------------------------------


def test_pending_tail_ignores_partial_closing_fence():
    # a partial "```" line is a fence, not code -> excluded from the tail
    seg = StreamSegmenter()
    seg.feed("```repl\nx = 1\n```")
    assert seg.pending_tail() == ""


def test_finish_emits_trailing_partial_line():
    # a partial (no-newline) line is folded into the block on finish()
    seg = StreamSegmenter()
    seg.feed("```repl\nx = 1\npartial")
    assert [s.source for s in seg.finish()] == ["partial"]


def test_finish_initializes_lines_for_partial_only_block():
    # a block with only a partial line (no complete line) has lines=None
    seg = StreamSegmenter()
    seg.feed("```repl\npartial")
    assert [s.source for s in seg.finish()] == ["partial"]


def test_dead_block_stops_emitting():
    # once a block hits a SyntaxError it goes dead and emits nothing further
    seg = StreamSegmenter()
    seg.feed("```repl\nx =\n")  # "x =" is a SyntaxError -> dead
    assert seg.feed("y = 2\n") == []


def test_compound_continues_over_blank_lines():
    # blank lines inside a compound keep it open (they belong to the body)
    seg = StreamSegmenter()
    out = seg.feed("```repl\nfor i in range(3):\n    pass\n\nnext = 1\n```\n")
    assert [s.source for s in out] == [
        "for i in range(3):\n    pass\n",
        "next = 1",
    ]


def test_scan_line_state_handles_escaped_quote():
    # a backslash-escaped quote inside a single-quoted string is not a closer
    seg = StreamSegmenter()
    out = seg.feed("```repl\ns = 'a\\'b'\nx = 1\n```\n")
    assert [s.source for s in out] == ["s = 'a\\'b'", "x = 1"]


def test_scan_line_state_stops_at_comment():
    # a '#' outside a string ends bracket scanning for the rest of the line
    seg = StreamSegmenter()
    out = seg.feed("```repl\nx = 1  # comment\ny = 2\n```\n")
    assert [s.source for s in out] == ["x = 1  # comment", "y = 2"]


def test_scan_line_state_resets_unclosed_single_string():
    # a single-quoted string can't span lines; the scan resets at EOL so the
    # enclosing compound is still treated as open (never parsed -> no error)
    seg = StreamSegmenter()
    seg.feed("```repl\nfor i in range(3):\n    s = 'abc\n")
    assert seg.pending_tail() == "for i in range(3):\n    s = 'abc\n"


# -- repair_tail / _bracket_closers edge cases ---------------------------------


def test_repair_tail_drops_final_partial_line():
    # an unrepairable final line is dropped and the rest is retried
    assert repair_tail("x = 1\nif :") == "x = 1"


def test_bracket_closers_ignores_brackets_in_triple_string():
    # brackets inside a closed triple-quoted string are not counted
    assert repair_tail('s = """a[b"""') == 's = """a[b"""'


def test_bracket_closers_handles_escaped_quote():
    # a backslash-escaped quote inside a single-quoted string is not a closer
    assert repair_tail("s = 'a\\'b'") == "s = 'a\\'b'"


def test_bracket_closers_handles_open_triple_string():
    # an open triple-quoted string swallows the trailing '['; the tail can't
    # be repaired cheaply so repair_tail gives up
    assert repair_tail('s = """a\n[') is None


def test_bracket_closers_ignores_brackets_in_comments():
    # a '[' inside a comment must not be treated as an open bracket
    assert repair_tail("x = 1  # [unclosed\n") == "x = 1  # [unclosed\n"


def test_repair_tail_does_not_recursively_overflow_on_large_unrepairable_tail():
    # A large multi-line tail that resists every cheap repair used to recurse
    # once per dropped line and blow the stack (RecursionError). It must now
    # give up iteratively and return None.
    tail = "\n".join(f"x{i} = )" for i in range(1500))
    assert repair_tail(tail) is None


# -- safe_eval edge cases ------------------------------------------------------


def test_safe_eval_depth_limit():
    with pytest.raises(Unresolvable):
        safe_eval(ast.parse("x", mode="eval").body, {"x": 1}, depth=41)


def test_safe_eval_fstring():
    assert safe_eval(ast.parse("f'val={x}'", mode="eval").body, {"x": 3}) == "val=3"


def test_safe_eval_fstring_unresolvable_value():
    # a JoinedStr value that is neither a FormattedValue nor a Constant
    node = ast.JoinedStr(values=[ast.Name(id="x", ctx=ast.Load())])
    with pytest.raises(Unresolvable):
        safe_eval(node, {})


def test_safe_eval_more_binops():
    assert safe_eval(ast.parse("7 % 3", mode="eval").body, {}) == 1
    assert safe_eval(ast.parse("5 - 2", mode="eval").body, {}) == 3
    assert safe_eval(ast.parse("7 // 2", mode="eval").body, {}) == 3


def test_safe_eval_subscript_and_slice():
    assert safe_eval(ast.parse("d['a']", mode="eval").body, {"d": {"a": 1}}) == 1
    assert safe_eval(ast.parse("xs[1:3]", mode="eval").body, {"xs": [1, 2, 3]}) == [
        2,
        3,
    ]


def test_safe_eval_tuple_and_list():
    assert safe_eval(ast.parse("(1, 2)", mode="eval").body, {}) == (1, 2)
    assert safe_eval(ast.parse("[1, 2]", mode="eval").body, {}) == [1, 2]


def test_safe_eval_dict():
    assert safe_eval(ast.parse("{'a': 1}", mode="eval").body, {}) == {"a": 1}


def test_safe_eval_dict_unpack_unresolvable():
    with pytest.raises(Unresolvable):
        safe_eval(ast.parse("{**d}", mode="eval").body, {"d": {}})


def test_safe_eval_non_pure_method_unresolvable():
    with pytest.raises(Unresolvable):
        safe_eval(ast.parse("s.count('l')", mode="eval").body, {"s": "hello"})


def test_safe_eval_listcomp_too_big():
    with pytest.raises(Unresolvable):
        safe_eval(ast.parse("[i for i in range(20000)]", mode="eval").body, {})


def test_safe_eval_compare_ops():
    assert safe_eval(ast.parse("1 == 1", mode="eval").body, {}) is True
    assert safe_eval(ast.parse("1 != 2", mode="eval").body, {}) is True
    assert safe_eval(ast.parse("1 < 2", mode="eval").body, {}) is True
    assert safe_eval(ast.parse("1 <= 1", mode="eval").body, {}) is True
    assert safe_eval(ast.parse("2 > 1", mode="eval").body, {}) is True
    assert safe_eval(ast.parse("2 >= 2", mode="eval").body, {}) is True
    assert safe_eval(ast.parse("1 in [1]", mode="eval").body, {}) is True
    assert safe_eval(ast.parse("2 not in [1]", mode="eval").body, {}) is True


def test_safe_eval_compare_unresolvable_op():
    # `is` is not in the whitelisted comparison ops
    with pytest.raises(Unresolvable):
        safe_eval(ast.parse("1 is 1", mode="eval").body, {})


def test_safe_eval_ifexp():
    assert safe_eval(ast.parse("1 if True else 2", mode="eval").body, {}) == 1
    assert safe_eval(ast.parse("1 if False else 2", mode="eval").body, {}) == 2


def test_safe_eval_listcomp_tuple_unpack():
    assert safe_eval(ast.parse("[a for a, b in [(1, 2)]]", mode="eval").body, {}) == [1]


def test_safe_eval_bind_unresolvable_target():
    # a subscript comprehension target can't be bound -> Unresolvable
    with pytest.raises(Unresolvable):
        safe_eval(ast.parse("[x for x[0] in [[1]]]", mode="eval").body, {})


def test_safe_eval_rejects_weird_values():
    class SpecValue:
        pass

    with pytest.raises(Unresolvable):
        safe_eval(ast.parse("x", mode="eval").body, {"x": SpecValue()})


# -- plan_peeks edge cases -----------------------------------------------------


def test_plan_peeks_empty_tail():
    assert plan_peeks("", SPEC, {}) == []


def test_plan_peeks_unroll_unresolvable_iter():
    # the loop iterable can't be resolved -> no unroll
    assert plan_peeks("for q in unknown:\n    llm_query(q)\n", SPEC, {}) == []


def test_plan_peeks_unroll_skips_when_calls_after_control_flow():
    # a hooked call hiding after an `if` inside the loop -> no unroll at all
    plans = plan_peeks(
        "for q in questions:\n    if q:\n        pass\n    llm_query(q)\n",
        SPEC,
        {"questions": ["a"]},
    )
    assert plans == []


def test_plan_peeks_unroll_bind_unresolvable():
    # a subscript loop target can't be bound per item -> no unroll
    plans = plan_peeks(
        "for q[0] in questions:\n    llm_query(q)\n",
        SPEC,
        {"questions": [["a"]]},
    )
    assert plans == []


def test_plan_peeks_skips_unresolvable_args():
    plans = plan_peeks("llm_query(unknown)\n", SPEC, {})
    assert plans == []


def test_plan_peeks_taints_subscript_store():
    # `data[0] = 1` taints `data`, so a later `llm_query(data)` is skipped
    plans = plan_peeks("data[0] = 1\nllm_query(data)\n", SPEC, {"data": [1]})
    assert plans == []


def test_plan_peeks_taints_mutating_method():
    # `data.append(1)` taints `data` (mutation blind spot) -> skip later call
    plans = plan_peeks("data.append(1)\nllm_query(data)\n", SPEC, {"data": []})
    assert plans == []


def test_plan_peeks_taints_nested_subscript():
    # `data[0][1] = 2` unwraps to the base name `data`
    plans = plan_peeks("data[0][1] = 2\nllm_query(data)\n", SPEC, {"data": [[1]]})
    assert plans == []


def test_plan_peeks_unroll_no_calls_after_control_flow():
    # control flow with no calls after it -> the straight-line prefix is planned
    plans = plan_peeks(
        "for q in questions:\n    llm_query(q)\n    if q:\n        pass\n",
        SPEC,
        {"questions": ["a"]},
    )
    assert [p.args for p in plans] == [("a",)]


# -- _call_closed_in (private helper) ------------------------------------------


def test_call_closed_in_missing_end():
    # a call node without end_lineno/end_col_offset is treated as not closed
    call = ast.Call(func=ast.Name(id="llm_query", ctx=ast.Load()), args=[], keywords=[])
    assert _call_closed_in("llm_query()", call) is False


def test_call_closed_in_end_lineno_beyond():
    # a call whose end line is past the raw tail is not textually complete
    call = ast.Call(func=ast.Name(id="llm_query", ctx=ast.Load()), args=[], keywords=[])
    call.end_lineno = 5
    call.end_col_offset = 0
    assert _call_closed_in("llm_query()", call) is False
