"""Planning side: find the calls in the tail worth pre-dispatching and plan
them — concrete :class:`Plan` bets plus chained :class:`ChainPlan`
continuations whose args depend on speculated producers.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Any

from dspy_rlm_hooks.speculation.streaming.evaluator import (
    Unresolvable,
    _bind,
    safe_eval,
)
from dspy_rlm_hooks.speculation.streaming.segmenter import repair_tail
from dspy_rlm_hooks.speculation.tool import SpecKey, canonical_hash

# =============================================================================
# 4. Find + plan: which calls in the tail do we bet on?
# =============================================================================

MAX_UNROLL = 64  # cap on per-loop pre-dispatch


@dataclass
class Plan:
    """One pre-dispatchable call in the tail, in program order.

    ``key`` is the claim identity (``SpecKey``) built via ``canonical_hash`` so
    the shadow dispatch and the real REPL claim agree on the same key.
    """

    tool: str
    args: tuple
    kwargs: dict = field(default_factory=dict)
    key: SpecKey | None = None


# dep descriptor: ("key", SpecKey) — a directly planned producing call — or
# ("cont", cont_id) — an upstream chained call whose key is realized at fire
# time. Kept picklable: these cross the shadow pipe as chain metadata.
DepRef = tuple[str, Any]


@dataclass
class ChainPlan:
    """A call whose args depend on a speculated predecessor's result.

    Planned once (ASTs stay in the worker); FIRED when every dependency has
    resolved: the worker evaluates the unresolved arg ASTs against the
    namespace extended with the dep results and dispatches the call like any
    shadow call. Args that already resolved at plan time are frozen as
    constants (loop variables would not exist at fire time).
    """

    cont_id: int
    tool: str
    arg_specs: list  # list[("const", value) | ("expr", ast.expr)]
    kwarg_specs: dict  # dict[str, ("const", value) | ("expr", ast.expr)]
    deps: dict  # dep name -> DepRef


@dataclass
class ChainMeta:
    """Parent-visible summary of a :class:`ChainPlan` (crosses the pipe)."""

    cont_id: int
    tool: str
    deps: dict  # dep name -> DepRef


def plan_peeks(tail: str, spec_names: set[str], ns: dict[str, Any]) -> list[Plan]:
    """All calls in the repaired tail worth pre-dispatching, in program order.

    Rails that keep waste rare (each skips a call rather than risking it):
      - the call's own closing paren must exist in the RAW tail (we never
        guess unfinished arguments);
      - calls under `if`/`while`/`try`/`def` in the tail are skipped — the
        shadow will resolve those branches with real values when they close;
      - calls whose args read names assigned EARLIER IN THE TAIL are skipped
        (those assignments haven't reached the shadow namespace yet), except
        the loop variable of an unrolled `for`, which we bind per item.

    Chained continuations (calls whose args depend on a planned producer's
    future result) are NOT included here — use :func:`plan_peeks_with_chains`.
    """
    plans, _chains, _metas = plan_peeks_with_chains(tail, spec_names, ns)
    return plans


def plan_peeks_with_chains(
    tail: str,
    spec_names: set[str],
    ns: dict[str, Any],
    segment_productions: dict[str, DepRef] | None = None,
) -> tuple[list[Plan], list[ChainPlan], list[ChainMeta]]:
    """All calls in the repaired tail worth pre-dispatching, in program order,
    plus CHAINED calls whose args depend on speculated predecessors.

    A call whose args read a name assigned earlier in the tail is skipped by
    :func:`plan_peeks` (the value is stale/unknown). When that name is bound to
    a producing call — planned earlier in this tail (``x = search(..)``) or
    executed by an already-closed segment (``segment_productions``) — the call
    becomes a :class:`ChainPlan` instead: it fires when the producer's
    speculation resolves, with the arg ASTs evaluated against the resolved
    value. This pipelines whole dataflow chains (``search -> read ->
    llm_query``) under the model's still-streaming output.

    Returns ``(plans, chain_plans, chain_metas)`` — metas are the picklable
    summaries the parent watches.
    """
    plans, chain_plans, metas = [], [], []
    counter = _ContCounter()
    # segment-level productions come first; tail-local producers override
    productions: dict[str, DepRef] = dict(segment_productions or {})
    _plan_body(
        tree_body=_parse_repaired(tail),
        spec_names=spec_names,
        ns=ns,
        tail=tail,
        plans=plans,
        chain_plans=chain_plans,
        metas=metas,
        productions=productions,
        counter=counter,
        dep_capable=set(productions),
    )
    return plans, chain_plans, metas


class _ContCounter:
    def __init__(self) -> None:
        self.n = 0

    def next(self) -> int:
        self.n += 1
        return self.n


def _parse_repaired(tail: str) -> list:
    repaired = repair_tail(tail)
    if repaired is None:
        return []
    try:
        tree = ast.parse(repaired)
    except SyntaxError:
        return []
    return tree.body


def _plan_body(
    *,
    tree_body: list,
    spec_names: set[str],
    ns: dict[str, Any],
    tail: str,
    plans: list[Plan],
    chain_plans: list[ChainPlan],
    metas: list[ChainMeta],
    productions: dict[str, DepRef],
    counter: "_ContCounter",
    dep_capable: set[str] | None = None,
) -> None:
    """Plan one straight-line statement sequence (shared by top level and the
    plannable prefix of unrolled for-bodies).

    ``dep_capable`` is the set of names a chained arg may read: every producer
    registered so far (tail-local and segment-level).
    """
    dep_capable = dep_capable if dep_capable is not None else set(productions)
    for stmt in tree_body:
        if isinstance(stmt, ast.For) and not isinstance(stmt.target, ast.Starred):
            assigned_before = _assigned_names_set(tree_body, stmt)
            loop_plans, loop_chains = _unroll_for(
                stmt,
                spec_names,
                ns,
                assigned_before,
                tail,
                counter,
                productions,
                dep_capable=dep_capable,
            )
            plans.extend(loop_plans)
            for cp in loop_chains:
                chain_plans.append(cp)
                metas.append(ChainMeta(cont_id=cp.cont_id, tool=cp.tool, deps=cp.deps))
            continue
        if isinstance(stmt, (ast.Expr, ast.Assign, ast.AugAssign, ast.AnnAssign)):
            assigned_before = _assigned_names_set(tree_body, stmt)
            for call in _hooked_calls(stmt, spec_names):
                resolved = _resolve_call_or_chain(
                    call,
                    ns,
                    assigned_before,
                    tail,
                    productions,
                    counter,
                    dep_capable=dep_capable,
                )
                if isinstance(resolved, Plan):
                    plans.append(resolved)
                elif isinstance(resolved, ChainPlan):
                    chain_plans.append(resolved)
                    metas.append(
                        ChainMeta(
                            cont_id=resolved.cont_id,
                            tool=resolved.tool,
                            deps=resolved.deps,
                        )
                    )
                # bind a plain assignment target to the producing call so later
                # calls in the tail can chain off it
                target = _single_assign_target(stmt)
                if target is not None and resolved is not None:
                    if isinstance(resolved, Plan):
                        productions[target] = ("key", resolved.key)
                    else:
                        productions[target] = ("cont", resolved.cont_id)
        # anything else (If/While/Try/def/class): too uncertain — skip


def _single_assign_target(stmt: ast.stmt) -> str | None:
    """The single Name target of a simple assignment, else None."""
    if not isinstance(stmt, ast.Assign):
        return None
    if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
        return None
    return stmt.targets[0].id


def _assigned_names_set(body: list, up_to: ast.stmt | None = None) -> set[str]:
    out: set[str] = set()
    for stmt in body:
        if up_to is not None and stmt is up_to:
            break
        out |= _assigned_names(stmt)
    return out


def _free_names(node: ast.AST) -> set[str]:
    return {
        n.id
        for n in ast.walk(node)
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
    }


def _resolve_call_or_chain(
    call: ast.Call,
    ns: dict[str, Any],
    assigned_in_tail: set[str],
    raw_tail: str,
    productions: dict[str, DepRef],
    counter: "_ContCounter",
    extra_env: dict[str, Any] | None = None,
    loop_var_ok: set[str] | frozenset[str] = frozenset(),
    dep_capable: set[str] | None = None,
) -> Plan | ChainPlan | None:
    """Plan a call concretely when every argument resolves against live state;
    otherwise, when its unresolved arguments read names bound to planned
    producer calls, plan it as a chained continuation.

    ``extra_env`` extends the evaluation namespace (loop-variable bindings for
    unrolled bodies); ``loop_var_ok`` marks tail-assigned names the concrete
    path may read anyway because they are bound per item in ``extra_env``.
    """
    if not isinstance(call.func, ast.Name):
        return None
    # the call must be textually complete in the RAW tail: its closing paren
    # was streamed, so every argument token is final.
    if not _call_closed_in(raw_tail, call):
        return None

    env = dict(ns)
    if extra_env:
        env.update(extra_env)
    _dep_capable: set[str] = (
        dep_capable if dep_capable is not None else set(productions)
    )

    # per-arg working state: (expr, kind, payload, dep_names)
    arg_state: list[tuple[ast.expr, str, Any, set[str]]] = []
    kw_state: list[tuple[str, ast.expr, str, Any, set[str]]] = []
    chained = False

    def _eval(expr: ast.expr) -> tuple[str, Any, set[str]]:
        """(kind, payload, dep_names) for one argument expression."""
        try:
            return "const", safe_eval(expr, env), set()
        except (Unresolvable, Exception):
            dep_names = {
                n
                for n in _free_names(expr)
                if n in assigned_in_tail or n in _dep_capable
            }
            if not dep_names or not all(n in productions for n in dep_names):
                raise  # stale/unknown value with no producer to wait on
            return "expr", expr, dep_names

    try:
        for a in call.args:
            kind, payload, dep_names = _eval(a)
            if dep_names:
                chained = True
            arg_state.append((a, kind, payload, dep_names))
        for kw in call.keywords:
            if kw.arg is None:  # **unpack — cannot statically resolve
                return None
            kind, payload, dep_names = _eval(kw.value)
            if dep_names:
                chained = True
            kw_state.append((kw.arg, kw.value, kind, payload, dep_names))
    except (Unresolvable, Exception):
        return None

    if not chained:
        # concrete path: staleness rail — an arg reading a tail-assigned name
        # (other than per-item loop vars) must NOT silently use the stale
        # namespace value the safe evaluator found.
        for a, _kind, _payload, _deps in arg_state:
            if any(
                n in assigned_in_tail and n not in loop_var_ok for n in _free_names(a)
            ):
                return None
        for _name, e, _kind, _payload, _deps in kw_state:
            if any(
                n in assigned_in_tail and n not in loop_var_ok for n in _free_names(e)
            ):
                return None
        args = tuple(payload for _e, _k, payload, _d in arg_state)
        kwargs = {name: payload for name, _e, _k, payload, _d in kw_state}
        key = canonical_hash(call.func.id, args, kwargs)
        return Plan(
            tool=call.func.id, args=args, kwargs=kwargs, key=(call.func.id, key)
        )

    # chained path: concrete args freeze as constants (loop variables would
    # not exist at fire time); dep-reading args keep their ASTs for the fire
    # -time evaluation against the resolved dep values.
    deps: dict[str, DepRef] = {}
    arg_specs: list[tuple[str, Any]] = []
    kwarg_specs: dict[str, tuple[str, Any]] = {}
    for _a, kind, payload, dep_names in arg_state:
        for n in dep_names:
            deps[n] = productions[n]
        arg_specs.append(("const", payload) if kind == "const" else ("expr", payload))
    for name, _e, kind, payload, dep_names in kw_state:
        for n in dep_names:
            deps[n] = productions[n]
        kwarg_specs[name] = ("const", payload) if kind == "const" else ("expr", payload)
    return ChainPlan(
        cont_id=counter.next(),
        tool=call.func.id,
        arg_specs=arg_specs,
        kwarg_specs=kwarg_specs,
        deps=deps,
    )


def _unroll_for(
    loop: ast.For,
    spec_names: set[str],
    ns: dict[str, Any],
    assigned: set[str],
    raw_tail: str,
    counter: "_ContCounter",
    productions: dict[str, DepRef] | None = None,
    dep_capable: set[str] | None = None,
) -> tuple[list[Plan], list[ChainPlan]]:
    """Pre-dispatch every iteration of `for X in ITER:` when ITER and the body
    call's args resolve — the flagship streaming win: the whole map fans out
    while the model is still writing the loop (or the code after it).

    The body SHAPE is analyzed once (not per item): statements up to the
    first nested control-flow construct are plannable for every iteration;
    if hooked-callable code could hide inside or after that construct, we
    don't unroll at all. Per-item analysis here caused spurious bet
    retraction (a re-plan after `if ...:` streamed in shrank the plan to
    item 0 and evicted valid bets).

    Body calls reading names bound to planned producers chain like anywhere
    else (one continuation per item, all fired when the producer resolves);
    loop variables freeze into per-item constants.
    """
    productions = productions if productions is not None else {}
    try:
        items = list(safe_eval(loop.iter, ns))
    except (Unresolvable, Exception):
        return [], []
    if len(items) > MAX_UNROLL:
        items = items[:MAX_UNROLL]
    plannable: list[ast.stmt] = []
    for stmt in loop.body:
        if isinstance(
            stmt,
            (
                ast.If,
                ast.While,
                ast.Try,
                ast.For,
                ast.FunctionDef,
                ast.AsyncFor,
                ast.AsyncWith,
                ast.AsyncFunctionDef,
            ),
        ):
            if not _no_calls_after(loop.body, stmt):
                return [], []  # calls hide in/after control flow: no bet
            break  # plan the straight-line prefix, all items
        plannable.append(stmt)
    plans: list[Plan] = []
    chain_plans: list[ChainPlan] = []
    body_assigned = set(assigned)
    loop_vars = _target_names(loop.target)
    for item in items:
        item_env = dict(ns)
        try:
            _bind(item_env, loop.target, item)
        except Unresolvable:
            return [], []
        item_assigned = set(body_assigned)
        known_locals = set(loop_vars)
        for stmt in plannable:
            target = _single_assign_target(stmt)
            if target is not None and isinstance(stmt, ast.Assign):
                reads = _free_names(stmt.value)
                if not (reads & (item_assigned - known_locals)):
                    try:
                        value = safe_eval(stmt.value, item_env)
                    except (
                        Unresolvable,
                        TypeError,
                        ValueError,
                        LookupError,
                        ArithmeticError,
                    ):
                        pass
                    else:
                        item_env[target] = value
                        known_locals.add(target)
                        item_assigned.add(target)
                        continue
            for call in _hooked_calls(stmt, spec_names):
                resolved = _resolve_call_or_chain(
                    call,
                    ns,
                    item_assigned,
                    raw_tail,
                    productions,
                    counter,
                    extra_env={n: item_env[n] for n in known_locals},
                    loop_var_ok=known_locals,
                    dep_capable=dep_capable,
                )
                # unresolvable calls (e.g. stage-2 of a per-item chain whose
                # arg is stage-1's result) are left for the shadow — the
                # resolvable stage-1 calls still fan out here
                if isinstance(resolved, Plan):
                    plans.append(resolved)
                elif isinstance(resolved, ChainPlan):
                    chain_plans.append(resolved)
                # bind single-target assignments so LATER body statements can
                # chain off this call (same as _plan_body does at top level)
                target = _single_assign_target(stmt)
                if target is not None and resolved is not None:
                    if isinstance(resolved, Plan):
                        productions[target] = ("key", resolved.key)
                    else:
                        productions[target] = ("cont", resolved.cont_id)
            changed = _assigned_names(stmt)
            known_locals -= changed
            item_assigned |= changed
    return plans, chain_plans


def _call_closed_in(raw: str, call: ast.Call) -> bool:
    """True if this call's closing paren exists in the raw (unrepaired) text."""
    end_lineno = getattr(call, "end_lineno", None)
    end_col = getattr(call, "end_col_offset", None)
    if end_lineno is None or end_col is None:
        return False
    lines = raw.split("\n")
    if end_lineno > len(lines):
        return False
    if end_lineno == len(lines) and end_col > len(lines[-1]):
        return False
    return True


def _hooked_calls(stmt: ast.stmt, spec_names: set[str]):
    for n in ast.walk(stmt):
        if (
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id in spec_names
        ):
            yield n


_MUTATING_METHODS = {
    "append",
    "extend",
    "insert",
    "pop",
    "remove",
    "clear",
    "sort",
    "reverse",
    "update",
    "setdefault",
    "popitem",
    "add",
    "discard",
}


def _assigned_names(stmt: ast.stmt) -> set[str]:
    """Names whose VALUE may change when this statement runs — the stale-name
    rail for peeks. Beyond plain Name stores this taints the base name of
    subscript/attribute stores (`data[i] = x`, `obj.f = x`) and of mutating
    method calls (`data.append(x)`) — the mutation blind spot."""
    out: set[str] = set()
    for n in ast.walk(stmt):
        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            out.add(n.id)
        elif isinstance(n, (ast.Subscript, ast.Attribute)) and isinstance(
            n.ctx, (ast.Store, ast.Del)
        ):
            out |= _base_names(n.value)
        elif (
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr in _MUTATING_METHODS
        ):
            out |= _base_names(n.func.value)
    return out


def _base_names(node: ast.expr) -> set[str]:
    while isinstance(node, (ast.Subscript, ast.Attribute)):
        node = node.value
    return {node.id} if isinstance(node, ast.Name) else set()


def _target_names(target: ast.expr) -> set[str]:
    return {n.id for n in ast.walk(target) if isinstance(n, ast.Name)}


def _no_calls_after(body: list[ast.stmt], from_stmt: ast.stmt) -> bool:
    seen = False
    for s in body:
        if s is from_stmt:
            seen = True
        if seen:
            for n in ast.walk(s):
                if isinstance(n, ast.Call):
                    return False
    return True
