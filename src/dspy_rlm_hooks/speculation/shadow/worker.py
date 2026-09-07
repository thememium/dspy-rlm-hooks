"""Worker-process side of the shadow: the subprocess message loop, statement
execution under the SIGALRM runaway guard, and chained-continuation planning."""

from __future__ import annotations

import ast
import builtins as _builtins
import multiprocessing
import pickle
import signal
from types import FunctionType
from typing import Any

from dspy_rlm_hooks.speculation.shadow.analysis import (
    _bound_names,
    _comp_local_names,
    _read_names,
)
from dspy_rlm_hooks.speculation.shadow.builtins import shadow_builtins
from dspy_rlm_hooks.speculation.shadow.snapshot import _rebind, load_ns
from dspy_rlm_hooks.speculation.streaming import (
    ChainPlan,
    Segment,
    _ContCounter,
    _plan_body,
    safe_eval,
)
from dspy_rlm_hooks.speculation.tool import (
    NonSpeculated,
    canonical_hash,
    contains_nonspec,
)


class ShadowAborted(Exception):
    pass


def _make_record_hook(conn, name: str):
    """Shadow-side tool hook: record the call to the parent, return a marker.

    The marker is a :class:`NonSpeculated` so taint flows by value: a later
    statement that reads this call's result is skipped and its targets poisoned,
    instead of aborting the whole shadow.
    """

    def hook(*args, **kwargs):
        conn.send(("tool", name, args, kwargs))
        return NonSpeculated(name)

    hook.__name__ = name
    return hook


def _mp_context() -> Any:
    """Use fork for faster shadow construction on Unix.

    Fork is ~10x faster than spawn because it doesn't need to re-import modules.
    The risk is fork-safety with threads, but the worker is carefully designed
    to be fork-safe (no locks held at fork time, clean namespace).
    """
    try:
        return multiprocessing.get_context("fork")
    except ValueError:
        # fork not available (Windows), fall back to default
        return multiprocessing.get_context()


# =============================================================================
# worker process
# =============================================================================


def _shadow_worker(conn, parent_conn, payload: dict) -> None:
    """Run in the subprocess: execute segments, record predicted calls, plan peeks.

    The namespace seed is kept un-pickled once; a persistent runner restores it
    on every ``("reset",)`` so per-turn state never leaks across turns.
    """
    parent_conn.close()
    seed: dict = payload["ns_seed"]  # classified bytes; re-loaded per turn
    spec_names: set[str] = payload["spec_names"]
    taint_skip: bool = payload["taint_skip"]
    budget: float = payload["budget"]

    def fresh_ns() -> dict:
        # re-load from bytes: every turn gets a fresh independent copy of the
        # seed (same isolation as a per-turn spawn), with no deepcopy of
        # Opaque sentinels (they cannot be copied and must pass through)
        ns = load_ns(seed)
        ns["__builtins__"] = shadow_builtins(dict(_builtins.__dict__))
        ns.setdefault("__name__", "__main__")  # class stmts need it
        # recording hooks live in the worker so they share its pipe end
        for name in spec_names:
            ns[name] = _make_record_hook(conn, name)
        # cross-turn helpers keep __globals__ on the shadow namespace
        for k, v in list(ns.items()):
            if isinstance(v, FunctionType) and k not in spec_names:
                ns[k] = _rebind(v, ns)
        return ns

    ns = fresh_ns()
    chains: dict[int, ChainPlan] = {}  # cont_id -> pending continuation
    chain_counter = _ContCounter()
    # names bound to a hooked call by an EXECUTED segment this turn: the
    # producer's speculation is already dispatched, so later tail peeks can
    # chain off it even though the producer statement left the tail
    seg_productions: dict[str, tuple] = {}
    try:
        while True:
            msg = conn.recv()
            if msg is None:
                break
            if isinstance(msg, tuple) and msg[0] == "reset":
                if len(msg) > 1 and msg[1] is not None:
                    seed = msg[1]  # reseed: subsequent fresh_ns() rebuilds from it
                ns = fresh_ns()
                chains = {}
                chain_counter = _ContCounter()
                seg_productions = {}
            elif isinstance(msg, tuple) and msg[0] == "peek":
                _worker_peek(conn, msg[1], spec_names, ns, chains, seg_productions)
            elif isinstance(msg, tuple) and msg[0] == "end_turn":
                conn.send(("turn_ended",))
            elif isinstance(msg, tuple) and msg[0] == "fire_chain":
                _worker_fire_chain(conn, msg[1], msg[2], chains, ns)
            else:
                _worker_exec(
                    conn,
                    msg,
                    ns,
                    spec_names,
                    taint_skip,
                    budget,
                    seg_productions,
                    chains,
                    chain_counter,
                )
    except (EOFError, OSError):
        pass
    finally:
        conn.close()


def _register_segment_productions(
    seg: Segment, ns: dict, hooks: set[str], seg_productions: dict[str, tuple]
) -> None:
    """After a segment executes, remember which names were bound to a hooked
    call (``doc = fetch(..)``): later tail peeks chain consumers off them.

    The producer identity is a RAW-material hash ((tool, canonical_hash) of the
    call as written) that the parent maps to the real claim key it dispatched.
    """
    try:
        tree = ast.parse(seg.source)
    except SyntaxError:
        return
    for stmt in tree.body:
        if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
            continue
        target = stmt.targets[0]
        if not isinstance(target, ast.Name) or not isinstance(stmt.value, ast.Call):
            continue
        call = stmt.value
        if not (isinstance(call.func, ast.Name) and call.func.id in hooks):
            continue
        try:
            args = tuple(safe_eval(a, ns) for a in call.args)
            kwargs = {k.arg: safe_eval(k.value, ns) for k in call.keywords if k.arg}
        except Exception:
            continue  # args not statically resolvable: no production recorded
        seg_productions[target.id] = (
            "segkey",
            (call.func.id, canonical_hash(call.func.id, args, kwargs)),
        )


def _plan_tainted_segment(
    conn,
    source: str,
    tree: ast.Module,
    ns: dict,
    hooks: set[str],
    seg_productions: dict[str, tuple],
    chains: dict[int, ChainPlan],
    counter: "_ContCounter",
) -> None:
    """Plan a taint-skipped segment's hooked calls: concrete ones dispatch
    now, marker-reading ones become chained continuations."""
    try:
        plans: list = []
        chain_plans: list = []
        metas: list = []
        _plan_body(
            tree_body=tree.body,
            spec_names=hooks,
            ns=ns,
            tail=source,  # the segment IS the complete (closed) text
            plans=plans,
            chain_plans=chain_plans,
            metas=metas,
            productions=seg_productions,
            counter=counter,
            dep_capable=set(seg_productions),
        )
    except Exception:
        return
    for cp in chain_plans:
        chains[cp.cont_id] = cp
    if plans or metas:
        conn.send(("plans", plans, metas))


def _worker_exec(
    conn,
    seg: Segment,
    ns: dict,
    hooks: set[str],
    taint_skip: bool,
    budget: float,
    seg_productions: dict[str, tuple] | None = None,
    chains: dict[int, ChainPlan] | None = None,
    chain_counter: "_ContCounter | None" = None,
) -> None:
    try:
        tree = ast.parse(seg.source)
    except SyntaxError:
        conn.send(("abort", "syntax"))
        return
    # guard: rebinding a hooked name evicts + aborts
    for name in _bound_names(tree):
        if name in hooks:
            conn.send(("evict_tool", name))
            conn.send(("abort", f"rebind:{name}"))
            return
    reads = _read_names(tree)
    # a statement reading a NonSpeculated marker is skipped and its targets
    # poisoned — taint flows by value instead of killing the turn. The
    # statement's own hooked calls become CHAINED continuations: they fire
    # when the producers' speculations resolve (pipelining the dataflow chain
    # under the still-streaming output), and concrete calls still dispatch.
    if taint_skip:
        tainted = sorted(n for n in reads if contains_nonspec(ns.get(n)))
        if tainted:
            for name in _bound_names(tree) - _comp_local_names(tree):
                ns[name] = NonSpeculated("tainted:" + "+".join(tainted))
            if (
                seg_productions is not None
                and chains is not None
                and chain_counter is not None
            ):
                _plan_tainted_segment(
                    conn,
                    seg.source,
                    tree,
                    ns,
                    hooks,
                    seg_productions,
                    chains,
                    chain_counter,
                )
            return

    # Runaway guard: SIGALRM raises ShadowAborted in this process when a
    # statement's wall-clock compute exceeds its budget. There is no spec-wait
    # in the subprocess (hooks return markers, never block), so the budget is
    # pure wall-clock — exactly right for `while True:` spinning.
    def _alarm(signum, frame):
        raise ShadowAborted("runaway")

    old = signal.signal(signal.SIGALRM, _alarm)
    signal.setitimer(signal.ITIMER_REAL, budget)
    try:
        exec(compile(tree, "<shadow>", "exec"), ns, ns)
        if seg_productions is not None:
            _register_segment_productions(seg, ns, hooks, seg_productions)
        conn.send(("executed",))
    except BaseException as e:
        conn.send(("abort", f"{type(e).__name__}: {e}"))
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def _worker_peek(
    conn,
    tail: str,
    spec_names: set[str],
    ns: dict,
    chains: dict[int, ChainPlan],
    seg_productions: dict[str, tuple],
) -> None:
    try:
        # Patch-seam preservation (tests monkeypatch
        # ``dspy_rlm_hooks.speculation.shadow.plan_peeks_with_chains``): in the
        # pre-split single module this call resolved the name from the shadow
        # module namespace. Resolve through the package namespace at call time
        # so patching the package attribute still reaches the worker.
        from dspy_rlm_hooks.speculation import shadow as _shadow_pkg

        plans, chain_plans, metas = _shadow_pkg.plan_peeks_with_chains(
            tail, spec_names, ns, segment_productions=seg_productions
        )
    except Exception:
        plans, chain_plans, metas = [], [], []
    # the newest plan generation replaces the chain table entirely; a chain
    # dropped here is re-planned by the next generation, and the parent fires
    # re-planned chains immediately when their deps are already resolved
    chains.clear()
    for cp in chain_plans:
        chains[cp.cont_id] = cp
    conn.send(("plans", plans, metas))


def _worker_fire_chain(
    conn, cont_id: int, dep_values: dict, chains: dict[int, ChainPlan], ns: dict
) -> None:
    """Evaluate a chained call's args against the resolved dep values and
    dispatch it through the normal record path (same dedup/adopt machinery)."""
    chain = chains.get(cont_id)
    if chain is None:
        return
    env = dict(ns)
    for name, blob in dep_values.items():
        try:
            env[name] = pickle.loads(blob)
        except Exception:
            return  # a dep value that cannot cross the pipe: drop the chain
    try:
        args: list = []
        for kind, payload in chain.arg_specs:
            args.append(payload if kind == "const" else safe_eval(payload, env))
        kwargs: dict = {}
        for name, (kind, payload) in chain.kwarg_specs.items():
            kwargs[name] = payload if kind == "const" else safe_eval(payload, env)
    except Exception:
        return  # fire-time evaluation failed: the real run executes the call
    # cont_id rides along so the parent can map this chain to its claim key
    conn.send(("tool", chain.tool, tuple(args), kwargs, cont_id))
