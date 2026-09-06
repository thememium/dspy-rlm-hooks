"""ShadowRunner: subprocess-isolated speculative execution of generated code.

Task 4 — builds against the frozen contracts from Task 1 (``tool.py`` /
``config.py``), the store (Task 2 ``store.py``) and streaming (Task 3
``streaming.py``).

Per the SPIKE (Task 0) decision, the shadow runs in a SEPARATE SUBPROCESS, not
in the host process: the in-process jail blocks ``__import__``/``open``/``eval``
but cannot block the object-introspection escape
(``().__class__.__mro__[1].__subclasses__()``), which reaches real host objects.
In a subprocess that escape can only touch the subprocess's own memory, never
the host's. The real Deno interpreter already runs in a subprocess, so a
subprocess shadow is architecturally consistent.

The worker process is seeded with the assembled namespace (deepcopy-forked via
``snapshot_ns``) + jailed builtins + recording shadow hooks. It executes each
:class:`Segment`, records the predicted ``(tool, args)`` calls, and streams them
back to the parent over a ``multiprocessing`` pipe for claim-hook dispatch. A
runaway watchdog (SIGALRM in the worker) bounds each statement's wall-clock
time, and the parent's ``join(timeout)`` bounds the whole shadow so a hang never
blocks real execution.
"""

from __future__ import annotations

import ast
import builtins as _builtins
import copy
import multiprocessing
import pickle
import signal
import threading
from types import FunctionType, SimpleNamespace
from typing import Any

from dspy_rlm_hooks.speculation.store import SpecStore
from dspy_rlm_hooks.speculation.streaming import (
    ChainMeta,
    ChainPlan,
    Segment,
    _ContCounter,
    _plan_body,
    plan_peeks_with_chains,
    safe_eval,
)
from dspy_rlm_hooks.speculation.tool import (
    NonSpeculated,
    canonical_hash,
    contains_nonspec,
    spec_key,
    split_batch_call,
)

STMT_WALL_BUDGET_S = 2.0  # runaway guard: max wall time per shadow statement


class Opaque:
    """Marker for values that refused deepcopy; any use raises -> opaque-abort."""

    def __init__(self, name: str) -> None:
        self._name = name

    def __getattr__(self, k: str) -> Any:
        raise RuntimeError(f"opaque value {self._name!r} touched in shadow")

    def __getitem__(self, k: Any) -> Any:
        raise RuntimeError(f"opaque value {self._name!r} touched in shadow")

    def __iter__(self):
        raise RuntimeError(f"opaque value {self._name!r} touched in shadow")


def _rebind(fn: FunctionType, ns: dict) -> FunctionType:
    """Same code, resolving globals in the shadow namespace instead of the real one."""
    out = FunctionType(fn.__code__, ns, fn.__name__, fn.__defaults__, fn.__closure__)
    out.__dict__.update(fn.__dict__)
    out.__kwdefaults__ = fn.__kwdefaults__
    return out


def snapshot_ns(ns: dict) -> dict:
    """Deepcopy the non-dunder locals; un-copyable values become :class:`Opaque`.

    The copy is what the subprocess executes against, so mutations stay inside
    the fork and never reach the host's objects.
    """
    out: dict = {}
    for k, v in ns.items():
        if k.startswith("__"):
            continue
        try:
            out[k] = copy.deepcopy(v)
        except Exception:
            # dict/list subclasses with un-copyable attrs: a plain cast keeps the
            # DATA and drops the behavior — mutations stay inside the fork.
            if isinstance(v, dict):
                try:
                    out[k] = {kk: copy.deepcopy(vv) for kk, vv in v.items()}
                    continue
                except Exception:
                    pass
            if isinstance(v, (list, tuple)):
                try:
                    out[k] = type(v)(copy.deepcopy(x) for x in v)
                    continue
                except Exception:
                    pass
            out[k] = Opaque(k)
    return out


# Pure stdlib modules with no import-time side effects: model code imports
# these constantly (re, json, collections...) and blocking them silenced whole
# turns. random/os/time stay blocked (nondeterminism / effects).
_SHADOW_IMPORT_WHITELIST = {
    "re",
    "asyncio",  # async tools: model code needs run/gather to reach the hooks
    "json",
    "time",  # perf_counter measurements in model code; runaway sleeps are SIGALRM-bounded
    "math",
    "itertools",
    "collections",
    "functools",
    "operator",
    "statistics",
    "string",
    "textwrap",
    "heapq",
    "bisect",
    "difflib",
    "ast",
    "unicodedata",
    "fractions",
    "decimal",
    "copy",
    "typing",
    "dataclasses",
}


def _shadow_import(name, *args, **kwargs):
    root = name.split(".")[0]
    if root in _SHADOW_IMPORT_WHITELIST:
        return __import__(name, *args, **kwargs)
    raise RuntimeError(f"import {name!r} blocked in shadow (not in pure whitelist)")


_SHADOW_BLOCKED = {
    "open",
    "eval",
    "exec",
    "compile",
    "input",
    "exit",
    "quit",
    "help",
    "breakpoint",
}


def shadow_builtins(real_builtins: dict) -> dict:
    """Jailed builtins: block dangerous names, restrict ``__import__``, drop ``print``."""
    b = dict(real_builtins)
    for name in _SHADOW_BLOCKED:
        b[name] = _blocked(name)
    b["__import__"] = _shadow_import
    b["print"] = _shadow_print  # captured-and-discarded
    return b


def _shadow_print(*a, **k):
    pass


def _blocked(name: str):
    def fn(*a, **k):
        raise RuntimeError(f"{name}() blocked in shadow")

    return fn


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


def _picklable_ns(ns: dict) -> dict:
    """Replace any value that can't cross the spawn boundary with :class:`Opaque`.

    ``NonSpeculated`` markers are deliberately converted to ``Opaque``: they
    cannot be pickled (their ``__getattr__`` breaks the pickle protocol), and a
    marker passed in via ``real_locals`` is meaningless across a process
    boundary anyway — the shadow's own taint markers are generated in the worker
    and never cross it.
    """
    out: dict = {}
    for k, v in ns.items():
        if isinstance(v, NonSpeculated):
            out[k] = Opaque(k)
            continue
        try:
            pickle.dumps(v)
            out[k] = v
        except Exception:
            out[k] = Opaque(k)
    return out


def classify_ns(ns: dict) -> dict[str, bytes | None]:
    """One pass over the host locals: pickle each non-dunder value to bytes
    (validating picklability), ``None`` for anything that cannot cross the
    spawn boundary (becomes :class:`Opaque` in the worker).

    This replaces the old ``deepcopy + probe-pickle + spawn-pickle`` triple
    serialization with a single pickle per value; the spawn transfer then ships
    the already-serialized bytes. ``NonSpeculated`` markers never cross.
    """
    out: dict[str, bytes | None] = {}
    for k, v in ns.items():
        if k.startswith("__"):
            continue
        if isinstance(v, NonSpeculated):
            out[k] = None  # marker is meaningless in the worker; taint it
            continue
        try:
            out[k] = pickle.dumps(v, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            # dict/list subclasses whose BEHAVIOR breaks pickling: a plain cast
            # keeps the DATA (mutations stay inside the fork either way).
            if isinstance(v, dict):
                try:
                    out[k] = pickle.dumps(dict(v), protocol=pickle.HIGHEST_PROTOCOL)
                    continue
                except Exception:
                    pass
            if isinstance(v, (list, tuple)):
                try:
                    out[k] = pickle.dumps(
                        list(v) if isinstance(v, list) else tuple(v),
                        protocol=pickle.HIGHEST_PROTOCOL,
                    )
                    continue
                except Exception:
                    pass
            out[k] = None
    return out


def load_ns(classified: dict[str, bytes | None]) -> dict:
    """Inverse of :func:`classify_ns` (worker side): bytes -> values, ``None``
    -> :class:`Opaque`."""
    out: dict = {}
    for k, blob in classified.items():
        if blob is None:
            out[k] = Opaque(k)
        else:
            out[k] = pickle.loads(blob)
    return out


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
        plans, chain_plans, metas = plan_peeks_with_chains(
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


# =============================================================================
# parent side
# =============================================================================


class ShadowRunner:
    """Subprocess-isolated speculative executor of generated code.

    The worker process executes each :class:`Segment` against a deepcopy-forked
    namespace with jailed builtins and recording hooks. Predicted ``(tool, args)``
    calls and peek plans stream back to the parent for claim-hook dispatch.
    """

    def __init__(
        self,
        real_locals: dict,
        shadow_hooks: dict,
        store: SpecStore,
        real_builtins: dict,
        launcher=None,
        registry=None,
        taint_skip: bool = True,
        stmt_budget: float = STMT_WALL_BUDGET_S,
        persistent: bool = False,
    ) -> None:
        self.store = store
        self.launcher = launcher  # needed only for peeks
        self.registry = registry
        self.taint_skip = taint_skip
        self.hooks = dict(shadow_hooks)
        self.persistent = persistent  # stay alive across turns (reset per turn)
        self.aborted: str | None = None
        self.executed = 0
        self.predicted: list[tuple[str, tuple]] = []  # (tool_name, args)
        self._last_peek_tally: dict = {}  # spec_key -> count from the last plan
        self._done = threading.Event()
        self._turn_ended = threading.Event()
        self._turn_open = False  # worker is between reset and end_turn
        # chained-continuation state (see plan_peeks_with_chains)
        self._chain_lock = threading.Lock()
        self._pending_chains: dict[int, ChainMeta] = {}
        self._cont_keys: dict[int, Any] = {}  # cont_id -> realized claim key
        self._segkey_to_real: dict[tuple, Any] = {}  # raw-material id -> claim key
        # batched-call decomposition: element peeks under per-element claim
        # keys, assembled back into the ordered list when all resolve
        self._batch_seq = 0
        self._batch_keys: dict[Any, list] = {}  # batch key -> [element keys]
        self._batch_values: dict[Any, dict[int, Any]] = {}  # batch key -> {idx: value}
        self._key_to_batch: dict[
            Any, tuple[Any, int]
        ] = {}  # element key -> (batch key, idx)
        self._chain_ready: dict[int, set[str]] = {}  # cont_id -> satisfied deps
        self._dep_values: dict[int, dict[str, Any]] = {}  # cont_id -> dep results
        self._send_lock = threading.Lock()  # the pipe is not thread-safe
        self._closed = False
        # classified seed — kept so a crashed worker can respawn without
        # re-serializing the host namespace
        self._seed = classify_ns(real_locals)
        self._real_builtins = real_builtins
        self._stmt_budget = stmt_budget
        # last per-turn seed pushed via begin_turn(ns) — skipping an identical
        # reseed avoids re-serializing the whole namespace every turn
        self._last_turn_seed = dict(real_locals)
        self._spawn()

    # -- process lifecycle ----------------------------------------------------
    def _spawn(self) -> None:
        """Start (or restart) the worker subprocess from the classified seed."""
        self._closed = False
        ctx = _mp_context()
        self._parent_conn, child_conn = ctx.Pipe()
        payload = {
            "ns_seed": self._seed,
            "spec_names": set(self.hooks),
            "taint_skip": self.taint_skip,
            "budget": self._stmt_budget,
        }
        self._proc = ctx.Process(
            target=_shadow_worker,
            args=(child_conn, self._parent_conn, payload),
            daemon=True,
        )
        self._proc.start()
        child_conn.close()
        self._conn = self._parent_conn
        self._done = threading.Event()
        self._turn_ended = threading.Event()
        self._turn_open = True  # a freshly spawned worker must be drained too
        self._reader = threading.Thread(
            target=self._read, daemon=True, name="shadow-reader"
        )
        self._reader.start()
        if (
            self.launcher is not None
            and getattr(self.launcher, "bus", None) is not None
        ):
            self.launcher.bus.subscribe(self._on_bus_event)

    # -- chained continuations -------------------------------------------------
    def _on_bus_event(self, kind: str, data: dict) -> None:
        """Watch speculation resolutions and fire continuations whose
        dependencies are satisfied."""
        if self._closed or kind != "ready":
            return
        key = data.get("key")
        spec = data.get("spec")
        try:
            self._fire_chains_for_key(key, spec)
        except Exception:
            pass  # continuation fires must never break dispatch

    def _chain_dep_matches(self, meta: ChainMeta, key: Any) -> bool:
        """True when any dep of the chain refers to the given claim key."""
        for _name, (ref_kind, ref) in meta.deps.items():
            if ref_kind == "key" and ref == key:
                return True
            if ref_kind == "cont" and self._cont_keys.get(ref) == key:
                return True
        return False

    def _fire_chains_for_key(self, key: Any, spec: Any) -> None:
        """Record a producer resolution; fire every chain whose deps are now
        all satisfied. A FAILED producer drops its dependent chains (the real
        run will surface the same error through its own claim)."""
        if key in self._key_to_batch:
            self._record_batch_element(key, spec)
            return
        if spec is not None and spec.error is not None:
            with self._chain_lock:
                self._pending_chains = {
                    cid: m
                    for cid, m in self._pending_chains.items()
                    if not self._chain_dep_matches(m, key)
                }
            return
        value = spec._result if spec is not None else None
        to_fire: list[tuple[int, dict[str, Any]]] = []
        with self._chain_lock:
            for cont_id, meta in self._pending_chains.items():
                if not self._chain_dep_matches(meta, key):
                    continue
                values = self._dep_values.setdefault(cont_id, {})
                for name, (ref_kind, ref) in meta.deps.items():
                    if name in values:
                        continue
                    if (ref_kind == "key" and ref == key) or (
                        ref_kind == "cont" and self._cont_keys.get(ref) == key
                    ):
                        values[name] = value
                    elif ref_kind == "key":
                        # dep already ready from an earlier resolution
                        prior = self._result_for_key(ref)
                        if prior is not None or self._key_resolved(ref):
                            values[name] = prior
                if meta.deps and all(n in values for n in meta.deps):
                    to_fire.append((cont_id, dict(values)))
        for cont_id, values in to_fire:
            self._send_chain_fire(cont_id, values)

    def _key_resolved(self, key: Any) -> bool:
        """True when some speculation for the key resolved (value may be None)."""
        with self.store._lock:
            for spec in self.store._q.get(key, ()):
                if spec.state in ("ready", "claimed"):
                    return True
        return False

    def _result_for_key(self, key: Any) -> Any:
        """Best-effort resolved value for a key (for deps already ready when a
        chain registers later)."""
        with self.store._lock:
            for spec in self.store._q.get(key, ()):
                if spec.state in ("ready", "claimed") and spec.error is None:
                    return spec._result
        return None

    def _send_chain_fire(self, cont_id: int, values: dict[str, Any]) -> None:
        blobs: dict[str, bytes] = {}
        for name, value in values.items():
            try:
                blobs[name] = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
            except Exception:
                return  # unpicklable dep result: drop the chain
        try:
            with self._send_lock:
                self._conn.send(("fire_chain", cont_id, blobs))
        except Exception:
            pass

    @property
    def is_alive(self) -> bool:
        return self._proc.is_alive()

    def ensure_alive(self) -> None:
        """Respawn the worker if it died (a persistent runner outlives crashes)."""
        if not self._proc.is_alive():
            self.aborted = None
            self._spawn()

    # -- producer side --------------------------------------------------------
    def feed(self, seg: Segment) -> None:
        with self._send_lock:
            self._conn.send(seg)

    def feed_peek(self, tail: str) -> None:
        """Queue a peek over the current unclosed tail. Runs in the worker AFTER
        all fed statements, so the namespace it evaluates against is exactly the
        state those statements produced."""
        self._conn.send(("peek", tail))

    def begin_turn(self, ns: dict | None = None) -> None:
        """Start a turn on a persistent runner: reset worker state to the seed
        and clear this turn's parent-side accumulators. Pipe ordering
        guarantees the reset lands before any segments fed after it.

        ``ns`` (when given) RESEEDS the worker: the original seed captures only
        ``input_args``, but cross-iteration state sync produces a fresh live
        snapshot each turn; without reseeding, a persistent worker would reset
        to stale values every turn. An identical namespace skips the reseed
        (no re-serialization)."""
        self.ensure_alive()
        self.predicted = []
        self.executed = 0
        self.aborted = None
        self._last_peek_tally = {}
        with self._chain_lock:
            self._pending_chains = {}
            self._cont_keys = {}
            self._segkey_to_real = {}
            self._batch_keys = {}
            self._batch_values = {}
            self._key_to_batch = {}
            self._chain_ready = {}
            self._dep_values = {}
        self._turn_ended.clear()
        reseed = None
        if ns is not None and ns != self._last_turn_seed:
            self._seed = classify_ns(ns)
            self._last_turn_seed = dict(ns)
            reseed = self._seed
        with self._send_lock:
            self._conn.send(("reset", reseed))
        self._turn_open = True

    def end_turn(self, timeout: float | None = None) -> bool:
        """Drain a persistent runner's turn: wait until the worker has
        processed every message fed so far. Returns True when acknowledged."""
        if not self._turn_open:
            return True
        self._turn_open = False
        if not self._proc.is_alive():
            return False
        try:
            with self._send_lock:
                self._conn.send(("end_turn",))
        except (OSError, BrokenPipeError):
            return False
        return self._turn_ended.wait(timeout)

    def shutdown(self) -> None:
        """Terminate a persistent runner gracefully (session close)."""
        self._closed = True
        self._turn_open = False
        self.finish()
        self.join(5)

    def finish(self) -> None:
        try:
            with self._send_lock:
                self._conn.send(None)
        except (BrokenPipeError, OSError):
            pass  # already terminated (e.g. abort() raced ahead of finish())

    def join(self, timeout: float | None = None) -> bool:
        """Wait for the worker to finish; terminate it on timeout. Returns True
        if it completed (a hang never blocks real execution)."""
        self._proc.join(timeout)
        if self._proc.is_alive():
            self._proc.terminate()
            self._proc.join(5)
            return False
        self._reader.join(timeout or 5)
        return True

    def abort(self, why: str = "external") -> None:
        self._closed = True
        self.aborted = why
        if self._proc.is_alive():
            self._proc.terminate()
            self._proc.join(2)  # deterministic death for respawn checks

    # -- reader ---------------------------------------------------------------
    def _read(self) -> None:
        try:
            while True:
                msg = self._conn.recv()
                if msg is None:
                    break
                kind = msg[0]
                if kind == "tool":
                    _, name, args, kwargs = msg[:4]
                    cont_id = msg[4] if len(msg) > 4 else None
                    self.predicted.append((name, args))
                    key = self._dispatch(name, args, kwargs)
                    if key is not None:
                        seg_id = (name, canonical_hash(name, args, kwargs))
                        with self._chain_lock:
                            self._segkey_to_real[seg_id] = key
                            if cont_id is not None:
                                self._cont_keys[cont_id] = key
                elif kind == "plans":
                    self._handle_plans(msg[1], msg[2] if len(msg) > 2 else ())
                elif kind == "executed":
                    self.executed += 1
                elif kind == "abort":
                    self.aborted = msg[1]
                elif kind == "turn_ended":
                    self._turn_ended.set()
                elif kind == "evict_tool":
                    self.store.evict_tool(msg[1], "shadow-rebind")
        except (EOFError, OSError):
            pass
        finally:
            self._done.set()

    def _dispatch(self, name: str, args: tuple, kwargs: dict) -> Any:
        """Dispatch one shadow-recorded call; returns its claim key (or None).

        The call comes from an EXECUTED segment, i.e. the model's statement
        closed: its speculation is ADOPTED so tail-shrink bet retraction can
        no longer evict it before the real run claims it.
        """
        if self.launcher is None:
            return None
        tool = self.registry.get(name) if self.registry else None
        if tool is None or not tool.speculatable:
            return None
        if tool.gate_fn and not tool.gate_fn(args, kwargs):
            return None
        if contains_nonspec(args) or contains_nonspec(kwargs):
            # the sandbox recorded this call with a TAINT MARKER in its args
            # (assigned and used within the same segment): never dispatch a
            # speculative call with garbage arguments
            return None
        # Batched tools (llm_query_batched & friends) are claimed PER ELEMENT
        # by the real run, so a whole-batch peek could never be claimed —
        # dispatch one peek per prompt element instead.
        decomposed = self._decompose_batched(name, args, kwargs, needed=1, adopt=True)
        if decomposed is not None:
            return decomposed[0]
        self.launcher.ensure_peeked(tool, args, kwargs, 1)
        key = spec_key(tool, args, kwargs)
        self.store.adopt(key)
        return key

    def _batch_key_for(self, name: str, args: tuple, kwargs: dict) -> Any:
        """Synthetic stable key for a batched call's ASSEMBLED result, or None
        when the call is not batched-shaped (or has no single-tool registry
        entry to claim elements against)."""
        registry = self.registry
        if registry is None or not name.endswith("_batched"):
            return None
        split = split_batch_call(args, kwargs)
        if split is None:
            return None
        prompts, rest, clean, _kwname = split
        single = registry.get(name[: -len("_batched")])
        if single is None:
            return None
        return ("__batch__", f"{name}|{spec_key(single, (prompts,) + rest, clean)[1]}")

    def _decompose_batched(
        self, name: str, args: tuple, kwargs: dict, needed: int, adopt: bool
    ) -> tuple[Any, list] | None:
        """Dispatch one peek PER ELEMENT of a batched call, under the same
        per-element claim keys the real run uses. Registers a batch group so
        the ordered list of element results can be assembled and published to
        dependent chains when every element resolves. Returns
        ``(batch_key, elem_keys)``, or None when the call is not batched-shaped
        (the caller falls back to a whole-batch dispatch)."""
        if self.launcher is None or self.registry is None:
            return None
        batch_key = self._batch_key_for(name, args, kwargs)
        if batch_key is None:
            return None
        split = split_batch_call(args, kwargs)
        if split is None:
            return None
        prompts, rest, clean, _kwname = split
        single = self.registry.get(name[: -len("_batched")])
        assert single is not None
        elem_keys: list = []
        for elem in prompts:
            elem_args = (elem,) + rest
            key = spec_key(single, elem_args, clean)
            elem_keys.append(key)
            self.launcher.ensure_peeked(single, elem_args, clean, needed)
            if adopt:
                self.store.adopt(key)
        with self._chain_lock:
            self._batch_keys[batch_key] = elem_keys
            self._batch_values.setdefault(batch_key, {})
            for i, k in enumerate(elem_keys):
                self._key_to_batch[k] = (batch_key, i)
        return batch_key, elem_keys

    def _record_batch_element(self, key: Any, spec: Any) -> None:
        """Record one batch element's resolved value; assemble and publish the
        ordered list to dependent chains when the last element lands."""
        batch_key, idx = self._key_to_batch[key]
        with self._chain_lock:
            values = self._batch_values.setdefault(batch_key, {})
            if spec is not None and spec.error is None:
                values[idx] = spec._result
            else:
                # a failed element means the batch result is garbage: never
                # assemble (dependent chains stay unfired, like failed producers)
                self._batch_keys.pop(batch_key, None)
                return
            keys = self._batch_keys.get(batch_key, [])
            complete = bool(keys) and all(i in values for i in range(len(keys)))
            assembled = [values[i] for i in range(len(keys))] if complete else None
        if complete:
            # publish the assembled list under the batch key (recursion depth 1:
            # the batch key is never itself a batch element)
            self._fire_chains_for_key(
                batch_key, SimpleNamespace(_result=assembled, error=None)
            )

    def _handle_plans(self, plans, chain_metas=()) -> None:
        if self.launcher is None:
            return
        new_tally: dict = {}
        # dedupe by HASHABLE identity (batched args contain lists)
        seen: dict[tuple, int] = {}
        ordered: list = []
        for p in plans:
            tool = self.registry.get(p.tool) if self.registry else None
            if tool is None or not tool.speculatable:
                continue
            if tool.gate_fn and not tool.gate_fn(p.args, p.kwargs):
                continue
            ident = (p.tool, repr(p.args), repr(sorted(p.kwargs.items())))
            if ident in seen:
                seen[ident] += 1
                continue
            seen[ident] = 1
            ordered.append((tool, p.args, p.kwargs))
        for tool, args, kwargs in ordered:
            needed = seen[(tool.name, repr(args), repr(sorted(kwargs.items())))]
            decomposed = self._decompose_batched(
                tool.name, args, kwargs, needed, adopt=False
            )
            if decomposed is not None:
                batch_key, elem_keys = decomposed
                # element-level tally: retraction must evict per-element peeks
                for elem_key in elem_keys:
                    new_tally[elem_key] = needed
                continue
            self.launcher.ensure_peeked(tool, args, kwargs, needed)
            new_tally[spec_key(tool, args, kwargs)] = needed
        # BET RETRACTION: a key the previous plan justified but this one doesn't
        # means new tokens invalidated the bet — evict the stale peeks now.
        for key, old_n in self._last_peek_tally.items():
            keep = new_tally.get(key, 0)
            if keep < old_n:
                self.store.evict_unadopted_peeks(key, keep, "peek-retracted")
        self._last_peek_tally = new_tally

        # CHAIN REGISTRATION: the newest generation replaces pending chains.
        # Worker-side Plan keys hash raw args; real claim keys hash the
        # signature-bound material — translate before watching.
        plan_key_to_real: dict = {}
        for p in plans:
            if p.key is None:
                continue
            tool = self.registry.get(p.tool) if self.registry else None
            if tool is not None:
                batch_key = self._batch_key_for(p.tool, p.args, p.kwargs)
                if batch_key is not None:
                    # the batch's ASSEMBLED list is what dependent chains see
                    plan_key_to_real[p.key] = batch_key
                else:
                    plan_key_to_real[p.key] = spec_key(tool, p.args, p.kwargs)
        translated: list[ChainMeta] = []
        for m in chain_metas:
            deps: dict[str, Any] = {}
            unwatchable = False
            for name, (ref_kind, ref) in m.deps.items():
                if ref_kind == "key":
                    deps[name] = (ref_kind, plan_key_to_real.get(ref, ref))
                elif ref_kind == "segkey":
                    real = self._segkey_to_real.get(ref)
                    if real is None:
                        # the producer dispatch has not been processed yet:
                        # drop the chain (a later peek re-plans it)
                        unwatchable = True
                        break
                    deps[name] = ("key", real)
                else:
                    deps[name] = (ref_kind, ref)
            if not unwatchable:
                translated.append(ChainMeta(cont_id=m.cont_id, tool=m.tool, deps=deps))
        with self._chain_lock:
            self._pending_chains = {m.cont_id: m for m in translated}
        for meta in translated:
            for name, (ref_kind, ref) in meta.deps.items():
                if ref_kind != "key":
                    continue
                if self._key_resolved(ref):
                    spec = self._spec_for_key(ref)
                    if spec is not None:
                        self._fire_chains_for_key(ref, spec)
                        break

    def _spec_for_key(self, key: Any) -> Any:
        with self.store._lock:
            for spec in self.store._q.get(key, ()):
                if spec.state in ("ready", "claimed"):
                    return spec
        return None


def _read_names(tree: ast.AST) -> set[str]:
    return {
        n.id
        for n in ast.walk(tree)
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
    }


def _comp_local_names(tree: ast.AST) -> set[str]:
    """Comprehension/lambda-local targets. They never leak in Python 3, so
    poisoning them would taint unrelated later statements reusing the name."""
    out: set[str] = set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            for g in n.generators:
                out |= {x.id for x in ast.walk(g.target) if isinstance(x, ast.Name)}
        elif isinstance(n, ast.Lambda):
            out |= {a.arg for a in n.args.args}
    return out


def _bound_names(tree: ast.AST) -> set[str]:
    out: set[str] = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            out.add(n.id)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(n.name)
    return out
