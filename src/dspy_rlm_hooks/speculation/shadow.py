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
from collections import Counter
from types import FunctionType
from typing import Any

from dspy_rlm_hooks.speculation.store import SpecStore
from dspy_rlm_hooks.speculation.streaming import Segment, plan_peeks
from dspy_rlm_hooks.speculation.tool import NonSpeculated, contains_nonspec, spec_key

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
    """Use the default start method (``spawn`` on macOS/Windows).

    ``spawn`` is safe to fork from a multi-threaded host (no fork deadlock risk)
    at the cost of pickling the payload — which ``_picklable_ns`` guarantees.
    """
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


# =============================================================================
# worker process
# =============================================================================


def _shadow_worker(conn, parent_conn, payload: dict) -> None:
    """Run in the subprocess: execute segments, record predicted calls, plan peeks."""
    parent_conn.close()
    ns: dict = payload["ns"]
    spec_names: set[str] = payload["spec_names"]
    taint_skip: bool = payload["taint_skip"]
    budget: float = payload["budget"]
    ns["__builtins__"] = shadow_builtins(dict(_builtins.__dict__))
    ns.setdefault("__name__", "__main__")  # class stmts need it
    # recording hooks live in the worker so they share its pipe end
    for name in spec_names:
        ns[name] = _make_record_hook(conn, name)
    # cross-turn helpers keep __globals__ on the shadow namespace
    for k, v in list(ns.items()):
        if isinstance(v, FunctionType) and k not in spec_names:
            ns[k] = _rebind(v, ns)
    try:
        while True:
            msg = conn.recv()
            if msg is None:
                break
            if isinstance(msg, tuple) and msg[0] == "peek":
                _worker_peek(conn, msg[1], spec_names, ns)
            else:
                _worker_exec(conn, msg, ns, spec_names, taint_skip, budget)
    except (EOFError, OSError):
        pass
    finally:
        conn.close()


def _worker_exec(
    conn, seg: Segment, ns: dict, hooks: set[str], taint_skip: bool, budget: float
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
    # poisoned — taint flows by value instead of killing the turn
    if taint_skip:
        tainted = sorted(n for n in reads if contains_nonspec(ns.get(n)))
        if tainted:
            for name in _bound_names(tree) - _comp_local_names(tree):
                ns[name] = NonSpeculated("tainted:" + "+".join(tainted))
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
        conn.send(("executed",))
    except BaseException as e:
        conn.send(("abort", f"{type(e).__name__}: {e}"))
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def _worker_peek(conn, tail: str, spec_names: set[str], ns: dict) -> None:
    try:
        plans = plan_peeks(tail, spec_names, ns)
    except Exception:
        plans = []
    conn.send(("plans", plans))


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
    ) -> None:
        self.store = store
        self.launcher = launcher  # needed only for peeks
        self.registry = registry
        self.taint_skip = taint_skip
        self.hooks = dict(shadow_hooks)
        self.aborted: str | None = None
        self.executed = 0
        self.predicted: list[tuple[str, tuple]] = []  # (tool_name, args)
        self._last_peek_tally: dict = {}  # spec_key -> count from the last plan
        self._done = threading.Event()

        ctx = _mp_context()
        self._parent_conn, child_conn = ctx.Pipe()
        ns = _picklable_ns(snapshot_ns(real_locals))
        ns.setdefault("__name__", "__main__")
        payload = {
            "ns": ns,
            "spec_names": set(shadow_hooks),
            "taint_skip": taint_skip,
            "budget": stmt_budget,
        }
        self._proc = ctx.Process(
            target=_shadow_worker,
            args=(child_conn, self._parent_conn, payload),
            daemon=True,
        )
        self._proc.start()
        child_conn.close()
        self._conn = self._parent_conn
        self._reader = threading.Thread(
            target=self._read, daemon=True, name="shadow-reader"
        )
        self._reader.start()

    # -- producer side --------------------------------------------------------
    def feed(self, seg: Segment) -> None:
        self._conn.send(seg)

    def feed_peek(self, tail: str) -> None:
        """Queue a peek over the current unclosed tail. Runs in the worker AFTER
        all fed statements, so the namespace it evaluates against is exactly the
        state those statements produced."""
        self._conn.send(("peek", tail))

    def finish(self) -> None:
        self._conn.send(None)

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
        self.aborted = why
        if self._proc.is_alive():
            self._proc.terminate()

    # -- reader ---------------------------------------------------------------
    def _read(self) -> None:
        try:
            while True:
                msg = self._conn.recv()
                if msg is None:
                    break
                kind = msg[0]
                if kind == "tool":
                    _, name, args, kwargs = msg
                    self.predicted.append((name, args))
                    self._dispatch(name, args, kwargs)
                elif kind == "plans":
                    self._handle_plans(msg[1])
                elif kind == "executed":
                    self.executed += 1
                elif kind == "abort":
                    self.aborted = msg[1]
                elif kind == "evict_tool":
                    self.store.evict_tool(msg[1], "shadow-rebind")
        except (EOFError, OSError):
            pass
        finally:
            self._done.set()

    def _dispatch(self, name: str, args: tuple, kwargs: dict) -> None:
        if self.launcher is None:
            return
        tool = self.registry.get(name) if self.registry else None
        if tool is None or not tool.speculatable:
            return
        if tool.gate_fn and not tool.gate_fn(args, kwargs):
            return
        self.launcher.ensure_peeked(tool, args, kwargs, 1)

    def _handle_plans(self, plans) -> None:
        if self.launcher is None:
            return
        tally = Counter(
            (p.tool, p.args, tuple(sorted(p.kwargs.items()))) for p in plans
        )
        new_tally: dict = {}
        for (tool_name, args, kwargs_items), needed in tally.items():
            kwargs = dict(kwargs_items)
            tool = self.registry.get(tool_name) if self.registry else None
            if tool is None or not tool.speculatable:
                continue
            if tool.gate_fn and not tool.gate_fn(args, kwargs):
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
