"""Namespace snapshotting for the shadow subprocess: opaque markers for
un-copyable values and single-pass pickling across the process boundary."""

from __future__ import annotations

import copy
import pickle
from types import FunctionType
from typing import Any

from dspy_rlm_hooks.speculation.tool import NonSpeculated


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
