"""Jailed builtins for the shadow subprocess: blocked dangerous names, a
pure-stdlib import whitelist, and a captured-and-discarded ``print``."""

from __future__ import annotations

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
