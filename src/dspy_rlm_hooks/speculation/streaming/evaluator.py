"""SafeEval side: pure, bounded expression evaluation over the shadow
namespace. ``safe_eval`` uses only side-effect-free operations; anything
outside the whitelist raises :class:`Unresolvable`.
"""

from __future__ import annotations

import ast
from collections.abc import Callable
from typing import Any


# =============================================================================
# 2/3. SafeEval: pure, bounded expression evaluation against live REPL state
# =============================================================================
class Unresolvable(Exception):
    """Raised when an expression needs anything outside the pure whitelist."""


_PURE_STR_METHODS = {
    "join",
    "strip",
    "lstrip",
    "rstrip",
    "upper",
    "lower",
    "replace",
    "split",
    "format",
    "startswith",
    "endswith",
    "title",
    "capitalize",
}
_PURE_BUILTINS: dict[str, Callable] = {
    "str": str,
    "int": int,
    "float": float,
    "len": len,
    "min": min,
    "max": max,
    "sum": sum,
    "sorted": sorted,
    "list": list,
    "tuple": tuple,
    "range": range,
    "enumerate": enumerate,
    "zip": zip,
    "repr": repr,
    "abs": abs,
    "round": round,
}


def safe_eval(node: ast.expr, ns: dict[str, Any], depth: int = 0) -> Any:
    """Evaluate an argument expression using only pure, side-effect-free
    operations over `ns` (the shadow namespace). Anything else raises
    Unresolvable — the call is then left for the shadow to handle normally."""
    if depth > 40:
        raise Unresolvable("depth")

    def ev(n: ast.expr) -> Any:
        return safe_eval(n, ns, depth + 1)

    match node:
        case ast.Constant():
            return node.value
        case ast.Name():
            if node.id in ns:
                v = ns[node.id]
                _reject_weird(v)
                return v
            raise Unresolvable(node.id)
        case ast.JoinedStr():
            parts: list[str] = []
            for v in node.values:
                if isinstance(v, ast.FormattedValue):
                    value = ev(v.value)
                    converters = {ord("s"): str, ord("r"): repr, ord("a"): ascii}
                    if v.conversion != -1:
                        value = converters[v.conversion](value)
                    spec = ev(v.format_spec) if v.format_spec is not None else ""
                    parts.append(format(value, spec))
                elif isinstance(v, ast.Constant):
                    parts.append(str(v.value))
                else:
                    raise Unresolvable("fstring")
            return "".join(parts)
        case ast.BinOp(op=ast.Add()):
            return ev(node.left) + ev(node.right)
        case ast.BinOp(op=ast.Mod()):
            return ev(node.left) % ev(node.right)
        case ast.BinOp(op=ast.Mult()):
            return ev(node.left) * ev(node.right)
        case ast.BinOp(op=ast.Sub()):
            return ev(node.left) - ev(node.right)
        case ast.BinOp(op=ast.FloorDiv()):
            return ev(node.left) // ev(node.right)
        case ast.Subscript():
            return ev(node.value)[ev(node.slice)]
        case ast.Slice():
            return slice(
                ev(node.lower) if node.lower else None,
                ev(node.upper) if node.upper else None,
                ev(node.step) if node.step else None,
            )
        case ast.Tuple() | ast.List():
            vals = [ev(e) for e in node.elts]
            return tuple(vals) if isinstance(node, ast.Tuple) else vals
        case ast.Dict():
            out: dict[Any, Any] = {}
            for k, v in zip(node.keys, node.values, strict=True):
                if k is None:
                    raise Unresolvable("dict-unpack")
                out[ev(k)] = ev(v)
            return out
        case ast.Call(func=ast.Name() as f):
            if f.id in _PURE_BUILTINS:
                return _PURE_BUILTINS[f.id](*[ev(a) for a in node.args])
            raise Unresolvable(f"call:{f.id}")
        case ast.Call(func=ast.Attribute() as attr):
            obj = ev(attr.value)
            if isinstance(obj, str) and attr.attr in _PURE_STR_METHODS:
                return getattr(obj, attr.attr)(*[ev(a) for a in node.args])
            raise Unresolvable(f"method:{attr.attr}")
        case ast.ListComp(generators=[gen]) if not gen.is_async:
            it = ev(gen.iter)
            out = []
            for item in it:
                sub = dict(ns)
                _bind(sub, gen.target, item)
                if all(safe_eval(c, sub, depth + 1) for c in gen.ifs):
                    out.append(safe_eval(node.elt, sub, depth + 1))
                if len(out) > 10_000:
                    raise Unresolvable("comp too big")
            return out
        case ast.Compare(ops=[op], comparators=[right]):
            lv, rv = ev(node.left), ev(right)
            match op:
                case ast.Eq():
                    return lv == rv
                case ast.NotEq():
                    return lv != rv
                case ast.Lt():
                    return lv < rv
                case ast.LtE():
                    return lv <= rv
                case ast.Gt():
                    return lv > rv
                case ast.GtE():
                    return lv >= rv
                case ast.In():
                    return lv in rv
                case ast.NotIn():
                    return lv not in rv
            raise Unresolvable("cmp")
        case ast.IfExp():
            return ev(node.body) if ev(node.test) else ev(node.orelse)
        case _:
            raise Unresolvable(type(node).__name__)


def _bind(ns: dict, target: ast.expr, value: Any) -> None:
    if isinstance(target, ast.Name):
        ns[target.id] = value
    elif isinstance(target, (ast.Tuple, ast.List)):
        vs = list(value)
        for t, v in zip(target.elts, vs, strict=False):
            _bind(ns, t, v)
    else:
        raise Unresolvable("bind")


def _reject_weird(v: Any) -> None:
    """Refuse to feed shadow-only artifacts (lazy proxies, opaque markers)
    into peek arguments — their concrete value isn't cheaply known yet."""
    tn = type(v).__name__
    if tn in ("SpecValue", "Opaque", "NonSpeculated"):
        raise Unresolvable(tn)
