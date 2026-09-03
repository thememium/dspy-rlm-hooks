"""Streaming side: statement segmentation of the token stream + tail peeking
(SafeEval over live state, pre-close loop unrolling).

Task 3 — builds against the frozen contracts from Task 1 (``tool.py`` /
``config.py``). This module is pure parsing/planning: it never executes code.
The shadow (Task 4) consumes the emitted :class:`Segment` objects and the
planned :class:`Plan` objects; the session/hooks (Task 5) build against this
surface.

The Lazy/JIT stage uses :meth:`StreamSegmenter.feed_complete` (wrap the whole
code block as one block, then feed). The streaming follow-up uses
:meth:`StreamSegmenter.feed` with raw model-output deltas.
"""

from __future__ import annotations

import ast
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from dspy_rlm_hooks.speculation.tool import SpecKey, canonical_hash

_COMPOUND = (
    "if ",
    "if(",
    "for ",
    "for(",
    "while ",
    "while(",
    "def ",
    "class ",
    "with ",
    "with(",
    "try:",
    "try ",
    "@",
    "async ",
    "match ",
)
_CONTINUATION = ("elif", "else", "except", "finally")

# Opening-fence languages treated as a Python REPL block (mirrors the RLM's
# own ``_PYTHON_FENCE_LANGS``). The streamed ``code`` field is markdown-fenced
# (`` ```python ... ``` ``); the peek/segmenter only emits inside a recognized
# block, so we accept the python family in addition to the synthetic ``repl``.
_PYTHON_FENCE_LANGS = {"repl", "python", "py", "python3", "py3", ""}


def _is_repl_open(line: str) -> bool:
    """True if *line* opens a Python REPL fence (`` ```repl``/`` ```python``/
    `` ```py``/... or a bare `` ``` ``)."""
    if not line.startswith("```"):
        return False
    rest = line[3:].strip()
    lang = rest.split(maxsplit=1)[0] if rest else ""
    return lang in _PYTHON_FENCE_LANGS


@dataclass
class Segment:
    block_id: int
    index: int
    source: str
    has_call: bool = False  # any ast.Call — decides whether real exec waits on shadow


@dataclass
class _BlockState:
    buf: str = ""
    emitted_upto: int = 0  # char offset of last emitted statement end
    stmt_index: int = 0
    dead: bool = False  # unparsable content seen -> stop emitting
    # incremental-scan cache: complete lines since emitted_upto, plus resume
    # state for the open statement's scan (keeps a growing compound O(n))
    lines: list[str] | None = None
    scan: tuple | None = None  # (j, depth, in_str) resumable scan position


class StreamSegmenter:
    """Feed raw model-output deltas; yields closed statements inside ```repl blocks."""

    def __init__(self) -> None:
        self.text = ""
        self.blocks: list[_BlockState] = []
        self._in_block = False
        self._scan_pos = 0

    def feed_complete(self, code: str) -> list[Segment]:
        """Lazy/JIT convenience: wrap ``code`` as one ``` ``repl`` block and feed it.

        The whole code block is treated as a single block; ``finish()`` closes
        it and emits every provably-closed top-level statement.
        """
        self.feed("```repl\n")
        out = self.feed(code)
        out.extend(self.finish())
        return out

    def feed(self, delta: str) -> list[Segment]:
        self.text += delta
        out: list[Segment] = []
        # scan for fence transitions line by line
        while True:
            nl = self.text.find("\n", self._scan_pos)
            if nl == -1:
                break
            line = self.text[self._scan_pos : nl]
            self._scan_pos = nl + 1
            stripped = line.strip()
            if not self._in_block:
                if _is_repl_open(stripped):
                    self._in_block = True
                    self.blocks.append(_BlockState())
            else:
                if stripped == "```":
                    self._in_block = False
                    out.extend(self._drain(final=True))
                else:
                    blk = self.blocks[-1]
                    blk.buf += line + "\n"
                    if blk.lines is None:
                        blk.lines = []
                    blk.lines.append(line)
                    out.extend(self._drain(final=False))
        return out

    def pending_tail(self) -> str:
        """The current block's not-yet-emitted text (incl. the partial line) —
        the peek engine's input. Empty when not inside a ``` ``repl block."""
        if not self._in_block or not self.blocks:
            return ""
        blk = self.blocks[-1]
        tail = blk.buf[blk.emitted_upto :]
        partial = self.text[self._scan_pos :]
        if partial.strip().startswith("```"):
            partial = ""
        return tail + partial

    def finish(self) -> list[Segment]:
        """Generation ended; close any open block."""
        if self._in_block and self._scan_pos < len(self.text):
            # trailing partial line — only complete lines were added; add rest
            rest = self.text[self._scan_pos :]
            if rest.strip() and not rest.strip().startswith("```"):
                blk = self.blocks[-1]
                blk.buf += rest + "\n"
                if blk.lines is None:
                    blk.lines = []
                blk.lines.append(rest)
        if self._in_block:
            self._in_block = False
            return self._drain(final=True)
        return []

    # -- statement closing ---------------------------------------------------
    def _drain(self, final: bool) -> list[Segment]:
        blk = self.blocks[-1]
        if blk.dead:
            return []
        block_id = len(self.blocks) - 1
        out: list[Segment] = []
        while True:
            src = self._next_closed(blk, final)
            if src is None:
                break
            try:
                tree = ast.parse(src)
            except SyntaxError:
                blk.dead = True  # model wrote broken code; real run will error too
                break
            has_call = any(isinstance(n, ast.Call) for n in ast.walk(tree))
            out.append(
                Segment(
                    block_id=block_id,
                    index=blk.stmt_index,
                    source=src,
                    has_call=has_call,
                )
            )
            blk.stmt_index += 1
        return out

    def _next_closed(self, blk: _BlockState, final: bool) -> str | None:
        """Return source of the next closed top-level statement, advancing the cursor."""
        start = blk.emitted_upto
        lines = blk.lines if blk.lines is not None else []
        # find first non-blank line
        i = 0
        while i < len(lines) and (
            not lines[i].strip() or lines[i].lstrip().startswith("#")
        ):
            i += 1
        if i >= len(lines):
            if final:
                blk.emitted_upto = len(blk.buf)
                blk.lines = []
                blk.scan = None
            return None
        first = lines[i]
        is_compound = first.lstrip().startswith(_COMPOUND)
        if blk.scan is not None and blk.scan[0] > i + 1:
            j, depth, in_str = blk.scan  # resume where the last drain stopped
        else:
            depth, in_str = _scan_line_state(first, 0, None)
            j = i + 1
        # extend while: inside brackets/triple-string, backslash continuation,
        # or (compound) subsequent indented/continuation lines
        while True:
            open_phys = (
                depth > 0
                or in_str is not None
                or (
                    j - 1 >= i
                    and lines[j - 1].rstrip().endswith("\\")
                    and in_str is None
                )
            )
            if j >= len(lines):
                if open_phys or (is_compound and not final):
                    blk.scan = (j, depth, in_str)  # resume here next drain
                    return None  # can't prove closed yet
                if is_compound and final:
                    break
                break
            line = lines[j]
            if open_phys:
                depth, in_str = _scan_line_state(line, depth, in_str)
                j += 1
                continue
            if not is_compound:
                break  # simple stmt closed at its newline
            # compound: continues while indented / blank / continuation kw at col 0
            if not line.strip():
                j += 1
                continue
            if line[0] in " \t":
                depth, in_str = _scan_line_state(line, depth, in_str)
                j += 1
                continue
            # decorators: while everything consumed so far is @-lines, a col-0
            # @/def/class/async line is part of the same (decorated) statement
            if all(
                lines[k].lstrip().startswith("@") for k in range(i, j)
            ) and line.startswith(("@", "def ", "class ", "async ")):
                depth, in_str = _scan_line_state(line, depth, in_str)
                j += 1
                continue
            head = line.split(":")[0].split("(")[0].strip()
            if any(head == k or head.startswith(k + " ") for k in _CONTINUATION):
                depth, in_str = _scan_line_state(line, depth, in_str)
                j += 1
                continue
            # a col-0 non-continuation line: previous compound is closed,
            # but trailing blank lines belong to nobody
            break
        src = "\n".join(lines[i:j])
        consumed = sum(len(ln) + 1 for ln in lines[:j])
        blk.emitted_upto = start + consumed
        blk.lines = lines[j:]
        blk.scan = None
        return src


def _scan_line_state(
    line: str, depth: int, in_str: str | None
) -> tuple[int, str | None]:
    """Track bracket depth and open (triple)strings across one physical line."""
    k, n = 0, len(line)
    while k < n:
        c = line[k]
        if in_str is not None:
            if in_str in ('"""', "'''"):
                if line.startswith(in_str, k):
                    in_str = None
                    k += 3
                    continue
            else:
                if c == "\\":
                    k += 2
                    continue
                if c == in_str:
                    in_str = None
            k += 1
            continue
        if c == "#":
            break
        if line.startswith('"""', k) or line.startswith("'''", k):
            in_str = line[k : k + 3]
            k += 3
            continue
        if c in "\"'":
            in_str = c
            k += 1
            continue
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth = max(0, depth - 1)
        k += 1
    # single-quote strings don't span physical lines (unless backslash — rare; ignored)
    if in_str is not None and in_str not in ('"""', "'''"):
        in_str = None
    return depth, in_str


# ==========================================================================
# tail peeking
# ==========================================================================

MAX_UNROLL = 64  # cap on per-loop pre-dispatch
_TAIL_LIMIT = 20_000  # don't re-parse absurd tails


# =============================================================================
# 1. Repair: make an incomplete tail parseable
# =============================================================================
def repair_tail(tail: str) -> str | None:
    """Close open brackets/strings and add `pass` bodies until `ast.parse`
    accepts the tail. Returns None if it can't be repaired cheaply."""
    if not tail.strip() or len(tail) > _TAIL_LIMIT:
        return None
    lines = tail.split("\n")
    # Iterative (was recursive) so a large unrepairable tail cannot blow the stack.
    for drop in range(len(lines)):
        text = "\n".join(lines[: len(lines) - drop])
        candidates = [text]
        closers = _bracket_closers(text)
        if closers:
            candidates.append(text + closers)
        for base in list(candidates):
            stripped = base.rstrip()
            if stripped.endswith(":"):  # bare compound header
                candidates.append(stripped + "\n    pass")
            candidates.append(
                stripped + "\n    pass" if _last_line_indented(base) else base
            )
        for cand in candidates:
            try:
                ast.parse(cand)
                return cand
            except SyntaxError:
                continue
    return None


def _bracket_closers(text: str) -> str:
    """Best-effort closing sequence for unbalanced brackets outside strings."""
    stack: list[str] = []
    in_str: str | None = None
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if in_str:
            if in_str in ('"""', "'''") and text.startswith(in_str, i):
                in_str = None
                i += 3
                continue
            if len(in_str) == 1:
                if c == "\\":
                    i += 2
                    continue
                if c == in_str or c == "\n":
                    in_str = None
            i += 1
            continue
        if text.startswith('"""', i) or text.startswith("'''", i):
            in_str = text[i : i + 3]
            i += 3
            continue
        if c in "\"'":
            in_str = c
        elif c == "#":
            while i < n and text[i] != "\n":
                i += 1
            continue
        elif c in "([{":
            stack.append({"(": ")", "[": "]", "{": "}"}[c])
        elif c in ")]}" and stack:
            stack.pop()
        i += 1
    out = ""
    if in_str and len(in_str) == 1:
        out += in_str
    return out + "".join(reversed(stack))


def _last_line_indented(text: str) -> bool:
    lines = [ln for ln in text.split("\n") if ln.strip()]
    return bool(lines) and lines[-1][0] in " \t"


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
                    parts.append(str(ev(v.value)))
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
    if tn in ("SpecValue", "Opaque"):
        raise Unresolvable(tn)


# =============================================================================
# 4. Find + plan: which calls in the tail do we bet on?
# =============================================================================
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


def _free_names(node: ast.expr) -> set[str]:
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
        for stmt in plannable:
            for call in _hooked_calls(stmt, spec_names):
                resolved = _resolve_call_or_chain(
                    call,
                    ns,
                    item_assigned,
                    raw_tail,
                    productions,
                    counter,
                    extra_env={n: item_env[n] for n in loop_vars},
                    loop_var_ok=loop_vars,
                    dep_capable=dep_capable,
                )
                # unresolvable calls (e.g. stage-2 of a per-item chain whose
                # arg is stage-1's result) are left for the shadow — the
                # resolvable stage-1 calls still fan out here
                if isinstance(resolved, Plan):
                    plans.append(resolved)
                elif isinstance(resolved, ChainPlan):
                    chain_plans.append(resolved)
            item_assigned |= _assigned_names(stmt)
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
