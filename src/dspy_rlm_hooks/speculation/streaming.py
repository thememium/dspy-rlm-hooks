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
            if not tree.body:
                continue
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
        # never emit the last line group unless another line follows or final
        if j >= len(lines) and not final:
            if is_compound:
                return None
            if not lines[j - 1 :]:
                return None
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
    candidates = [tail]
    closers = _bracket_closers(tail)
    if closers:
        candidates.append(tail + closers)
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
    # last resort: drop the final (partial) line and retry once
    lines = tail.split("\n")
    if len(lines) > 1:
        return repair_tail("\n".join(lines[:-1]))
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
    """
    repaired = repair_tail(tail)
    if repaired is None:
        return []
    try:
        tree = ast.parse(repaired)
    except SyntaxError:
        return []
    plans: list[Plan] = []
    assigned: set[str] = set()
    for stmt in tree.body:
        if isinstance(stmt, ast.For) and not isinstance(stmt.target, ast.Starred):
            plans.extend(_unroll_for(stmt, spec_names, ns, assigned, tail))
        elif isinstance(stmt, (ast.Expr, ast.Assign, ast.AugAssign, ast.AnnAssign)):
            for call in _hooked_calls(stmt, spec_names):
                plan = _resolve_call(call, ns, assigned, tail)
                if plan:
                    plans.append(plan)
        # anything else (If/While/Try/def/class): too uncertain — skip
        assigned |= _assigned_names(stmt)
    return plans


def _unroll_for(
    loop: ast.For,
    spec_names: set[str],
    ns: dict[str, Any],
    assigned: set[str],
    raw_tail: str,
) -> list[Plan]:
    """Pre-dispatch every iteration of `for X in ITER:` when ITER and the body
    call's args resolve — the flagship streaming win: the whole map fans out
    while the model is still writing the loop (or the code after it).

    The body SHAPE is analyzed once (not per item): statements up to the
    first nested control-flow construct are plannable for every iteration;
    if hooked-callable code could hide inside or after that construct, we
    don't unroll at all. Per-item analysis here caused spurious bet
    retraction (a re-plan after `if ...:` streamed in shrank the plan to
    item 0 and evicted valid bets).
    """
    try:
        items = list(safe_eval(loop.iter, ns))
    except (Unresolvable, Exception):
        return []
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
                return []  # calls hide in/after control flow: no bet
            break  # plan the straight-line prefix, all items
        plannable.append(stmt)
    plans: list[Plan] = []
    body_assigned = set(assigned)
    loop_vars = _target_names(loop.target)
    for item in items:
        item_env = dict(ns)
        try:
            _bind(item_env, loop.target, item)
        except Unresolvable:
            return []
        item_assigned = set(body_assigned)
        for stmt in plannable:
            for call in _hooked_calls(stmt, spec_names):
                plan = _resolve_call(
                    call, item_env, item_assigned, raw_tail, loop_var_ok=loop_vars
                )
                # unresolvable calls (e.g. stage-2 of a per-item chain whose
                # arg is stage-1's result) are left for the shadow — the
                # resolvable stage-1 calls still fan out here
                if plan is not None:
                    plans.append(plan)
            item_assigned |= _assigned_names(stmt)
    return plans


def _resolve_call(
    call: ast.Call,
    ns: dict[str, Any],
    assigned_in_tail: set[str],
    raw_tail: str,
    loop_var_ok: set[str] | frozenset[str] = frozenset(),
) -> Plan | None:
    if call.keywords:
        return None
    if not isinstance(call.func, ast.Name):
        return None
    # the call must be textually complete in the RAW tail: its closing paren
    # was streamed, so every argument token is final.
    if not _call_closed_in(raw_tail, call):
        return None
    # args must not read names whose assignment is still in the tail
    for n in ast.walk(ast.Tuple(elts=call.args, ctx=ast.Load())):
        if (
            isinstance(n, ast.Name)
            and n.id in assigned_in_tail
            and n.id not in loop_var_ok
        ):
            return None
    try:
        args = tuple(safe_eval(a, ns) for a in call.args)
    except (Unresolvable, Exception):
        return None
    key = canonical_hash(call.func.id, args, {})
    return Plan(tool=call.func.id, args=args, key=(call.func.id, key))


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
