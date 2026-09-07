"""Segmentation side: ``StreamSegmenter`` closes statements inside ```repl
fences of the streamed model output; ``repair_tail`` makes an incomplete tail
parseable for the peek engine.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

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
    # EXCEPT when the string contains an unescaped newline (e.g., "\n\n"),
    # which means the string literal spans multiple physical lines.
    if in_str is not None and in_str not in ('"""', "'''"):
        # Check if the string is still open (no closing quote found)
        # If so, keep tracking it across lines
        pass  # Don't reset - let the string span across lines
    return depth, in_str


# ==========================================================================
# tail peeking
# ==========================================================================
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
