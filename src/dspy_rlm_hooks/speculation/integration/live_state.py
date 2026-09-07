"""Cross-iteration live-state snapshot: merge live REPL values into the seed.

The persistent shadow resets to its ORIGINAL seed each turn, so reads of
variables created by EARLIER iterations would miss speculation. A probe
executed in the LIVE sandbox serializes safe top-level variables; literal-repr
values are re-hydrated into the shadow seed.
"""

from __future__ import annotations

import ast
import builtins
from typing import Any

from dspy_rlm_hooks.speculation.speculator import Speculator
from dspy_rlm_hooks.speculation.streaming import _free_names

# -- cross-iteration state sync -----------------------------------------------

# The persistent shadow resets to its ORIGINAL seed (input_args) each turn, so
# calls reading variables created by EARLIER iterations would miss speculation.
# A probe executed in the LIVE sandbox serializes safe top-level variables;
# literal-repr values are re-hydrated into the shadow seed. Values that do not
# round-trip are left out — the shadow treats the name as unknown and the real
# path handles the call (no claim is ever corrupted by a stale value).

_SNAPSHOT_MAX_VALUE_CHARS = 100_000

_SNAPSHOT_PROBE = (
    "_spec_out = {}\n"
    "for _spec_k in _spec_requested:\n"
    "    try:\n"
    "        _spec_v = globals().get(_spec_k)\n"
    "        if _spec_v is not None and not callable(_spec_v) and not isinstance(_spec_v, type):\n"
    "            _spec_r = repr(_spec_v)\n"
    "            if len(_spec_r) <= 100000:\n"
    "                _spec_out[_spec_k] = _spec_r\n"
    "    except Exception:\n"
    "        pass\n"
    "print(repr(_spec_out))\n"
)


def _snapshot_reads(tree: ast.Module) -> set[str]:
    required: set[str] = set()
    bound: set[str] = set()
    imports: set[str] = set()  # survive bound.clear(); always available at module level
    for statement in tree.body:
        reads = _free_names(statement)
        for node in ast.walk(statement):
            if isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
                reads.add(node.target.id)
        required.update(reads - bound - imports)
        if isinstance(statement, ast.Assign):
            bound.update(
                target.id
                for target in statement.targets
                if isinstance(target, ast.Name)
            )
        elif isinstance(statement, (ast.Import, ast.ImportFrom)):
            # import X / from X import Y bind names just like assignments.
            # Without this, `import time` followed by `time.perf_counter`
            # would treat `time` as a free name needing snapshot, triggering
            # an expensive REPL probe that the snapshot filter would discard
            # anyway (modules are filtered by isinstance check in the probe).
            for alias in statement.names:
                imports.add(alias.asname or alias.name)
        else:
            bound.clear()
    return required


def _pure_assigned_names(tree: ast.Module) -> set[str]:
    """Names that are assigned WITHOUT being read in the same assignment's value.

    For ``now = time.perf_counter``, the target ``now`` does not appear in the
    value — it's a pure overwrite and the REPL snapshot is useless (the code's
    own assignment will replace whatever the snapshot provides).

    For ``value = value + 'new'``, the target ``value`` IS read in the value —
    the snapshot is needed because the code reads the old value.

    Only top-level (non-nested) assignments are considered, matching the
    scoping rules of ``_snapshot_reads``.
    """
    out: set[str] = set()
    for stmt in tree.body:
        if isinstance(stmt, ast.Assign):
            value_reads = {
                n.id for n in ast.walk(stmt.value) if isinstance(n, ast.Name)
            }
            for target in stmt.targets:
                if isinstance(target, ast.Name) and target.id not in value_reads:
                    out.add(target.id)
        elif isinstance(stmt, (ast.Import, ast.ImportFrom)):
            for alias in stmt.names:
                out.add(alias.asname or alias.name)
    return out


def _live_state_seed(
    repl: Any, code: str, input_args: dict[str, Any], spec: Speculator
) -> dict[str, Any]:
    """Merge a live REPL snapshot into the shadow seed (input_args win).

    Gated: the sandbox round trip runs only when the block reads names the seed
    cannot provide (free names beyond ``input_args``, tool names, and builtins).
    Non-literal or oversized values are skipped — the shadow treats the name as
    unknown and the real path handles the call.
    """
    seed = dict(input_args)
    try:
        tree = ast.parse(code)
        free = _snapshot_reads(tree)
    except (SyntaxError, ValueError):
        return seed
    tool_names = set(spec.registry.names()) if spec.registry else set()
    requested = free - seed.keys() - tool_names - set(dir(builtins))
    if not requested:
        return seed
    # Single-pass AST analysis: collect pure assignments, loop targets,
    # all stored names, and all read names in one walk.
    pure_assigned = _pure_assigned_names(tree)
    requested -= pure_assigned
    if not requested:
        return seed
    loop_targets: set[str] = set()
    all_stored: set[str] = set()
    all_reads: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.For):
            iter_node = node.iter
            non_empty = (
                isinstance(iter_node, (ast.List, ast.Tuple)) and len(iter_node.elts) > 0
            ) or (
                isinstance(iter_node, ast.Constant)
                and isinstance(iter_node.value, (str, bytes, list, tuple))
                and len(iter_node.value) > 0
            )
            if non_empty:
                for n in ast.walk(node.target):
                    if isinstance(n, ast.Name):
                        loop_targets.add(n.id)
        if isinstance(node, ast.Name):
            if isinstance(node.ctx, (ast.Store, ast.Del)):
                all_stored.add(node.id)
            elif isinstance(node.ctx, ast.Load):
                all_reads.add(node.id)
        elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
            all_reads.add(node.target.id)
    stored_not_read = (all_stored - all_reads) | loop_targets
    remaining = requested - stored_not_read
    if not remaining:
        return seed
    requested = remaining
    try:
        out = repl.execute(
            f"_spec_requested = {sorted(requested)!r}\n" + _SNAPSHOT_PROBE
        )
        line = out.strip().splitlines()[-1] if out and out.strip() else ""
        snap = ast.literal_eval(line)
    except Exception:
        return seed
    if not isinstance(snap, dict):
        return seed
    for k, r in snap.items():
        if k not in requested:
            continue
        if not isinstance(r, str) or len(r) > _SNAPSHOT_MAX_VALUE_CHARS:
            continue
        try:
            seed[k] = ast.literal_eval(r)
        except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
            continue
    return seed
