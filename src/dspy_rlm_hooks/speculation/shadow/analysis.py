"""AST name analysis for shadow statements, plus the per-statement wall-clock
budget constant."""

from __future__ import annotations

import ast

STMT_WALL_BUDGET_S = 2.0  # runaway guard: max wall time per shadow statement


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
