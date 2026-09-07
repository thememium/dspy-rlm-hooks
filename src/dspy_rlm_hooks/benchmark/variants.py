"""Named execution variants plus variant apply/teardown plumbing."""

from __future__ import annotations

from typing import Any, Callable


def _variant_default(rlm: Any, tools: dict[str, Callable]) -> None:
    """Plain RLM — no hooks at all."""


def _variant_spec(rlm: Any, tools: dict[str, Callable]) -> None:
    """Speculative programmatic tool calling (the engine under test)."""
    from dspy_rlm_hooks import enable_rlm_speculation

    enable_rlm_speculation(
        rlm,
        tools={name: (fn, {"deterministic": True}) for name, fn in tools.items()},
        speculate_user_tools=True,
        streaming=True,
        timeout_s=5.0,
    )


def _variant_hooks(rlm: Any, tools: dict[str, Callable]) -> None:
    """Lifecycle hooks enabled with a no-op hook (for timing hook overhead)."""
    from dspy_rlm_hooks import enable_rlm_hooks

    enable_rlm_hooks(rlm)


NAMED_VARIANTS: dict[str, Callable[[Any, dict[str, Callable]], None]] = {
    "default": _variant_default,
    "spec": _variant_spec,
    "hooks": _variant_hooks,
}


def _apply_variant(program: Any, variant: Any, tool_fns: dict[str, Callable]) -> str:
    if isinstance(variant, str):
        fn = NAMED_VARIANTS.get(variant)
        if fn is None:
            raise ValueError(
                f"unknown variant {variant!r}; known: {sorted(NAMED_VARIANTS)}"
            )
        name = variant
    elif callable(variant):
        fn, name = variant, getattr(variant, "__name__", "custom")
    else:
        raise TypeError("variant must be a name or a callable(rlm, tools)")
    fn(program, tool_fns)
    return name


def _teardown(program: Any, variant_name: str) -> None:
    from dspy_rlm_hooks import disable_rlm_speculation

    try:
        disable_rlm_speculation(program)
    except Exception:
        pass
    try:
        from dspy_rlm_hooks import disable_rlm_hooks

        disable_rlm_hooks(program)
    except Exception:
        pass
