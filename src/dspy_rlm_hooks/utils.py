"""Small internal utilities for the RLM hook system."""

from __future__ import annotations


def _strip_code_fences(code: str) -> str:
    """Remove markdown code fences from LLM-generated code.

    Handles both plain `` ``` `` fences and language-tagged ones
    (`` ```python ``).  Raises :class:`SyntaxError` when a non-Python
    language fence is detected.

    Args:
        code: Raw code string, potentially wrapped in markdown fences.

    Returns:
        Clean code with fences stripped.

    Raises:
        SyntaxError: If the fence specifies a language other than Python.
    """
    code = code.strip()
    if "```" not in code:
        return code

    lines = code.splitlines()
    while len(lines) >= 2 and lines[0].strip() == "```" and lines[-1].strip() == "```":
        lines.pop(0)
        lines.pop()
    code = "\n".join(lines).strip()

    if "```" not in code:
        return code

    # If there are still fences after stripping plain pairs,
    # it may be a language-tagged block or multiple blocks.
    # Strip one more language-tagged fence if present.
    fence_start = code.find("```")
    lang_line, separator, remainder = code[fence_start + 3 :].partition("\n")
    if not separator:
        return code

    lang = (lang_line.strip().split(maxsplit=1)[0] if lang_line.strip() else "").lower()
    if lang not in {"python", "py", "python3", "py3", ""}:
        raise SyntaxError(
            f"Expected Python code but got ```{lang} fence. "
            f"Write Python code, not {lang}."
        )

    block_end = remainder.find("```")
    if block_end == -1:
        return remainder.strip()
    if block_end == 0:
        # Opening and closing fences are adjacent — return everything after
        return remainder[3:].strip()
    return remainder[:block_end].strip()
