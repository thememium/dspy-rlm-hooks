"""Tests for utility functions."""

from __future__ import annotations

import pytest

from dspy_rlm_hooks.core.utils import (
    _assemble_execution_code,
    _prepend_python_code,
    _strip_code_fences,
    _with_prompt_context,
)


class TestPrependPythonCode:
    def test_prepends_iteration_code(self):
        assert _prepend_python_code("generated()", "current = True") == (
            "current = True\ngenerated()"
        )

    def test_empty_iteration_code_leaves_generated_code_unchanged(self):
        assert _prepend_python_code("generated()", "") == "generated()"


class TestPromptContext:
    def test_appends_labeled_context_without_mutating_input(self):
        variables_info = ["Variable: question"]
        result = _with_prompt_context(variables_info, "Explore another branch.")

        assert variables_info == ["Variable: question"]
        assert result[0] == "Variable: question"
        assert "not a Python variable" in result[1]
        assert "Explore another branch." in result[1]

    def test_empty_context_reuses_existing_list(self):
        variables_info = ["Variable: question"]
        assert _with_prompt_context(variables_info, "") is variables_info


class TestAssembleExecutionCode:
    def test_returns_generated_code_without_persisted_globals(self):
        class Repl:
            repl_globals = ""

        assert _assemble_execution_code(Repl(), "print('hello')") == "print('hello')"

    def test_prepends_persisted_globals_exactly_as_the_interpreter_receives_them(self):
        class Repl:
            repl_globals = "seed = 7"

        assert _assemble_execution_code(Repl(), "print(seed)") == (
            "seed = 7\nprint(seed)"
        )

    def test_supports_interpreters_without_repl_globals(self):
        assert _assemble_execution_code(object(), "print('hello')") == "print('hello')"


class TestStripCodeFences:
    """Tests for the _strip_code_fences utility."""

    def test_plain_code_no_fences(self):
        """Test that code without fences is returned unchanged."""
        code = "print('hello')"
        result = _strip_code_fences(code)
        assert result == "print('hello')"

    def test_simple_fences(self):
        """Test stripping simple ``` fences."""
        code = "```\nprint('hello')\n```"
        result = _strip_code_fences(code)
        assert result == "print('hello')"

    def test_python_fences(self):
        """Test stripping ```python fences."""
        code = "```python\nprint('hello')\n```"
        result = _strip_code_fences(code)
        assert result == "print('hello')"

    def test_py_fences(self):
        """Test stripping ```py fences."""
        code = "```py\nprint('hello')\n```"
        result = _strip_code_fences(code)
        assert result == "print('hello')"

    def test_python3_fences(self):
        """Test stripping ```python3 fences."""
        code = "```python3\nprint('hello')\n```"
        result = _strip_code_fences(code)
        assert result == "print('hello')"

    def test_empty_fences(self):
        """Test stripping fences with empty code."""
        code = "```\n\n```"
        result = _strip_code_fences(code)
        assert result == ""

    def test_code_with_whitespace(self):
        """Test that surrounding whitespace is stripped."""
        code = "  ```python\nprint('hello')\n```  "
        result = _strip_code_fences(code)
        assert result == "print('hello')"

    def test_nested_fences_raises(self):
        """Test that non-Python language fences raise SyntaxError."""
        code = '```json\n{"key": "value"}\n```'
        with pytest.raises(SyntaxError, match="Expected Python code"):
            _strip_code_fences(code)

    def test_javascript_fences_raises(self):
        """Test that JavaScript fences raise SyntaxError."""
        code = "```javascript\nconsole.log('hello')\n```"
        with pytest.raises(SyntaxError, match="Expected Python code"):
            _strip_code_fences(code)

    def test_bash_fences_raises(self):
        """Test that bash fences raise SyntaxError."""
        code = "```bash\necho hello\n```"
        with pytest.raises(SyntaxError, match="Expected Python code"):
            _strip_code_fences(code)

    def test_inline_backticks_not_confused(self):
        """Test that inline backticks are not confused with fences."""
        code = "x = `not a fence`"
        result = _strip_code_fences(code)
        assert result == "x = `not a fence`"

    def test_multiple_fence_pairs(self):
        """Multiple fence pairs are not supported; function extracts last block."""
        code = "```\nprint('a')\n```\n```\nprint('b')\n```"
        result = _strip_code_fences(code)
        # The function is designed for a single fenced block.
        # With multiple pairs it falls back to last-block extraction.
        assert "print('b')" in result

    def test_fences_with_language_and_params(self):
        """Test fences with language and additional params."""
        code = "```python line_numbers\nprint('hello')\n```"
        result = _strip_code_fences(code)
        assert result == "print('hello')"

    def test_unclosed_fences(self):
        """Test unclosed fences return best-effort content."""
        code = "```python\nprint('hello')"
        result = _strip_code_fences(code)
        assert result == "print('hello')"

    def test_fence_without_newline_after_language(self):
        """Test fence marker without newline after language tag."""
        # This should hit line 41 - early return when no separator
        code = "```python"
        result = _strip_code_fences(code)
        # Should return the code as-is since there's no content after the fence
        assert "```python" in result
