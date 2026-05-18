"""Tests for utility functions."""

from __future__ import annotations

import pytest

from dspy_rlm_hooks.utils import _strip_code_fences


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
        code = "```json\n{\"key\": \"value\"}\n```"
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
        """Test that multiple fence pairs return the first block only."""
        code = "```\nprint('a')\n```\n```\nprint('b')\n```"
        result = _strip_code_fences(code)
        # The function extracts the first Python block it finds
        assert "print('a')" in result or result == "print('a')"
        assert "print('b')" not in result

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
