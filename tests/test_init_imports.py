"""Tests to cover __init__.py import exception paths.

These tests use importlib.reload() to reimport the module with mocked
dependencies to trigger the exception handlers.
"""

from __future__ import annotations

import importlib
import sys
from unittest.mock import patch

import pytest


class TestVersionFallback:
    """Tests for version detection fallback path."""

    def test_version_fallback_when_metadata_fails(self):
        """Test that __version__ is 'unknown' when importlib.metadata.version fails."""
        # Save the original module
        original_module = sys.modules.get("dspy_rlm_hooks")

        try:
            # Remove the module from cache to force reimport
            if "dspy_rlm_hooks" in sys.modules:
                del sys.modules["dspy_rlm_hooks"]

            # Mock importlib.metadata.version to raise ImportError
            with patch("importlib.metadata.version", side_effect=ImportError("mock")):
                # Reimport the module
                import dspy_rlm_hooks

                # The version should fall back to "unknown"
                assert dspy_rlm_hooks.__version__ == "unknown"
        finally:
            # Restore the original module
            if original_module is not None:
                sys.modules["dspy_rlm_hooks"] = original_module
            else:
                # Remove if it wasn't there before
                sys.modules.pop("dspy_rlm_hooks", None)


class TestPredictRLMImportFallback:
    """Tests for predict_rlm compatibility import fallback path."""

    def test_predict_rlm_import_failure(self):
        """Test that _is_predict_rlm is not exported when predict_rlm_compat import fails."""
        # Save the original module
        original_module = sys.modules.get("dspy_rlm_hooks")

        try:
            # Remove the module from cache to force reimport
            if "dspy_rlm_hooks" in sys.modules:
                del sys.modules["dspy_rlm_hooks"]

            # Mock the predict_rlm_compat import to fail
            with patch.dict(sys.modules, {"dspy_rlm_hooks.predict_rlm_compat": None}):
                # Reimport the module
                import dspy_rlm_hooks

                # _is_predict_rlm should not be in __all__ when import fails
                assert "_is_predict_rlm" not in dspy_rlm_hooks.__all__
        finally:
            # Restore the original module
            if original_module is not None:
                sys.modules["dspy_rlm_hooks"] = original_module
            else:
                sys.modules.pop("dspy_rlm_hooks", None)
