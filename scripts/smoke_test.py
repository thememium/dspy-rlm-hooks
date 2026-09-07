"""Smoke test for built distributions.

Runs the package installed from a wheel or source distribution in an
isolated environment and verifies the crucial files made it into the build:

    uv run --isolated --no-project --python 3.12 --with dist/*.whl scripts/smoke_test.py
    uv run --isolated --no-project --python 3.12 --with dist/*.tar.gz scripts/smoke_test.py

Exits non-zero on failure. No pytest dependency — safe to run standalone.
"""

from __future__ import annotations

import importlib.metadata
import pathlib
import tomllib

import dspy_rlm_hooks

PYPROJECT = pathlib.Path(__file__).parent.parent / "pyproject.toml"
PYPROJECT_VERSION = tomllib.loads(PYPROJECT.read_text())["project"]["version"]


def main() -> None:
    version = dspy_rlm_hooks.__version__
    if version != PYPROJECT_VERSION:
        raise AssertionError(
            f"Installed version ({version!r}) does not match "
            f"pyproject.toml ({PYPROJECT_VERSION!r})"
        )
    importlib.metadata.version("dspy-rlm-hooks")

    missing = [
        name for name in dspy_rlm_hooks.__all__ if not hasattr(dspy_rlm_hooks, name)
    ]
    if missing:
        raise AssertionError(f"Missing public API: {', '.join(missing)}")

    print(
        f"Smoke test OK: dspy-rlm-hooks {version} "
        f"({len(dspy_rlm_hooks.__all__)} exports)"
    )


if __name__ == "__main__":
    main()
