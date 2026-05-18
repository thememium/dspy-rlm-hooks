"""Shared test fixtures for the dspy-rlm-hooks test suite."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from dotenv import load_dotenv

# Load environment variables from .env files if present
for _dotenv in [".env", ".env.local"]:
    _path = Path(__file__).resolve().parents[1] / _dotenv
    if _path.exists():
        load_dotenv(_path, override=True)


def pytest_addoption(parser):
    """Add custom pytest options."""
    parser.addoption(
        "--run-e2e",
        action="store_true",
        default=False,
        help="Run end-to-end tests that call a real LLM",
    )


@pytest.fixture
def mock_rlm():
    """Return a fully-patched mock RLM instance with all required attributes."""
    rlm = MagicMock()
    rlm._execute_iteration = MagicMock(return_value=MagicMock())
    rlm._aexecute_iteration = MagicMock(return_value=MagicMock())
    rlm._process_execution_result = MagicMock(return_value=MagicMock())
    rlm.generate_action = MagicMock()
    rlm.generate_action.acall = AsyncMock(return_value=MagicMock())
    rlm.max_iterations = 5
    rlm.verbose = False
    rlm._execute_code = MagicMock(return_value="mock_result")
    return rlm


@pytest.fixture
def mock_repl():
    """Return a mock REPL instance."""
    repl = MagicMock()
    repl.repl_globals = ""
    repl.execute = MagicMock(return_value="mock_result")
    return repl


@pytest.fixture
def mock_history():
    """Return a mock REPLHistory."""
    from dspy.primitives.repl_types import REPLHistory

    history = MagicMock(spec=REPLHistory)
    history.entries = []
    return history


@pytest.fixture
def mock_variables():
    """Return a list of mock REPLVariable objects."""
    var = MagicMock()
    var.format = MagicMock(return_value="mock_var_info")
    return [var]


@pytest.fixture
def mock_action():
    """Return a mock action with code and reasoning."""
    action = MagicMock()
    action.code = "print('hello')"
    action.reasoning = "test"
    return action


@pytest.fixture(scope="session")
def dspy_lm():
    """Configure DSPy with a real LLM for end-to-end tests."""
    import dspy

    application_name = "dspy-rlm-hooks-test"
    lm = dspy.LM(
        "openrouter/openai/gpt-oss-120b",
        cache=False,
        extra_body={"provider": {"order": ["groq"], "allow_fallbacks": False}},
        extra_headers={
            "HTTP-Referer": f"http://{application_name}.local",
            "X-Title": application_name,
        },
    )
    dspy.configure(lm=lm)
    return lm


@pytest.fixture
def pre_iteration_hook():
    """Return a simple pre-iteration hook that injects a variable."""
    from dspy_rlm_hooks import PreIterationOutput

    def hook(iteration, variables, history, input_args):
        return PreIterationOutput(extra_vars={"injected": True})

    return hook


@pytest.fixture
def pre_execution_hook():
    """Return a pre-execution hook that rewrites code."""
    from dspy_rlm_hooks import PreExecutionOutput

    def hook(iteration, code, variables, history, input_args):
        return PreExecutionOutput(code=f"# modified\n{code}")

    return hook


@pytest.fixture
def post_execution_hook():
    """Return a post-execution hook that transforms results."""
    from dspy_rlm_hooks import PostExecutionOutput

    def hook(iteration, code, result, variables, history, input_args):
        return PostExecutionOutput(result=f"transformed: {result}")

    return hook


@pytest.fixture
def post_iteration_hook():
    """Return a post-iteration hook that returns history unchanged."""

    from dspy_rlm_hooks import PostIterationOutput

    def hook(iteration, pred, code, result, history):
        return PostIterationOutput(history=history)

    return hook
