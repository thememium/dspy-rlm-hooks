"""A dead sandbox must not consume every remaining generation iteration."""

from types import SimpleNamespace
from typing import Any

import pytest
from dspy.primitives import code_interpreter

from dspy_rlm_hooks.core.patcher import _execute_code


def test_terminal_interpreter_error_propagates(monkeypatch):
    class Recoverable(code_interpreter.CodeInterpreterError):
        pass

    monkeypatch.setattr(
        code_interpreter, "CodeExecutionError", Recoverable, raising=False
    )
    failure = code_interpreter.CodeInterpreterError(
        "PythonInterpreter session has ended"
    )

    def execute(code, variables):
        raise failure

    with pytest.raises(code_interpreter.CodeInterpreterError) as caught:
        _execute_code(None, SimpleNamespace(execute=execute), "print(1)", {})
    assert caught.value is failure


def test_recoverable_execution_error_still_returns_model_feedback(monkeypatch):
    class Recoverable(code_interpreter.CodeInterpreterError):
        pass

    monkeypatch.setattr(
        code_interpreter, "CodeExecutionError", Recoverable, raising=False
    )

    def execute(code, variables):
        raise Recoverable("NameError: unknown variable")

    result = _execute_code(None, SimpleNamespace(execute=execute), "missing", {})
    assert result == "[Error] NameError: unknown variable"


@pytest.mark.skipif(
    not hasattr(code_interpreter, "CodeExecutionError"),
    reason="requires DSPy >=3.4 error categories",
)
def test_real_rlm_stops_after_first_terminal_error():
    import dspy
    from dspy.utils import DummyLM

    from dspy_rlm_hooks import enable_rlm_hooks

    class DeadInterpreter:
        def __init__(self):
            self.tools = {}
            self.executions = 0

        def start(self):
            pass

        def shutdown(self):
            pass

        def execute(self, code, variables=None):
            self.executions += 1
            raise code_interpreter.CodeInterpreterError("Deno exited during execution")

    interpreter = DeadInterpreter()
    lm = DummyLM([{"reasoning": "run one block", "code": "print(1)"}])
    # kwargs keep this version-gated test importable/type-checkable against
    # the older DSPy supported by this package as well.
    options: dict[str, Any] = {
        "max_iters": 20,
        "interpreter_factory": lambda: interpreter,
    }
    rlm = dspy.RLM("task -> answer", **options)
    enable_rlm_hooks(rlm)
    with dspy.context(lm=lm):
        with pytest.raises(code_interpreter.CodeInterpreterError, match="Deno exited"):
            rlm(task="test recovery")
    assert interpreter.executions == 1
    assert len(lm.history) == 1
