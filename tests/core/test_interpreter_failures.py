"""A dead sandbox must not consume every remaining generation iteration."""

from types import SimpleNamespace
from typing import Any, cast

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
            if (
                code.lstrip().startswith("def __dspy_bootstrap")
                or "Sandbox side of the dspy facade" in code
            ):
                # DSPy installs its sandbox facade before generation
                # (>=3.5 bootstrap, 3.4 dspy facade source).
                return None
            self.executions += 1
            raise code_interpreter.CodeInterpreterError("Deno exited during execution")

    interpreter = DeadInterpreter()
    # kwargs keep this version-gated test importable/type-checkable against
    # the older DSPy supported by this package as well.
    options: dict[str, Any] = {
        "max_iters": 20,
        "interpreter_factory": lambda: interpreter,
    }
    rlm = dspy.RLM("task -> answer", **options)
    # DSPy >=3.5 serializes the reasoning output under its fallback name
    # (``plan``); older releases use the field name itself.
    action_signature = rlm.generate_action.signature
    assert action_signature is not None
    reasoning_field = action_signature.output_fields["reasoning"]
    schema_extra = reasoning_field.json_schema_extra
    extra = cast(dict[str, Any], schema_extra) if isinstance(schema_extra, dict) else {}
    reasoning_key = extra.get("__dspy_reasoning_fallback") or "reasoning"
    lm = DummyLM([{reasoning_key: "run one block", "code": "print(1)"}])
    enable_rlm_hooks(rlm)
    with dspy.context(lm=lm):
        with pytest.raises(code_interpreter.CodeInterpreterError, match="Deno exited"):
            rlm(task="test recovery")
    assert interpreter.executions == 1
    assert len(lm.history) == 1
