<a name="readme-top"></a>

<div align="center">
  <h3 align="center">DSPy RLM Hooks</h3>

  <p align="center">
    Lifecycle instrumentation for DSPy's RLM (Recursive Language Model).
    <br />
    <a href="#table-of-contents"><strong>Explore the Documentation »</strong></a>
    <br />
    <a href="https://github.com/thememium/dspy-rlm-hooks/issues">Report Bug</a>
    ·
    <a href="https://github.com/thememium/dspy-rlm-hooks/issues">Request Feature</a>
  </p>
</div>

<!-- TABLE OF CONTENTS -->

<a name="table-of-contents"></a>

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about">About</a></li>
    <li><a href="#quick-start">Quick Start</a></li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#predictrlm-support">PredictRLM Support</a></li>
    <li><a href="#development">Development</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
  </ol>
</details>

<!-- ABOUT -->

## About

DSPy RLM Hooks injects **lifecycle hooks** into DSPy's internal `RLM` iteration loop, giving you full control over every stage of code generation, execution, and history tracking.

- **Code Rewriting** — Fix or augment LLM-generated code before it runs
- **Variable Injection** — Seed the interpreter with persistent variables and imports
- **Result Auditing** — Transform, validate, or retry on errors
- **History Management** — Inspect and modify the REPL history between iterations
- **Sync & Async** — Hooks work in either mode; coroutines are auto-detected
- **PredictRLM Support** — Same hook API works on [PredictRLM](https://github.com/Trampoline-AI/predict-rlm) instances

Requires **DSPy 3.1+** and **Pydantic 2+**.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ARCHITECTURE -->

## Architecture

### RLM Hook Lifecycle

``` 
┌──────────────────────────┐
│    pre_iteration_hook    │
└────────────┬─────────────┘
             │
             │ inject vars,
             │ prepend code
             ▼
┌──────────────────────────┐
│      Generate Code       │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│    pre_execution_hook    │
└────────────┬─────────────┘
             │
             │ rewrite code
             ▼
┌──────────────────────────┐
│       Execute Code       │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│   post_execution_hook    │
└────────────┬─────────────┘
             │
             │ transform result
             ▼
┌──────────────────────────┐
│   post_iteration_hook    │
└──────────────────────────┘
```

Hooks fire at each stage of an RLM iteration, allowing inspection and modification of behaviour.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- QUICK START -->

## Quick Start

### Install

Install dspy-rlm-hooks with uv (recommended):

```bash
uv add dspy-rlm-hooks
```

Or with pip:

```bash
pip install dspy-rlm-hooks
```

### Basic Usage

```python
import dspy
from dspy_rlm_hooks import enable_rlm_hooks, PreIterationOutput

rlm = dspy.RLM(...)

def inject_math(iteration, variables, history, input_args):
    return PreIterationOutput(
        extra_vars={"tool": "calculator"},
        python_code="import math",
    )

enable_rlm_hooks(rlm, pre_iteration_hook=inject_math)

result = rlm(question="What is the square root of 1764?")
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE -->

## Usage

### All Four Hooks

A realistic example showing how each hook can be used to build a **safe, instrumented agent**:

```python
from dspy_rlm_hooks import (
    enable_rlm_hooks,
    PreIterationOutput,
    PreExecutionOutput,
    PostExecutionOutput,
    PostIterationOutput,
)
from dspy.primitives.repl_types import REPLHistory
import re

# ── Pre-iteration: seed interpreter with a regex toolkit ──

def pre_iteration(iteration, variables, history, input_args):
    """Inject a regex helper and seed variables before every iteration."""
    return PreIterationOutput(
        extra_vars={"search_pattern": r"TODO|FIXME|HACK"},
        python_code="""
import re

def grep(pattern, text):
    return re.findall(pattern, text)
""",
    )

# ── Pre-execution: block dangerous code ──

FORBIDDEN = re.compile(r"\b(eval|exec|compile|__import__)\b")

def pre_execution(iteration, code, variables, history, input_args):
    """Sanitise generated code before it reaches the interpreter."""
    if FORBIDDEN.search(code):
        safe_code = FORBIDDEN.sub("# BLOCKED", code)
        return PreExecutionOutput(code=safe_code)
    return PreExecutionOutput(code=code)

# ── Post-execution: retry on error ──

def post_execution(iteration, code, result, variables, history, input_args):
    """If execution raised an error, wrap a hint so the LLM retries next round."""
    if isinstance(result, str) and result.startswith("[Error]"):
        return PostExecutionOutput(
            result=f"{result}\n# Hint: the variable 'search_pattern' is already in scope."
        )
    return PostExecutionOutput(result=result)

# ── Post-iteration: enforce a price budget ──

MAX_COST_USD = 0.50

def _estimate_cost(pred):
    # In production, derive this from response.usage or similar.
    return 0.015

def make_budget_hook(max_cost=MAX_COST_USD):
    """Return a post_iteration hook with isolated, per-request state.

    Create a new hook for every RLM session so budgets don't leak
    across concurrent requests on a multi-threaded or async server.
    """
    accumulated_cost = 0.0

    def post_iteration(iteration, pred, code, result, history: REPLHistory):
        nonlocal accumulated_cost
        accumulated_cost += _estimate_cost(pred)
        if accumulated_cost >= max_cost:
            return PostIterationOutput(history=history, stop=True)
        return PostIterationOutput(history=history)

    return post_iteration

# ── Wire everything up ──

enable_rlm_hooks(
    rlm,
    pre_iteration_hook=pre_iteration,
    pre_execution_hook=pre_execution,
    post_execution_hook=post_execution,
    post_iteration_hook=make_budget_hook(max_cost=0.50),
)

result = rlm(question="Find all TODO comments in the codebase")
```

### Async Hooks

Return a coroutine and the system handles it automatically:

```python
async def fetch_context(iteration, variables, history, input_args):
    context = await remote_cache.get(input_args["question"])
    return PreIterationOutput(extra_vars={"cached_context": context})

enable_rlm_hooks(rlm, pre_iteration_hook=fetch_context)
```

### Disabling Hooks

```python
from dspy_rlm_hooks import disable_rlm_hooks

disable_rlm_hooks(rlm)
```

Removes all monkey-patched overrides and reverts to original behaviour.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- PREDICTRLM SUPPORT -->

## PredictRLM Support

`enable_rlm_hooks` works on both `dspy.RLM` and
[PredictRLM](https://github.com/Trampoline-AI/predict-rlm) with the same API.
The function auto-detects the RLM type and uses the appropriate mechanism.

Install with the `predict-rlm` extra:

```bash
uv add "dspy-rlm-hooks[predict-rlm]"
```

### Quick Example

```python
from predict_rlm import PredictRLM
from dspy_rlm_hooks import enable_rlm_hooks, PreExecutionOutput

rlm = PredictRLM("query -> answer")

def sanitize_code(iteration, code, variables, history, input_args):
    """Block dangerous code patterns."""
    if "os.system" in code:
        code = code.replace("os.system", "# BLOCKED")
    return PreExecutionOutput(code=code)

enable_rlm_hooks(rlm, pre_execution_hook=sanitize_code)
result = rlm(query="...")
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Hook Reference

| Hook | When it fires | What it can do |
| --- | --- | --- |
| **PreIteration** | Before action generation | Inject variables (`extra_vars`) and persistent code (`python_code`) |
| **PreExecution** | After code generation, before running | Rewrite or sanitise the generated `code` string |
| **PostExecution** | After code runs, before history processing | Transform, audit, or replace the raw `result` |
| **PostIteration** | After result is folded into history | Save learnings, trigger side effects, modify `history`, or set `stop=True` to force final extraction |

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- DEVELOPMENT -->

## Development

### Code Quality

This project uses several tools to maintain code quality:

- **Ruff:** Linting and formatting
- **isort:** Import sorting
- **pytest:** Testing framework
- **ty:** Type checking
- **deptry:** Dependency analysis

**Available commands:**

```sh
# Run all quality checks
uv run poe clean-full

# Individual checks
uv run poe lint          # Ruff linting
uv run poe format        # Ruff formatting
uv run poe sort          # Import sorting
uv run poe typecheck     # Type checking
uv run poe deptry        # Dependency analysis
```

### Testing

Run tests using pytest:

```sh
# Run all tests
uv run pytest

# Run specific test
uv run pytest path/to/test.py::test_name
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTRIBUTING -->

## Contributing

Quick workflow:

1. Fork and branch: `git checkout -b feature/name`
2. Make changes
3. Run checks: `uv run poe clean-full`
4. Commit and push
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- LICENSE -->

## License

MIT (as declared in `pyproject.toml`).

---

<div align="center">
  <p>
    <sub>Built by <a href="https://github.com/thememium">thememium</a></sub>
  </p>
</div>
