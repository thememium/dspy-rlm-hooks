# Autoresearch: 100% Test Coverage

## Objective
Achieve 100% test coverage for the `dspy-rlm-hooks` package using `uv run poe test-cov`. The source files are:
- `src/dspy_rlm_hooks/__init__.py`
- `src/dspy_rlm_hooks/types.py`
- `src/dspy_rlm_hooks/utils.py`
- `src/dspy_rlm_hooks/patcher.py`
- `src/dspy_rlm_hooks/predict_rlm_compat.py`

## Metrics
- **Primary**: coverage_pct (%, higher is better) — overall test coverage percentage
- **Secondary**: total_lines, missed_lines

## How to Run
`./.auto/measure.sh` — outputs `METRIC coverage_pct=XX.X` and secondary metrics.

## Files in Scope
- `src/dspy_rlm_hooks/__init__.py` — Package init, exports, version detection
- `src/dspy_rlm_hooks/types.py` — Pydantic models and Protocol types for hooks
- `src/dspy_rlm_hooks/utils.py` — `_strip_code_fences()` helper
- `src/dspy_rlm_hooks/patcher.py` — Core monkey-patching logic for RLM hooks (sync + async)
- `src/dspy_rlm_hooks/predict_rlm_compat.py` — PredictRLM compatibility layer

## Off Limits
- Do NOT modify source files to make them easier to test
- Do NOT add `# pragma: no cover` or `noqa` comments to suppress coverage
- Do NOT delete or weaken existing tests
- Do NOT change the package's public API

## Constraints
- All existing tests must continue to pass
- No new dependencies
- Tests must be real behavioral tests, not just importing modules
- Do NOT cheat by using `# pragma: no cover` or similar directives

## Coverage Gaps (as of baseline)
1. **predict_rlm_compat.py (14%)** — Almost entirely untested. Key gaps:
   - `_run_async()` else branch (running event loop → new loop)
   - `_is_predict_rlm()` returning True
   - `enable_predict_rlm_hooks()` full function
   - `disable_predict_rlm_hooks()` full function
   - `_StopIteration` exception handling
   - All wrapped iteration logic (sync + async)

2. **patcher.py (87%)** — Missing lines:
   - `_run_async()` else branch (lines 46-50)
   - `self.verbose` logging paths (lines 124, 210-211)
   - `_aexecute_iteration` pre_execution hook (lines 233-236)
   - `enable_rlm_hooks` PredictRLM branch (lines 335-344)
   - `disable_rlm_hooks` PredictRLM branch (lines 376-379)

3. **__init__.py (71%)** — Missing:
   - `__version__ = version("dspy-rlm-hooks")` (lines 62-63)
   - PredictRLM compatibility import (lines 84-85)

4. **utils.py (96%)** — Missing:
   - Non-Python language fence detection (line 41)

## What's Been Tried
- (baseline) Existing tests cover types.py 100%, utils.py 96%, patcher.py 87%
- predict_rlm_compat.py tests exist but are skipped when predict-rlm not installed
