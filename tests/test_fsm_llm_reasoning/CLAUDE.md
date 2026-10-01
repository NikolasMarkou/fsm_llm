# test_fsm_llm_reasoning

Path: `tests/test_fsm_llm_reasoning`
Purpose: Offline unit tests (126) for the reasoning engine subpackage `fsm_llm.reasoning` (source in `src/fsm_llm/reasoning/`).

## Scope

Covers `constants.py`, `definitions.py`, `exceptions.py`, `handlers.py`, `utilities.map_reasoning_type`, `reasoning_modes.py` (absence of removed functions), selected `engine.py` internals, and `__main__.py` (JSON output helpers and `--verbose`). No test calls a real LLM or needs network or env keys. `test_engine_scripted.py` runs whole `ReasoningEngine.solve_problem` solves through core with a scripted `LLMInterface` (`_ScriptedLLM`, injected as `llm_interface=`) and asserts every call: no user exchange and no "Continue reasoning" in history, push and pop of the strategy FSM, the retry loop reaching `max_retries_reached`, the hybrid back edge bounded, handler-only verdict keys, the budget error with `partial_context`, critical handlers, classifier value normalisation.

## Architecture

Tests call static methods and model constructors directly with small context dicts keyed by `ContextKeys` constants, or drive whole solves on a scripted interface (`test_engine_scripted.py`). Three further techniques recur:

- Bare engine: `object.__new__(ReasoningEngine)` plus a hand-set `engine.reasoning_fsms` dict, then call `engine._prepare_reasoning_execution(context)` (`test_engine.py::TestReasoningTypeFallback`).
- Source assertions: `inspect.getsource(...)` on `ReasoningEngine`, `ReasoningEngine._classify_problem`, `ReasoningEngine._solve_problem_locked`, the `engine` module, and the `handlers` module, then string checks (`test_audit_fixes.py`).
- Subprocess probe: `test_cli_logging.py` runs `sys.executable -c _PROBE`. The probe imports `fsm_llm.reasoning.__main__`, replaces `solve_problem_with_engine` with a stub that logs `I-PROBE` (info) and `D-PROBE` (debug) from a namespace named `fsm_llm.reasoning.engine`, sets `sys.argv` to `['fsm-llm-reasoning', 'what is 2+2?', '-o', 'json', ...]`, and exits with `m.main()`.

## Key files

| File | Role | Notes |
| --- | --- | --- |
| `test_audit_fixes.py` | 28 regression tests for audit findings F-003, F-005, F-011 and senior-review fixes | Many assertions match literal source strings |
| `test_cli_logging.py` | 2 parametrized cases: `["--verbose"]` gives 1 `I-PROBE` line in stderr, `[]` gives 0; `D-PROBE` never appears | Subprocess, `timeout=60`, deletes `FSM_LLM_LOG_LEVEL` |
| `test_cli_output.py` | 3 tests: `_format_json_output` and `_save_as_json` fed a real `ReasoningTrace(...).model_dump()` write `reasoning_types_used` as a sorted list, never redacted | Uses `tmp_path` |
| `test_constants.py` | 19 tests on constant values | Pins literal strings and numbers |
| `test_definitions.py` | 28 tests on Pydantic models | Validators, computed properties, thresholds |
| `test_engine.py` | 10 tests: models, handlers, `map_reasoning_type`, ANALYTICAL-only fallback (D-009) | Uses loguru sink for the warning |
| `test_engine_scripted.py` | Whole solves through core on `_ScriptedLLM` (also imported by `tests/test_fsm_llm_regression/test_engine_integration.py`) | Asserts calls, history, stack, retries, budget and failure details |
| `test_exceptions.py` | 8 tests on the exception hierarchy | `details`, `reasoning_type` attributes |
| `test_handlers.py` | 28 tests on `ReasoningHandlers`, `ContextManager`, `OutputFormatter` | Validation, trace, pruning, merge, final solution |
| `__init__.py` | Empty package marker | |

## Public interface under test

- `ReasoningType` (str enum), exactly 9 values: `simple_calculator`, `analytical`, `deductive`, `inductive`, `abductive`, `analogical`, `creative`, `critical`, `hybrid`. Unknown value raises `ValueError`.
- `OrchestratorStates`: `problem_analysis`, `strategy_selection`, `execute_reasoning`, `synthesize_solution`, `validate_refine`, `final_answer`.
- `ClassifierStates`: `analyze_domain`, `analyze_structure`, `identify_reasoning_needs`, `recommend_strategy`.
- `HandlerNames`: `OrchestratorProblemClassifier`, `OrchestratorStrategyExecutor`, `OrchestratorSolutionValidator`, `ReasoningTracer`, `ContextPruner`, `RetryLimiter`, `RetryKeyClearer`, `HybridLoopCounter`.
- `ErrorMessages` placeholders: `INVALID_REASONING_TYPE` `{type}`, `FSM_NOT_FOUND` `{name}`; `MAX_RETRIES_EXCEEDED` non-empty. `CALCULATION_ERROR`, `VALIDATION_FAILED`, `CONTEXT_TOO_LARGE` and `Defaults.MAX_CONTEXT_SIZE` are pinned absent (removed in plan 07ad3f8c).
- `LogMessages` placeholders: `ENGINE_INITIALIZED` `{model}`, `CLASSIFICATION_STARTED` `{context}`, `CLASSIFICATION_COMPLETE` `{type}`, `FSM_PUSHED` `{name}`, `PROBLEM_SOLVED` `{steps}`.
- `ReasoningHandlers.validate_solution(context) -> dict`: sets `VALIDATION_RESULT`, `SOLUTION_CONFIDENCE`, `VALIDATION_CHECKS` (`has_solution`, `has_insights`, `sufficient_detail`, `addresses_problem`), `RETRY_COUNT`, `MAX_RETRIES_REACHED`. Callable on the class or an instance.
- `ReasoningHandlers.update_reasoning_trace(context) -> dict`: appends `{"from", "to", "context_snapshot", ...}` from `_previous_state`/`_current_state`.
- `ReasoningHandlers.prune_context(context) -> dict`: returns only the updated keys; `{}` under threshold.
- `ContextManager.extract_relevant_context(source_context, target_keys, max_size=None) -> dict` and `ContextManager.merge_reasoning_results(orchestrator_context, sub_fsm_context, reasoning_type) -> dict`.
- `OutputFormatter.extract_final_solution(context) -> str` and `OutputFormatter.format_reasoning_summary(trace_info) -> str`.
- `map_reasoning_type(str) -> str`: case-insensitive, unknown maps to `"analytical"`.
- `ReasoningEngine._prepare_reasoning_execution(context)` reads `ContextKeys.PREFERRED_REASONING_TYPE`, writes `REASONING_TYPE_SELECTED` and `REASONING_PUSH_PENDING` (`ContextKeys.REASONING_FSM_TO_PUSH` is pinned absent).
- `fsm_llm.reasoning.__main__._format_json_output(solution, trace_info)` and `_save_as_json(save_path, problem, solution, trace_info)`.

## Data shapes

Pinned `Defaults`: `TEMPERATURE == 0.7`, `MAX_TOKENS == 2000`, `MAX_RETRIES == 3`, `MAX_SOLVE_STEPS == 170`, `MAX_HYBRID_LOOPS == 2`, `MAX_TRACE_STEPS == 50`, `CONTEXT_PRUNE_THRESHOLD == 8000`, `MIN_SOLUTION_LENGTH == 20`, `PRUNE_LIST_MAX_LENGTH == 10`, `PRUNE_STRING_MAX_LENGTH == 1000`. `MODEL` is only checked to be a `str`.

Pinned `ContextKeys` strings include `problem_statement`, `problem_type`, `problem_components`, `proposed_solution`, `final_solution`, `key_insights`, `validation_result`, `solution_confidence`, `reasoning_push_pending`, `reasoning_type_selected`, `retry_count`, `max_retries_reached`, `operand1`, `operand2`, `operator`, `calculation_result`, `deductive_conclusion`, `inductive_hypothesis`, `best_creative_solution`, `critical_assessment`, `final_hybrid_solution`, `best_explanation`, `analogical_solution`.

Model behaviour pinned in `test_definitions.py`:

- `TimestampedModel.timestamp` is timezone-aware; `age_seconds >= 0`.
- `ReasoningStep`: default `confidence == 0.0`, bounds 0 to 1; content is stripped and whitespace-only raises `ValueError`; empty evidence entries dropped. `confidence_level` for 0.3/0.6/0.8/0.95 is `LOW`/`MEDIUM`/`HIGH`/`VERY_HIGH`.
- `ValidationResult`: `passed_checks`, `total_checks`, `pass_rate` (0.0 with no checks), `has_issues`, `validation_summary` contains `"Valid"` or `"Invalid"`.
- `ReasoningTrace`: `reasoning_complexity` is `simple` (3 steps, 1 type), `moderate` (8, 2), `complex` (15, 3), `highly_complex` (25, 4); `average_step_time = execution_time_seconds / steps`, `None` when empty; computed fields passed as input (`total_steps`, `unique_states_visited`) are ignored; `reasoning_types_used` dumps as a sorted list (python and JSON mode) that `redacting_json_default` leaves intact.
- `ReasoningClassificationResult`: `alternatives` de-duplicated in order.
- `ProblemContext`: statement shorter than 3 chars or blank raises; `priority="urgent"` makes `is_high_priority` true; `context_size > 0`.
- `SolutionResult`: blank `solution` raises; confidence 0.95 gives `VERY_HIGH`, `is_high_confidence`.

## Invariants and constraints

- Missing reasoning FSM: fall back to ANALYTICAL only, with warning text `Falling back to analytical reasoning`; if ANALYTICAL is also missing, or only another type (for example DEDUCTIVE) exists, raise `ReasoningExecutionError`. Never substitute an arbitrary type.
- `extract_relevant_context` with `max_size` removes keys until `len(json.dumps(result, default=str)) <= max_size`; keys whose value is `None` or absent are dropped.
- `extract_final_solution`: `FINAL_SOLUTION` beats every other key; falls back to `PROPOSED_SOLUTION`, then must find each merged result key (`INTEGRATED_ANALYSIS`, `DEDUCTIVE_CONCLUSION`, `INDUCTIVE_HYPOTHESIS`, `BEST_CREATIVE_SOLUTION`, `CRITICAL_ASSESSMENT`, `CALCULATION_RESULT`, `FINAL_HYBRID_SOLUTION`, `BEST_EXPLANATION`, `ANALOGICAL_SOLUTION`). `MAX_RETRIES_REACHED` alone gives `ErrorMessages.MAX_RETRIES_EXCEEDED`; empty context gives text containing "no explicit solution".
- `merge_reasoning_results` writes `<type>_reasoning_completed` for every `ReasoningType`, drops `None` values, maps deductive `CONCLUSION` to `DEDUCTIVE_CONCLUSION`, and for `simple_calculator` copies `CALCULATION_RESULT` into `PROPOSED_SOLUTION`.
- `validate_solution`: `simple_calculator` strategy accepts short answers and relaxes `has_insights`; a solution shorter than `MIN_SOLUTION_LENGTH` fails; failure increments `RETRY_COUNT`; `RETRY_COUNT == MAX_RETRIES` sets `MAX_RETRIES_REACHED`; empty `PROBLEM_STATEMENT` makes `addresses_problem` follow `has_solution`.
- `update_reasoning_trace` adds nothing without both state keys, snapshots only specific `ContextKeys` (never `_`-prefixed or arbitrary keys), and keeps length at most `MAX_TRACE_STEPS + 1`.
- `prune_context` never returns `PROBLEM_STATEMENT` or `PROPOSED_SOLUTION` (preserved keys).
- Source-level pins: the `engine` module contains no `converse(` and no `"Continue reasoning"` (the engine sends no user message); `_classify_problem` has `except ReasoningClassificationError:` before `except Exception as e:`, uses `ContextKeys.PROBLEM_DOMAIN` and `ContextKeys.ALTERNATIVE_APPROACHES`, and has no `problem_domain_classified`; `_solve_problem_locked` uses `default=redacting_json_default` and no `default=str`; the `engine` module has no `default=str` and at least 2 `default=redacting_json_default`; the `handlers` module contains the phrase `never emitted`.
- `reasoning_modes` must not define `get_fsm_by_name`, `list_available_fsms`, `get_reasoning_fsms_only`.
- CLI JSON output and saved results replace an object's `__str__` with `<redacted:ClassName>`.
- Exceptions: `ReasoningEngineError(msg, details=None)` has `details == {}` by default; `ReasoningExecutionError(msg, reasoning_type=None)`; both subclasses catchable as `ReasoningEngineError`.

## Dependencies

- `fsm_llm.reasoning` submodules: `constants`, `definitions`, `engine`, `exceptions`, `handlers`, `utilities`, `reasoning_modes`, `__main__`.
- `fsm_llm.logging.logger` (loguru) for the fallback-warning check.
- `pytest` only; no fixtures from `tests/conftest.py` are used here except the built-ins `tmp_path` and `monkeypatch`.

## Failure modes

- Library logging is disabled at import. A test that needs a log line must call `logger.enable("fsm_llm")`, add a sink, and restore with `logger.remove(sink_id)` and `logger.disable("fsm_llm")` in `finally`. `caplog` cannot see loguru output.
- Enabling logging is process-global, which is why the CLI test uses a subprocess. Do not move it in-process.
- Rewording engine or handler source can fail `test_audit_fixes.py` even when behaviour is unchanged.
- Pruning tests in `TestPruneContext` and `test_prune_context_uses_list_constant` assert only inside `if key in result`, so they are weak checks.

## Working here

- Run: `.venv/bin/python -m pytest tests/test_fsm_llm_reasoning/` (126 tests: audit_fixes 28, cli_logging 2, cli_output 3, constants 19, definitions 28, engine 10, exceptions 8, handlers 28).
- Conventions: files `test_<module>.py`, classes `Test<Feature>`, local helpers prefixed `_` (for example `_make_engine`, `_PROBE`).
- When changing a pinned constant, state name, or source string in `src/fsm_llm/reasoning/`, update the matching assertion here in the same change, and read any `# DECISION` anchors near the edited code first (D-009 fallback, D-016 verbose logging, D-004 logging off by default are referenced in these tests).
- Keep tests offline: build engines with `object.__new__` and hand-set attributes rather than constructing `ReasoningEngine` with a model.
- Adding or removing tests changes the collected count; the repo pins test counts in its root docs and `tests/test_packaging.py`, so re-measure with `pytest --collect-only -q | tail -1` and update those pins.
