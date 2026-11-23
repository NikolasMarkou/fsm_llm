# test_fsm_llm_reasoning

Path: `tests/test_fsm_llm_reasoning`
Purpose: Offline unit tests for the reasoning engine subpackage `fsm_llm.reasoning` (source in `src/fsm_llm/reasoning/`).

## Scope

Covers `constants.py`, `definitions.py`, `exceptions.py`, `handlers.py`, `utilities.map_reasoning_type`, `strategy_files` (the package-data loader; `test_strategy_files.py`), selected `engine.py` internals, and `__main__.py` (JSON output helpers and `--verbose`). No test calls a real LLM or needs network or env keys. `test_engine_scripted.py` runs whole `ReasoningEngine.solve_problem` solves through core with a scripted `LLMInterface` (`_ScriptedLLM`, injected as `llm_interface=`) and asserts every call: no user exchange and no "Continue reasoning" in history, push and pop of the strategy FSM, the retry loop reaching `max_retries_reached`, the hybrid back edge bounded, handler-only verdict keys, the budget error with `partial_context`, critical handlers, classifier value normalisation.

## Architecture

Tests call static methods and model constructors directly with small context dicts keyed by `ContextKeys` constants, or drive whole solves on a scripted interface (`test_engine_scripted.py`). Further techniques recur:

- The strategy choice is two small functions: `choice._strategy_for(recommended, preferred, choices, analytical)` (a module function in `reasoning/choice.py`, no engine needed; `engine` imports it back) and `engine._select_reasoning_type(context)` on a real engine built with a scripted interface (`test_engine.py`); a missing or over-budget strategy is injected by patching `executables.strategy_registry` before the engine is built (`test_engine_scripted.py::_registry_with`).
- Source assertions: `inspect.getsource(...)` on `ReasoningEngine`, `ReasoningEngine._classify_problem`, `ReasoningEngine._solve_problem_locked`, the `engine` module, and the `handlers` module, then string checks (`test_engine.py`, `test_handlers.py`, `test_constants.py`, `test_cli_output.py`).
- Subprocess probe: `test_cli_logging.py` runs `sys.executable -c _PROBE`. The probe imports `fsm_llm.reasoning.__main__`, replaces `solve_problem_with_engine` with a stub that logs `I-PROBE` (info) and `D-PROBE` (debug) from a namespace named `fsm_llm.reasoning.engine`, sets `sys.argv` to `['fsm-llm-reasoning', 'what is 2+2?', '-o', 'json', ...]`, and exits with `m.main()`.

## Key files

| File | Role | Notes |
| --- | --- | --- |
| `test_cli_logging.py` | The parametrized cases: `["--verbose"]` gives one `I-PROBE` line in stderr, `[]` gives none; `D-PROBE` never appears | Subprocess, `timeout=60`, deletes `FSM_LLM_LOG_LEVEL` |
| `test_cli_output.py` | `_format_json_output` and `_save_as_json` fed a real `ReasoningTrace(...).model_dump()` write `reasoning_types_used` as a sorted list, never redacted; CLI escape of unencodable output, `--quiet` output, summary facts stated once | Uses `tmp_path`; includes the plan 5980fbb7 step 18 CLI fixes |
| `test_calculator.py` | The calculator's evaluator is total: a table of adversarial inputs (the plain ones plus huge literals, chains and nests) each ends in a value or a `ValueError`, refusals, fixed messages, the text bound before the parser (spy), the pow pre-check (spy), inclusive exponent and node bounds | Runs on every Python of the matrix; a `bool`, complex or non-finite result is refused |
| `test_constants.py` | Constant values | Pins literal strings and numbers |
| `test_definitions.py` | Pydantic models | Validators, computed properties, thresholds |
| `test_engine.py` | Models, handlers, `map_reasoning_type`, ANALYTICAL-only fallback (D-009); audit regression classes (malformed user message, final solution keys, classification exception handling, removed dead code, `execution_time_seconds`, `direct computation` no special case, no classifier tracer, core truncation suffix, classifier cleanup warning) | Uses loguru sink for the warning; several assertions match literal source strings |
| `test_engine_scripted.py` | Whole solves through core on `_ScriptedLLM` (also imported by `tests/test_fsm_llm_regression/test_engine_integration.py`; its `complete` answers the strategy choice from `script["recommended_reasoning_type"]` and keeps those requests in `completions`) | Asserts calls, history, stack, retries, budget and failure details |
| `test_classifier_intents.py`, `classifier_file.py` | The closed-intent classification of `recommend_strategy` (plan baa37f40 step 5b, D-019); `classifier_file.py` is the writer of that state in `classifier.json` (`python -m tests.test_fsm_llm_reasoning.classifier_file --write`) | The file equals its derivation; one `complete` request; fallback, unknown and low confidence give `analytical` (the last with one WARNING); no `complete` is refused at construction |
| `test_solve_stream.py` | The streaming solve (plan baa37f40 step 19, D-036) | Lazy start and call-time limit checks, strategy replies yielded while the strategy frame is live, `.result()` only after exhaustion, the tuple wrapper equal to the stream's result with the same model requests, the lock and the conversation released by exhaustion, an error, `close()`, `with`, a dropped stream, a signal and a cancel; no thread |
| `test_solve_budgets.py` | The budgets of a solve (plan baa37f40 step 16, D-032) | Derived default (`solve_step_budget`), explicit `max_steps` and `max_seconds` (a ticking stand-in for the run loop's clock through `bind_api_clock`), refusal before any call, `SolveExecutable` mapping, the agents call surface by AST |
| `test_selection.py`, `test_selection_intents.py` | Selection over the registry (plan baa37f40 steps 17 and 17b, D-033, D-034) | `_document` builds a user strategy from the analytical body; the warnings are captured with a loguru sink; the one function `strategy_choices` and every consumer; named mutants (a string import, the kind filter widened, the cached solution keys, a second filter on the intents) were run on a git-archive copy |
| `test_cli_registry.py` | The reasoning CLI over the registry (plan baa37f40 step 18, D-035) | in-process `main()` with a stub engine that supports `with`; a real engine on `_ScriptedLLM` for one end-to-end row |
| `test_exceptions.py` | The exception hierarchy | `details`, `reasoning_type` attributes |
| `test_golden_tables.py` | sha256 of the canonical JSON of the FSM definitions and of the strategy tables (enum, registry keys, aliases, descriptions, merge ladder, calculator cases, solution order) | Captured before the prompt sentences, state ids and tables were generated; a hash moves only on purpose |
| `test_handlers.py` | `ReasoningHandlers`, `ContextManager`, `OutputFormatter` | Validation, trace, pruning, merge, final solution |
| `__init__.py` | Empty package marker | |

## Public interface under test

- `ReasoningType` (str enum), exactly these values: `simple_calculator`, `analytical`, `deductive`, `inductive`, `abductive`, `analogical`, `creative`, `critical`, `hybrid`. Unknown value raises `ValueError`.
- `OrchestratorStates`: `problem_analysis`, `strategy_selection`, `execute_reasoning`, `synthesize_solution`, `validate_refine`, `final_answer`.
- `ClassifierStates`: `analyze_domain`, `analyze_structure`, `identify_reasoning_needs`, `recommend_strategy`.
- `HandlerNames`: `OrchestratorProblemClassifier`, `OrchestratorStrategyExecutor`, `OrchestratorSolutionValidator`, `ReasoningTracer`, `ContextPruner`, `RetryLimiter`, `RetryKeyClearer`, `HybridLoopCounter`.
- `ErrorMessages` placeholders: `INVALID_REASONING_TYPE` `{type}`, `FSM_NOT_FOUND` `{name}`; `MAX_RETRIES_EXCEEDED` non-empty. `CALCULATION_ERROR`, `VALIDATION_FAILED`, `CONTEXT_TOO_LARGE` and `Defaults.MAX_CONTEXT_SIZE` are pinned absent (removed in plan 07ad3f8c).
- `LogMessages` placeholders: `ENGINE_INITIALIZED` `{model}`, `CLASSIFICATION_STARTED` `{context}`, `CLASSIFICATION_COMPLETE` `{type}`, `FSM_PUSHED` `{name}`, `PROBLEM_SOLVED` `{steps}`.
- `ReasoningHandlers.validate_solution(context) -> dict`: sets `VALIDATION_RESULT`, `SOLUTION_CONFIDENCE`, `VALIDATION_CHECKS` (`has_solution`, `has_insights`, `sufficient_detail`, `addresses_problem`), `RETRY_COUNT`, `MAX_RETRIES_REACHED`. Callable on the class or an instance.
- `ReasoningHandlers.update_reasoning_trace(context) -> dict`: appends `{"from", "to", "context_snapshot", ...}` from `_previous_state`/`_current_state`.
- `ReasoningHandlers.prune_context(context) -> dict`: returns only the updated keys; `{}` under threshold.
- `ContextManager.extract_relevant_context(source_context, target_keys, max_size=None) -> dict`.
- `OutputFormatter.extract_final_solution(context) -> str` and `OutputFormatter.format_reasoning_summary(trace_info) -> str`.
- `map_reasoning_type(str) -> str`: case-insensitive, unknown maps to `"analytical"`.
- `ReasoningEngine._select_reasoning_type(context)` writes `REASONING_TYPE_SELECTED` and defaults `CONSTRAINTS`; `_classify_problem` writes `CLASSIFIED_PROBLEM_TYPE` as the target id and reads `ContextKeys.PREFERRED_REASONING_TYPE` (`ContextKeys.REASONING_PUSH_PENDING` and `REASONING_FSM_TO_PUSH` are pinned absent).
- `fsm_llm.reasoning.__main__._format_json_output(solution, trace_info)` and `_save_as_json(save_path, problem, solution, trace_info)`.

## Data shapes

Pinned `Defaults`: `TEMPERATURE == 0.7`, `MAX_TOKENS == 2000`, `MAX_RETRIES == 3`, `MAX_HYBRID_LOOPS == 2`, `MAX_TRACE_STEPS == 50`, `CONTEXT_PRUNE_THRESHOLD == 8000`, `MIN_SOLUTION_LENGTH == 20`, `PRUNE_LIST_MAX_LENGTH == 10`, `PRUNE_STRING_MAX_LENGTH == 1000`. `MODEL` is only checked to be a `str`.

Pinned `ContextKeys` strings include `problem_statement`, `problem_type`, `problem_components`, `proposed_solution`, `final_solution`, `key_insights`, `validation_result`, `solution_confidence`, `reasoning_push_pending`, `reasoning_type_selected`, `retry_count`, `max_retries_reached`, `calculation_expression`, `calculation_result`, `calculation_error`, `deductive_conclusion`, `inductive_hypothesis`, `best_creative_solution`, `critical_assessment`, `final_hybrid_solution`, `best_explanation`, `analogical_solution`.

Model behaviour pinned in `test_definitions.py`:

- `TimestampedModel.timestamp` is timezone-aware; `age_seconds >= 0`.
- `ReasoningStep`: default `confidence == 0.0`, bounds 0 to 1; content is stripped and whitespace-only raises `ValueError`; empty evidence entries dropped. `confidence_level` for 0.3/0.6/0.8/0.95 is `LOW`/`MEDIUM`/`HIGH`/`VERY_HIGH`.
- `ValidationResult`: `passed_checks`, `total_checks`, `pass_rate` (0.0 with no checks), `has_issues`, `validation_summary` contains `"Valid"` or `"Invalid"`.
- `ReasoningTrace`: `reasoning_complexity` is `simple` (3 steps, 1 type), `moderate` (8, 2), `complex` (15, 3), `highly_complex` (25, 4); `average_step_time = execution_time_seconds / steps`, `None` when empty; computed fields passed as input (`total_steps`, `unique_states_visited`) are ignored; `reasoning_types_used` dumps as a sorted list (python and JSON mode) that `redacting_json_default` leaves intact.
- `ReasoningClassificationResult`: `alternatives` de-duplicated in order.
- `ProblemContext`: statement shorter than 3 chars or blank raises; `priority="urgent"` makes `is_high_priority` true; `context_size > 0`.

## Invariants and constraints

- Missing reasoning FSM: fall back to ANALYTICAL only, with warning text `Falling back to analytical reasoning`; if ANALYTICAL is also missing, or only another type (for example DEDUCTIVE) exists, raise `ReasoningExecutionError`. Never substitute an arbitrary type.
- `extract_relevant_context` with `max_size` removes keys until `len(json.dumps(result, default=str)) <= max_size`; keys whose value is `None` or absent are dropped.
- `extract_final_solution`: `FINAL_SOLUTION` beats every other key; falls back to `PROPOSED_SOLUTION`, then must find each merged result key (`INTEGRATED_ANALYSIS`, `DEDUCTIVE_CONCLUSION`, `INDUCTIVE_HYPOTHESIS`, `BEST_CREATIVE_SOLUTION`, `CRITICAL_ASSESSMENT`, `CALCULATION_RESULT`, `FINAL_HYBRID_SOLUTION`, `BEST_EXPLANATION`, `ANALOGICAL_SOLUTION`). `MAX_RETRIES_REACHED` alone gives `ErrorMessages.MAX_RETRIES_EXCEEDED`; empty context gives text containing "no explicit solution".
- A strategy returns the keys of its `results` map (one table, pinned by the `results_maps` hash of `test_golden_tables.py`), which core `settle` merges (`test_call_settle.py::TestTheTargetsOwnOutputMap`).
- `validate_solution`: `simple_calculator` strategy accepts short answers and relaxes `has_insights`; a solution shorter than `MIN_SOLUTION_LENGTH` fails; failure increments `RETRY_COUNT`; `RETRY_COUNT == MAX_RETRIES` sets `MAX_RETRIES_REACHED`; empty `PROBLEM_STATEMENT` makes `addresses_problem` follow `has_solution`.
- `update_reasoning_trace` adds nothing without both state keys, snapshots only specific `ContextKeys` (never `_`-prefixed or arbitrary keys), and keeps length at most `MAX_TRACE_STEPS + 1`.
- `prune_context` never returns `PROBLEM_STATEMENT` or `PROPOSED_SOLUTION` (preserved keys).
- Source-level pins: the `engine` module contains no `converse(` and no `"Continue reasoning"` (the engine sends no user message); `_classify_problem` has `except ReasoningClassificationError:` before `except Exception as e:`, uses `ContextKeys.PROBLEM_DOMAIN` and `ContextKeys.ALTERNATIVE_APPROACHES`, and has no `problem_domain_classified`; `_solve_problem_locked` uses `default=redacting_json_default` and no `default=str`; the `engine` module has no `default=str` and at least 2 `default=redacting_json_default`; the `handlers` module contains the phrase `never emitted`.
- `strategy_files` must not define `get_fsm_by_name`, `list_available_fsms`, `get_reasoning_fsms_only`; `fsm_llm.reasoning.reasoning_modes`, `STRATEGIES`, `StrategySpec` and `_TYPE_MAP` are gone (`test_strategy_files.py`).
- CLI JSON output and saved results replace an object's `__str__` with `<redacted:ClassName>`.
- Exceptions: `ReasoningError(msg, details=None)` has `details == {}` by default; `ReasoningExecutionError(msg, reasoning_type=None)`; both subclasses catchable as `ReasoningError`.

## Dependencies

- `fsm_llm.reasoning` submodules: `constants`, `definitions`, `engine`, `exceptions`, `handlers`, `utilities`, `strategy_files`, `__main__`.
- `fsm_llm.logging.logger` (loguru) for the fallback-warning check.
- `pytest` only; no fixtures from `tests/conftest.py` are used here except the built-ins `tmp_path` and `monkeypatch`.

## Failure modes

- Library logging is disabled at import. A test that needs a log line must call `logger.enable("fsm_llm")`, add a sink, and restore with `logger.remove(sink_id)` and `logger.disable("fsm_llm")` in `finally`. `caplog` cannot see loguru output.
- Enabling logging is process-global, which is why the CLI test uses a subprocess. Do not move it in-process.
- Rewording engine or handler source can fail the source-assertion tests of `test_engine.py`, `test_handlers.py`, `test_constants.py` and `test_cli_output.py` even when behaviour is unchanged.
- Pruning tests in `TestPruneContext` and `test_prune_context_uses_list_constant` assert only inside `if key in result`, so they are weak checks.

## Working here

- Run: `.venv/bin/python -m pytest tests/test_fsm_llm_reasoning/`.
- Conventions: files `test_<module>.py`, classes `Test<Feature>`, local helpers prefixed `_` (for example `_make_engine`, `_PROBE`).
- When changing a pinned constant, state name, or source string in `src/fsm_llm/reasoning/`, update the matching assertion here in the same change, and read any `# DECISION` anchors near the edited code first (D-009 fallback, D-016 verbose logging, D-004 logging off by default are referenced in these tests).
- Keep tests offline: build engines with `object.__new__` and hand-set attributes rather than constructing `ReasoningEngine` with a model.
- No document states this suite's test count; measure it with `.venv/bin/python -m pytest tests/test_fsm_llm_reasoning --collect-only -q | tail -1`.
