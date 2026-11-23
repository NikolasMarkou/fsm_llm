# test_fsm_llm_reasoning

Unit tests for the reasoning engine in the FSM-LLM repository. The engine lives in `src/fsm_llm/reasoning/` and solves a problem by picking one of the reasoning strategies (each a finite state machine, or FSM) and running it through an orchestrator FSM.

## What it is for

These tests check the reasoning engine's building blocks without calling a real LLM. They cover the constants, the Pydantic data models, the exception classes, the handler functions that validate solutions and trim context, and a few engine internals. A group of tests also locks in fixes from earlier code audits so they cannot quietly come back. Nothing here needs a network, an API key or a running model.

## How it works

Each test imports from `fsm_llm.reasoning.*` and calls classes or static methods directly with small hand-built context dicts. Engine tests that need a `ReasoningEngine` build one with `object.__new__` and set `reasoning_fsms` by hand, so no FSM files are loaded and no LLM is set up. Some audit tests read source code with `inspect.getsource` and assert that certain strings are, or are not, present. The CLI logging test starts a fresh Python process with a stubbed solver, so global logging state does not leak into the test run.

## Files

- `__init__.py` - empty, marks the directory as a package.
- `test_cli_logging.py` - tests that the CLI `--verbose` flag turns library logging on and that it stays off without it.
- `test_cli_output.py` - tests that CLI JSON output and JSON save files list `reasoning_types_used` as a sorted list built from a real `ReasoningTrace`, that JSON output survives a value `json.dumps` cannot encode, that unencodable output is escaped on an ASCII terminal, and that `--quiet` prints the formatted output.
- `test_constants.py` - tests for `ReasoningType`, state name constants, `ContextKeys`, `HandlerNames`, `Defaults`, and message templates.
- `test_definitions.py` - tests for the Pydantic models (`ReasoningStep`, `ValidationResult`, `ReasoningTrace`, `ProblemContext`, and others).
- `test_engine.py` - tests for models, handlers, `map_reasoning_type`, the strategy choice (`_strategy_for`, `_select_reasoning_type`), and the engine's regression classes (malformed user message, final solution keys, classification errors, removed dead code, execution time, only the orchestrator traces, classifier cleanup).
- `test_engine_scripted.py` - tests that run whole solves offline through the core engine with a scripted `LLMInterface`: orchestrator, classifier FSM, pushed and popped strategy FSM, validation retry loop, forced pop, spent step budget, failures that keep `partial_context`, caller context that cannot drive the push.
- `test_classifier_intents.py` - tests that the strategy choice is a core classification with the nine strategies as closed intents: the classifier file equals its derivation (`classifier_file.py`, the writer), one `complete` request with a closed `enum`, the fallback, unknown and low-confidence picks and an outage end on `analytical`, an interface without `complete` is refused at construction.
- `test_orchestrator_call.py` - tests that the orchestrator pushes its strategy through the `call` state `execute_reasoning` (plan baa37f40 step 15, D-031): the file, the model call sequence of a solve, the retry clear, the choice (classified, preferred, never a model reply), absent `constraints`, the reply order, an interrupt inside the pushed frame, the event log of a call frame and the declared-inputs-only census row.
- `test_solve_budgets.py` - tests the budgets of a solve (plan baa37f40 step 16, D-032): the default step budget derived from the `call.max_steps` of the orchestrator file (170 for the shipped one, no literal), `solve_problem(..., max_steps=, max_seconds=)` bounding the one run, a bad limit refused before any model call, `reasoning.solve` mapping the limits of its run context, and `ReasoningReactAgent`'s call surface unchanged.
- `test_selection.py`, `test_selection_intents.py` - tests the selection over the registry (plan baa37f40 steps 17 and 17b, D-033, D-034): `strategy_choices` and its eligibility rules with one warning per skipped strategy, `ReasoningEngine(registry=)`, a user strategy chosen by `preferred_reasoning_type` or by the classifier and returning its own answer, the classifier intents derived from the same list as `choose.targets` (cap 15, `analytical` kept, one line each), and the named engine errors.
- `test_cli_registry.py` - tests the reasoning CLI over the registry (plan baa37f40 step 18, D-035): `--executables DIR`, `--type` checked against the registry names (exit 2), `--list-types`, one warning line per unreadable file or collision, the engine closed on every way out and Ctrl-C as 130.
- `test_strategy_contract.py` - tests that each strategy file declares one input and no output, keeps its body byte equal, and that the registry resolves a strategy with its own output map (plan baa37f40 step 14, D-030).
- `test_exceptions.py` - tests for `ReasoningError`, `ReasoningExecutionError`, `ReasoningClassificationError`.
- `test_golden_tables.py` - tests that pin the FSM definitions and the strategy tables by sha256.
- `test_handlers.py` - tests for `ReasoningHandlers`, `ContextManager`, and `OutputFormatter`.

## How to use it

From the repository root, with the project virtualenv:

```bash
.venv/bin/python -m pytest tests/test_fsm_llm_reasoning/
.venv/bin/python -m pytest tests/test_fsm_llm_reasoning/test_handlers.py -v
```

## Things to know

- The tests import the engine as `fsm_llm.reasoning`, a subpackage of the core `fsm_llm` package.
- Library logging is off by default. Tests that check a log line turn it on with `logger.enable("fsm_llm")`, add a temporary sink, and turn it off again. `pytest`'s `caplog` does not work here because the logger is loguru, not the standard `logging` module.
- `test_cli_logging.py` runs a child Python process with a 60 second timeout and clears `FSM_LLM_LOG_LEVEL` for it.
- Several audit tests match exact strings in the source. The engine module must contain no `converse(`, no `"Continue reasoning"`, no `json.dumps` and no `default=str`; in `ReasoningEngine._classify_problem` the text `except RunBudgetExceededError as e:` must come before `except Exception as e:`; `handlers.py` must keep the words "never emitted" in a comment. Renaming or rewording that code will fail them.
- `test_engine_scripted.py` runs whole solves through the core engine with a scripted LLM interface and checks every call (no user message is ever sent, the strategy FSM is pushed and popped, retries reach their limit, failures carry `partial_context`).
- Some pruning tests only assert when the key appears in the result (`if ... in result`), so they pass if pruning leaves the key untouched.
