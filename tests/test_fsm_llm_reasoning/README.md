# test_fsm_llm_reasoning

Unit tests for the reasoning engine in the FSM-LLM repository. The engine lives in `src/fsm_llm/reasoning/` and solves a problem by picking one of 9 reasoning strategies (each a finite state machine, or FSM) and running it through an orchestrator FSM.

## What it is for

These tests check the reasoning engine's building blocks without calling a real LLM. They cover the constants, the Pydantic data models, the exception classes, the handler functions that validate solutions and trim context, and a few engine internals. A group of tests also locks in fixes from earlier code audits so they cannot quietly come back. Nothing here needs a network, an API key or a running model.

## How it works

Each test imports from `fsm_llm.reasoning.*` and calls classes or static methods directly with small hand-built context dicts. Engine tests that need a `ReasoningEngine` build one with `object.__new__` and set `reasoning_fsms` by hand, so no FSM files are loaded and no LLM is set up. Some audit tests read source code with `inspect.getsource` and assert that certain strings are, or are not, present. The CLI logging test starts a fresh Python process with a stubbed solver, so global logging state does not leak into the test run.

## Files

- `__init__.py` - empty, marks the directory as a package.
- `test_audit_fixes.py` - 27 tests that pin audit fixes: context size limits, pruning constants, solution key coverage, exception re-raise order, removed dead code, and redacted JSON output.
- `test_cli_logging.py` - 2 tests that the CLI `--verbose` flag turns library logging on and that it stays off without it.
- `test_cli_output.py` - 3 tests that CLI JSON output and JSON save files list `reasoning_types_used` as a sorted list built from a real `ReasoningTrace`.
- `test_constants.py` - 26 tests for `ReasoningType`, state name constants, `ContextKeys`, `HandlerNames`, `Defaults`, and message templates.
- `test_definitions.py` - 28 tests for the Pydantic models (`ReasoningStep`, `ValidationResult`, `ReasoningTrace`, `ProblemContext`, `SolutionResult`, and others).
- `test_engine.py` - 10 tests for models, handlers, `map_reasoning_type`, and the rule that a missing reasoning FSM may only fall back to analytical.
- `test_exceptions.py` - 8 tests for `ReasoningEngineError`, `ReasoningExecutionError`, `ReasoningClassificationError`.
- `test_handlers.py` - 39 tests for `ReasoningHandlers`, `ContextManager`, and `OutputFormatter`.

191 tests in total.

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
- Several audit tests match exact strings in the engine source (for example that it contains no `converse(` and no `"Continue reasoning"`, and does contain `"except ReasoningClassificationError:"` and `"default=redacting_json_default"`). Renaming or rewording that code will fail them.
- `test_engine_scripted.py` runs whole solves through the core engine with a scripted LLM interface and checks every call (no user message is ever sent, the strategy FSM is pushed and popped, retries reach their limit, failures carry `partial_context`).
- Some pruning tests only assert when the key appears in the result (`if ... in result`), so they pass if pruning leaves the key untouched.
