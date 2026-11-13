# test_fsm_llm_regression

Regression tests for the `fsm_llm` framework, in `tests/test_fsm_llm_regression/`. Each test pins one bug that was found and fixed, so the bug cannot come back unnoticed.

## What it is for

`fsm_llm` runs chatbots as JSON-defined finite state machines (FSMs) driven by an LLM. Over many review rounds ("plans" and audits), bugs were found in the core API, the transition evaluator, the JsonLogic expression engine, prompt building, the LiteLLM wrapper, the visualizer and validator, and in the reasoning and workflow subpackages. This folder keeps one test class per bug. The class docstring names the bug id (for example `VB14b`, `P3-B10`, `ED-001`, `F-005`) and says what the fixed behavior is. No test calls a real LLM: they use mocks, hand-built objects, or source inspection.

## How it works

Tests fall into four styles:

- Behavior tests call real code with mock LLMs and check results (for example, `less("10", "2")` must be `False` because numeric strings compare as numbers).
- End-to-end tests drive `API` or `WorkflowEngine` through a full conversation or workflow with a mock LLM.
- Source and file checks read Python source (`inspect.getsource`) or repo files (`README.md`, `pyproject.toml`, `tox.ini`, `docs/quickstart.md`) and assert that a bad pattern is absent or a required value is present.
- Hand-built objects: some tests create `API.__new__(API)` without running `__init__` and set only the attributes the method under test reads.

## Files

- `__init__.py` - empty package marker.
- `test_bugs.py` - bugs B1 to B8, B-NEW-1 to B-NEW-6, CR3, plan 3 (P3-B1 to P3-B11) and C1: JSON extraction escapes, no `logs/` dir at import, sync handlers, no md5, BFS with deque, error masking, API key env mutation, `soft_equals` with booleans, forbidden context patterns, review fixes H1 (LLM `timeout`), H2 (one `import re` in `llm.py`) and M1 (no async handlers).
- `test_engine_integration.py` - end-to-end runs of `ReasoningEngine.solve_problem` (including that the caller's initial context is not changed) and `WorkflowEngine` (linear, condition, API, parallel, event, custom step, 10 sequential instances), plus ED-001 (falsy solutions such as `0` kept), ED-002 (event listeners and timers registered automatically), ED-003 (reasoning type aliases such as `"math"`), iter-2 F-004 (WaitEventStep state validation) and review C2, C3, H4 (reasoning context pruning, `ContextKeys` constants, no `MergeStrategy` alias).
- `test_functional.py` - full conversation lifecycle through `API` with the mock 2-pass LLM: start, converse, transitions, context data, push/pop of FSMs, definition validation, history size.
- `test_regression_cli_and_exports.py` - plan 13: CLI `main_cli` entry points, `ContextMergeStrategy` export, no `fsm_llm.workflows.cli`, README badge and headings, `docs/quickstart.md` links, `py.typed`, shared version, and review fixes C1 (version `0.12.0`), H5 (litellm spec excludes 1.82.7 and 1.82.8), M3 (static `__all__`) and M6 (no Sphinx in `tox.ini`).
- `test_regression_core_bugs.py` - plan 6 (VB1 to VB25): self-transitions, reasoning never shown as the reply, error wrapping, terminal-state rejection, visualizer depth and sorting, validator cascades, numeric string comparison; iter-2 F-003 (`api_key` forbidden pattern), F-005 (LRU FSM cache) and F-006 (`less`). Iter-2 F-001 and F-002 are tested elsewhere.
- `test_regression_expressions_and_transitions.py` - plan 5: `ValueError` propagation in `converse`, `max_history_size=0`, high-priority single transitions, JsonLogic `and`/`or` values, chained `>`, prompt tag escaping, `--version` flag.
- `test_regression_handlers_and_state.py` - plan 4: POST_TRANSITION sees the new state, handler `handler_error_mode`, missing or `None` LLM content.
- `test_regression_jsonlogic_and_merge.py` - plan 9: multi-key JsonLogic dicts raise, reserved LLM kwargs, `transition_decision` never gets structured output, context merge, LIFO stack teardown in `end_conversation`, history role mapping (an unknown role becomes system), exception chaining, `min`/`max`/`missing_some`, `None` comparisons, removed dead code.
- `test_regression_logic_and_visualizer.py` - plan 7: empty `and`/`or`, visualizer box widths, instance cleanup when a START_CONVERSATION handler fails, removed `early_termination`.
- `test_regression_messages_and_operators.py` - plan 12: message truncation length, allowed JsonLogic operator names, validator output without emoji, Python 3.11 floor, no `HANDLER_ERROR_SKIP`.
- `test_api_snapshot.py` - the public API snapshot: `docs/api.txt` must equal the output of `scripts/api_snapshot.py` (3 tests); regenerate it with `make api-snapshot` after a deliberate public change.
- `test_regression_transition_eval.py` - plan 8: strict condition matching stops on error, priority tie-break, dict content in response parsing, no CONTEXT_UPDATE handlers for empty data, `pop_fsm` failure path.
- `test_no_model_paths.py` - review gate 1 of plan d454fd62: with `LLM_MODEL` deleted by the test itself (the root autouse fixture sets it for every test), every public constructor and runner that takes an injected `llm_interface` or needs no LLM builds and runs (agents, SOPs, patterns, eval, reasoning, workflows, monitor, synthesis, meta), and every path that has nothing injected still raises `ModelNotConfiguredError`.

## How to use it

From the repo root, with the project virtualenv:

```bash
.venv/bin/python -m pytest tests/test_fsm_llm_regression/
.venv/bin/python -m pytest tests/test_fsm_llm_regression/test_regression_core_bugs.py -k VB14b
```

At the time of writing, 268 tests are collected: 266 pass and 2 are skipped (the two `requirements.txt` checks in `test_regression_cli_and_exports.py`). The run takes about a second.

## Things to know

- `test_regression_cli_and_exports.py` pins the package version to `"0.12.0"` (`TestVersionAlignment`). Update it on every release.
- Many tests call private methods (for example `TransitionEvaluator._determine_evaluation_result`, `LiteLLMInterface._parse_response_generation_response`, `manager._pipeline._execute_state_transition`). Renaming them breaks these tests.
- Source checks fail on text, not behavior: reintroducing the string `md5` in `src/api.py` or `import asyncio` in `handlers.py` fails a test even if unused.
- Tests that build `API.__new__(API)` must set every attribute the method reads (`_stack_lock`, `_last_accessed`, `_pending_push_ids`, `_ended_conversations`, `_MAX_ENDED_CACHE`, `_temp_fsm_definitions`). A new attribute in `API.__init__` may need adding here.
- The fixtures `mock_llm2_interface` and `sample_fsm_definition_v2` come from `tests/conftest.py`.
- Async tests in `test_engine_integration.py` have no marker and rely on `asyncio_mode = "auto"` in `pyproject.toml`.
