# fsm_llm.testing

Public test helpers for code built on FSM-LLM, living at `src/testing` inside the core `fsm_llm` package: fake LLM interfaces that need no network, a scripted user that drives a conversation, and assertions that name a state, a context value, a visited path or a run result in one line.

## What it is for

A test of a bot, an agent or a workflow should not call a real model: it is slow, costs money and gives a different answer each run. This package will hold the doubles and assertions the FSM-LLM repository uses for its own tests, so your project can use the same ones instead of copying them from `tests/conftest.py`.

## Status

The subpackage is wired into the build and has no public name yet: `fsm_llm.testing.__all__` is empty. The names arrive in later steps of the plan that added it (`MockLLM2Interface` and `PromptGroundedLLM` promoted from `tests/conftest.py`, then `ScriptedLLM`, reply builders, `ScriptedUser`, the assertions and `PathRecorder`). Do not rely on anything here before that lands.

## How it fits

```mermaid
flowchart TD
    testing[fsm_llm.testing] --> core[fsm_llm core]
```

- Import it by name: `from fsm_llm.testing import ...`. `import fsm_llm` never loads it and no library module imports it.
- It imports the core and `fsm_llm.codes` only. It never imports `pytest` (an opt-in pytest plugin module may, later, and is imported only by name), `litellm` or another subpackage.
- The `testing` extra in `pyproject.toml` is `[]` (no third-party dependency); `all`, `make install-dev`, tox and CI request it.
- Importing it has no side effect: no clock, no randomness, no socket.

## Files

- `__init__.py` - module docstring and the static, empty `__all__`.
- `__version__.py` - re-exports the package version.

## Tests

`tests/test_fsm_llm_testing/` (run `.venv/bin/python -m pytest tests/test_fsm_llm_testing/`). The layering checks there fail if a library module imports this package, if it imports `pytest`, `litellm` or another subpackage, or if `import fsm_llm` loads it.
