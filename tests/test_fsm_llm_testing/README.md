# test_fsm_llm_testing

The pytest suite for `fsm_llm.testing`, the public test-helper subpackage of the FSM-LLM repository. It lives at `tests/test_fsm_llm_testing/`.

## What it is for

`fsm_llm.testing` will hold the fake LLM interfaces, the scripted user and the assertions that other projects use to test code built on `fsm_llm`. For now only the wiring exists, so this suite checks the layering: the module is imported by name, no library module imports it, it imports no `pytest`, `litellm` or other subpackage, and `import fsm_llm` does not load it. Tests for each public name are added together with the name.

## How it works

- Import probes run in a child process (`python -I`) with the model-server variables pointed at a closed port, so a leak is seen in a clean `sys.modules`.
- Source checks read every module with `ast`, so a stray import is found without running it.

## Files

- `__init__.py` - empty package marker.
- `test_layering.py` - the import probes, the library-wide scan for imports of `fsm_llm.testing`, the `pytest` and `litellm` ban, the no-other-subpackage rule, and the skeleton checks (docstring, one static `__all__`).

## How to use it

```bash
.venv/bin/python -m pytest tests/test_fsm_llm_testing/
```

## Things to know

- Run from the repository root. No test calls a model or a network.
- The repository root `AGENTS.md` and `tests/AGENTS.md` state the suite counts; `tests/test_packaging.py` checks them against a fresh collection.
