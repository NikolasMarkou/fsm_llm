# fsm_llm.testing

Path: `src/testing`
Purpose: The seventh subpackage of `fsm_llm`: public test doubles, a scripted user and assertions for code built on the package. Plan d454fd62 (roadmap row 0.9). Today only the wiring and the skeleton exist: `__all__` is empty and the names are added by the plan's steps 22 to 25.

## Scope

- In: fake `LLMInterface` implementations that need no network (the promoted `MockLLM2Interface` and `PromptGroundedLLM`, then `ScriptedLLM`), reply builders, a `hash_embed` embedder, `ScriptedUser` and `ConversationRun`, plain assertion functions (`assert_state`, `assert_context`, `assert_path`, `assert_result`) and an opt-in `PathRecorder` handler.
- Out: anything that reaches a model, a network or a clock; `unittest.mock` helpers (they stay in `tests/conftest.py`); `block_network`, `ollama_available`.

## Architecture

```mermaid
flowchart TD
    testing[fsm_llm.testing] --> core[fsm_llm core and fsm_llm.codes]
```

## Key files

| File | Role | Notes |
| --- | --- | --- |
| `__init__.py` | Module docstring, one static `__all__` (empty until step 22) | Names are added to `__all__` and to `docs/api.txt` in the same commit |
| `__version__.py` | Re-export of `fsm_llm.__version__` | Same shape as every subpackage |

## Invariants and constraints

- Imports the core and `fsm_llm.codes` only; never `pytest`, `litellm` or another subpackage (`tests/test_fsm_llm_testing/test_layering.py` reads the source with `ast`).
- No library module imports it and `fsm_llm/__init__.py` never loads it; it is imported by name.
- Importing it has no side effect: no clock, no randomness, no socket.
- Assertions (later steps) raise `AssertionError` explicitly, never `assert` (stripped by `python -O`).
- Names stay out of `src/constants.py`, `src/eval/cases.py` and `src/utilities.py`: they are bench-pinned (see the root `AGENTS.md`).
- Every public name gets a docstring with an interface contract (parameters, return shape, failure mode) and a static `__all__` entry; `docs/api.txt` is regenerated in the same commit.

## Testing

`tests/test_fsm_llm_testing/`. The `testing` extra is `[]`.
