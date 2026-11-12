# fsm_llm.testing

Path: `src/testing`
Purpose: The seventh subpackage of `fsm_llm`: public test doubles, a scripted user and assertions for code built on the package. Plan d454fd62 (roadmap row 0.9). Today `__all__` holds the two promoted doubles (`MockLLM2Interface`, `PromptGroundedLLM`, step 22), `ScriptedLLM`, `UnscriptedCallError`, `state_in_prompt`, `hash_embed` and the three reply builders (step 23); the rest is added by steps 24 and 25.

## Scope

- In: fake `LLMInterface` implementations that need no network (the promoted `MockLLM2Interface` and `PromptGroundedLLM`, `ScriptedLLM` with its `UnscriptedCallError`), reply builders, `state_in_prompt`, a `hash_embed` embedder, `ScriptedUser` and `ConversationRun`, plain assertion functions (`assert_state`, `assert_context`, `assert_path`, `assert_result`) and an opt-in `PathRecorder` handler.
- Out: anything that reaches a model, a network or a clock; `unittest.mock` helpers (they stay in `tests/conftest.py`); `block_network`, `ollama_available`.

## Architecture

```mermaid
flowchart TD
    testing[fsm_llm.testing] --> core[fsm_llm core and fsm_llm.codes]
```

## Key files

| File | Role | Notes |
| --- | --- | --- |
| `__init__.py` | Module docstring, one static `__all__` | Names are added to `__all__` and to `docs/api.txt` in the same commit |
| `doubles.py` | `MockLLM2Interface` (fixed answers) and `PromptGroundedLLM` (a fact only when the prompt shows its evidence), moved verbatim from `tests/conftest.py`; the private `_CURRENT_STATE_TAG` regex (the one tag regex, read through `state_in_prompt`); `ScriptedLLM` (reply queue, `extraction_data`, `responses` per state, `stream_chunks`, `unexpected`, `assert_exhausted()`, metered through core's `_Metered`), `UnscriptedCallError`, `hash_embed` | Plain classes with mutable attributes, subclassed by the repo's own tests; the dead `transition_target` argument was dropped; Interface-contract docstrings |
| `builders.py` | `final_reply`, `tool_call_reply`, `malformed_reply`: the three `CompletionResponse` shapes | `kind` values from `fsm_llm.codes.CompletionKind`; Interface-contract docstrings |
| `__version__.py` | Re-export of `fsm_llm.__version__` | Same shape as every subpackage |

## Invariants and constraints

- Imports the core and `fsm_llm.codes` only; never `pytest`, `litellm` or another subpackage (`tests/test_fsm_llm_testing/test_layering.py` reads the source with `ast`).
- No library module imports it and `fsm_llm/__init__.py` never loads it; it is imported by name.
- Importing it has no side effect: no clock, no randomness, no socket.
- `ScriptedLLM` never answers an unscripted call with an invented reply: it records it in `unexpected`, counts an error and raises `UnscriptedCallError` (an `AssertionError`). The pipeline swallows a raised extraction error, so `assert_exhausted()` is what fails the test. It reuses core's private `_Metered`/`_meter_of` (one meter, not a second copy); a rename there must update `doubles.py`.
- Assertions (later steps) raise `AssertionError` explicitly, never `assert` (stripped by `python -O`).
- Names stay out of `src/constants.py`, `src/eval/cases.py` and `src/utilities.py`: they are bench-pinned (see the root `AGENTS.md`).
- Every public name gets a docstring with an interface contract (parameters, return shape, failure mode) and a static `__all__` entry; `docs/api.txt` is regenerated in the same commit.

## Testing

`tests/test_fsm_llm_testing/`. The `testing` extra is `[]`.
