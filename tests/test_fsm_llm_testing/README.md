# test_fsm_llm_testing

The pytest suite for `fsm_llm.testing`, the public test-helper subpackage of the FSM-LLM repository. It lives at `tests/test_fsm_llm_testing/`.

## What it is for

`fsm_llm.testing` will hold the fake LLM interfaces, the scripted user and the assertions that other projects use to test code built on `fsm_llm`. Today it checks the contract of the doubles (promoted and scripted), the reply builders, `state_in_prompt`, `hash_embed`, the scripted user and its `ConversationRun` record, the assertions and `PathRecorder`, and the layering: the module is imported by name, no library module imports it, it imports no `pytest`, `litellm` or other subpackage, and `import fsm_llm` does not load it. Tests for each public name are added together with the name.

## How it works

- Import probes run in a child process (`python -I`) with the model-server variables pointed at a closed port, so a leak is seen in a clean `sys.modules`.
- Source checks read every module with `ast`, so a stray import is found without running it.

## Files

- `__init__.py` - empty package marker.
- `test_doubles.py` - the contract of `MockLLM2Interface` and `PromptGroundedLLM` (mutable attributes, `call_history.clear()`, subclassing, cut flags, evidence grounding, bulk quoting, `complete` raising), a guard that no test module imports the doubles from `tests.conftest`, the dead `transition_target` scan, a child process with every socket refused.
- `test_scripted_llm.py` - the contract of `ScriptedLLM`: queue, extraction, per-state replies, streaming, `unexpected` and `assert_exhausted()` (also under `python -O`), metered usage with errors, and a pipeline test that a swallowed unscripted extraction call stays detectable.
- `test_builders.py` - `final_reply`, `tool_call_reply`, `malformed_reply`, `state_in_prompt` (one regex, drift test against the real prompt builder) and `hash_embed`.
- `test_user.py` - `ScriptedUser` (messages, rules, fallback, `max_turns`, stopping on an ended conversation, `close`, propagating exceptions, argument checks), `ConversationRun` (frozen, path equal to `fsm_llm.eval` `visited_states`) and `advance_until_terminal`, on the `simple_greeting` and `form_filling` example FSM files (read only) and inline FSMs.
- `test_assertions.py` - `assert_state`, `assert_context`, `assert_path`, `assert_result` (pass, first-divergence message wording, typo `ValueError`, sentinel semantics of `reason`), the explicit-raise guarantees (no `assert` statement, hidden frames, a `python -O` child) and `PathRecorder` (path of an example FSM, stays, pushed frames, read-only behaviour, the deep-copy side effect observed).
- `test_guide.py` - runs every Python block of `docs/testing.md` offline, in order, in one namespace, and checks the guide names every public helper and keeps its promised notes.
- `test_layering.py` - the import probes, the library-wide scan for imports of `fsm_llm.testing`, the `pytest` and `litellm` ban, the no-other-subpackage rule, and the skeleton checks (docstring, one static `__all__`).

## How to use it

```bash
.venv/bin/python -m pytest tests/test_fsm_llm_testing/
```

## Things to know

- Run from the repository root. No test calls a model or a network.
- No document states this suite's test count; measure it with `.venv/bin/python -m pytest tests/test_fsm_llm_testing --collect-only -q | tail -1`.
