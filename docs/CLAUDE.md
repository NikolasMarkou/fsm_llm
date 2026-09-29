# docs

Path: `docs/`
Purpose: Long-form Markdown guides for FSM-LLM v0.11.0 (quickstart, FSM design, handlers, architecture, API reference) plus three historical Strands design records.

## Scope

- In scope: 8 Markdown files at `docs/` with no subdirectories. FSM-LLM is one distribution (`fsm-llm`). The code lives in `src/fsm_llm/`, and the extensions are subpackages: `fsm_llm.reasoning`, `fsm_llm.workflows`, `fsm_llm.agents`, `fsm_llm.monitor`, `fsm_llm.harness`, `fsm_llm.eval`.
- Out of scope: runnable code, per-package READMEs, `CHANGELOG.md`, `EVALUATE.md`, planning artifacts under `plans/`. By user preference, monitor docs belong in `api_reference.md`. Do not create a separate `docs/monitor.md`.

## Architecture

Two groups of files:

| Group | Files | Header | Status |
| --- | --- | --- | --- |
| Current guides | `quickstart.md`, `fsm_design.md`, `handlers.md`, `architecture.md`, `api_reference.md` | `> Covers FSM-LLM v0.11.0` (matches `src/fsm_llm/__version__.py`) | Must match the code |
| Historical records | `strands_features.md`, `strands_features_phase_1.md`, `strands_features_phase_2.md` | `> **Historical design record.** ...` blockquote | Frozen snapshots. The blockquote says the counts are from the time of writing |

Links between the docs: `fsm_design.md` ends with `**Next:** [Handler Development](./handlers.md)`. Each Strands file links to `api_reference.md`.

## Key files

| File | Role | Notes |
| --- | --- | --- |
| `quickstart.md` | First-run tutorial | Env var table (`LLM_MODEL`, `LLM_TEMPERATURE` default `0.5`, `LLM_MAX_TOKENS` default `1000`, `FSM_PATH`). Full FSM `friendly_greeter` in a Python block. Handler example at `CONTEXT_UPDATE`. `converse_stream`, `FileSessionStore`. Example commands point at `examples/basic/form_filling`, `examples/basic/story_time`, `examples/intermediate/book_recommendation` |
| `fsm_design.md` | Design guide | Only state-level JSON fragments, no full FSM. Covers the priority rule, `evaluation_priority` (default 100, range 0-1000), `llm_description` (max 300 chars), `context_scope.read_keys`/`write_keys`, `handler_only_keys`, correction provenance with `<rejected_corrections>`, stacking merge strategies, WorkingMemory reach |
| `handlers.md` | Handler guide | 8 timings, `HandlerBuilder` table including `.critical()`, `error_mode`, `HandlerSystem.handlers_at` fast path, ERROR-handler rules (return dict not merged, re-entry raises `FSMError`) |
| `architecture.md` | System design | Layer diagram, component list, message, start and stream flows, `FSMContext` fields, special `_` keys, stacked-FSM merge, extension table, a long harness section (gates read from disk, driver vs worker, safety bounds table), extension points |
| `api_reference.md` | Reference | `API(...)` signature, session, query, stacking methods, `LLMInterface`, `WorkingMemory`, `TransitionEvaluatorConfig`, classification, reasoning, agents, workflows, harness, eval (CLI flags, exit codes, `EvalConfig` table, dataset schema, output layout), monitor, exception tree, constants |
| `strands_features.md` | Historical | 12 features to adapt, plus "Features NOT Recommended" |
| `strands_features_phase_1.md` | Historical | 4 delivered features, commit `a7e3d88`, test counts from that time |
| `strands_features_phase_2.md` | Historical | 8 features with estimated LOC and "Not yet implemented" notes from that time |

## Public interface

Other files read or point to these docs:

- `tests/test_fsm_llm/test_docs_snippets.py`: `_REFERENCE_DOCS` holds `docs/api_reference.md`, `docs/architecture.md`, `docs/fsm_design.md`, `docs/handlers.md`. `docs/quickstart.md` is in `_FULL_FSM_DOCS`. The test regex is ```` ```(json|python)\n(.*?)``` ````. Any block that contains `"initial_state"` must load through `FSMDefinition(**definition)`. For JSON the whole block is parsed. For Python it takes the first dict literal with `initial_state`, found through `ast`, so the dict must be a pure literal. `quickstart.md` must yield at least one such snippet. Every `_REFERENCE_DOCS` file must exist.
- `tests/test_fsm_llm_regression/test_regression_cli_and_exports.py` (`TestQuickstartReferences`): `quickstart.md` must not contain `examples/basic/quiz`, `examples/intermediate/customer_service` or `python main.py`.
- `src/fsm_llm/fsm.py`: the module docstring points to `docs/architecture.md`. A comment near line 617 quotes `docs/handlers.md` on ERROR handlers.

## Data shapes

A full FSM snippet has to pass the loader. Minimum per state: `id`, `description`, `purpose`. Top level: `name`, `description`, `initial_state`, `states`. Classification entries hold `intents` (at least 2) and a `fallback_intent` that is one of them, directly on the entry with no nested `schema`. Gating uses a transition `conditions` entry with `description`, `requires_context_keys`, `logic`.

Facts the guides state that must stay in step with the code:

- `API.__init__` parameters and defaults: `max_history_size=5`, `max_message_length=1000`, `handler_error_mode="continue"`, `max_fsm_cache_size=64`, `**llm_kwargs`.
- `DEFAULT_LLM_MODEL = "ollama_chat/qwen3.5:4b"`, `DEFAULT_MAX_STACK_DEPTH = 10`.
- `TransitionEvaluatorConfig` fields: `ambiguity_threshold`, `minimum_confidence`, `strict_condition_matching`, `evidence_conditions_normalizer`, `detailed_logging`.
- `fsm_llm.harness.CHECKS` has 30 entries.
- Console scripts (`pyproject.toml`): `fsm-llm`, `fsm-llm-visualize`, `fsm-llm-validate`, `fsm-llm-monitor`, `fsm-llm-meta`, `fsm-llm-harness`, `fsm-llm-eval`.

## Invariants and constraints

- `required_context_keys` never blocks a transition. Every guide gates with a condition. New examples must do the same.
- Among passing transitions, the single lowest `priority` wins. Only a tie at the lowest priority is AMBIGUOUS and goes to the LLM classifier. The guides already say this; keep it that way.
- Keep the `> Covers FSM-LLM v<version>` header on the five current guides and update it on release.
- Do not rewrite the Strands files to current facts. They are records. Their header already warns that they are dated.
- Style follows the repo: plain language, `--` used as a dash in existing text, no emojis.

## Dependencies

No runtime dependencies. The snippets import from `fsm_llm` and from `fsm_llm.agents`, `fsm_llm.reasoning`, `fsm_llm.workflows`, `fsm_llm.harness`, `fsm_llm.eval`, `fsm_llm.monitor`. The monitor snippet also uses `uvicorn`.

## Failure modes

No known mismatches between these docs and the code. When one is found, list it here until it is fixed.

## Working here

- Before stating a count, signature or default, check it against `src/fsm_llm/` (for example `.venv/bin/python -c "import fsm_llm.harness as h; print(len(h.__all__))"`).
- After any edit, run:
  `.venv/bin/python -m pytest tests/test_fsm_llm/test_docs_snippets.py tests/test_fsm_llm_regression/test_regression_cli_and_exports.py`
- A new full FSM example must include `description` on the FSM and on every state, `id` and `purpose` on every state, and a reachable terminal state.
- Renaming or deleting a current guide means updating `_REFERENCE_DOCS` or `_FULL_FSM_DOCS` in `test_docs_snippets.py` and the pointers in `src/fsm_llm/fsm.py`.
- Example paths cited here must exist under `examples/`. Do not edit `examples/` to fit the docs: they are evaluation baselines.
