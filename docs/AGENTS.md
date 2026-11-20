# docs

Path: `docs/`
Purpose: Markdown documentation for FSM-LLM v0.13.0 (landing page, quickstart, concepts, FSM design, handlers, synthesis, testing, API reference, command line, architecture, examples), the wrapper pages of the root project files and the published roadmaps. It is also the source of the documentation site.

## Scope

- In scope: the Markdown files of this folder, flat (no subfolder). FSM-LLM is one distribution (`llm-fsm`); the code lives in `src/fsm_llm/` and the extensions are subpackages: `fsm_llm.reasoning`, `fsm_llm.workflows`, `fsm_llm.agents`, `fsm_llm.monitor`, `fsm_llm.eval`, `fsm_llm.synthesis`, `fsm_llm.testing`.
- Out of scope: runnable code, per-package READMEs (they stay in `src/fsm_llm/<package>/`), `CHANGELOG.md`, `EVALUATE.md`, planning artifacts under `plans/`, the public API snapshot (the test fixture `tests/fixtures/api_snapshot.txt`, never documentation). By user preference, monitor docs belong in `api_reference.md`. Do not create a separate `docs/monitor.md`.
- Removed on purpose (step 123 of plan d454fd62, D-052, the user's instruction), never recreate under another name: `docs/reference/` (generated module pages), `docs/packages/` (includes of the package READMEs), `docs/upgrade-guide/` (the changelog is the migration record), `docs/decisions/` (the generated decision index), `docs/api.txt` and `docs/agents_roadmap.md`.

## Architecture

The files by kind:

| Group | Files | Header | Status |
| --- | --- | --- | --- |
| Current pages | `index.md`, `quickstart.md`, `concepts.md`, `fsm_design.md`, `handlers.md`, `synthesis.md`, `testing.md`, `api_reference.md`, `cli.md`, `architecture.md`, `examples.md` | `> Covers FSM-LLM v0.13.0` (matches `src/fsm_llm/__version__.py`; the form `v<version>` is pinned by `tests/test_fsm_llm_regression/test_release_version.py`) | Must match the code |
| Wrappers | `contributing.md`, `security.md`, `evaluate.md`, `changelog.md` | none | One `include-markdown` line each of the root file; never edit the wrapper |
| Roadmaps | `2026_roadmap.md`, and any other `*_roadmap.md` | none | Published as written: no front matter, no directive. `2026_roadmap.md` is user-owned: editable by an agent to keep a status line true or when asked, in the same published-as-written form |

Every page is in the `nav` of `mkdocs.yml` or in its `exclude_docs` (`AGENTS.md` and `/README.md`); `tests/test_fsm_llm/test_docs_site.py` fails on a page that is neither.

## Key files

| File | Role | Notes |
| --- | --- | --- |
| `index.md` | Landing page | What it is, install, the one runnable no-model example (a `ScriptedLLM`; `test_docs_content.py` runs it), where to go next, the subpackage table linking the READMEs by absolute GitHub URL |
| `quickstart.md` | First-run tutorial | Env var table (`LLM_MODEL`, `LLM_TEMPERATURE` default `0.5`, `LLM_MAX_TOKENS` default `1000`, `FSM_PATH`). Full FSM `friendly_greeter` in a Python block. Handler at `CONTEXT_UPDATE`, `converse_stream`, `FileSessionStore`. Section 6 "More Ways to Build": builders, an agent, a workflow, the reasoning engine; fences are plain `python` |
| `concepts.md` | Why, how it works, features, status, references | The mermaid diagrams, the subpackage table linking the READMEs on GitHub |
| `fsm_design.md` | Design guide | Only state-level JSON fragments, no full FSM. The priority rule, `evaluation_priority` (default 100, range 0-1000), `llm_description` (max 300 chars), `context_scope`, `handler_only_keys`, stacking merge strategies |
| `handlers.md` | Handler guide | The handler timings, `HandlerBuilder` table including `.critical()`, `handler_error_mode`, ERROR-handler rules (return dict not merged, re-entry raises `FSMError`) |
| `synthesis.md` | Guide | Runtime FSM synthesis. Holds the one full JSON FSM of the guides (loaded by the snippet test) and the live results of the bench blocks s0 to s2 (all INCONCLUSIVE, one model); update it together with `src/fsm_llm/synthesis/*.md` and `scripts/bench_data/README.md` when the bench runs |
| `testing.md` | Guide | `fsm_llm.testing` for users. Every `python` block is executed in order, in one namespace, offline, by `tests/test_fsm_llm_testing/test_guide.py`; a block that must not run is fenced `python notest` |
| `api_reference.md` | Reference | Hand written: the public surface of the core and of every subpackage, exceptions, constants. A signature, default or name changes here in the same commit as the code |
| `cli.md` | Command line | One `## fsm-llm-<tool>` section per console script of `pyproject.toml`, its entry point, `python -m fsm_llm.reasoning`, `--model`, `--env-file`, exit codes (`tests/test_fsm_llm/test_docs_content.py` compares it with `[project.scripts]`: add the section when you add a script) |
| `architecture.md` | System design | The 2-pass turn, the message-free step, the layer table and the contracts of `.importlinter` with the named debts, state kinds and executors, extension points |
| `examples.md` | Examples table | Absolute GitHub URLs (`https://github.com/NikolasMarkou/fsm_llm/tree/main/examples/...`): the examples are not part of the site and a test checks every linked folder exists |
| `contributing.md`, `security.md`, `evaluate.md`, `changelog.md` | Wrappers | `changelog.md` also has the Material front matter `search: exclude: true`. A relative link in a root file is rewritten against the included file, so a link to a file outside `docs/` must be an absolute GitHub URL |
| `README.md` | Folder index on GitHub | Page table, reading order, how the site is published (owner steps, risk), how to build it; not part of the site |

## Public interface

Other files read or point to these docs:

- `tests/test_fsm_llm/test_docs_snippets.py`: loads every fenced `json` or `python` block that contains `"initial_state"` in the pages through `FSMDefinition`. The regex is ```` ```(json|python)\n(.*?)``` ````; for Python it takes the first dict literal with `initial_state` found through `ast`, so the dict must be a pure literal. `quickstart.md` must yield at least one such snippet.
- `tests/test_fsm_llm/test_docs_site.py`, `test_docs_content.py`, `test_docs_links.py`: the site configuration, the page set and the nav groups, the wrappers, the roadmaps, the CLI page, the landing example, and every relative link and anchor.
- `tests/test_fsm_llm_testing/test_guide.py`: executes every `python` block of `testing.md`.
- `tests/test_fsm_llm_regression/test_regression_cli_and_exports.py` (`TestQuickstartReferences`): `quickstart.md` must not contain `examples/basic/quiz`, `examples/intermediate/customer_service` or `python main.py`.
- `src/fsm_llm/engine/fsm/manager.py` points to `docs/architecture.md`; a comment in `engine/fsm/turns.py` quotes `docs/handlers.md` on ERROR handlers.

## Data shapes

A full FSM snippet has to pass the loader. Minimum per state: `id`, `description`, `purpose`. Top level: `name`, `description`, `initial_state`, `states`. Classification entries hold `intents` (at least 2) and a `fallback_intent` that is one of them, directly on the entry with no nested `schema`. Gating uses a transition `conditions` entry with `description`, `requires_context_keys`, `logic`.

Facts the pages state that must stay in step with the code:

- `API.__init__` defaults: `max_history_size=5`, `max_message_length=1000`, `handler_error_mode="continue"`, `max_fsm_cache_size=64`, keyword-only `executors=None` (internal, no builder setter).
- `DEFAULT_MAX_STACK_DEPTH = 10`; the core has no default model: a missing model raises `ModelNotConfiguredError`; `DEFAULT_TOOL_MODEL` (`ollama_chat/bonsai-2:27b`) is the last tier of `fsm-llm-eval` and the monitor launch routes only.
- Core step driver: `API.advance`, `advance_stream`, `run_until_terminal(conversation_id, *, max_steps, max_seconds=None, before_step=None, seconds_exempt_states=())`, `run_until_terminal_stream`, `AdvanceResult`, `RunBudgetExceededError`.
- Console scripts (`pyproject.toml`): `fsm-llm`, `fsm-llm-visualize`, `fsm-llm-validate`, `fsm-llm-monitor`, `fsm-llm-meta`, `fsm-llm-eval`, `fsm-llm-okf`, `fsm-llm-skills`.

## Invariants and constraints

- `required_context_keys` never blocks a transition. Every page gates with a condition. New examples must do the same.
- Among passing transitions, the single lowest `priority` wins. Only a tie at the lowest priority is AMBIGUOUS and goes to the LLM classifier.
- Keep the `> Covers FSM-LLM v<version>` header on every current page and update it on release (`test_release_version.py` compares it with `pyproject.toml`).
- No plan id and no step number in a page, no emoji, no em dash (a ratchet test counts them), no stale path, no count of tests, files, modules, lines, contracts or examples (a review rule; `mkdocs build --strict` catches a broken link).
- A page has a title, a one-line purpose, the `Covers` line, sections and a `Next steps` section. No topic is written twice: link instead.
- `2026_roadmap.md` is the maintainer's: edit it only to keep a status line true or when asked, never move or rename it.

## Dependencies

No runtime dependencies. The snippets import from `fsm_llm` and from `fsm_llm.agents`, `fsm_llm.reasoning`, `fsm_llm.workflows`, `fsm_llm.eval`, `fsm_llm.monitor`, `fsm_llm.synthesis`, `fsm_llm.testing`. The monitor snippet also uses `uvicorn`. The site tools are the `docs` dependency group of `pyproject.toml`, pinned in `constraints.txt`.

## Failure modes

No known mismatches between these pages and the code. When one is found, list it here until it is fixed.

## Working here

- State no count of tests, files, modules, lines, contracts or examples: measure it when it is needed. Before stating a signature or default, check it against `src/fsm_llm/`.
- After any edit, run `.venv/bin/python -m pytest tests/test_fsm_llm/test_docs_snippets.py tests/test_fsm_llm/test_docs_site.py tests/test_fsm_llm/test_docs_content.py tests/test_fsm_llm/test_docs_links.py tests/test_fsm_llm_regression/test_regression_cli_and_exports.py`, and a strict build with `make docs` when the docs tools are installed.
- A new full FSM example must include `description` on the FSM and on every state, `id` and `purpose` on every state, and a reachable terminal state.
- Renaming or deleting a page means updating `mkdocs.yml`, the tests above and the pointers in `src/fsm_llm/engine/fsm/manager.py`.
- Example paths cited here must exist under `examples/`. Do not edit `examples/` to fit the docs: they are evaluation baselines.
