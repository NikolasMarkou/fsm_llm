# docs

Path: `docs/`
Purpose: Long-form Markdown guides for FSM-LLM v0.12.0 (quickstart, FSM design, handlers, architecture, API reference, runtime FSM synthesis) plus the two dated roadmaps and the pages of the documentation site.

## Scope

- In scope: the Markdown files of the groups below and one generated text file (`api.txt`, the public API snapshot written by `scripts/api_snapshot.py`; never edit it by hand, run `make api-snapshot`) at `docs/`, and four subdirectories: `decisions/`, holding the generated decision index `decisions/index.md` (written by `scripts/decision_index.py`; never edit it by hand, run `make decision-index`), `reference/` (one mkdocstrings stub per module of `scripts/api_snapshot.py` `MODULES`, named `<dotted.module>.md`, plus `api-snapshot.md` and `cli.md`) `packages/` (one page per subpackage README, an include of `src/fsm_llm/<pkg>/README.md`) and `upgrade-guide/` (the hand-written migration guides of a release, `upgrade-guide/v<version>/<theme>.md`, one `##` heading per `**BREAKING:**` bullet of `CHANGELOG.md`, each with Before and After code). FSM-LLM is one distribution (`llm-fsm`). The code lives in `src/fsm_llm/`, and the extensions are subpackages: `fsm_llm.reasoning`, `fsm_llm.workflows`, `fsm_llm.agents`, `fsm_llm.monitor`, `fsm_llm.eval`, `fsm_llm.synthesis`.
- Out of scope: runnable code, per-package READMEs, `CHANGELOG.md`, `EVALUATE.md`, planning artifacts under `plans/`. By user preference, monitor docs belong in `api_reference.md`. Do not create a separate `docs/monitor.md`.

## Architecture

Three groups of files:

| Group | Files | Header | Status |
| --- | --- | --- | --- |
| Current guides | `quickstart.md`, `fsm_design.md`, `handlers.md`, `architecture.md`, `api_reference.md`, `synthesis.md`, `testing.md` | `> Covers FSM-LLM v0.12.0` (matches `src/fsm_llm/__version__.py`) | Must match the code |
| Roadmaps | `2026_roadmap.md`, `agents_roadmap.md` | none | Dated records, published as written. `2026_roadmap.md` is user-owned: never edit it. `agents_roadmap.md` keeps its counts as recorded on their dates |
| Site pages | `index.md`, `concepts.md`, `examples.md`, the four wrappers, `reference/`, `packages/`, `decisions/index.md`, `upgrade-guide/` | none | Described in the key files below; every one is in the `nav` of `mkdocs.yml` |

Links between the docs: `fsm_design.md` ends with `**Next:** [Handler Development](./handlers.md)`.

## Key files

| File | Role | Notes |
| --- | --- | --- |
| `reference/` | Generated reference pages | Each `<dotted.module>.md` is a title and one `::: <dotted.module>` block (mkdocstrings reads `src/` statically; `docstring_style: null`); the set equals `MODULES` of `scripts/api_snapshot.py` (`tests/test_fsm_llm/test_docs_reference.py`). `api-snapshot.md` puts `api.txt` in a `text` fence through `include-markdown` (`../api.txt`, so the file must hold no triple backtick). `cli.md` has one `## fsm-llm-<tool>` section per console script of `pyproject.toml`: add the section when you add a script. Nothing here is edited to change an API: change the code and run `make api-snapshot` |
| `packages/` | Package pages | One page per `src/fsm_llm/<pkg>/README.md`, one include line each; the README is the source, edit it there. A README that a page includes must not carry a relative link or image (it would resolve against the page) |
| `quickstart.md` | First-run tutorial | Env var table (`LLM_MODEL`, `LLM_TEMPERATURE` default `0.5`, `LLM_MAX_TOKENS` default `1000`, `FSM_PATH`). Full FSM `friendly_greeter` in a Python block. Handler example at `CONTEXT_UPDATE`. `converse_stream`, `FileSessionStore`. Section 6 "More Ways to Build" (moved from the README Quick Start 3 to 6 at step 105: builders, an agent, a workflow, the reasoning engine; fences are plain `python`, never a bare word after the language). Example commands point at `examples/basic/form_filling`, `examples/basic/story_time`, `examples/intermediate/book_recommendation` |
| `fsm_design.md` | Design guide | Only state-level JSON fragments, no full FSM (the `synthesize` section too). Covers the priority rule, `evaluation_priority` (default 100, range 0-1000), `llm_description` (max 300 chars), `context_scope.read_keys`/`write_keys`, `handler_only_keys`, correction provenance with `<rejected_corrections>`, stacking merge strategies, WorkingMemory reach |
| `handlers.md` | Handler guide | 8 timings, `HandlerBuilder` table including `.critical()`, `handler_error_mode`, `HandlerSystem.handlers_at` fast path, ERROR-handler rules (return dict not merged, re-entry raises `FSMError`) |
| `architecture.md` | System design | Layer diagram, component list, message, message-free step, start and stream flows, `FSMContext` fields, special `_` keys, stacked-FSM merge, extension table, extension points |
| `api_reference.md` | Reference | `API(...)` signature, steps without a user message (`advance`, `run_until_terminal`), session, query, stacking methods, visualization, `LLMInterface`, `WorkingMemory`, `TransitionEvaluatorConfig`, classification, reasoning, agents, workflows, eval (CLI flags, exit codes, `EvalConfig` table, dataset schema, output layout), monitor, Open Knowledge Format (core API, `fsm-llm-okf`, agents API, safety model), Agent Skills (core API, `fsm-llm-skills`, agents tools, safety model, known limits), the `State.synthesize` field, Synthesis (compile and reuse, library, runner, agent tools, traces and repair, safety model, known limits), Testing (`fsm_llm.testing`: doubles, scripted user, assertions, `PathRecorder`), exception tree, constants |
| `synthesis.md` | Guide | Runtime FSM synthesis: concierge pattern, `FSMSynthesizer`, gates G1 to G7, `FSMLibrary`, `ProcedureRunner`, agent tools, approval, traces and repair, limits, safety model, known limits. Holds the one full JSON FSM of the reference docs (a `json` block with `"initial_state"`, loaded by the snippet test) and a Python block that reads it from `concierge.json`. Holds the live results of blocks s0 to s2 (all INCONCLUSIVE, one model) and claims no win for synthesis; update it together with `src/fsm_llm/synthesis/*.md` and `scripts/bench_data/README.md` when the bench runs |
| `testing.md` | Guide | `fsm_llm.testing` for users: the doubles and when each fits, `ScriptedUser`, the four assertions, `PathRecorder` (deep-copy caveat), pytest notes, migration from a copied `tests/conftest.py`. Holds two small FSMs as Python dict literals (loaded by the snippet test) and every `python` block is executed in order, in one namespace, offline by `tests/test_fsm_llm_testing/test_guide.py`; a block that must not run is fenced `python notest`. Update it together with `src/fsm_llm/testing/README.md` and `docs/api_reference.md` |
| `concepts.md` | Site page | Standalone prose copied from the README sections Why, How It Works, Key Features, Status, Acknowledgments and References (the README text moved out at step 105 of plan d454fd62, so this page is the one copy; the README keeps only the intro, install, two Quick Starts and links). Two mermaid diagrams, the subpackage table linking `packages/*.md`. No count token of the shapes `scripts/doc_counts.py` pins (`N tests`, `N test files`) |
| `examples.md` | Site page | The examples table with absolute GitHub URLs (`https://github.com/NikolasMarkou/fsm_llm/tree/main/examples/...`): the examples are not part of the site and a test checks every linked folder exists |
| `contributing.md`, `security.md`, `evaluate.md`, `changelog.md` | Wrapper pages | One `include-markdown` line each of the root file (`../CONTRIBUTING.md`, `../SECURITY.md`, `../EVALUATE.md`, `../CHANGELOG.md`); `changelog.md` also has the Material front matter `search: exclude: true` (the changelog is the largest page). Edit the root file, never the wrapper. A relative link in a root file is rewritten against the included file, so `tests/test_fsm_llm/test_docs_links.py` does not judge it; the strict build does, and a link to a file outside `docs/` must be an absolute GitHub URL |
| `2026_roadmap.md`, `agents_roadmap.md` | Roadmaps | Published as written (no front matter, no directive; `test_docs_content.py`). `2026_roadmap.md` is user-owned: never edit it |
| `index.md` | Site landing page | The home page of the mkdocs site (`mkdocs.yml`, `make docs`): a short intro and links to the guides, the decision index and the roadmaps. Not a copy of the repository README. Every page of this folder must be in the `nav` of `mkdocs.yml` or in its `exclude_docs` (`tests/test_fsm_llm/test_docs_site.py`); `AGENTS.md`, `api.txt` and `README.md` are excluded |
| `decisions/index.md` | Generated | The decision index: every `# DECISION` anchor of the code grouped by plan, with the manifest date and rationale, the places (path and enclosing definition, no line numbers) and a Retired section for `[STALE]` anchors. GENERATED by `make decision-index` from the anchors and `plans/ANCHORS.md`; never edit it by hand. `make decision-check` (and a default-suite test) fails when it is stale or an anchor has no manifest line, so regenerate it in the same commit as any anchor change |
| `upgrade-guide/` | Hand-written migration guides | `v<version>/<theme>.md` (today `v0.13.0/`: `distribution-rename`, `python-floor`, `default-model-and-env`, `module-paths`, `tools-and-policy`, `monitor-routes`, `core-renames`, `definition-version`, `workflows-vocabulary`, `agents-renames`, `monitor-and-reasoning`, `removals`) plus the overview page `upgrade-guide/index.md` (no `Covers` line: it lists the guides in the order to migrate). A page opens `# Title` and `Covers FSM-LLM <old> to <new>.`, then one `##` heading per `**BREAKING:**` bullet of the root `CHANGELOG.md` (the bullet links `docs/upgrade-guide/v<version>/<theme>.md#<slug>`; `scripts/changelog_check.py` reads the slug rule, keep headings plain ASCII). A python fence is introduced by the bold word `**Before**` (old code, may name a removed path), `**After**` (new code, RUN by `tests/test_fsm_llm/test_upgrade_guides.py` with no model: keep it offline and self-contained) or `**Sketch**` (parsed and import-checked only); fence info strings are bare (`python`, `bash`, `text`, `json`). The guides name old paths and counts on purpose (`scripts/doc_counts.py` HISTORICAL). Every page is in the `nav` |

## Public interface

Other files read or point to these docs:

- `tests/test_fsm_llm/test_docs_snippets.py`: `_REFERENCE_DOCS` holds `docs/api_reference.md`, `docs/architecture.md`, `docs/fsm_design.md`, `docs/handlers.md`, `docs/synthesis.md`, `docs/testing.md`. `docs/quickstart.md` is in `_FULL_FSM_DOCS`. The test regex is ```` ```(json|python)\n(.*?)``` ````. Any block that contains `"initial_state"` must load through `FSMDefinition(**definition)`. For JSON the whole block is parsed. For Python it takes the first dict literal with `initial_state`, found through `ast`, so the dict must be a pure literal. `quickstart.md` must yield at least one such snippet. Every `_REFERENCE_DOCS` file must exist.
- `tests/test_fsm_llm_testing/test_guide.py`: executes every ```` ```python ```` block of `docs/testing.md` in document order in one namespace (a later block uses the names of an earlier one), offline, and checks that every name in `fsm_llm.testing.__all__` appears in the guide.
- `tests/test_fsm_llm_regression/test_regression_cli_and_exports.py` (`TestQuickstartReferences`): `quickstart.md` must not contain `examples/basic/quiz`, `examples/intermediate/customer_service` or `python main.py`.
- `src/fsm_llm/engine/fsm/manager.py`: the module docstring points to `docs/architecture.md`. A comment in `engine/fsm/turns.py` (`_process_message_locked`) quotes `docs/handlers.md` on ERROR handlers.

## Data shapes

A full FSM snippet has to pass the loader. Minimum per state: `id`, `description`, `purpose`. Top level: `name`, `description`, `initial_state`, `states`. Classification entries hold `intents` (at least 2) and a `fallback_intent` that is one of them, directly on the entry with no nested `schema`. Gating uses a transition `conditions` entry with `description`, `requires_context_keys`, `logic`.

Facts the guides state that must stay in step with the code:

- `API.__init__` parameters and defaults: `max_history_size=5` (exchanges stored; the last N messages shown in the prompts, within the history token budget of 3000), `max_message_length=1000`, `handler_error_mode="continue"`, `max_fsm_cache_size=64`, keyword-only `executors=None` (internal, no builder setter), `**llm_kwargs`.
- `DEFAULT_MAX_STACK_DEPTH = 10` (there is no default model constant: a missing model raises `ModelNotConfiguredError`).
- `TransitionEvaluatorConfig` fields: `strict_condition_matching`, `detailed_logging` (the three no-op confidence fields were removed).
- Core step driver: `API.advance`, `advance_stream`, `run_until_terminal(conv_id, *, max_steps, max_seconds=None, before_step=None)`, `run_until_terminal_stream`, `AdvanceResult`, `RunBudgetExceededError`; a silent state makes no call, returns `""` and adds nothing to history. Graph export: `build_fsm_graph`, `to_mermaid`, `to_dot`, `fsm-llm-visualize --format`.
- Console scripts (`pyproject.toml`): `fsm-llm`, `fsm-llm-visualize`, `fsm-llm-validate`, `fsm-llm-monitor`, `fsm-llm-meta`, `fsm-llm-eval`, `fsm-llm-okf`, `fsm-llm-skills`.

## Invariants and constraints

- `required_context_keys` never blocks a transition. Every guide gates with a condition. New examples must do the same.
- Among passing transitions, the single lowest `priority` wins. Only a tie at the lowest priority is AMBIGUOUS and goes to the LLM classifier. The guides already say this; keep it that way.
- Keep the `> Covers FSM-LLM v<version>` header on the seven current guides and update it on release.
- Do not rewrite `agents_roadmap.md` to current facts: it is a dated record. `2026_roadmap.md` is the maintainer's: never edit it.
- Style follows the repo: plain language, `--` used as a dash in existing text, no emojis.

## Dependencies

No runtime dependencies. The snippets import from `fsm_llm` and from `fsm_llm.agents`, `fsm_llm.reasoning`, `fsm_llm.workflows`, `fsm_llm.eval`, `fsm_llm.monitor`, `fsm_llm.synthesis`. The monitor snippet also uses `uvicorn`.

## Failure modes

No known mismatches between these docs and the code. When one is found, list it here until it is fixed.

## Working here

- Before stating a count, signature or default, check it against `src/fsm_llm/` (for example `.venv/bin/python -c "import fsm_llm.eval as e; print(len(e.__all__))"`).
- After any edit, run:
  `.venv/bin/python -m pytest tests/test_fsm_llm/test_docs_snippets.py tests/test_fsm_llm/test_docs_site.py tests/test_fsm_llm_regression/test_regression_cli_and_exports.py`
- A new full FSM example must include `description` on the FSM and on every state, `id` and `purpose` on every state, and a reachable terminal state.
- Renaming or deleting a current guide means updating `_REFERENCE_DOCS` or `_FULL_FSM_DOCS` in `test_docs_snippets.py` and the pointers in `src/fsm_llm/engine/fsm/manager.py`.
- Example paths cited here must exist under `examples/`. Do not edit `examples/` to fit the docs: they are evaluation baselines.
