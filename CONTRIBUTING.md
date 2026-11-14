# Contributing to FSM-LLM

Contributions are welcome, from a typo fix to a new agent pattern. Report vulnerabilities as described in [SECURITY.md](https://github.com/NikolasMarkou/fsm_llm/blob/main/SECURITY.md).

## Development setup

Work inside the project virtualenv. Every command below assumes `.venv/bin` is on `PATH` (or call `.venv/bin/python` directly).

```bash
python -m venv .venv
source .venv/bin/activate
make install-dev    # pip install -c constraints.txt -e ".[dev,workflows,reasoning,agents,monitor,eval,synthesis,testing]" + pre-commit install
```

Upgrading a clone from before the 2026-09-29 restructure (extensions moved under `fsm_llm/`): delete the old `src/fsm_llm_<sub>/` directories by hand (`make clean` no longer removes them), then reinstall with `pip install -e .`. After the package moved back to `src/fsm_llm/` (plan d454fd62, step 62), reinstall with `pip install -e .` again; the old editable install pointed at the flat `src/`. A leftover directory imports as a namespace package.

`constraints.txt` pins every direct dependency to one exact version, litellm to a verified-safe release (1.82.7 and 1.82.8 were compromised; `pyproject.toml` excludes them too). Run `make audit` after installing new packages; it scans site-packages for suspicious `.pth` files.

## Tests, lint and types

```bash
make install-dev    # editable install with every extra, plus pre-commit hooks
make test           # Run full test suite (14,043 tests)
make lint           # ruff check src/ tests/
make format         # ruff format src/ tests/
make type-check     # mypy on the fsm_llm package (src/fsm_llm/)
make lint-imports   # import contracts in .importlinter (the Section 5.2 layers, kernel, litellm, testing, pytest, entry modules, ...)
make audit          # scan site-packages for malicious .pth files
make counts         # fail when a test count written in a .md file differs from a collection
make build          # wheel and sdist
pytest tests/test_fsm_llm/            # one suite (core)
pytest -m "not slow"                  # skip the slow tests
```

- The default suite mocks the LLM: it needs no network and no model. Live tests need a reachable Ollama and skip without one.
- CI (`.github/workflows/python-package.yml`) runs a `lint` job once (ruff lint and format check, mypy, the `.pth` audit, the import contracts, the decision-anchor gate and the workflow linters `actionlint` and `zizmor`), a `supply-chain` job (`pip-audit --skip-editable` over the installed environment, plus a canary that must see `pip-audit` fail on `tests/fixtures/vulnerable-requirements.txt`: never install or "fix" that file), a `dependency-review` job on pull requests (`actions/dependency-review-action`, fails on a high advisory) and a `test` matrix of nine legs: Linux on Python 3.11 to 3.15, macOS and Windows on 3.11 and 3.15. Every leg runs the `.pth` audit and pytest without the `slow`, `real_llm` and `integration` markers; the Linux 3.12 leg also measures branch coverage and runs the ratchet (`scripts/coverage_ratchet.py`). The Windows legs run `python -m` commands in bash, never `make`. Check an edit to a workflow with `uvx --from actionlint-py actionlint .github/workflows/*.yml` and `uvx --from zizmor zizmor --offline .github/workflows` (the lint job runs both at the pins of `constraints.txt`); every `uses:` takes a 40-hex SHA with its release tag in a comment, in every workflow file; `tests/test_fsm_llm_regression/test_ci_workflow.py` pins the matrix.
- Pre-commit runs ruff (lint with `--fix`, and format) and the file hygiene hooks on commit, and a quick pytest run on push.
- A behaviour change needs a test that fails on the code before the change. Run it against the parent commit to confirm.
- Test counts are written out as literals in `AGENTS.md` and in `CONTRIBUTING.md` (the `make test` line above; the README is the PyPI page and holds no count), and `tests/test_packaging.py` checks them against a fresh collection. When you add or remove tests, re-measure with `pytest --collect-only -q | tail -1` (and per suite) and update every literal. Never guess or adjust a count by hand.
- `tests/test_fsm_llm/test_docs_snippets.py` loads every fenced block that names `"initial_state"` in the first-touch docs. A full FSM snippet in the docs must be a valid FSM.
- Naming for new code (ids, units, constants, enums, builders, exceptions, verbs) follows the `Naming conventions` section of [AGENTS.md](https://github.com/NikolasMarkou/fsm_llm/blob/main/AGENTS.md).
- litellm is imported in exactly one module, `src/fsm_llm/llm/providers/litellm.py`. Do not add a second request builder or provider call elsewhere.
- Internal-key and secret-key checks go through `fsm_llm.constants.has_internal_prefix` and `is_forbidden_context_entry`. Do not re-inline them.
- Anything new that is configured with many settings gets a step-by-step builder that follows the convention in [src/fsm_llm/AGENTS.md](https://github.com/NikolasMarkou/fsm_llm/blob/main/src/fsm_llm/AGENTS.md): setters only record, `build()` is the only validator.
- Commit messages follow `<type>(<scope>): <summary>`.

## Project structure

The repository is one distribution (`fsm-llm`, setuptools, src layout). The directory `src/fsm_llm/` is the import package `fsm_llm` (found by `packages.find` in `pyproject.toml`), so `src/fsm_llm/engine/api/conversation.py` is `fsm_llm.engine.api.conversation`, and a subpackage is imported by name, for example `from fsm_llm.agents import create_agent`.

```
fsm_llm/
├── src/fsm_llm/      # THE LIBRARY (the import package)
│   ├── engine/api/, engine/fsm/,   # API, FSMManager, MessagePipeline (the 2-pass turn)
│   │   engine/turn/, engine/kinds/
│   ├── core/definitions/      # Pydantic FSM models and core exceptions (a package)
│   ├── core/transition_evaluator.py, core/expressions/   # JsonLogic evaluation, no LLM
│   ├── core/verify/, core/visualize/   # Validator, data-flow and liveness analysis; diagrams and graph export
│   ├── handlers/              # Handler system, HandlerBuilder, create_handler
│   ├── llm/                   # LLMInterface and LiteLLMInterface (llm/providers/litellm.py is the only litellm import), reply parsing
│   ├── context/               # Context cleaning, compaction, filtering and the prompt builders (context/layout/)
│   ├── engine/classification.py, session.py, builders.py   # Classifier, session store, API/FSMManager builders
│   ├── security/, constants.py, logging.py, memory.py, env.py   # Key filtering, path confinement, limits, logging, working memory, .env loader
│   ├── okf/, text.py                       # Open Knowledge Format, text helpers (okf_bridge is engine/okf_bridge.py)
│   ├── skills/                # Agent Skills (read-only), the fsm-llm-skills command
│   ├── tools/                 # Tool registry, tool models and the @tool decorator (re-exported by agents)
│   ├── policy/                # Approval policy and the human-in-the-loop manager (re-exported by agents)
│   ├── frontmatter.py, cli_support.py   # The one YAML frontmatter loader, the shared CLI skeleton
│   ├── runner.py, __main__.py           # The interactive fsm-llm command
│   ├── reasoning/             # 9 strategies as stacked FSMs
│   ├── workflows/             # Async workflow engine, 11 step types, DSL
│   ├── agents/                # 18 agent patterns, tools, HITL, MCP, A2A, meta-builder
│   ├── monitor/               # FastAPI dashboard, OTEL exporter
│   ├── eval/                  # Examples scorer, conversation cases
│   ├── synthesis/             # Runtime FSM synthesis: compile, gates, library, runner
│   └── testing/               # Public test doubles, scripted user, assertions
├── tests/                     # One folder per subpackage plus root-level files
├── examples/                  # 100 runnable examples in 8 categories (evaluation baselines)
├── docs/                      # Guides, API reference and the site pages (mkdocs.yml builds them)
├── evaluation/                # Datasets and recorded eval runs
├── scripts/                   # Supply-chain audit, benches, doc-count check
├── images/                    # Logo and diagrams
├── .github/workflows/         # CI: 9-leg matrix (Python 3.11 to 3.15), docs
└── CHANGELOG.md, EVALUATE.md, CONTRIBUTING.md, SECURITY.md
```

## Examples are frozen

Do not modify anything under `examples/` unless a maintainer asks for it. The examples are the evaluation baselines for `fsm-llm-eval examples`, and every shipped FSM JSON must keep loading (`tests/test_examples`).

## Plans, decisions and findings

Non-trivial work is done as a plan: explore, write a plan, execute it one step per commit, then review. Each plan lives in its own directory, `plans/plan-<YYYY-MM-DD>T<HHMMSS>-<8 hex>/`, holding `plan.md` (goal, steps, success criteria), `decisions.md` (numbered entries `D-001`, `D-002`, ... with context, decision, trade-off and reasoning), `findings.md` (what exploration found, with evidence), `progress.md` and `state.md`.

`plans/` is gitignored. The one committed file in it is `plans/ANCHORS.md`, an append-only manifest with one line per anchored decision (`<plan-id>/D-NNN | date | one-line rationale`), so an anchor in the code still resolves after its plan directory is deleted. Never edit, reorder or trim its lines.

A commit made for a plan step starts with the plan tag: `[plan-YYYY-MM-DD-<8 hex>/iter-N/step-M] <type>(<scope>): <summary>`.

## Decision anchors

Code whose reason is not obvious from reading it carries a decision anchor, a comment that links it to a `decisions.md` entry:

```python
# DECISION plan-2026-09-22T080837-8b258a25/D-036
# _fsm_cache_lock is a LEAF lock: while holding it do NOT take `_lock`,
# a conv_lock, log, or call fsm_loader. See D-036.
```

- Format: `# DECISION <plan-id>/D-NNN`, with the full plan directory name as the plan id. Older anchors use the legacy form `plan_YYYY-MM-DD_<8 hex>`. A bare `D-NNN` with no plan id is not allowed in new code.
- Add one when the code takes an approach chosen after another one failed, when a simpler-looking alternative was rejected on purpose, when it works around a library or framework constraint, or when a reader would reasonably ask "why not just do X?".
- The body says what NOT to do and why. A new anchor is at most 6 lines; the full reasoning belongs in `decisions.md`.
- Every anchor needs a line in `plans/ANCHORS.md`, the append-only manifest (one line per decision: plan id, number, date, one-line rationale). The line is produced when the plan closes; never edit or reorder an existing line. `plans/ANCHORS.md` is the one tracked file under `plans/`, so a fresh clone has it. The owner commits the appended lines before the first push.
- `make decision-check` is the gate: it fails when an anchor has no manifest line (and is not `[STALE]`), when an anchor has a bad prefix (not a plan id, or a bare `D-NNN`), and when `docs/decisions/index.md` is stale. A default-suite test runs the same check.
- `docs/decisions/index.md` is generated from the anchors and the manifest: do not edit it. After you add, move, reword or remove an anchor (or when `plans/ANCHORS.md` gains lines) run `make decision-index` and commit the regenerated file with the change. The index lists every place of each decision (path and enclosing definition, no line numbers), so it is the lookup path from a decision to its code; the `**Anchor-Refs**:` line in a plan's `decisions.md` is no longer the way to find the places.
- Read the anchors near code before you change it, and do not undo what they forbid. If a rule no longer fits, record a new decision that supersedes it instead of deleting the anchor quietly.

Two markers can follow an anchor's id:

- `[STALE]`: the plan directory that recorded the decision has been retired, so the full entry may be gone (its one-line summary is in `plans/ANCHORS.md`). The gate counts such an anchor as resolved by its tag (it is listed in the index under Retired). The rule itself may still bind. Read the anchor text and the code around it before changing anything.
- `[SUPERSEDED BY D-nnn]` (or an inline `[supersedes ...]` note on the newer anchor): the rule was replaced. Follow the newer decision; the old text is kept only as history.

## Building the docs

The documentation site is built from `docs/` and `mkdocs.yml` with mkdocs-material. Its tools are a dependency group, not an extra (they need Python 3.11 or newer and are for maintainers), pinned exactly in `constraints.txt`:

```bash
.venv/bin/pip install --group docs -c constraints.txt   # needs pip 25.1 or newer
make docs                                               # a strict build into site/ (ignored by git)
make docs-serve                                         # live reload at http://127.0.0.1:8000
```

A strict build fails on any warning: a page missing from the `nav` in `mkdocs.yml`, a broken link or anchor, an `include-markdown` target that does not exist. A new page under `docs/` goes into the `nav`; `AGENTS.md` files are never published (`exclude_docs`). `tests/test_fsm_llm/test_docs_site.py` checks the same structure without mkdocs installed.

`.github/workflows/docs.yml` builds the site strictly on every pull request and every tag `v*`, and a tag deploys it with `mike` to the `gh-pages` branch after checking that the tag equals `src/fsm_llm/__version__.py` (`docs/README.md`, "Publishing the site", lists the owner steps). Its text is pinned by `tests/test_fsm_llm_regression/test_docs_workflow.py`; check a change to it with `uvx --from actionlint-py actionlint .github/workflows/docs.yml`. Every `uses:` in a workflow is a full 40-character commit SHA with the release tag in a trailing comment.

## Good places to help

All taken from recorded open items:

- **Agents** ([docs/agents_roadmap.md](https://github.com/NikolasMarkou/fsm_llm/blob/main/docs/agents_roadmap.md), Known open items and Track B): `AgentGraph` has no failure routing, so a fallback branch cannot be built; Orchestrator subtasks dropped over `max_workers` do not affect `success`; a reply cut off by `max_tokens` is detected (`finish_reason`, a WARNING, a named cause in extraction errors, no unreadable half envelope shown as a reply) and a cut final reply (or answer field) of an agent run ends it with `success=False` and `stop_reason="truncated"` (REWOO stops on a cut plan), but a cut `native_fc` tool call, a `SelfConsistencyAgent` sample, a long meta-builder artifact, a streamed reply and a cut in an earlier turn are still not flagged; Swarm never hands off by itself; real human-in-the-loop for REWOO, PlanExecute, ParallelReact and native_fc; MCP v2 and A2A v1.0.
- **Core** (`CHANGELOG.md`, Unreleased, Known open): validating an `FSMDefinition` for the first time from several threads at once can race; the message-free `classification_extractions` path has offline coverage only; nothing was checked live on a non-Ollama provider.
- **Reasoning** (`CHANGELOG.md`, Known open): cheaper typed per-key reasoning fields were tried and reverted after losing answers on a pre-registered probe, so a new attempt needs its own probe; a `final_context` passed back as `initial_context` still carries the previous problem's analysis.
- **Measurement**: every live number here comes from one small local model. Runs on other models and providers are welcome, as new pre-registered blocks (see `scripts/bench_data/README.md`).

## Pull requests

1. Fork the repository and create a feature branch.
2. Make the change with tests (see above).
3. Make sure `make lint`, `make type-check` and `make test` pass.
4. Add an entry under `Unreleased` in `CHANGELOG.md`. Name every public removal, rename and deprecation.
5. Open the pull request.
