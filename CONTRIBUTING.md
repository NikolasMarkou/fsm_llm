# Contributing to FSM-LLM

Contributions are welcome, from a typo fix to a new agent pattern. Report vulnerabilities as described in [SECURITY.md](https://github.com/NikolasMarkou/fsm_llm/blob/main/SECURITY.md).

## Development setup

Work inside the project virtualenv. Every command below assumes `.venv/bin` is on `PATH` (or call `.venv/bin/python` directly).

```bash
python -m venv .venv
source .venv/bin/activate
make install-dev    # pip install -c constraints.txt -e ".[dev,workflows,reasoning,agents,monitor,eval,synthesis,testing,otel,a2a]" + pre-commit install
```

Upgrading a clone from before the 2026-09-29 restructure (extensions moved under `fsm_llm/`): delete the old `src/fsm_llm_<sub>/` directories by hand (`make clean` no longer removes them), then reinstall with `pip install -e .`. After the package moved back to `src/fsm_llm/` (plan d454fd62, step 62), reinstall with `pip install -e .` again; the old editable install pointed at the flat `src/`. A leftover directory imports as a namespace package.

`constraints.txt` pins every direct dependency to one exact version, litellm to a verified-safe release (1.82.7 and 1.82.8 were compromised; `pyproject.toml` excludes them too). Run `make audit` after installing new packages; it scans site-packages for suspicious `.pth` files.

## Tests, lint and types

```bash
make install-dev    # editable install with every extra, plus pre-commit hooks
make test           # Run full test suite
make lint           # ruff check src/ tests/
make format         # ruff format src/ tests/
make type-check     # mypy on the fsm_llm package (src/fsm_llm/)
make lint-imports   # import contracts in .importlinter (the Section 5.2 layers, kernel, litellm, testing, pytest, entry modules, ...)
make audit          # scan site-packages for malicious .pth files
make gate-step      # the lean step gate (scripts/gate.sh step), before a commit
make gate-push      # the lean push gate (scripts/gate.sh push), the pre-push hook
make build          # wheel and sdist
pytest tests/test_fsm_llm/            # one suite (core)
pytest -m "not slow"                  # skip the slow tests
```

- The default suite mocks the LLM: it needs no network and no model. Live tests need a reachable Ollama and skip without one.
- CI (`.github/workflows/python-package.yml`) runs a `lint` job once (ruff lint and format check, mypy, the `.pth` audit, the import contracts, the decision-anchor gate and the workflow linters `actionlint` and `zizmor`), a `supply-chain` job (`pip-audit --skip-editable` over the installed environment, plus a canary that must see `pip-audit` fail on `tests/fixtures/canary-vulnerable-pins.txt`: never install or "fix" that file), a `dependency-review` job on pull requests (`actions/dependency-review-action`, fails on a high advisory) and a `test` matrix of legs: Linux on Python 3.11 to 3.15, Windows on 3.11 and 3.15 (macOS is not tested in CI). Every leg runs the `.pth` audit and pytest without the `slow`, `real_llm` and `integration` markers; the Linux 3.12 leg also measures branch coverage and runs the ratchet (`scripts/coverage_ratchet.py`). The Windows legs run `python -m` commands in bash, never `make`. Check an edit to a workflow with `uvx --from actionlint-py actionlint .github/workflows/*.yml` and `uvx --from zizmor zizmor --offline .github/workflows` (the lint job runs both at the pins of `constraints.txt`); every `uses:` takes a 40-hex SHA with its release tag in a comment, in every workflow file; `tests/test_fsm_llm_regression/test_ci_workflow.py` pins the matrix.
- Pre-commit runs ruff (lint with `--fix`, and format) and the file hygiene hooks on commit, and the lean push gate (`scripts/gate.sh push`) on push. `make gate-step` is the same script for a step before a commit: ruff on the touched files, the import contracts, the decision anchors, the generated fixtures and one pytest process over the files that guard the invariants (name more with `scripts/gate.sh step --tests PATH...`). CI is the wide run.
- A behaviour change needs a test that fails on the code before the change. Run it against the parent commit to confirm.
- No document states a count of tests, test files, modules, lines, contracts or examples: measure with `pytest --collect-only -q | tail -1` (or `make test`). `tests/test_fsm_llm_regression/test_doc_count_ratchet.py` fails when a tracked `.md` gains one, so do not write one.
- `tests/test_fsm_llm/test_docs_snippets.py` loads every fenced block that names `"initial_state"` in the first-touch docs. A full FSM snippet in the docs must be a valid FSM.
- Naming for new code (ids, units, constants, enums, builders, exceptions, verbs) follows the `Naming conventions` section of [AGENTS.md](https://github.com/NikolasMarkou/fsm_llm/blob/main/AGENTS.md).
- litellm is imported in exactly one module, `src/fsm_llm/llm/providers/litellm.py`. Do not add a second request builder or provider call elsewhere.
- Internal-key and secret-key checks go through `fsm_llm.constants.has_internal_prefix` and `is_forbidden_context_entry`. Do not re-inline them.
- Anything new that is configured with many settings gets a step-by-step builder that follows the convention in [src/fsm_llm/AGENTS.md](https://github.com/NikolasMarkou/fsm_llm/blob/main/src/fsm_llm/AGENTS.md): setters only record, `build()` is the only validator.
- Commit messages follow `<type>(<scope>): <summary>`.

## Project structure

The repository is one distribution (`llm-fsm`, setuptools, src layout). The directory `src/fsm_llm/` is the import package `fsm_llm` (found by `packages.find` in `pyproject.toml`), so `src/fsm_llm/engine/api/conversation.py` is `fsm_llm.engine.api.conversation`, and a subpackage is imported by name, for example `from fsm_llm.agents import create_agent`.

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
│   ├── runtime/               # The opt-in durable event log: EventLog, JSONL and blob stores, turn scope, session and stack projections (API.fork and API.resume read them), privacy levels, retention, usage fold (imports core, llm and the kernel only)
│   ├── context/               # Context cleaning, compaction, filtering and the prompt builders (context/layout/)
│   ├── engine/classification.py, session.py, builders.py   # Classifier, session store, API/FSMManager builders
│   ├── security/, constants.py, logging.py, memory.py, env.py   # Key filtering, path confinement, limits, logging, working memory, .env loader
│   ├── okf/, text.py                       # Open Knowledge Format, text helpers (okf_bridge is engine/okf_bridge.py)
│   ├── skills/                # Agent Skills (read-only), the fsm-llm-skills command
│   ├── tools/                 # Tool registry, tool models and the @tool decorator (re-exported by agents)
│   ├── policy/                # Approval policy and the human-in-the-loop manager (re-exported by agents)
│   ├── frontmatter.py, cli_support.py   # The one YAML frontmatter loader, the shared CLI skeleton
│   ├── runner.py, __main__.py           # The interactive fsm-llm command
│   ├── reasoning/             # Reasoning strategies as stacked FSMs
│   ├── workflows/             # Async workflow engine, step types, DSL
│   ├── agents/                # Agent patterns, tools, HITL, MCP, A2A, meta-builder
│   ├── monitor/               # FastAPI dashboard, OTEL exporter
│   ├── eval/                  # Examples scorer, conversation cases
│   ├── synthesis/             # Runtime FSM synthesis: compile, gates, library, runner
│   └── testing/               # Public test doubles, scripted user, assertions
├── tests/                     # One folder per subpackage plus root-level files
├── examples/                  # Runnable examples by category (evaluation baselines)
├── docs/                      # The documentation pages of the site (mkdocs.yml builds them)
├── evaluation/                # Datasets and recorded eval runs
├── scripts/                   # Supply-chain audit, benches
├── images/                    # Logo and diagrams
├── .github/workflows/         # CI: the test matrix (Python 3.11 to 3.15), docs, pin refresh, release
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
- `make decision-check` (`python scripts/decision_index.py --check`) is the gate: it fails when an anchor has no manifest line (and is not `[STALE]`) and when an anchor has a bad prefix (not a plan id, or a bare `D-NNN`). It reads the committed `plans/ANCHORS.md` only and takes no argument. A default-suite test runs the same check, and so does the `lint` job of CI.
- `make decision-index` writes a report of every anchor to `build/decision-index.md` (ignored by git, never committed): each decision with its manifest rationale and every place (path and enclosing definition, no line numbers), so it is the lookup path from a decision to its code on a machine that has no plan directory. `python scripts/decision_index.py --report` prints the counts.
- Read the anchors near code before you change it, and do not undo what they forbid. If a rule no longer fits, record a new decision that supersedes it instead of deleting the anchor quietly.

Two markers can follow an anchor's id:

- `[STALE]`: the plan directory that recorded the decision has been retired, so the full entry may be gone (its one-line summary is in `plans/ANCHORS.md`). The gate counts such an anchor as resolved by its tag (it is listed in the index under Retired). The rule itself may still bind. Read the anchor text and the code around it before changing anything.
- `[SUPERSEDED BY D-nnn]` (or an inline `[supersedes ...]` note on the newer anchor): the rule was replaced. Follow the newer decision; the old text is kept only as history.

## Changelog

`CHANGELOG.md` follows Keep a Changelog 1.1.0: a `## [Unreleased]` section first, then `## [X.Y.Z] - YYYY-MM-DD` releases, newest first; inside a section the `###` headings are only Added, Changed, Deprecated, Removed, Fixed and Security, each at most once, in that order; and `[Unreleased]: .../compare/v<newest release>...HEAD` sits with the reference links at the bottom. Releases already published stay as they were written.

A change that breaks a caller (a removed or renamed public name or path, a changed default, a new refusal) gets one bullet that starts its breaking part with the exact marker `**BREAKING:**` (not `**Breaking:**`, not prose). The bullet names the old form and the new one: the changelog is the migration record of a release, and no separate upgrade guide is published. A `**BREAKING:**` marker outside a list item (a table row, a paragraph) is refused, because only a bullet is read for the spelling.

```bash
python scripts/changelog_check.py check                   # the form and the spelling of every BREAKING marker
python scripts/changelog_check.py check --from 0.12.0     # widen the scope to older sections
python scripts/changelog_check.py extract 0.13.0          # the notes of one release (also v0.13.0 or Unreleased)
make changelog-check                                      # the same check as `check`
```

The checker (stdlib only, read-only) judges `[Unreleased]`, plus the newest release while `[Unreleased]` is empty. It is enforced three ways: `tests/test_packaging.py::TestTheRealChangelog` runs it on the real file in every test run, the `lint` job of the CI workflow runs the same command, and the release workflow reads the release notes with `extract`. In the release body a repository-relative link `docs/...` is made absolute under the tag by `extract --link-base` (a GitHub release body does not resolve a relative link); the changelog holds no such link today, and if you add one write it relative, never as an absolute URL to `main`.

## Building the docs

The documentation site is built from `docs/` and `mkdocs.yml` with mkdocs-material. Its tools are a dependency group, not an extra (they need Python 3.11 or newer and are for maintainers), pinned exactly in `constraints.txt`:

```bash
.venv/bin/pip install --group docs -c constraints.txt   # needs pip 25.1 or newer
make docs                                               # a strict build into site/ (ignored by git)
make docs-serve                                         # live reload at http://127.0.0.1:8000
```

A strict build fails on any warning: a page missing from the `nav` in `mkdocs.yml`, a broken link or anchor, an `include-markdown` target that does not exist. A new page under `docs/` goes into the `nav`; `AGENTS.md` files are never published (`exclude_docs`). `tests/test_fsm_llm/test_docs_site.py` checks the same structure without mkdocs installed.

`.github/workflows/refresh-pins.yml` refreshes the `litellm` and `mcp` pins of `constraints.txt` once a week (Mondays 05:17 UTC) and on demand (Actions tab, "Refresh pins", Run workflow). `scripts/refresh_pins.py` picks, per package, the newest release that is stable, not yanked, **at least 7 days old** (the cooldown: litellm 1.82.7 and 1.82.8 were malware for hours, and a bad release is usually yanked within days), whose `Requires-Python` admits every Python of the CI matrix, and that stays in the major of the current pin; it never downgrades and never crosses a major (a new major is a decision, not a refresh). When a pin moves, the `test` job installs it and runs the non-slow suite (the `mcp` stdio tests included), mypy, the `.pth` audit and `pip-audit`, and only if all pass does the `propose` job push a `chore/refresh-pins-<date>` branch and open a pull request whose body is the script's report (old and new pins, release dates, ages, the newer releases skipped and why). The `test` and `propose` jobs are separate on purpose: `test` executes a brand-new third-party release, so it has the read-only token and no credentials, and hands over only an artifact (`constraints.txt`, `pyproject.toml`, the report, uploaded before anything is installed); `propose` holds the write token, installs nothing and runs no project code, and a guard step refuses anything but those regular files and any change beyond the litellm and mcp pin and range lines. A pull request opened with the default `GITHUB_TOKEN` does not trigger the CI workflow, which is why the gate runs in the refresh workflow itself. **Owner action, once:** Settings, Actions, General, Workflow permissions: tick "Allow GitHub Actions to create and approve pull requests" (without it the last step fails with a permission error and no pull request appears; the branch is still pushed). **When a pull request appears:** read the table, skim the upstream release notes of the new version, merge it, and close an older refresh pull request it supersedes. A red `test` job means the new pin broke the suite or has a known vulnerability: `propose` does not run, nothing is pushed, the pin stays, and the log names the failing step. Scheduled workflows run only from the default branch and GitHub stops them after 60 days without repository activity (re-enable in the Actions tab). Try the script without touching the repository: `python scripts/refresh_pins.py --dry-run` (reads PyPI, prints the table and the diff, writes nothing; `--json-dir DIR` reads canned `<package>.json` files instead). The tests are `tests/test_fsm_llm_regression/test_refresh_pins.py` (the script, canned JSON, no network) and `test_refresh_workflow.py` (the workflow).

`.github/workflows/docs.yml` builds the site strictly on every pull request and every tag `v*`, and a tag deploys it with `mike` to the `gh-pages` branch after checking that the tag equals `src/fsm_llm/__version__.py` (`docs/README.md`, "Publishing the site", lists the owner steps). Its text is pinned by `tests/test_fsm_llm_regression/test_docs_workflow.py`; check a change to it with `uvx --from actionlint-py actionlint .github/workflows/docs.yml`. Every `uses:` in a workflow is a full 40-character commit SHA with the release tag in a trailing comment.

## Releasing

A release is one pushed tag. `.github/workflows/release.yml` starts on a push of a tag `v*` and on nothing else (no manual dispatch, no pull request, no branch push), so only a pushed tag releases. It builds the sdist and the wheel once, checks them (`twine check --strict`, the tag against `src/fsm_llm/__version__.py`, a final release only, the section of the version in `CHANGELOG.md` through `scripts/changelog_check.py extract`), writes an SBOM (`sbom.cdx.json`, CycloneDX, from a fresh environment with the wheel installed with every extra), then publishes to PyPI by trusted publishing with PEP 740 attestations (no API token exists anywhere), and finally creates the GitHub release with the distributions, the SBOM and the notes. `tests/test_fsm_llm_regression/test_release_workflow.py` pins the workflow; the distribution is `llm-fsm` (the import package stays `fsm_llm`).

**The version bump** (what the release commit of 0.13.0 did; the next one repeats it): set `version` in `pyproject.toml` and `__version__` in `src/fsm_llm/__version__.py` (the subpackage `__version__.py` files re-export it); the literal pins of the version in `tests/test_fsm_llm_monitor/test_app.py` and `tests/test_fsm_llm_regression/test_regression_cli_and_exports.py`; the README badge; the `> Covers FSM-LLM v<version>` header of the current guides in `docs/` (and the lines that say the same in `docs/AGENTS.md`, `docs/README.md`, `docs/concepts.md`, `AGENTS.md`, `src/fsm_llm/AGENTS.md`, `src/fsm_llm/README.md`); in `CHANGELOG.md`, `## [Unreleased]` becomes `## [X.Y.Z] - YYYY-MM-DD` under a fresh empty `## [Unreleased]`, with the reference links `[Unreleased]: .../compare/vX.Y.Z...HEAD` and `[X.Y.Z]: .../compare/v<previous>...vX.Y.Z`. The release section opens with a short `**Highlights**` block before its first `###` heading (under 8,000 characters, links in the `docs/...` form): the release body on GitHub holds at most 125,000 characters, `release.yml` cuts a longer body at 120,000 and points at the full text, so the Highlights are the part that always survives. `tests/test_fsm_llm_regression/test_release_version.py` fails on any place that was missed and on a section without Highlights.

```bash
python scripts/changelog_check.py check                   # the release section is judged while [Unreleased] is empty
python scripts/changelog_check.py extract X.Y.Z --link-base https://github.com/NikolasMarkou/fsm_llm/blob/vX.Y.Z/ > /tmp/notes.md   # the body GitHub will show, before the cut; `head -40 /tmp/notes.md` is the Highlights
```

Owner steps, in this order (the commands need `gh` logged in as the owner). Steps 1 to 3 are done once, before the first push of a plan's work:

1. Commit the anchor manifest first: `git add plans/ANCHORS.md && git commit -m "docs: anchor manifest lines"`. It holds the appended lines that `make decision-check` (and the `lint` job) need, so a push before this commit is a red first run on purpose. Never edit an existing line of it.
2. In any other clone or virtualenv, after the layout move of 0.13.0: `pip uninstall fsm-llm`, delete a leftover `src/fsm_llm.egg-info`, then `pip install -e .` once (console scripts and the distribution name moved; `python scripts/audit_pth.py` flags a stale code-bearing finder).
3. On PyPI, confirm `llm-fsm` is in "Your projects" and add a Trusted Publisher with exactly these names: owner `NikolasMarkou`, repository `fsm_llm`, workflow `release.yml`, environment `pypi`. (A TestPyPI pending publisher with the same names is an optional dry run of the whole path before the real tag.)
4. Create the GitHub environment `pypi` with a required reviewer (you, with two-factor authentication on) and restrict it to tags `v*`. Without the reviewer `publish` runs unattended on any pushed `v*` tag; without the tag restriction a branch can deploy to the environment. One `PUT` sets the reviewer and the restriction to custom policies, then one `POST` adds the tag pattern. The environments API takes `reviewers` as an array of objects, which `gh api` field flags cannot express (they document scalar arrays only), so the `PUT` body is JSON on standard input, with real JSON booleans: the booleans as `-f` fields would be sent as the strings `"true"` and `"false"` and the API refuses them (422); with `-F` they would be booleans, but the reviewer would still be missing. The `type` of the `POST` is a string, so `-f` is right there:

   ```bash
   owner_id=$(gh api users/NikolasMarkou --jq .id)
   body='{"prevent_self_review":false,"reviewers":[{"type":"User","id":'"$owner_id"'}],"deployment_branch_policy":{"protected_branches":false,"custom_branch_policies":true}}'
   gh api -X PUT repos/NikolasMarkou/fsm_llm/environments/pypi --input - <<< "$body"
   gh api -X POST repos/NikolasMarkou/fsm_llm/environments/pypi/deployment-branch-policies \
     -f name='v*' -f type=tag
   gh api repos/NikolasMarkou/fsm_llm/environments/pypi --jq '{protection_rules, deployment_branch_policy}'
   ```

   The last command is the check: `protection_rules` must hold a `required_reviewers` rule naming you and `deployment_branch_policy` must read `{"custom_branch_policies":true,"protected_branches":false}`; the tag pattern is listed by `gh api repos/NikolasMarkou/fsm_llm/environments/pypi/deployment-branch-policies --jq '.branch_policies'` (one entry, name `v*`, type `tag`). The same in the web UI: Settings, Environments, `pypi`: tick "Required reviewers" and add yourself, leave "Prevent self-review" off (you are the only maintainer: with it on, nobody could approve your own tag), then "Deployment branches and tags", "Selected branches and tags", "Add deployment branch or tag rule", type "Tag", pattern `v*`. Required reviewers need a public repository or a paid plan. The `PUT` creates the environment when it does not exist and replaces its whole protection configuration, so run it once, before the first tag.

5. Settings, Actions, General: tick "Allow GitHub Actions to create and approve pull requests" (the weekly pin refresh opens its pull request with it). Once the matrix is green, Settings, Branches: protect `main`, require a pull request before merging, and require the `test (<os>, Python <version>)` checks of the CI workflow (and `lint` and `supply-chain`). The pull request rule is not a formality: the coverage ratchet's `guard` (the baseline in `coverage-baseline.json` may only rise) runs on pull requests only, so a direct push to `main` that lowers the baseline file would pass `check` against the lowered file. As the sole maintainer, set the required approvals to 0 (a pull request, no second reviewer).
6. `git push origin main` and watch the first matrix. The Windows legs had never run before the first push: a failure there is a finding to fix, not a reason to skip the leg. The coverage tolerance is meant to be tightened after three green runs.
7. Release from a clean `main` whose CI is green and whose `CHANGELOG.md` date is the day of the tag (edit `## [X.Y.Z] - YYYY-MM-DD` if the tag is cut on another day, commit, and run `python scripts/changelog_check.py check` again):

   ```bash
   git tag -a v0.13.0 -m "v0.13.0"
   git push origin v0.13.0
   ```

8. Open the run in the Actions tab. `build` and `sbom` run first; `publish` then waits for the environment review: approve the deployment. `github-release` follows. Check the result: the PyPI page of `llm-fsm` shows the release with its attestations ("Provenance"), the GitHub release carries the wheel, the sdist, `sbom.cdx.json` and the notes (they open with the Highlights), and `pip install llm-fsm==0.13.0` works in a fresh environment on Linux and Windows.
9. After the first docs deploy (the tag also starts `.github/workflows/docs.yml`): Settings, Pages, source "Deploy from a branch", branch `gh-pages`, folder `/ (root)`; then, from a clone of `main` with the docs tools installed (`pip install --group docs -c constraints.txt`), run `mike set-default --push latest`. [`docs/README.md`](https://github.com/NikolasMarkou/fsm_llm/blob/main/docs/README.md), "Publishing the site", has the detail.

If a run goes wrong after the tag:

- `github-release` failed after `publish` succeeded (the files are on PyPI): use "Re-run failed jobs" only. It reuses the artifacts of the run. "Re-run all jobs" rebuilds, and PyPI refuses files it already has ("File already exists"), so the GitHub release would never be cut. A failure before `publish` is fixed by deleting the tag (below) and tagging again.
- `.github/workflows/docs.yml` also starts on the tag and deploys the site on its own, whether or not the release publishes. If a publish is abandoned (the tag deleted or PyPI refused it), remove that version from the site with `mike delete <version> --push` from a clone with the docs tools installed (deleting a version deletes its aliases too, so `latest` goes with it), then give `latest` back to the previous release with `mike alias --push --update-aliases <previous version> latest`. The two workflows are not chained on purpose: a `workflow_run` chain cannot be tested without a real run, so this sentence is the guard.

A tag that fails a guard publishes nothing: delete it (`git push origin :refs/tags/vX.Y.Z` and `git tag -d vX.Y.Z`), fix the cause and tag again. A version that reached PyPI cannot be replaced: cut the next one (the files of a release are immutable there).

## Good places to help

All taken from recorded open items:

- **Agents** (`src/fsm_llm/agents/README.md` and the [2026 roadmap](https://github.com/NikolasMarkou/fsm_llm/blob/main/docs/2026_roadmap.md), open items): `AgentGraph` has no failure routing, so a fallback branch cannot be built; Orchestrator subtasks dropped over `max_workers` do not affect `success`; a reply cut off by `max_tokens` is detected (`finish_reason`, a WARNING, a named cause in extraction errors, no unreadable half envelope shown as a reply) and a cut final reply (or answer field) of an agent run ends it with `success=False` and `stop_reason="truncated"` (REWOO stops on a cut plan), but a cut `native_fc` tool call, a `SelfConsistencyAgent` sample, a long meta-builder artifact, a streamed reply and a cut in an earlier turn are still not flagged; Swarm never hands off by itself; real human-in-the-loop for REWOO, PlanExecute, ParallelReact and native_fc; MCP v2 and A2A v1.0.
- **Core** (`CHANGELOG.md`, 0.12.0, Known open): validating an `FSMDefinition` for the first time from several threads at once can race; the message-free `classification_extractions` path has offline coverage only; nothing was checked live on a non-Ollama provider.
- **Reasoning** (`CHANGELOG.md`, Known open): cheaper typed per-key reasoning fields were tried and reverted after losing answers on a pre-registered probe, so a new attempt needs its own probe; a `final_context` passed back as `initial_context` still carries the previous problem's analysis.
- **Measurement**: every live number here comes from one small local model. Runs on other models and providers are welcome, as new pre-registered blocks (see `scripts/bench_data/README.md`).

## Pull requests

1. Fork the repository and create a feature branch.
2. Make the change with tests (see above).
3. Make sure `make lint`, `make type-check` and `make test` pass.
4. Add an entry under `Unreleased` in `CHANGELOG.md`. Name every public removal, rename and deprecation.
5. Open the pull request.
