# Contributing to FSM-LLM

## Development setup

Work inside the project virtualenv. Every command below assumes `.venv/bin` is on `PATH` (or call `.venv/bin/python` directly).

```bash
python -m venv .venv
source .venv/bin/activate
make install-dev    # pip install -c constraints.txt -e ".[dev,workflows,reasoning,agents,monitor,harness]" + pre-commit install
```

`constraints.txt` pins litellm to a verified-safe release (1.82.7 and 1.82.8 were compromised; `pyproject.toml` excludes them too). Run `make audit` after installing new packages; it scans site-packages for suspicious `.pth` files.

## Tests, lint and types

```bash
make test           # full suite (pytest -v)
make lint           # ruff check src/ tests/
make format         # ruff format src/ tests/
make type-check     # mypy across all six packages
pytest tests/test_fsm_llm/            # one package's suite
pytest -m "not slow"                  # skip the slow tests
```

- The default suite mocks the LLM. Live tests need a reachable Ollama and skip without one; the harness live tests also need `FSM_LLM_HARNESS_LIVE=1`.
- Pre-commit runs ruff (lint with `--fix`, and format) and the file hygiene hooks on commit, and a quick pytest run on push.
- A behaviour change needs a test that fails on the code before the change. Run it against the parent commit to confirm.
- Test counts are written out as literals in `CLAUDE.md`, `README.md` and `src/fsm_llm_harness/CLAUDE.md`, and `tests/test_packaging.py` checks them against a fresh collection. When you add or remove tests, re-measure with `pytest --collect-only -q | tail -1` (and per suite) and update every literal. Never guess or adjust a count by hand.
- `tests/test_fsm_llm/test_docs_snippets.py` loads every fenced block that names `"initial_state"` in the first-touch docs. A full FSM snippet in the docs must be a valid FSM.

## Examples are frozen

Do not modify anything under `examples/` unless a maintainer asks for it. The examples are the evaluation baselines for `scripts/eval.py`, and every shipped FSM JSON must keep loading (`tests/test_examples`).

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
- When you place or move an anchor, update the entry's `**Anchor-Refs**:` line (`path:line`) in the same commit.
- Read the anchors near code before you change it, and do not undo what they forbid. If a rule no longer fits, record a new decision that supersedes it instead of deleting the anchor quietly.

Two markers can follow an anchor's id:

- `[STALE]`: the plan directory that recorded the decision has been retired, so the full entry may be gone (its one-line summary is in `plans/ANCHORS.md`). The rule itself may still bind. Read the anchor text and the code around it before changing anything.
- `[SUPERSEDED BY D-nnn]` (or an inline `[supersedes ...]` note on the newer anchor): the rule was replaced. Follow the newer decision; the old text is kept only as history.

## Pull requests

1. Fork the repository and create a feature branch.
2. Make the change with tests (see above).
3. Make sure `make lint`, `make type-check` and `make test` pass.
4. Add an entry under `Unreleased` in `CHANGELOG.md`. Name every public removal, rename and deprecation.
5. Open the pull request.
