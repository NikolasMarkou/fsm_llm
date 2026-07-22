"""Tests for ``fsm_llm_harness.plan_validator``.

Every fixture below is lifted from this repository's own ``plans/`` tree.
``plans/`` is gitignored (``.gitignore:183``) so the content is EMBEDDED here
rather than read from disk -- but it is real content, and the details that make
it real are exactly the details a synthetic stub would smooth away:

* ``PLAN_MD`` keeps the 11 sections in the validator's order, and it keeps the
  two backticked angle-bracket literals (``` `<think>` ```, ``` `<hex8>` ```)
  that a naive template-slot detector reports as unfilled placeholders.
* ``DECISIONS_MD`` keeps a ``**Trade-off**:`` whose ``at the cost of`` wraps
  across a line break, an ``**Anchor-Refs**:`` line, and an ``**Outcome ...**``
  line with NO colon -- which the decisions parser folds into the preceding
  field's value, dragging prose into the reference list unless only the first
  line is read.
* ``DECISIONS_MD`` also quotes ``<!-- COMPRESSED-SUMMARY -->`` inside backticks,
  the way this plan's real D-020 does.  A marker scan without code-span
  awareness reports that plan as having an unclosed compression block.
* ``VERIFICATION_MD`` keeps a ``PENDING`` row beside claimed ones, because
  evidence quality is only asserted for rows that claim an outcome.
* ``CHANGELOG_MD`` keeps the header line that QUOTES the pipe-delimited format,
  which an entry-shaped test applied first reads as a malformed ledger line.
* ``STATE_MD`` keeps the trailing HTML-comment block the real file carries
  below ``## Transition History:``.

The base fixture is deliberately AUDIT-CLEAN: ``test_a_healthy_plan_directory``
pins ``audit() == []``, and every other audit test mutates exactly one thing and
asserts exactly one new issue.  That is what makes the suite mutation-sensitive
-- a check that stops firing shows up as a missing issue, and a check that fires
too eagerly shows up in the clean-directory test.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from fsm_llm_harness.constants import ArtifactNames, Defaults, GateSlug, Severity
from fsm_llm_harness.plan_validator import (
    _TEMPLATE_SLOT_RE,
    CHECKS,
    GateResult,
    Issue,
    _claimed_result,
    _commit_tag_form,
    _is_placeholder,
    _same_file,
    _strip_code,
    audit,
    pre_step_gate,
)
from fsm_llm_harness.storage import DRIVER_READ_MAX_BYTES
from fsm_llm_harness.tools import MAX_READ_BYTES

# ---------------------------------------------------------------------------
# Real fixtures
# ---------------------------------------------------------------------------

PLAN_ID = "plan-2026-07-21T191807-bf7ffe24"
COMMIT_TAG = "plan-2026-07-21-bf7ffe24"

STATE_MD = """# Current State: EXECUTE
*Skill: iterative-planner v2.56.0*
## Iteration: 1
## Current Plan Step: 9 (plan_validator.py)
## Pre-Step Checklist (reset before each EXECUTE step)
- [ ] Re-read state.md (this file)
- [x] Re-read plan.md
## Fix Attempts (resets per plan step)
- (none yet for current step)
## Change Manifest (current iteration)
- step 8 (`cc35681`): `storage.py` NEW (813), `test_storage.py` NEW (93 tests).
## Last Transition: INIT → EXPLORE (2026-07-21T19:18:07Z)
## Transition History:
- INIT → EXPLORE (task started)
- EXPLORE → PLAN (gathered enough context, 2026-07-21T19:52:00Z)
- PLAN → EXECUTE (user approved, 2026-07-21T20:05:00Z)
- EXECUTE step 1 PASS (`f63104f`). Pre-Mortem #2 did NOT fire.
<!-- When logging EXPLORE → PLAN, add Exploration Confidence on the line below.
See references/planning-rigor.md for definitions. -->
"""

PLAN_MD = """# Plan v1: Make the harness actually run on `:4b`, then finish it

## Goal
Make `src/fsm_llm_harness` run the iterative-planner protocol end-to-end on
`ollama_chat/qwen3.5:4b`.

## Problem Statement
The two agent arms available to a harness role have COMPLEMENTARY failures and
neither is complete alone.

## Context
Read `findings.md` (index, Key Constraints, Corrections) and the four detail
files. All five live claims below are measured, not asserted.

## Files To Modify
| File | Change | Reason |
|---|---|---|
| `tests/test_fsm_llm_harness/test_hardening.py` | new — fenced/`<think>`/prose recovery | small-model hardening |

## Steps
1. [x] **`native_fc` Ollama-helper repair.** Apply `apply_ollama_params` gated
   by `is_ollama_model`. [RISK: medium] [deps: none]
2. [ ] **`storage.py::PlanDirectory`.** Plan-id minting
   (`plan-YYYY-MM-DDTHHMMSS-<hex8>`), path layout. [RISK: medium] [deps: 1]

## Assumptions
- **A1.** `tools=` and `response_format=` in the SAME litellm call are NOT
  required by the chosen design. Falsified if step 2's repair turn fails.

## Failure Modes
| Dependency | Slow | Bad data | Down | Blast radius |
|---|---|---|---|---|
| Ollama at `localhost:11434` | 1-18s/call | empty `content` | steps 2, 5 unverifiable | live criteria only |

## Pre-Mortem & Falsification Signals
1. **The terminal-turn `response_format` will not work.** → **STOP IF** the
   repair turn yields a schema-valid payload in fewer than 4 of 5 runs.

## Success Criteria
1. A role dispatch that calls a tool AND writes a file to disk.
2. `pytest` full suite holds its 3629-test baseline.

## Verification Strategy
| # | Criterion | Method | Command | Pass condition |
|---|---|---|---|---|
| 1 | Tool call + file on disk | Live, n≥5 | `pytest -k L4 -q` | ≥4/5 runs |

## Complexity Budget
| Metric | Budget | Notes |
|---|---|---|
| Files added — harness source | 4/4 | fixed by the module boundary |
"""

DECISIONS_MD = f"""# Decision Log
*Plan: {PLAN_ID}*
*Skill: iterative-planner v2.56.0*

*Append-only. Never edit past entries.*

<!-- Schema example — DO NOT REMOVE. Real entries follow this shape.
## D-001 | EXPLORE → PLAN | YYYY-MM-DD
**Trade-off**: <X> **at the cost of** <Y>
-->

## D-001 | EXPLORE → PLAN | 2026-07-21
**Title**: `native_fc` is the role arm, not stock ReAct
**Context**: The two agent arms have complementary failures.
**Decision**: Back harness roles with `NativeFunctionCallingReactAgent`.
**Trade-off**: Reliable, measured tool selection under role-weight prompts **at the cost of**
losing the free constrained decoding ReAct gets from `output_schema`.
**Reasoning**: The failure ReAct exhibits is the one the protocol cannot tolerate.
**Anchor-Refs**: `src/fsm_llm_harness/roles.py:12`
**Outcome (step 2, iter 1)** — Pre-Mortem #1's trigger did NOT fire; see
`bench_step2.py:40` for the harness used. This line carries NO colon after its
bold run, so the decisions parser folds it into the PRECEDING field's value.

## D-002 | EXECUTE | 2026-07-22
**Title**: The compressed-summary block is protected STRUCTURALLY
**Context**: A compressed summary must never be summarised into itself. The
rejected implementation prepends a fresh `<!-- COMPRESSED-SUMMARY -->` pair each
pass, which looks equivalent on the first CLOSE and nests on the second.
**Decision**: New bullets are inserted INSIDE the existing block.
**Trade-off**: Nesting is unreachable rather than merely detected **at the cost of**
the window depending on the block's heading text.
**Reasoning**: A check-afterwards design would have to decide what to DO on detection.
"""

FINDINGS_MD = """# Findings
*Summary and index of all findings. Detailed files go in findings/ directory.*

## Index
1. `findings/native-fc-live-repair.md` — option (b) measured 0/3 → 3/3
2. `findings/react-tool-selection-live.md` — cause is prompt WEIGHT
3. `findings/harness-remaining-tier.md` — independence verdict

## Key Constraints
- **HARD**: `plans/` is gitignored, so protocol memory has no VCS backstop.
"""

FINDINGS_TOPIC_MD = """# Finding: remaining harness tier

## Summary
The remaining tier has exactly one call-site seam into the built driver.

## Key Findings
- `harness.py:1416` is the only seam (`_pre_step_gate`).

## Constraints
- **HARD** `rules.py::OWNERSHIP` is read-only from both sides.

## Code Patterns
- `[REUSE]` every mechanical gate is a pure `(planDir, issues) -> None`.

## Risks & Unknowns
- The external `validate-plan.mjs` cannot be read from this repository.
"""

PROGRESS_MD = """# Progress

## Completed
- [x] EXPLORE: four parallel investigations, all live-measured

## In Progress
- [ ] Step 9: `plan_validator.py`

## Remaining
- [ ] 10. Wire artifacts + storage + validator into `harness.py`

## Blocked
*Nothing currently.*
"""

VERIFICATION_MD = """# Verification Results (Iteration 1)
*Rewritten each iteration — not append-only.*

## Criteria Verification
| # | Criterion (from plan.md) | Method | Command/Action | Result | Evidence |
|---|---|---|---|---|---|
| 1 | Tool call + file on disk | Live, n≥5 | `pytest -k L4 -q` | PASS | 5/5 runs wrote bytes |
| 2 | Full suite baseline | Automated | `pytest -q` | PENDING | - |
| 3 | Lint and types | Automated | `make lint` | PASS | exit 0; "All checks passed" |

## Additional Checks
| Check | Command/Action | Result | Details |
|---|---|---|---|
| Regression | `pytest -q` | PASS | 869/869 |
| Scope drift | manifest vs plan.md | PASS | manual review — observed no unplanned file |
| Diff review | `git diff` | PASS | manual review — observed no stray prints |

## Not Verified
| What | Why |
|---|---|
| Non-Ollama providers end-to-end | no paid-provider budget |

## Verdict
- Criteria passed: 2/3
- Regressions: none
- Scope drift: none
- Simplification blockers: none
- Recommendation: → EXECUTE (continue with step 10)
"""

CHANGELOG_MD = """# Changelog
*Append-only per-edit ledger. One line per file edit. Owner: ip-executor.*
*Format: `UTC | iter-N/step-M | commit | path | OP(+N,-M) | radius:TIER(score) | D-NNN-or-dash | reason`*
2026-07-21T20:10:30Z | iter-1/step-1 | f63104f | src/fsm_llm_agents/native_fc.py | EDIT(+59,-10) | radius:LOW(0) | D-001 | apply ollama call prep behind is_ollama_model
2026-07-22T00:05:00Z | iter-1/step-3 | 9101369 | src/fsm_llm_harness/tools.py | EDIT(+77,-2) | radius:LOW(2) | - | repair sentinel-prefixed absolute paths
"""

CHECKPOINT_MD = """# Checkpoint cp-000-iter1

## Created
2026-07-21T20:10:00Z — EXECUTE, iteration 1, before step 1.

## Reason
Nuclear fallback / full-revert restore point for the whole iteration.

## Git State
- Commit: `14e27a18d9c21ee68127670bb74312dcd3763dad` (`14e27a1`)

## Lockfiles snapshotted:
- none (no package manager touched)

## Rollback:
```bash
git reset --hard 14e27a1
```
"""

LESSONS_MD = """# Lessons Learned
*Cross-plan lessons. Updated and consolidated on close. Max 200 lines.*

## Recurring Patterns
- [I:5] A safety cap the caller can reset from inside its own callback is not a cap.

## Failed Approaches (+ why)
- [I:3] Strengthening a prompt a third time — measured 0/5 twice already.

## Successful Strategies
- [I:4] Mechanical verification beats a stronger instruction.

## Codebase Gotchas
- [I:5] `plans/` is gitignored; embed real content as fixtures instead.
"""

SYSTEM_MD = """# System Atlas
*Last refreshed: 2026-07-22*

## Identity
A Python framework for stateful conversational AI combining LLMs with FSMs.

## Components
- `fsm_llm` — the core 2-pass engine.

## Boundaries
- Core never imports `fsm_llm_agents`.

## Invariants
- The driver is the sole writer of all nine gate flags.

## Flows
- User input → Pass 1 → transition → Pass 2 → output.

## Known Patterns
- Filesystem-as-memory: context window = RAM, filesystem = disk.
"""

#: keyword slug -> (path relative to the plan directory, default content).
#: ``../`` addresses the cross-plan tier, which lives BESIDE the plan dirs.
BASE_FILES: dict[str, tuple[str, str]] = {
    "state_md": (ArtifactNames.STATE, STATE_MD),
    "plan_md": (ArtifactNames.PLAN, PLAN_MD),
    "decisions_md": (ArtifactNames.DECISIONS, DECISIONS_MD),
    "findings_md": (ArtifactNames.FINDINGS_INDEX, FINDINGS_MD),
    "progress_md": (ArtifactNames.PROGRESS, PROGRESS_MD),
    "verification_md": (ArtifactNames.VERIFICATION, VERIFICATION_MD),
    "changelog_md": (ArtifactNames.CHANGELOG, CHANGELOG_MD),
    "findings_topic_md": ("findings/harness-remaining-tier.md", FINDINGS_TOPIC_MD),
    "checkpoint_md": ("checkpoints/cp-000-iter1.md", CHECKPOINT_MD),
    "lessons_md": (f"../{ArtifactNames.LESSONS}", LESSONS_MD),
    "system_md": (f"../{ArtifactNames.SYSTEM}", SYSTEM_MD),
}


def make_plan_dir(tmp_path: Path, **overrides: str | None) -> Path:
    """Write a complete, AUDIT-CLEAN plan directory; override or delete files.

    Keyword keys are the :data:`BASE_FILES` slugs; a value of ``None`` deletes
    that file instead of writing it.
    """
    unknown = set(overrides) - set(BASE_FILES)
    assert not unknown, f"unknown fixture override(s): {sorted(unknown)}"
    plan_dir = tmp_path / "plans" / PLAN_ID
    plan_dir.mkdir(parents=True, exist_ok=True)
    for slug, (relative, default) in BASE_FILES.items():
        content = overrides.get(slug, default)
        if content is None:
            continue
        target = (plan_dir / relative).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    return plan_dir


def tags(issues: list[Issue]) -> list[str]:
    return [issue.check for issue in issues]


def only(issues: list[Issue], check: str) -> list[Issue]:
    return [issue for issue in issues if issue.check == check]


def state_with(*, state: str = "EXECUTE", iteration: int = 1, attempts: int = 0) -> str:
    """``state.md`` with the three fields every gate/audit tier reads."""
    lines = (
        [f"- Step 9, attempt {index}" for index in range(1, attempts + 1)]
        if attempts
        else ["- (none yet for current step)"]
    )
    return (
        f"# Current State: {state}\n"
        f"## Iteration: {iteration}\n"
        "## Current Plan Step: 9\n"
        "## Pre-Step Checklist (reset before each EXECUTE step)\n"
        "- [ ] Re-read state.md (this file)\n"
        "## Fix Attempts (resets per plan step)\n" + "\n".join(lines) + "\n"
        "## Change Manifest (current iteration)\n"
        "- step 8 (`cc35681`): storage.py\n"
        "## Last Transition: PLAN → EXECUTE\n"
        "## Transition History:\n"
        "- PLAN → EXECUTE (user approved)\n"
    )


# ---------------------------------------------------------------------------
# The result types
# ---------------------------------------------------------------------------


class TestResultTypes:
    def test_a_passing_gate_carries_no_slug_and_exits_zero(self) -> None:
        result = GateResult(passed=True)
        assert result.slug is None
        assert result.exit_code == 0
        assert str(result) == "GATE:PASS"

    def test_a_failing_gate_exits_two_and_renders_its_slug(self) -> None:
        result = GateResult(
            passed=False, slug=GateSlug.LEASH_CAP, detail="attempts=2 cap=2"
        )
        assert result.exit_code == 2
        assert str(result) == "GATE:FAIL [leash-cap] attempts=2 cap=2"

    def test_a_passing_gate_may_not_carry_a_slug(self) -> None:
        with pytest.raises(ValueError, match="passing gate cannot carry"):
            GateResult(passed=True, slug=GateSlug.NO_PLAN)

    def test_a_failing_gate_must_name_a_known_slug(self) -> None:
        with pytest.raises(ValueError, match="not a pre-step gate slug"):
            GateResult(passed=False, slug="made-up")

    def test_every_gate_failure_is_hard(self) -> None:
        with pytest.raises(ValueError, match="must be hard"):
            GateResult(passed=False, slug=GateSlug.NO_PLAN, hard=False)

    def test_an_issue_rejects_an_unknown_severity(self) -> None:
        with pytest.raises(ValueError, match="unknown severity"):
            Issue(severity="critical", check="state", message="x")

    def test_an_issue_rejects_an_unregistered_check_tag(self) -> None:
        with pytest.raises(ValueError, match="unknown check tag"):
            Issue(severity=Severity.ERROR, check="invented", message="x")

    def test_check_tags_are_sorted_and_unique(self) -> None:
        assert list(CHECKS) == sorted(set(CHECKS))

    def test_issue_renders_its_tag_and_artifact(self) -> None:
        issue = Issue(
            severity=Severity.WARNING,
            check="leash",
            message="3 attempts",
            artifact="state.md",
        )
        assert str(issue) == "[leash] state.md: 3 attempts"
        assert not issue.is_error


# ---------------------------------------------------------------------------
# pre_step_gate: the four slugs, their order, and the short circuit
# ---------------------------------------------------------------------------


class TestPreStepGate:
    def test_a_healthy_execute_state_passes(self, tmp_path: Path) -> None:
        assert pre_step_gate(make_plan_dir(tmp_path)) == GateResult(passed=True)

    def test_no_plan_when_the_directory_does_not_exist(self, tmp_path: Path) -> None:
        result = pre_step_gate(tmp_path / "nowhere")
        assert result.slug == GateSlug.NO_PLAN
        assert result.exit_code == 2

    def test_no_plan_when_state_md_is_absent(self, tmp_path: Path) -> None:
        assert (
            pre_step_gate(make_plan_dir(tmp_path, state_md=None)).slug
            == GateSlug.NO_PLAN
        )

    def test_no_plan_when_state_md_does_not_parse(self, tmp_path: Path) -> None:
        plan_dir = make_plan_dir(tmp_path, state_md="not a protocol artifact at all\n")
        assert pre_step_gate(plan_dir).slug == GateSlug.NO_PLAN

    def test_no_plan_never_creates_the_missing_directory(self, tmp_path: Path) -> None:
        # D-023: routing this through `PlanMemory` would `mkdir` the directory
        # whose absence the slug exists to report.
        missing = tmp_path / "nowhere"
        assert pre_step_gate(missing).slug == GateSlug.NO_PLAN
        assert not missing.exists()

    def test_the_gate_writes_nothing_at_all(self, tmp_path: Path) -> None:
        plan_dir = make_plan_dir(tmp_path)
        before = {
            path: path.stat().st_mtime_ns
            for path in sorted(plan_dir.rglob("*"))
            if path.is_file()
        }
        pre_step_gate(plan_dir)
        after = {
            path: path.stat().st_mtime_ns
            for path in sorted(plan_dir.rglob("*"))
            if path.is_file()
        }
        assert before == after

    @pytest.mark.parametrize("state", ["explore", "plan", "reflect", "pivot", "close"])
    def test_wrong_state_for_every_non_execute_state(
        self, tmp_path: Path, state: str
    ) -> None:
        plan_dir = make_plan_dir(tmp_path, state_md=state_with(state=state.upper()))
        result = pre_step_gate(plan_dir)
        assert result.slug == GateSlug.WRONG_STATE
        assert result.detail == f"expected=EXECUTE actual={state.upper()}"

    @pytest.mark.parametrize("attempts", [0, 1])
    def test_leash_cap_does_not_fire_below_the_cap(
        self, tmp_path: Path, attempts: int
    ) -> None:
        plan_dir = make_plan_dir(tmp_path, state_md=state_with(attempts=attempts))
        assert pre_step_gate(plan_dir).passed

    @pytest.mark.parametrize("attempts", [2, 3, 4])
    def test_leash_cap_hard_blocks_the_third_spawn(
        self, tmp_path: Path, attempts: int
    ) -> None:
        # THE GATE FIRES AT 2. `audit()` uses 3/4+ for the same counter; see
        # `test_the_two_leash_tiers_are_deliberately_different`.
        plan_dir = make_plan_dir(tmp_path, state_md=state_with(attempts=attempts))
        result = pre_step_gate(plan_dir)
        assert result.slug == GateSlug.LEASH_CAP
        assert result.detail == f"attempts={attempts} cap={Defaults.MAX_FIX_ATTEMPTS}"

    @pytest.mark.parametrize("iteration", [0, 1, 5])
    def test_iteration_cap_does_not_fire_below_six(
        self, tmp_path: Path, iteration: int
    ) -> None:
        plan_dir = make_plan_dir(tmp_path, state_md=state_with(iteration=iteration))
        assert pre_step_gate(plan_dir).passed

    @pytest.mark.parametrize("iteration", [6, 7, 12])
    def test_iteration_cap_fires_at_six(self, tmp_path: Path, iteration: int) -> None:
        plan_dir = make_plan_dir(tmp_path, state_md=state_with(iteration=iteration))
        result = pre_step_gate(plan_dir)
        assert result.slug == GateSlug.ITERATION_CAP
        assert (
            result.detail
            == f"iteration={iteration} hard-cap={Defaults.ITERATION_HARD_CAP}"
        )

    def test_iteration_is_derived_from_history_when_the_bump_was_forgotten(
        self, tmp_path: Path
    ) -> None:
        # A forgotten manual bump must never LOWER the safety cap.
        forgotten = state_with(iteration=0).replace(
            "- PLAN → EXECUTE (user approved)\n",
            "".join(f"- EXECUTE → REFLECT (iteration {n})\n" for n in range(1, 7)),
        )
        assert pre_step_gate(make_plan_dir(tmp_path, state_md=forgotten)).slug == (
            GateSlug.ITERATION_CAP
        )

    def test_the_declared_iteration_still_wins_when_it_is_higher(
        self, tmp_path: Path
    ) -> None:
        plan_dir = make_plan_dir(tmp_path, state_md=state_with(iteration=6))
        assert pre_step_gate(plan_dir).slug == GateSlug.ITERATION_CAP

    # -- order and short-circuit ----------------------------------------

    def test_the_slug_order_is_the_constants_order(self) -> None:
        assert GateSlug.ORDER == (
            "no-plan",
            "wrong-state",
            "leash-cap",
            "iteration-cap",
        )

    def test_no_plan_wins_over_every_other_failure(self, tmp_path: Path) -> None:
        # state.md is absent, so wrong-state / leash-cap / iteration-cap cannot
        # even be asked; `no-plan` is the only answer available.
        plan_dir = make_plan_dir(tmp_path, state_md=None)
        assert pre_step_gate(plan_dir).slug == GateSlug.NO_PLAN

    def test_wrong_state_wins_over_leash_and_iteration(self, tmp_path: Path) -> None:
        plan_dir = make_plan_dir(
            tmp_path, state_md=state_with(state="REFLECT", iteration=9, attempts=5)
        )
        assert pre_step_gate(plan_dir).slug == GateSlug.WRONG_STATE

    def test_leash_cap_wins_over_iteration_cap(self, tmp_path: Path) -> None:
        plan_dir = make_plan_dir(tmp_path, state_md=state_with(iteration=9, attempts=5))
        assert pre_step_gate(plan_dir).slug == GateSlug.LEASH_CAP

    def test_later_checks_are_not_evaluated_at_all(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Short-circuit, proven by non-evaluation rather than by outcome."""
        import fsm_llm_harness.plan_validator as module

        called: list[str] = []

        def spy(slug: str, real: object) -> object:
            def wrapper(*args: object, **kwargs: object) -> object:
                called.append(slug)
                return real(*args, **kwargs)  # type: ignore[operator]

            return wrapper

        monkeypatch.setattr(
            module,
            "_GATE_CHECKS",
            {slug: spy(slug, check) for slug, check in module._GATE_CHECKS.items()},
        )
        plan_dir = make_plan_dir(tmp_path, state_md=state_with(iteration=9, attempts=5))
        assert pre_step_gate(plan_dir).slug == GateSlug.LEASH_CAP
        assert called == [GateSlug.WRONG_STATE, GateSlug.LEASH_CAP]
        assert GateSlug.ITERATION_CAP not in called

    def test_no_plan_evaluates_no_predicate_whatsoever(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import fsm_llm_harness.plan_validator as module

        def explode(*args: object, **kwargs: object) -> str | None:
            raise AssertionError("no predicate may run once state.md is unreadable")

        monkeypatch.setattr(
            module, "_GATE_CHECKS", {slug: explode for slug in module._GATE_CHECKS}
        )
        assert pre_step_gate(tmp_path / "nowhere").slug == GateSlug.NO_PLAN

    def test_the_thresholds_are_caller_overridable(self, tmp_path: Path) -> None:
        plan_dir = make_plan_dir(tmp_path, state_md=state_with(iteration=3, attempts=1))
        assert pre_step_gate(plan_dir).passed
        assert pre_step_gate(plan_dir, max_fix_attempts=1).slug == GateSlug.LEASH_CAP
        assert pre_step_gate(plan_dir, iteration_cap=3).slug == GateSlug.ITERATION_CAP
        assert (
            pre_step_gate(plan_dir, expected_state="reflect").slug
            == GateSlug.WRONG_STATE
        )


# ---------------------------------------------------------------------------
# audit: the clean baseline every other test mutates away from
# ---------------------------------------------------------------------------


class TestAuditBaseline:
    def test_a_healthy_plan_directory_reports_nothing(self, tmp_path: Path) -> None:
        assert audit(make_plan_dir(tmp_path)) == []

    def test_a_missing_plan_directory_is_a_single_error(self, tmp_path: Path) -> None:
        issues = audit(tmp_path / "nowhere")
        assert [issue.severity for issue in issues] == [Severity.ERROR]
        assert "is absent" in issues[0].message

    def test_audit_creates_nothing_and_changes_nothing(self, tmp_path: Path) -> None:
        plan_dir = make_plan_dir(tmp_path)
        before = sorted(
            (path.relative_to(tmp_path), path.read_bytes())
            for path in tmp_path.rglob("*")
            if path.is_file()
        )
        audit(plan_dir, workspace_root=tmp_path)
        after = sorted(
            (path.relative_to(tmp_path), path.read_bytes())
            for path in tmp_path.rglob("*")
            if path.is_file()
        )
        assert before == after

    def test_audit_never_raises_for_an_unreadable_artifact(
        self, tmp_path: Path
    ) -> None:
        plan_dir = make_plan_dir(tmp_path)
        # Over the DRIVER's read bound (storage.py D-037), not the agent-facing
        # 64 KB cap: fail CLOSED, as an issue, not a crash.
        (plan_dir / ArtifactNames.PLAN).write_text(
            "# x\n" + "y" * DRIVER_READ_MAX_BYTES
        )
        issues = audit(plan_dir)
        assert "state" in tags(issues)
        assert any("could not complete" in issue.message for issue in issues)

    def test_a_real_sized_decisions_log_is_audited_whole(self, tmp_path: Path) -> None:
        """The 64 KB blocker: a REAL ``decisions.md`` outgrows the agent cap.

        This plan's own decision log is ~108 KB.  Read through
        ``PlanMemory.read_text`` it came back clipped, and every check that
        reads the TAIL -- the schema scan, the ``Anchor-Refs`` back-links, the
        compression-marker scan -- went dark on a document that was fine.
        Here the tail carries the only malformed entry, so a truncated read
        would report the file as clean.
        """
        padding = "\n".join(
            f"**Note {index}**: filler that pushes the tail past the agent cap"
            for index in range(1200)
        )
        oversized = f"{DECISIONS_MD}{padding}\n\n## D-004 | EXECUTE | 2026-07-22\n**Title**: no trade-off line\n"
        plan_dir = make_plan_dir(tmp_path, decisions_md=oversized)
        assert (plan_dir / ArtifactNames.DECISIONS).stat().st_size > MAX_READ_BYTES
        issues = only(audit(plan_dir), "decisions-schema")
        assert any("D-004" in issue.message for issue in issues), tags(audit(plan_dir))

    def test_every_emitted_tag_is_registered(self, tmp_path: Path) -> None:
        plan_dir = make_plan_dir(
            tmp_path, plan_md=None, decisions_md=None, state_md=None
        )
        assert set(tags(audit(plan_dir))) <= set(CHECKS)


# ---------------------------------------------------------------------------
# audit: state.md -- the two leash tiers and the two iteration tiers
# ---------------------------------------------------------------------------


class TestAuditState:
    def test_a_missing_state_is_an_error(self, tmp_path: Path) -> None:
        issues = only(audit(make_plan_dir(tmp_path, state_md=None)), "state")
        assert [issue.severity for issue in issues] == [Severity.ERROR]

    def test_an_unparseable_state_is_an_error(self, tmp_path: Path) -> None:
        plan_dir = make_plan_dir(tmp_path, state_md="# Current State: NOT-A-STATE\n")
        assert only(audit(plan_dir), "state")[0].severity == Severity.ERROR

    @pytest.mark.parametrize("attempts", [0, 1, 2])
    def test_two_recorded_attempts_is_legal_and_silent(
        self, tmp_path: Path, attempts: int
    ) -> None:
        plan_dir = make_plan_dir(tmp_path, state_md=state_with(attempts=attempts))
        assert only(audit(plan_dir), "leash") == []

    def test_three_attempts_warns_that_the_gate_was_passed(
        self, tmp_path: Path
    ) -> None:
        plan_dir = make_plan_dir(tmp_path, state_md=state_with(attempts=3))
        issues = only(audit(plan_dir), "leash")
        assert [issue.severity for issue in issues] == [Severity.WARNING]
        assert "was passed" in issues[0].message

    @pytest.mark.parametrize("attempts", [4, 5, 9])
    def test_four_or_more_attempts_errors_that_the_gate_was_bypassed(
        self, tmp_path: Path, attempts: int
    ) -> None:
        plan_dir = make_plan_dir(tmp_path, state_md=state_with(attempts=attempts))
        issues = only(audit(plan_dir), "leash")
        assert [issue.severity for issue in issues] == [Severity.ERROR]
        assert "bypassed" in issues[0].message

    def test_the_two_leash_tiers_are_deliberately_different(
        self, tmp_path: Path
    ) -> None:
        """D-022: the gate fires at 2; the audit is silent at 2 and WARNs at 3.

        These thresholds are INTENTIONALLY not aligned.  The gate runs while a
        step is live, where 2 recorded attempts means the budget is spent.  The
        audit runs over a finished plan, where a step is ALLOWED to have used
        both of its attempts -- so erroring at 2 would false-positive on every
        plan that correctly spent its leash and then pivoted.
        """
        at_cap = make_plan_dir(tmp_path / "a", state_md=state_with(attempts=2))
        past_cap = make_plan_dir(tmp_path / "b", state_md=state_with(attempts=3))

        assert pre_step_gate(at_cap).slug == GateSlug.LEASH_CAP  # HARD block
        assert only(audit(at_cap), "leash") == []  # legal in retrospect

        assert pre_step_gate(past_cap).slug == GateSlug.LEASH_CAP
        assert [i.severity for i in only(audit(past_cap), "leash")] == [
            Severity.WARNING
        ]

        assert Defaults.MAX_FIX_ATTEMPTS == 2
        assert Defaults.LEASH_AUDIT_WARN_ATTEMPTS == 3
        assert Defaults.LEASH_AUDIT_ERROR_ATTEMPTS == 4

    @pytest.mark.parametrize("iteration", [0, 1, 4])
    def test_iterations_below_five_are_silent(
        self, tmp_path: Path, iteration: int
    ) -> None:
        plan_dir = make_plan_dir(tmp_path, state_md=state_with(iteration=iteration))
        assert only(audit(plan_dir), "iteration") == []

    def test_iteration_five_warns_for_decomposition(self, tmp_path: Path) -> None:
        plan_dir = make_plan_dir(tmp_path, state_md=state_with(iteration=5))
        issues = only(audit(plan_dir), "iteration")
        assert [issue.severity for issue in issues] == [Severity.WARNING]
        assert "decomposition" in issues[0].message

    @pytest.mark.parametrize("iteration", [6, 8])
    def test_iteration_six_or_more_is_an_error(
        self, tmp_path: Path, iteration: int
    ) -> None:
        plan_dir = make_plan_dir(tmp_path, state_md=state_with(iteration=iteration))
        assert [i.severity for i in only(audit(plan_dir), "iteration")] == [
            Severity.ERROR
        ]


# ---------------------------------------------------------------------------
# audit: plan.md
# ---------------------------------------------------------------------------


class TestAuditPlan:
    def test_a_missing_plan_is_an_error(self, tmp_path: Path) -> None:
        issues = only(audit(make_plan_dir(tmp_path, plan_md=None)), "plan")
        assert [issue.severity for issue in issues] == [Severity.ERROR]

    def test_a_missing_section_is_a_section_error(self, tmp_path: Path) -> None:
        broken = PLAN_MD.replace("## Failure Modes", "## Failure Notes")
        issues = only(audit(make_plan_dir(tmp_path, plan_md=broken)), "plan-section")
        assert [issue.severity for issue in issues] == [Severity.ERROR]
        assert "Failure Modes" in issues[0].message

    def test_sections_out_of_order_are_a_section_error(self, tmp_path: Path) -> None:
        head, _, tail = PLAN_MD.partition("## Success Criteria")
        swapped = (
            head
            + "## Complexity Budget\n| Metric | Budget |\n|---|---|\n| Files | 4/4 |\n\n"
        )
        swapped += "## Success Criteria" + tail.replace(
            "## Complexity Budget", "## Leftover"
        )
        issues = only(audit(make_plan_dir(tmp_path, plan_md=swapped)), "plan-section")
        assert [issue.severity for issue in issues] == [Severity.ERROR]

    def test_an_unfilled_template_section_warns(self, tmp_path: Path) -> None:
        templated = PLAN_MD.replace(
            "The two agent arms available to a harness role have COMPLEMENTARY failures and\n"
            "neither is complete alone.",
            "<one-paragraph statement of the problem>",
        )
        issues = only(audit(make_plan_dir(tmp_path, plan_md=templated)), "plan-section")
        assert [issue.severity for issue in issues] == [Severity.WARNING]
        assert "Problem Statement" in issues[0].message

    def test_an_empty_section_warns(self, tmp_path: Path) -> None:
        emptied = PLAN_MD.replace(
            "- **A1.** `tools=` and `response_format=` in the SAME litellm call are NOT\n"
            "  required by the chosen design. Falsified if step 2's repair turn fails.\n",
            "",
        )
        issues = only(audit(make_plan_dir(tmp_path, plan_md=emptied)), "plan-section")
        assert "Assumptions" in issues[0].message

    def test_a_placeholder_complexity_budget_gets_its_own_tag(
        self, tmp_path: Path
    ) -> None:
        blanked = PLAN_MD.replace(
            "| Metric | Budget | Notes |\n|---|---|---|\n"
            "| Files added — harness source | 4/4 | fixed by the module boundary |\n",
            "TBD\n",
        )
        issues = audit(make_plan_dir(tmp_path, plan_md=blanked))
        assert tags(issues) == ["complexity"]

    def test_backticked_angle_brackets_are_not_placeholders(
        self, tmp_path: Path
    ) -> None:
        # `<think>` and `<hex8>` appear in the real plan.md; both are code spans.
        assert "`<think>`" in PLAN_MD and "-<hex8>`" in PLAN_MD
        assert _TEMPLATE_SLOT_RE.search("<think>") is not None  # would fire in prose
        assert only(audit(make_plan_dir(tmp_path)), "plan-section") == []


class TestPlaceholderDetection:
    @pytest.mark.parametrize(
        "body",
        [
            "",
            "   \n\n",
            "<one-paragraph background — what was discovered in EXPLORE>",
            "TBD",
            "- TODO\n- todo: write this",
            "<!-- only a comment -->",
            "```\nfenced only\n```",
        ],
    )
    def test_placeholders(self, body: str) -> None:
        assert _is_placeholder(body)

    @pytest.mark.parametrize(
        "body",
        [
            "Real prose about the problem.",
            "A row mentioning `<think>` inside a code span.",
            "Mint `plan-YYYY-MM-DDTHHMMSS-<hex8>` ids.",
            "- [x] step 1 done\n- [ ] step 2 pending",
            "5 < 6 and 7 > 4",
        ],
    )
    def test_not_placeholders(self, body: str) -> None:
        assert not _is_placeholder(body)

    def test_strip_code_preserves_line_numbers(self) -> None:
        text = "a\n```\nb\nc\n```\nd `e` f"
        assert _strip_code(text).split("\n") == ["a", "", "", "", "", "d  f"]


# ---------------------------------------------------------------------------
# audit: decisions.md
# ---------------------------------------------------------------------------


class TestAuditDecisions:
    def test_a_missing_decisions_log_is_an_error(self, tmp_path: Path) -> None:
        issues = only(
            audit(make_plan_dir(tmp_path, decisions_md=None)), "decisions-schema"
        )
        assert [issue.severity for issue in issues] == [Severity.ERROR]

    def test_a_missing_plan_id_preamble_is_an_error(self, tmp_path: Path) -> None:
        broken = DECISIONS_MD.replace(f"*Plan: {PLAN_ID}*\n", "")
        issues = audit(make_plan_dir(tmp_path, decisions_md=broken))
        assert tags(issues) == ["preamble-missing"]

    def test_a_preamble_naming_another_plan_is_an_error(self, tmp_path: Path) -> None:
        broken = DECISIONS_MD.replace(PLAN_ID, "plan-2026-01-01T000000-deadbeef", 1)
        issues = only(
            audit(make_plan_dir(tmp_path, decisions_md=broken)), "preamble-mismatch"
        )
        assert [issue.severity for issue in issues] == [Severity.ERROR]

    def test_a_gap_in_the_d_nnn_sequence_is_an_error(self, tmp_path: Path) -> None:
        broken = DECISIONS_MD.replace("## D-002 |", "## D-004 |")
        issues = only(
            audit(make_plan_dir(tmp_path, decisions_md=broken)), "decisions-schema"
        )
        assert [issue.severity for issue in issues] == [Severity.ERROR]
        assert "no gaps" in issues[0].message

    def test_entries_that_do_not_start_at_d_001_are_an_error(
        self, tmp_path: Path
    ) -> None:
        broken = DECISIONS_MD.replace("## D-001 |", "## D-002 |").replace(
            "## D-002 | EXECUTE", "## D-003 | EXECUTE"
        )
        assert only(
            audit(make_plan_dir(tmp_path, decisions_md=broken)), "decisions-schema"
        )

    def test_a_trailing_title_on_the_header_line_is_rejected(
        self, tmp_path: Path
    ) -> None:
        """A documented real-world gotcha: the header grammar is exact."""
        broken = DECISIONS_MD.replace(
            "## D-001 | EXPLORE → PLAN | 2026-07-21",
            "## D-001 | EXPLORE → PLAN | 2026-07-21 — native_fc is the role arm",
        )
        issues = only(
            audit(make_plan_dir(tmp_path, decisions_md=broken)), "decisions-schema"
        )
        assert [issue.severity for issue in issues] == [Severity.ERROR]
        assert "D-NNN | PHASE | YYYY-MM-DD" in issues[0].message

    def test_a_malformed_date_is_rejected(self, tmp_path: Path) -> None:
        broken = DECISIONS_MD.replace("| 2026-07-21", "| 21-07-2026")
        assert only(
            audit(make_plan_dir(tmp_path, decisions_md=broken)), "decisions-schema"
        )

    def test_a_missing_trade_off_line_is_an_error(self, tmp_path: Path) -> None:
        broken = re.sub(
            r"\*\*Trade-off\*\*: Reliable.*?`output_schema`\.\n",
            "",
            DECISIONS_MD,
            flags=re.S,
        )
        issues = only(
            audit(make_plan_dir(tmp_path, decisions_md=broken)), "decisions-schema"
        )
        assert "Trade-off" in issues[0].message

    def test_a_trade_off_without_at_the_cost_of_is_an_error(
        self, tmp_path: Path
    ) -> None:
        broken = DECISIONS_MD.replace(
            "**at the cost of**\nlosing the free constrained decoding ReAct gets from `output_schema`.",
            "and nothing is given up.",
        )
        issues = only(
            audit(make_plan_dir(tmp_path, decisions_md=broken)), "decisions-schema"
        )
        assert "at the cost of" in issues[0].message

    def test_a_trade_off_wrapped_across_a_line_break_is_accepted(
        self, tmp_path: Path
    ) -> None:
        # The base fixture's D-001 wraps `**at the cost of**` onto the next line.
        assert "**at the cost of**\n" in DECISIONS_MD
        assert only(audit(make_plan_dir(tmp_path)), "decisions-schema") == []

    def test_the_schema_example_in_a_comment_is_not_an_entry(
        self, tmp_path: Path
    ) -> None:
        assert "## D-001 | EXPLORE → PLAN | YYYY-MM-DD" in DECISIONS_MD
        assert only(audit(make_plan_dir(tmp_path)), "decisions-schema") == []


# ---------------------------------------------------------------------------
# audit: findings, progress, verification
# ---------------------------------------------------------------------------


class TestAuditFindings:
    def test_a_missing_index_warns(self, tmp_path: Path) -> None:
        issues = only(audit(make_plan_dir(tmp_path, findings_md=None)), "findings")
        assert [issue.severity for issue in issues] == [Severity.WARNING]

    @pytest.mark.parametrize("keep", [0, 1, 2])
    def test_fewer_than_three_indexed_findings_warns(
        self, tmp_path: Path, keep: int
    ) -> None:
        entries = FINDINGS_MD.split("## Index\n")[1].split("\n\n")[0].split("\n")
        thinned = FINDINGS_MD.replace("\n".join(entries), "\n".join(entries[:keep]))
        issues = only(audit(make_plan_dir(tmp_path, findings_md=thinned)), "findings")
        assert [issue.severity for issue in issues] == [Severity.WARNING]
        assert f"only {keep} indexed findings" in issues[0].message

    def test_three_indexed_findings_is_the_threshold(self, tmp_path: Path) -> None:
        assert Defaults.FINDINGS_THRESHOLD == 3
        assert only(audit(make_plan_dir(tmp_path)), "findings") == []

    def test_a_missing_index_section_warns(self, tmp_path: Path) -> None:
        broken = FINDINGS_MD.replace("## Key Constraints", "## Notes")
        issues = only(
            audit(make_plan_dir(tmp_path, findings_md=broken)), "findings-index"
        )
        assert [issue.severity for issue in issues] == [Severity.WARNING]

    def test_a_topic_file_missing_one_of_its_five_sections_warns(
        self, tmp_path: Path
    ) -> None:
        broken = FINDINGS_TOPIC_MD.replace("## Risks & Unknowns", "## Risks")
        plan_dir = make_plan_dir(tmp_path, findings_topic_md=broken)
        issues = only(audit(plan_dir), "findings-topic")
        assert [issue.severity for issue in issues] == [Severity.WARNING]
        assert "Risks & Unknowns" in issues[0].message

    def test_a_topic_file_that_is_not_markdown_at_all_is_an_error(
        self, tmp_path: Path
    ) -> None:
        plan_dir = make_plan_dir(tmp_path, findings_topic_md="no heading here\n")
        assert [i.severity for i in only(audit(plan_dir), "findings-topic")] == [
            Severity.ERROR
        ]


class TestAuditProgress:
    def test_a_missing_progress_warns(self, tmp_path: Path) -> None:
        issues = only(audit(make_plan_dir(tmp_path, progress_md=None)), "progress")
        assert [issue.severity for issue in issues] == [Severity.WARNING]

    def test_a_missing_section_warns(self, tmp_path: Path) -> None:
        broken = PROGRESS_MD.replace("## Blocked", "## Stuck")
        issues = only(audit(make_plan_dir(tmp_path, progress_md=broken)), "progress")
        assert "Blocked" in issues[0].message

    def test_sections_out_of_order_warn(self, tmp_path: Path) -> None:
        broken = (
            PROGRESS_MD.replace("## In Progress", "## TEMP")
            .replace("## Remaining", "## In Progress")
            .replace("## TEMP", "## Remaining")
        )
        issues = only(audit(make_plan_dir(tmp_path, progress_md=broken)), "progress")
        assert any("out of order" in issue.message for issue in issues)

    def test_all_four_sections_in_order_is_clean(self, tmp_path: Path) -> None:
        assert only(audit(make_plan_dir(tmp_path)), "progress") == []


class TestAuditVerification:
    def test_a_missing_verification_warns(self, tmp_path: Path) -> None:
        issues = only(audit(make_plan_dir(tmp_path, verification_md=None)), "verdict")
        assert [issue.severity for issue in issues] == [Severity.WARNING]

    @pytest.mark.parametrize(
        "bullet",
        [
            "Criteria passed",
            "Regressions",
            "Scope drift",
            "Simplification blockers",
            "Recommendation",
        ],
    )
    def test_each_missing_verdict_bullet_warns(
        self, tmp_path: Path, bullet: str
    ) -> None:
        broken = re.sub(rf"^- {re.escape(bullet)}:.*$", "", VERIFICATION_MD, flags=re.M)
        issues = only(audit(make_plan_dir(tmp_path, verification_md=broken)), "verdict")
        assert any(bullet in issue.message for issue in issues)

    def test_verdict_bullets_out_of_order_warn(self, tmp_path: Path) -> None:
        broken = VERIFICATION_MD.replace(
            "- Criteria passed: 2/3\n- Regressions: none\n",
            "- Regressions: none\n- Criteria passed: 2/3\n",
        )
        issues = only(audit(make_plan_dir(tmp_path, verification_md=broken)), "verdict")
        assert any("out of the required order" in issue.message for issue in issues)

    def test_an_illegal_recommendation_warns(self, tmp_path: Path) -> None:
        broken = VERIFICATION_MD.replace(
            "→ EXECUTE (continue with step 10)", "→ SHIP IT"
        )
        issues = only(audit(make_plan_dir(tmp_path, verification_md=broken)), "verdict")
        assert any("SHIP" in issue.message for issue in issues)

    @pytest.mark.parametrize("check", ["Regression", "Scope drift", "Diff review"])
    def test_each_missing_mandatory_additional_check_warns(
        self, tmp_path: Path, check: str
    ) -> None:
        broken = re.sub(
            rf"^\| {re.escape(check)} \|.*$", "", VERIFICATION_MD, flags=re.M
        )
        issues = only(audit(make_plan_dir(tmp_path, verification_md=broken)), "verdict")
        assert any(check in issue.message for issue in issues)

    @pytest.mark.parametrize("evidence", ["looks good", "LGTM", "-", "done"])
    def test_a_claimed_row_with_rejected_evidence_warns(
        self, tmp_path: Path, evidence: str
    ) -> None:
        broken = VERIFICATION_MD.replace(
            "| PASS | 5/5 runs wrote bytes |", f"| PASS | {evidence} |"
        )
        issues = only(
            audit(make_plan_dir(tmp_path, verification_md=broken)), "evidence"
        )
        assert [issue.severity for issue in issues] == [Severity.WARNING]
        assert "criterion 1 claims 'PASS'" in issues[0].message

    def test_an_unclaimed_row_is_never_asked_for_evidence(self, tmp_path: Path) -> None:
        # The base fixture's criterion 2 is PENDING with a `-` evidence cell.
        assert "| PENDING | - |" in VERIFICATION_MD
        assert only(audit(make_plan_dir(tmp_path)), "evidence") == []

    @pytest.mark.parametrize("result", ["PENDING", "BLOCKED", "N/A"])
    def test_every_unclaimed_verdict_word_is_exempt(
        self, tmp_path: Path, result: str
    ) -> None:
        broken = VERIFICATION_MD.replace(
            "| PASS | 5/5 runs wrote bytes |",
            f"| **{result} — see the RCA** | looks good |",
        )
        assert (
            only(audit(make_plan_dir(tmp_path, verification_md=broken)), "evidence")
            == []
        )

    @pytest.mark.parametrize(
        ("cell", "expected"),
        [
            ("**PASS (gating proof only)**", "PASS"),
            ("**BLOCKED — criterion untestable**", "BLOCKED"),
            ("PENDING", "PENDING"),
            ("", ""),
            ("  fail  ", "FAIL"),
        ],
    )
    def test_the_claimed_verdict_word_is_the_leading_token(
        self, cell: str, expected: str
    ) -> None:
        assert _claimed_result(cell) == expected


# ---------------------------------------------------------------------------
# audit: changelog.md
# ---------------------------------------------------------------------------


class TestAuditChangelog:
    def test_a_missing_changelog_warns(self, tmp_path: Path) -> None:
        issues = only(
            audit(make_plan_dir(tmp_path, changelog_md=None)), "changelog-malformed"
        )
        assert [issue.severity for issue in issues] == [Severity.WARNING]

    def test_a_well_formed_ledger_is_clean(self, tmp_path: Path) -> None:
        assert only(audit(make_plan_dir(tmp_path)), "changelog-malformed") == []

    def test_the_header_quoting_the_format_is_not_read_as_an_entry(
        self, tmp_path: Path
    ) -> None:
        assert "| iter-N/step-M | commit |" in CHANGELOG_MD
        assert only(audit(make_plan_dir(tmp_path)), "changelog-malformed") == []

    @pytest.mark.parametrize(
        ("bad", "field"),
        [
            ("radius:LOW(-1)", "radius"),
            ("radius:HUGE(3)", "radius"),
            ("iter-1/step-4b", "step"),
            ("CREATE(+)", "op"),
            ("D-1", "decision_ref"),
        ],
    )
    def test_each_malformed_field_warns_on_its_own_line(
        self, tmp_path: Path, bad: str, field: str
    ) -> None:
        good = {
            "radius": "radius:LOW(0)",
            "step": "iter-1/step-1",
            "op": "EDIT(+59,-10)",
            "decision_ref": "D-001",
        }[field]
        broken = CHANGELOG_MD.replace(good, bad, 1)
        issues = only(
            audit(make_plan_dir(tmp_path, changelog_md=broken)), "changelog-malformed"
        )
        assert [issue.severity for issue in issues] == [Severity.WARNING]
        assert field in issues[0].message

    def test_one_bad_line_does_not_hide_the_others(self, tmp_path: Path) -> None:
        """D-025: whole-file parsing reports one issue and loses the join."""
        broken = CHANGELOG_MD.replace("radius:LOW(0)", "radius:LOW(-1)", 1).replace(
            "| - | repair sentinel", "| D-404 | repair sentinel"
        )
        issues = audit(make_plan_dir(tmp_path, changelog_md=broken))
        assert sorted(tags(issues)) == ["changelog-dref-orphan", "changelog-malformed"]

    def test_every_line_is_joined_not_just_the_first(self, tmp_path: Path) -> None:
        """A defect on line 2 is as visible as one on line 1."""
        broken = CHANGELOG_MD.replace("| D-001 |", "| D-097 |").replace(
            "| - | repair sentinel", "| D-098 | repair sentinel"
        )
        issues = only(
            audit(make_plan_dir(tmp_path, changelog_md=broken)), "changelog-dref-orphan"
        )
        assert sorted(
            re.search(r"D-\d{3}", issue.message).group() for issue in issues
        ) == [
            "D-097",
            "D-098",
        ]

    def test_two_malformed_lines_are_two_issues(self, tmp_path: Path) -> None:
        broken = CHANGELOG_MD.replace("radius:LOW(0)", "radius:LOW(-1)").replace(
            "radius:LOW(2)", "radius:BLAST(2)"
        )
        issues = only(
            audit(make_plan_dir(tmp_path, changelog_md=broken)), "changelog-malformed"
        )
        assert [issue.message.split(":")[0] for issue in issues] == ["line 4", "line 5"]

    def test_a_decision_ref_with_no_matching_entry_warns(self, tmp_path: Path) -> None:
        broken = CHANGELOG_MD.replace("| D-001 |", "| D-099 |")
        issues = only(
            audit(make_plan_dir(tmp_path, changelog_md=broken)), "changelog-dref-orphan"
        )
        assert [issue.severity for issue in issues] == [Severity.WARNING]
        assert "D-099" in issues[0].message

    def test_a_dash_decision_ref_is_never_an_orphan(self, tmp_path: Path) -> None:
        assert only(audit(make_plan_dir(tmp_path)), "changelog-dref-orphan") == []

    def test_the_join_still_runs_when_decisions_fails_its_schema(
        self, tmp_path: Path
    ) -> None:
        # A gap in the D-NNN sequence rejects the DOCUMENT; the join reads the
        # headers directly so it keeps working, and stays permissive.
        broken = DECISIONS_MD.replace("## D-002 |", "## D-005 |")
        issues = audit(make_plan_dir(tmp_path, decisions_md=broken))
        assert "changelog-dref-orphan" not in tags(issues)
        assert "decisions-schema" in tags(issues)


# ---------------------------------------------------------------------------
# audit: checkpoints, cross-plan tier, ownership
# ---------------------------------------------------------------------------


class TestAuditCheckpoints:
    def test_no_checkpoints_at_all_warns(self, tmp_path: Path) -> None:
        issues = only(audit(make_plan_dir(tmp_path, checkpoint_md=None)), "checkpoints")
        assert [issue.severity for issue in issues] == [Severity.WARNING]
        assert "cp-000-iter1.md" in issues[0].message

    def test_a_missing_nuclear_checkpoint_warns(self, tmp_path: Path) -> None:
        plan_dir = make_plan_dir(tmp_path, checkpoint_md=None)
        (plan_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        (plan_dir / "checkpoints" / "cp-001-iter1.md").write_text(CHECKPOINT_MD)
        issues = only(audit(plan_dir), "checkpoints")
        assert any("cp-000-iter1.md" in issue.message for issue in issues)

    def test_a_badly_named_checkpoint_warns(self, tmp_path: Path) -> None:
        plan_dir = make_plan_dir(tmp_path)
        (plan_dir / "checkpoints" / "before-the-risky-bit.md").write_text(CHECKPOINT_MD)
        issues = only(audit(plan_dir), "checkpoints")
        assert any("cp-NNN-iterN.md" in issue.message for issue in issues)

    def test_a_checkpoint_without_the_lockfiles_section_warns(
        self, tmp_path: Path
    ) -> None:
        broken = CHECKPOINT_MD.replace(
            "## Lockfiles snapshotted:\n- none (no package manager touched)\n\n", ""
        )
        issues = only(
            audit(make_plan_dir(tmp_path, checkpoint_md=broken)), "checkpoints"
        )
        assert any("Lockfiles snapshotted" in issue.message for issue in issues)

    def test_a_complete_checkpoint_is_clean(self, tmp_path: Path) -> None:
        assert only(audit(make_plan_dir(tmp_path)), "checkpoints") == []


class TestAuditCrossPlan:
    def test_a_missing_lessons_file_warns(self, tmp_path: Path) -> None:
        issues = only(audit(make_plan_dir(tmp_path, lessons_md=None)), "lessons-absent")
        assert [issue.severity for issue in issues] == [Severity.WARNING]

    def test_a_missing_system_atlas_warns(self, tmp_path: Path) -> None:
        issues = only(audit(make_plan_dir(tmp_path, system_md=None)), "atlas-absent")
        assert [issue.severity for issue in issues] == [Severity.WARNING]

    def test_lessons_over_its_line_cap_is_an_error(self, tmp_path: Path) -> None:
        padded = LESSONS_MD.replace(
            "## Codebase Gotchas\n",
            "## Codebase Gotchas\n"
            + "- [I:1] filler lesson\n" * Defaults.LESSONS_LINE_CAP,
        )
        issues = only(audit(make_plan_dir(tmp_path, lessons_md=padded)), "lessons-cap")
        assert [issue.severity for issue in issues] == [Severity.ERROR]
        assert str(Defaults.LESSONS_LINE_CAP) in issues[0].message

    def test_lessons_at_its_line_cap_is_clean(self, tmp_path: Path) -> None:
        assert only(audit(make_plan_dir(tmp_path)), "lessons-cap") == []

    def test_the_system_atlas_over_its_line_cap_is_an_error(
        self, tmp_path: Path
    ) -> None:
        padded = SYSTEM_MD.replace(
            "## Known Patterns\n",
            "## Known Patterns\n" + "- filler\n" * Defaults.SYSTEM_LINE_CAP,
        )
        issues = only(audit(make_plan_dir(tmp_path, system_md=padded)), "atlas-cap")
        assert [issue.severity for issue in issues] == [Severity.ERROR]
        assert "rewritten, not truncated" in issues[0].message

    def test_an_evicted_protected_lesson_is_an_error(self, tmp_path: Path) -> None:
        plan_dir = make_plan_dir(tmp_path)
        (plan_dir.parent / ArtifactNames.LESSONS_ARCHIVE).write_text(
            "# Evicted\n- [I:2] a fair eviction\n- [I:5] never evict this one\n"
        )
        issues = only(audit(plan_dir), "lessons-eviction")
        assert [issue.severity for issue in issues] == [Severity.ERROR]
        assert "never evict this one" in issues[0].message

    def test_an_archive_of_only_unprotected_lessons_is_clean(
        self, tmp_path: Path
    ) -> None:
        plan_dir = make_plan_dir(tmp_path)
        (plan_dir.parent / ArtifactNames.LESSONS_ARCHIVE).write_text(
            "# Evicted\n- [I:1] cheap\n- [I:4] still not protected\n"
        )
        assert only(audit(plan_dir), "lessons-eviction") == []

    def test_an_unclosed_compression_marker_is_an_error(self, tmp_path: Path) -> None:
        plan_dir = make_plan_dir(tmp_path)
        (plan_dir.parent / ArtifactNames.CROSS_DECISIONS).write_text(
            "# Consolidated Decisions\n<!-- COMPRESSED-SUMMARY -->\n## Summary (compressed)\n- x\n"
        )
        issues = only(audit(plan_dir), "compress-markers")
        assert [issue.severity for issue in issues] == [Severity.ERROR]
        assert "unclosed" in issues[0].message

    def test_nested_compression_markers_are_an_error(self, tmp_path: Path) -> None:
        plan_dir = make_plan_dir(tmp_path)
        (plan_dir.parent / ArtifactNames.CROSS_FINDINGS).write_text(
            "# Consolidated Findings\n<!-- COMPRESSED-SUMMARY -->\n"
            "<!-- COMPRESSED-SUMMARY -->\n- x\n<!-- /COMPRESSED-SUMMARY -->\n"
            "<!-- /COMPRESSED-SUMMARY -->\n"
        )
        assert any(
            "nested" in issue.message
            for issue in only(audit(plan_dir), "compress-markers")
        )

    def test_a_balanced_block_is_clean(self, tmp_path: Path) -> None:
        plan_dir = make_plan_dir(tmp_path)
        (plan_dir.parent / ArtifactNames.CROSS_DECISIONS).write_text(
            "# Consolidated Decisions\n<!-- COMPRESSED-SUMMARY -->\n"
            "## Summary (compressed)\n- one bullet\n<!-- /COMPRESSED-SUMMARY -->\n"
        )
        assert only(audit(plan_dir), "compress-markers") == []

    def test_a_marker_quoted_as_code_is_not_a_marker(self, tmp_path: Path) -> None:
        """The base fixture's D-002 quotes the marker inside backticks."""
        assert "`<!-- COMPRESSED-SUMMARY -->`" in DECISIONS_MD
        assert only(audit(make_plan_dir(tmp_path)), "compress-markers") == []

    def test_an_oversized_consolidated_file_is_reported_not_crashed(
        self, tmp_path: Path
    ) -> None:
        plan_dir = make_plan_dir(tmp_path)
        (plan_dir.parent / ArtifactNames.CROSS_DECISIONS).write_text(
            "# D\n" + "x" * DRIVER_READ_MAX_BYTES
        )
        issues = only(audit(plan_dir), "compress-markers")
        assert [issue.severity for issue in issues] == [Severity.WARNING]
        assert "could not be checked" in issues[0].message
        # ...and the rest of the cross-plan check still ran.
        assert only(audit(plan_dir), "lessons-absent") == []


class TestAuditOwnership:
    def test_a_stray_file_no_role_owns_warns(self, tmp_path: Path) -> None:
        plan_dir = make_plan_dir(tmp_path)
        (plan_dir / "scratch-notes.md").write_text("thoughts\n")
        issues = only(audit(plan_dir), "ownership")
        assert [issue.severity for issue in issues] == [Severity.WARNING]
        assert "scratch-notes.md" in issues[0].message

    def test_every_owned_artifact_is_accepted(self, tmp_path: Path) -> None:
        plan_dir = make_plan_dir(tmp_path)
        (plan_dir / ArtifactNames.SUMMARY).write_text(
            "# Summary\n\n## Outcome\nx\n\n## Key Decisions\nx\n\n## Files Changed\nx\n\n"
            "## Decision Anchors Registry\nx\n\n## Lessons\nx\n"
        )
        assert only(audit(plan_dir), "ownership") == []

    def test_a_dotfile_is_not_a_stray_artifact(self, tmp_path: Path) -> None:
        plan_dir = make_plan_dir(tmp_path)
        (plan_dir / ".gitkeep").write_text("")
        assert only(audit(plan_dir), "ownership") == []


# ---------------------------------------------------------------------------
# audit: the bounded decision-anchor scan
# ---------------------------------------------------------------------------


#: The anchor marker word, spelled INDIRECTLY on purpose.  A literal
#: ``# DECISION <prefix>/D-NNN`` written into this file would be picked up as a
#: real anchor by the external ``validate-plan.mjs`` anchor audit, which scans
#: `tests/` too -- so the fixtures below would become 12 bad-prefix findings
#: against this repository's own plan directory.  Do NOT inline this constant.
ANCHOR_WORD = "DECISION"


def anchor(decision_id: str, *, prefix: str | None = PLAN_ID, marker: str = "#") -> str:
    """One ``<marker> DECISION [<prefix>/]D-NNN`` anchor line."""
    qualified = f"{prefix}/{decision_id}" if prefix else decision_id
    return f"{marker} {ANCHOR_WORD} {qualified}"


def write_source(root: Path, relative: str, body: str) -> Path:
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(body, encoding="utf-8")
    return target


class TestAnchorScan:
    def test_no_workspace_root_means_no_anchor_checks(self, tmp_path: Path) -> None:
        plan_dir = make_plan_dir(tmp_path)
        write_source(tmp_path / "src", "a.py", anchor("D-404") + "\nx = 1\n")
        assert [tag for tag in tags(audit(plan_dir)) if tag.startswith("anchor")] == []

    def test_an_absent_workspace_root_is_an_error(self, tmp_path: Path) -> None:
        issues = only(
            audit(make_plan_dir(tmp_path), workspace_root=tmp_path / "gone"),
            "anchor-orphan",
        )
        assert [issue.severity for issue in issues] == [Severity.ERROR]
        assert not (tmp_path / "gone").exists()

    def test_an_anchor_with_no_decision_entry_is_an_orphan(
        self, tmp_path: Path
    ) -> None:
        plan_dir = make_plan_dir(tmp_path)
        source = tmp_path / "src"
        write_source(source, "roles.py", anchor("D-404") + "\nx = 1\n")
        issues = only(audit(plan_dir, workspace_root=source), "anchor-orphan")
        assert [issue.severity for issue in issues] == [Severity.ERROR]
        assert "D-404" in issues[0].message

    def test_an_anchor_with_a_matching_entry_is_clean(self, tmp_path: Path) -> None:
        plan_dir = make_plan_dir(tmp_path)
        source = tmp_path / "src"
        write_source(
            source, "fsm_llm_harness/roles.py", "\n" * 11 + anchor("D-001") + "\n"
        )
        assert [
            tag
            for tag in tags(audit(plan_dir, workspace_root=source))
            if "anchor" in tag
        ] == []

    def test_the_commit_tag_form_is_rejected(self, tmp_path: Path) -> None:
        """D-014's real defect: anchors keep `THHMMSS`, commit tags drop it."""
        plan_dir = make_plan_dir(tmp_path)
        source = tmp_path / "src"
        write_source(source, "roles.py", anchor("D-001", prefix=COMMIT_TAG) + "\n")
        issues = only(audit(plan_dir, workspace_root=source), "anchor-badprefix")
        assert [issue.severity for issue in issues] == [Severity.ERROR]
        assert COMMIT_TAG in issues[0].message and PLAN_ID in issues[0].message

    def test_an_unqualified_anchor_warns(self, tmp_path: Path) -> None:
        plan_dir = make_plan_dir(tmp_path)
        source = tmp_path / "src"
        write_source(source, "roles.py", anchor("D-001", prefix=None) + "\n")
        issues = only(audit(plan_dir, workspace_root=source), "anchor-unqualified")
        assert [issue.severity for issue in issues] == [Severity.WARNING]
        assert "roles.py:1" in issues[0].artifact

    def test_another_plans_anchor_is_not_this_plans_business(
        self, tmp_path: Path
    ) -> None:
        plan_dir = make_plan_dir(tmp_path)
        source = tmp_path / "src"
        write_source(
            source,
            "roles.py",
            anchor("D-404", prefix="plan-2026-01-01T000000-deadbeef") + "\n",
        )
        assert [
            tag
            for tag in tags(audit(plan_dir, workspace_root=source))
            if "anchor" in tag
        ] == []

    def test_anchors_quoted_inside_the_plan_directory_are_skipped(
        self, tmp_path: Path
    ) -> None:
        plan_dir = make_plan_dir(tmp_path)
        (plan_dir / "findings" / "quoting.md").write_text(
            "# Finding: quoting\n\n## Summary\nThe anchor `"
            + anchor("D-404")
            + "` is prose.\n"
            "\n## Key Findings\n- x\n\n## Constraints\n- x\n\n## Code Patterns\n- x\n"
            "\n## Risks & Unknowns\n- x\n"
        )
        issues = audit(plan_dir, workspace_root=tmp_path)
        assert "anchor-orphan" not in tags(issues)

    @pytest.mark.parametrize("marker", ["#", "//", "--", "/*", "     #"])
    def test_the_anchor_is_recognised_in_any_comment_syntax(
        self, tmp_path: Path, marker: str
    ) -> None:
        plan_dir = make_plan_dir(tmp_path)
        source = tmp_path / "src"
        write_source(source, "a.txt", anchor("D-404", marker=marker) + " */\n")
        assert only(audit(plan_dir, workspace_root=source), "anchor-orphan")

    def test_a_truncated_scan_is_reported_as_unverified(self, tmp_path: Path) -> None:
        plan_dir = make_plan_dir(tmp_path)
        source = tmp_path / "src"
        # More files than the confined walker will open in one pass.
        for index in range(2100):
            write_source(source, f"pkg{index // 100}/mod{index}.py", "x = 1\n")
        issues = only(audit(plan_dir, workspace_root=source), "anchor-orphan")
        assert [issue.severity for issue in issues] == [Severity.WARNING]
        assert "UNVERIFIED" in issues[0].message

    # -- Anchor-Refs back-links -----------------------------------------

    def test_an_anchored_decision_without_anchor_refs_is_an_error(
        self, tmp_path: Path
    ) -> None:
        broken = DECISIONS_MD.replace(
            "**Anchor-Refs**: `src/fsm_llm_harness/roles.py:12`\n", ""
        )
        plan_dir = make_plan_dir(tmp_path, decisions_md=broken)
        source = tmp_path / "src"
        write_source(
            source, "fsm_llm_harness/roles.py", "\n" * 11 + anchor("D-001") + "\n"
        )
        issues = only(audit(plan_dir, workspace_root=source), "anchor-refs-missing")
        assert [issue.severity for issue in issues] == [Severity.ERROR]
        assert "D-001" in issues[0].message

    def test_an_unanchored_decision_needs_no_back_link(self, tmp_path: Path) -> None:
        # D-002 carries no Anchor-Refs and is anchored nowhere: that is correct.
        plan_dir = make_plan_dir(tmp_path)
        source = tmp_path / "src"
        write_source(
            source, "fsm_llm_harness/roles.py", "\n" * 11 + anchor("D-001") + "\n"
        )
        assert only(audit(plan_dir, workspace_root=source), "anchor-refs-missing") == []

    def test_a_back_link_to_a_file_with_no_such_anchor_warns(
        self, tmp_path: Path
    ) -> None:
        plan_dir = make_plan_dir(tmp_path)
        source = tmp_path / "src"
        write_source(
            source, "fsm_llm_harness/tools.py", "\n" * 11 + anchor("D-001") + "\n"
        )
        issues = only(audit(plan_dir, workspace_root=source), "anchor-refs-stale")
        assert [issue.severity for issue in issues] == [Severity.WARNING]
        assert "roles.py" in issues[0].message

    def test_a_back_link_with_a_drifted_line_number_is_info(
        self, tmp_path: Path
    ) -> None:
        plan_dir = make_plan_dir(tmp_path)
        source = tmp_path / "src"
        write_source(
            source, "fsm_llm_harness/roles.py", "\n" * 40 + anchor("D-001") + "\n"
        )
        issues = only(audit(plan_dir, workspace_root=source), "anchor-refs-stale")
        assert [issue.severity for issue in issues] == [Severity.INFO]
        assert "line 41" in issues[0].message

    def test_prose_after_the_anchor_refs_line_is_not_a_reference(
        self, tmp_path: Path
    ) -> None:
        """The decisions parser folds an `**Outcome ...**` line into the value."""
        assert "`bench_step2.py:40`" in DECISIONS_MD
        plan_dir = make_plan_dir(tmp_path)
        source = tmp_path / "src"
        write_source(
            source, "fsm_llm_harness/roles.py", "\n" * 11 + anchor("D-001") + "\n"
        )
        issues = only(audit(plan_dir, workspace_root=source), "anchor-refs-stale")
        assert issues == []


class TestAnchorHelpers:
    @pytest.mark.parametrize(
        ("plan_id", "expected"),
        [
            ("plan-2026-07-21T191807-bf7ffe24", "plan-2026-07-21-bf7ffe24"),
            ("plan-2026-07-21-bf7ffe24", None),
            ("plan_2026-05-07_7556fb98", None),
            ("nonsense", None),
        ],
    )
    def test_commit_tag_derivation(self, plan_id: str, expected: str | None) -> None:
        assert _commit_tag_form(plan_id) == expected

    @pytest.mark.parametrize(
        ("reference", "scanned", "same"),
        [
            ("src/fsm_llm_harness/roles.py", "fsm_llm_harness/roles.py", True),
            ("fsm_llm_harness/roles.py", "src/fsm_llm_harness/roles.py", True),
            ("src/fsm_llm_harness/roles.py", "src/fsm_llm_harness/roles.py", True),
            ("src/fsm_llm_harness/roles.py", "src/fsm_llm_harness/rules.py", False),
            ("roles.py", "src/fsm_llm_agents/heroles.py", False),
        ],
    )
    def test_same_file_matches_by_path_suffix_both_ways(
        self, reference: str, scanned: str, same: bool
    ) -> None:
        assert _same_file(reference, scanned) is same
