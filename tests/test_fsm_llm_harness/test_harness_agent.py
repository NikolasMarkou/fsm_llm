"""Falsifying tests for ``fsm_llm_harness.harness.HarnessAgent``.

These tests exist to CATCH REGRESSIONS in the driver, not to document that it
works.  Each class corresponds to a mechanism decision recorded in
``plans/plan-2026-07-21T125237-191b2eb2/decisions.md``:

===============================  ==========================================
Class                            Decision it pins
===============================  ==========================================
``TestDispatchLedger``           D-017, D-021, D-022 (dispatch once)
``TestLeash``                    D-018, D-019 (2-attempt autonomy leash)
``TestFixAttemptResets``         D-018 (the three reset edges)
``TestIterationAccounting``      D-018, D-019 (iteration semantics + cap)
``TestApprovals``                D-015, D-023 (never auto-approve)
``TestReentrancy``               D-014 (single coordinator, invariant I5)
``TestFailureSurvival``          worker-raises / degrade paths
``TestFailClosed``               invariant I8 at the driver allowlist
``TestContextCaps``              D-020 (prompt-size caps)
``TestExtractionCannotOpenAGate``  D-044, criterion 18 (writer provenance)
``TestDriverOwnedTable``         D-044, D-053 (the seed/revert mechanisms)
``TestEntryBookkeeping``         D-044 / review W4 (flags cleared on entry)
``TestRoutingExclusivity``       D-044 (the two-layer ranking)
``TestLeashIsBounded``           D-051, D-052 (review C3, both escapes)
``TestSingleRunPerInstance``     D-055 (review W7)
===============================  ==========================================

Everything runs against ``MockLLM2Interface``: no network, no sleeps, no
ollama.  By DEFAULT the mock extracts nothing, so the protocol advances only
through worker replies and driver bookkeeping.  That default is a deliberate
choice, not an accident (decisions.md D-056): the fabricating LLM -- the second
writer into gate context, and the one review C4 found the whole suite had
mocked into silence -- is a first-class fixture axis instead, exercised by
``TestExtractionCannotOpenAGate`` over EVERY driver-owned key.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import pytest

from fsm_llm.handlers import HandlerTiming
from fsm_llm_agents.definitions import AgentResult
from fsm_llm_agents.exceptions import AgentError
from fsm_llm_harness import harness as harness_module
from fsm_llm_harness import storage as storage_module
from fsm_llm_harness.artifacts import (
    PRESENTATION_CONTRACTS,
    PlanDoc,
    Section,
    StateDoc,
)
from fsm_llm_harness.constants import (
    DRIVER_OWNED_SEEDS,
    DRIVER_OWNED_UNSET,
    ArtifactNames,
    ContextKeys,
    Defaults,
    GateSlug,
    HandlerNames,
    HarnessStates,
    PlanSchema,
    Role,
)
from fsm_llm_harness.exceptions import (
    HarnessError,
    HarnessOwnershipError,
    HarnessReentrancyError,
)
from fsm_llm_harness.harness import (
    EXECUTE_TARGET_ASSIGNED,
    EXECUTE_TARGET_ASSIGNED_PROSE,
    EXECUTE_TARGET_NO_PLAN_DIR,
    EXECUTE_TARGET_NO_PLAN_DOC,
    EXECUTE_TARGET_NO_TOKEN,
    HarnessAgent,
    RoleRequest,
    _derive_prose_target,
    derive_execute_target,
)
from fsm_llm_harness.plan_validator import Issue, _is_placeholder
from fsm_llm_harness.roles import get_role_spec
from fsm_llm_harness.rules import EXPLORE_TOPICS, ROLE_BY_STATE, get_rules
from tests.conftest import MockLLM2Interface
from tests.test_fsm_llm_harness.conftest import (
    APPROVAL_CLOSE,
    APPROVAL_GATES,
    APPROVAL_LEASH,
    APPROVAL_PLAN,
    APPROVAL_REVERT,
    FABRICATED_DRIVER_OWNED,
    ApprovalRecorder,
    RecordingWorker,
)

Call = tuple[str, int, int, int]


# ---------------------------------------------------------------------------
# Script builders
# ---------------------------------------------------------------------------


def _plan_is_substantive(text: str) -> bool:
    """Mirror the driver's ``_plan_has_content`` APPROVABLE rule.

    True only when *text* parses as a valid ``PlanDoc`` AND is
    ALL-non-placeholder (EVERY section body carries real content) -- the same
    bar the honest approval gate uses (`_plan_is_approvable`), since approval
    denial does not redispatch.  Stated here rather than imported so a
    regression that loosens the driver's own bar cannot loosen the fixture's at
    the same time -- the same reason :func:`_is_exactly` restates ``_exactly``.
    """
    try:
        plan = PlanDoc.from_markdown(text)
    except Exception:
        return False
    return all(not _is_placeholder(section.body) for section in plan.sections)


def _write_plan_md(request: RoleRequest) -> None:
    """Put a real, substantive ``plan.md`` on disk -- a FAITHFUL plan-writer.

    Since plan-2026-07-23 the driver SEEDS a HEADERS-ONLY ``plan.md`` scaffold
    at PLAN entry (all 11 sections, EMPTY bodies), and ``_plan_has_content`` is
    SUBSTANTIVE (valid ``PlanDoc`` AND not-all placeholder).  A faithful
    plan-writer therefore FILLS that scaffold -- modelled here as writing the
    substantive ``_PLAN_MD`` over the bare headers, exactly as a live
    plan-writer's append turns produce a filled plan.  It overwrites a seeded
    scaffold or an empty file, stays idempotent for an already-substantive plan
    (a completion-fix re-entry), and never truncates a real plan.  Writes only
    when a plan directory is configured.
    """
    if request.plan_dir is None:
        return
    plan_path = Path(request.plan_dir) / ArtifactNames.PLAN
    if plan_path.exists() and _plan_is_substantive(plan_path.read_text()):
        return
    plan_path.write_text(_PLAN_MD)


def _traverse_script(*, total_steps: int = 1, findings: int = 3) -> dict[str, Any]:
    """A script that walks EXPLORE -> PLAN -> EXECUTE -> REFLECT -> CLOSE."""

    def _plan(request: RoleRequest) -> dict[str, Any]:
        _write_plan_md(request)
        return {"ctx": {ContextKeys.TOTAL_STEPS: total_steps}}

    return {
        HarnessStates.EXPLORE: {"ctx": {ContextKeys.FINDINGS_COUNT: findings}},
        HarnessStates.PLAN: _plan,
        HarnessStates.EXECUTE: {"ctx": {}},
        HarnessStates.REFLECT: {"ctx": {ContextKeys.ALL_CRITERIA_PASS: True}},
        HarnessStates.PIVOT: {"ctx": {ContextKeys.PIVOT_RESOLVED: True}},
        HarnessStates.CLOSE: {"ctx": {}},
    }


def _failing_execute_script(*, total_steps: int = 1) -> dict[str, Any]:
    """A script whose executor never succeeds; everything else is well-behaved."""
    script = _traverse_script(total_steps=total_steps)
    script[HarnessStates.EXECUTE] = {"success": False, "ctx": {}}
    #: REFLECT writes no routing flag -- the driver's own bookkeeping decides
    #: between a completion fix and a leash halt, which is what is under test.
    script[HarnessStates.REFLECT] = {"ctx": {}}
    return script


def _all_failing_script() -> dict[str, Any]:
    """Every dispatch fails -- the live spike's measured 6/6 shape.

    With no successful worker anywhere, the driver's own value for every
    driver-owned key is exactly its seed, which is what makes
    ``TestExtractionCannotOpenAGate``'s assertions exact rather than
    approximate.
    """
    return {state: {"success": False, "ctx": {}} for state in HarnessStates.ALL}


# ---------------------------------------------------------------------------
# Plan-directory fixtures for the artifact seam
# ---------------------------------------------------------------------------

#: A ``plan.md`` with all 11 sections in order and two annotated steps.  Real
#: shape, not a stub: ``PlanDoc`` REJECTS a plan missing any of them, so a
#: smoothed-over fixture would exercise only the unreadable branch.
_PLAN_MD = """# Plan v1: exercise the harness protocol

## Goal
Make the driver read and write the artifacts it owns.

## Problem Statement
The driver kept all protocol memory in the FSM context.

## Context
`findings.md` carries the index; `plan.md` carries the steps.

## Files To Modify
| File | Change | Reason |
|---|---|---|
| `harness.py` | wire the seam | step 10 |

## Steps
1. [x] **Wire the pre-step gate onto the on-disk validator.** [RISK: high] [deps: none]
2. [ ] **Emit the presentation contracts from the artifacts.** [RISK: low] [deps: 1]

## Assumptions
- **A1.** The plan directory is writable.

## Failure Modes
| Dependency | Slow | Bad data | Down | Blast radius |
|---|---|---|---|---|
| plan directory | n/a | truncated artifact | gate goes dark | the whole gate |

## Pre-Mortem & Falsification Signals
1. **The gate reads what it just wrote.** → **STOP IF** `wrong-state` is unreachable.

## Success Criteria
1. Each of the four slugs produces its own action.

## Verification Strategy
| # | Criterion | Method | Command | Pass condition |
|---|---|---|---|---|
| 1 | Four slugs, four actions | Automated | `pytest -k FourSlugs` | all pass |

## Complexity Budget
| Metric | Budget | Notes |
|---|---|---|
| Files added | 0/0 | the seam adds none |
"""


def _hollow_plan_doc() -> PlanDoc:
    """A schema-VALID ``plan.md`` PlanDoc with all 11 sections present in order
    but EMPTY bodies (every section reads as a placeholder).

    Test helper: mirrors what the removed ``_empty_plan_scaffold`` produced, so
    a test can assert the gate DENIES a hollow plan without the deleted seed
    machinery.  Callers may fill ``sections[i].body`` to build partially-filled
    plans.
    """
    return PlanDoc(
        title="Plan",
        sections=[Section(name=name, body="") for name in PlanSchema.SECTIONS],
    )


#: A ``changelog.md`` whose ledger line names the step the fixture runs.
_CHANGELOG_MD = """# Changelog
*Append-only per-edit ledger. One line per file edit. Owner: ip-executor.*
2026-07-22T09:00:00Z | iter-1/step-1 | abc1234 | src/fsm_llm_harness/harness.py | EDIT(+300,-40) | radius:HIGH(9) | D-040 | wire the four-slug action table
"""

_FINDINGS_MD = """# Findings
*Summary and index of all findings.*

## Index
1. `findings/seam.md` — the driver writes exactly one artifact
2. `findings/gate.md` — the four slugs are four events
3. `findings/caps.md` — the agent read cap is not the driver's

## Key Constraints
- **HARD**: `plans/` is gitignored, so protocol memory has no VCS backstop.
"""

_PROGRESS_MD = """# Progress

## Completed
- [x] step 9: `plan_validator.py`

## In Progress
- [ ] step 10: the integration seam

## Remaining
- [ ] step 12: the CLI

## Blocked
*Nothing currently.*
"""

_VERIFICATION_MD = """# Verification Results (Iteration 1)

## Criteria Verification
| # | Criterion (from plan.md) | Method | Command/Action | Result | Evidence |
|---|---|---|---|---|---|
| 1 | Four slugs, four actions | Automated | `pytest -k FourSlugs` | PASS | 8/8 |

## Additional Checks
| Check | Command/Action | Result | Details |
|---|---|---|---|
| Regression | `pytest -q` | PASS | 1186/1186 |
| Scope drift | manifest vs plan.md | PASS | manual review — no unplanned file |
| Diff review | `git diff` | PASS | manual review — no stray prints |

## Not Verified
| What | Why |
|---|---|
| Live `:4b` traverse | step 15 owns it |

## Verdict
- Criteria passed: 1/1
- Regressions: none
- Scope drift: none
- Simplification blockers: none
- Recommendation: → CLOSE
"""


def _seed_plan_directory(plan_dir: Path) -> None:
    """Write every artifact the contract builders quote.

    Without this the blocks are emitted with EMPTY floor fields -- which is the
    honest behaviour and has its own test, but makes a "the floor is filled"
    assertion vacuous.  The three ``findings/*.md`` files are what lets the run
    LEAVE explore at all: since D-032 the gate counts files on disk, not what a
    worker says it wrote.
    """
    (plan_dir / ArtifactNames.PLAN).write_text(_PLAN_MD)
    (plan_dir / ArtifactNames.CHANGELOG).write_text(_CHANGELOG_MD)
    (plan_dir / ArtifactNames.FINDINGS_INDEX).write_text(_FINDINGS_MD)
    (plan_dir / ArtifactNames.PROGRESS).write_text(_PROGRESS_MD)
    (plan_dir / ArtifactNames.VERIFICATION).write_text(_VERIFICATION_MD)
    (plan_dir / "checkpoints" / "cp-000-iter1.md").write_text("# Checkpoint\n")
    _seed_findings(plan_dir, "seam", "gate", "caps")


def _is_exactly(value: Any, expected: Any) -> bool:
    """Same runtime type AND equal -- ``False == 0`` must not pass for an int.

    Mirrors the driver's own ``_exactly``; stated here rather than imported so
    a regression that loosens the driver's comparison cannot loosen the test's
    at the same time.
    """
    return type(value) is type(expected) and bool(value == expected)


def _occupancies(calls: list[Call]) -> list[list[Call]]:
    """Split a dispatch sequence into maximal same-state runs.

    A state only dispatches while it is occupied, and every state in this FSM
    dispatches on entry, so a maximal run of consecutive dispatches for one
    state is exactly one state occupancy.  That is the unit invariant I4 is
    stated over (decisions.md D-022).
    """
    runs: list[list[Call]] = []
    for call in calls:
        if runs and runs[-1][0][0] == call[0]:
            runs[-1].append(call)
        else:
            runs.append([call])
    return runs


# ---------------------------------------------------------------------------
# Dispatch (invariant I4, D-017 / D-021 / D-022)
# ---------------------------------------------------------------------------


class TestDispatchLedger:
    """At most one dispatch per ``(state, iteration, step)`` per occupancy."""

    def test_full_traverse_dispatches_each_states_role(self, make_harness) -> None:
        """One dispatch per state, in protocol order, ending at CLOSE."""
        harness = make_harness(_traverse_script())
        result = harness.run()

        assert harness.worker.states == [
            HarnessStates.EXPLORE,
            HarnessStates.PLAN,
            HarnessStates.EXECUTE,
            HarnessStates.REFLECT,
            HarnessStates.CLOSE,
        ]
        assert harness.worker.roles == [
            Role.EXPLORER,
            Role.PLAN_WRITER,
            Role.EXECUTOR,
            Role.VERIFIER,
            Role.ARCHIVIST,
        ]
        assert result.success is True

    def test_initial_state_is_dispatched_exactly_once(self, make_harness) -> None:
        """``on_state_entry`` never fires for the initial state (assumption A1).

        The driver compensates with a ``START_CONVERSATION`` handler.  If that
        handler were dropped, this count is 0; if it double-fired with the
        entry handler, it is 2.
        """
        harness = make_harness(_traverse_script())
        harness.run()
        assert harness.worker.count_for(HarnessStates.EXPLORE) == 1

    def test_no_two_dispatches_within_one_occupancy(self, make_harness) -> None:
        """The ledger's occupancy marker makes a duplicate fire a no-op.

        Run a completion fix, which legitimately re-enters both EXECUTE and
        REFLECT at the *same* ``(state, iteration, step)`` key -- the case that
        forced D-022's wording.  Both re-entries are legal; two dispatches
        inside one occupancy would not be.
        """
        seen: dict[str, int] = {}

        def executor(request: RoleRequest) -> dict[str, Any]:
            seen["execute"] = seen.get("execute", 0) + 1
            return {"success": seen["execute"] > 1}

        script = _traverse_script()
        script[HarnessStates.EXECUTE] = executor
        script[HarnessStates.REFLECT] = lambda r: (
            {"ctx": {ContextKeys.ALL_CRITERIA_PASS: True}}
            if seen.get("execute", 0) > 1
            else {"ctx": {}}
        )

        harness = make_harness(script)
        harness.run()

        calls = harness.worker.calls
        # The same key really is dispatched twice across the whole run...
        assert calls.count((HarnessStates.EXECUTE, 1, 1, 0)) >= 1
        assert harness.worker.count_for(HarnessStates.EXECUTE) == 2
        # ...but never twice inside one occupancy.
        for run in _occupancies(calls):
            keys = [(state, it, step) for state, it, step, _ in run]
            assert len(keys) == len(set(keys)), (
                f"two dispatches for the same key inside one occupancy: {run}"
            )

    def test_multi_step_execute_dispatches_once_per_step(self, make_harness) -> None:
        """A held EXECUTE dispatches via ``_on_loop_iteration``, once per step.

        ``on_state_entry`` does not fire while a state is held, so without the
        loop hook a 3-step plan would run step 1 and then sit there.
        """
        harness = make_harness(_traverse_script(total_steps=3))
        harness.run()

        execute_calls = harness.worker.calls_for(HarnessStates.EXECUTE)
        assert [step for _, _, step, _ in execute_calls] == [1, 2, 3]
        assert harness.worker.count_for(HarnessStates.EXECUTE) == 3

    def test_duplicate_handler_fire_with_same_timestamp_is_a_noop(
        self, make_harness
    ) -> None:
        """Two fires for one transition: the second returns an empty delta."""
        harness = make_harness(_traverse_script(total_steps=3))
        context = {
            ContextKeys.ITERATION: 1,
            ContextKeys.STEP_NUMBER: 1,
            ContextKeys.TOTAL_STEPS: 3,
            ContextKeys.FIX_ATTEMPTS: 0,
            ContextKeys.DISPATCH_LEDGER: [],
            ContextKeys.ROLE_RESULTS: [],
            "_transition_timestamp": 12345.0,
        }

        first = harness.agent._on_state_entry(HarnessStates.EXECUTE, dict(context))
        assert harness.worker.count_for(HarnessStates.EXECUTE) == 1
        assert first, "the first fire must produce a delta"

        merged = {**context, **{k: v for k, v in first.items() if v is not None}}
        second = harness.agent._on_state_entry(HarnessStates.EXECUTE, merged)

        assert second == {}
        assert harness.worker.count_for(HarnessStates.EXECUTE) == 1

    def test_new_transition_timestamp_reauthorises_one_dispatch(
        self, make_harness
    ) -> None:
        """A genuine re-entry re-authorises exactly one further dispatch."""
        harness = make_harness(_traverse_script(total_steps=3))
        context = {
            ContextKeys.ITERATION: 1,
            ContextKeys.STEP_NUMBER: 1,
            ContextKeys.TOTAL_STEPS: 3,
            ContextKeys.FIX_ATTEMPTS: 0,
            ContextKeys.DISPATCH_LEDGER: [],
            ContextKeys.ROLE_RESULTS: [],
            "_transition_timestamp": 12345.0,
        }

        first = harness.agent._on_state_entry(HarnessStates.EXECUTE, dict(context))
        merged = {**context, **{k: v for k, v in first.items() if v is not None}}
        merged["_transition_timestamp"] = 99999.0
        harness.agent._on_state_entry(HarnessStates.EXECUTE, merged)

        assert harness.worker.count_for(HarnessStates.EXECUTE) == 2

    def test_ledger_records_every_dispatch_key(self, make_harness) -> None:
        """The on-context ledger mirrors the dispatch sequence."""
        harness = make_harness(_traverse_script(total_steps=2))
        result = harness.run()

        keys = [
            entry
            for entry in result.final_context[ContextKeys.DISPATCH_LEDGER]
            if entry.startswith("dispatch:")
        ]
        assert "dispatch:explore:0:0" in keys
        assert "dispatch:execute:1:1" in keys
        assert "dispatch:execute:1:2" in keys
        assert "dispatch:close:1:2" in keys


# ---------------------------------------------------------------------------
# Autonomy leash (invariant I2)
# ---------------------------------------------------------------------------


class TestLeash:
    """The 2-attempt leash caps executor dispatches, not just messages."""

    def test_repeated_failure_stops_at_exactly_two_dispatches(
        self, make_harness
    ) -> None:
        """Exactly 2 executor dispatches, never 3.

        The dispatch COUNT is what the leash actually protects; a run that
        reported ``leash-cap`` while still having dispatched a third attempt
        would satisfy a slug-only assertion and violate the invariant.
        """
        harness = make_harness(
            _failing_execute_script(),
            approvals=ApprovalRecorder({APPROVAL_LEASH: False}),
        )
        result = harness.run()

        assert harness.worker.count_for(HarnessStates.EXECUTE) == 2
        assert result.final_context[ContextKeys.FIX_ATTEMPTS] == 2
        assert result.final_context[ContextKeys.LAST_GATE_SLUG] == GateSlug.LEASH_CAP

    def test_leash_halt_routes_into_reflect(self, make_harness) -> None:
        """The leash reports from REFLECT; it does not end the run in EXECUTE."""
        harness = make_harness(
            _failing_execute_script(),
            approvals=ApprovalRecorder({APPROVAL_LEASH: False}),
        )
        result = harness.run()

        assert harness.worker.states[-1] == HarnessStates.REFLECT
        assert harness.worker.count_for(HarnessStates.CLOSE) == 0
        assert "REFLECT" in result.answer.upper()
        assert result.success is False

    def test_leash_is_configurable(self, make_harness) -> None:
        """A wider leash really does permit more attempts."""
        harness = make_harness(
            _failing_execute_script(),
            approvals=ApprovalRecorder({APPROVAL_LEASH: False}),
            max_fix_attempts=3,
        )
        result = harness.run()

        assert harness.worker.count_for(HarnessStates.EXECUTE) == 3
        assert result.final_context[ContextKeys.FIX_ATTEMPTS] == 3

    def test_leash_consults_the_human_before_reporting(self, make_harness) -> None:
        """A leash halt is a turn-gate, not a silent stop."""
        harness = make_harness(
            _failing_execute_script(),
            approvals=ApprovalRecorder({APPROVAL_LEASH: False}),
        )
        harness.run()
        assert harness.approvals.count(APPROVAL_LEASH) >= 1


class TestFixAttemptResets:
    """``fix_attempts`` resets on a new step, on a leash-continue, and on PIVOT."""

    def test_reset_on_a_new_step(self, make_harness) -> None:
        """A fresh step starts at 0 attempts, not at the previous step's count.

        Without the reset, step 2's first failure would read
        ``fix_attempts == 1`` and its second dispatch would be refused, so the
        executor count would be 3 and the ``(execute, 1, 2, 0)`` dispatch below
        would never appear.
        """

        def executor(request: RoleRequest) -> dict[str, Any]:
            first_try_of_step_one = (
                request.step_number == 1 and request.fix_attempts == 0
            )
            succeed = request.step_number == 1 and not first_try_of_step_one
            return {"success": succeed}

        script = _failing_execute_script(total_steps=2)
        script[HarnessStates.EXECUTE] = executor

        harness = make_harness(
            script, approvals=ApprovalRecorder({APPROVAL_LEASH: False})
        )
        result = harness.run()

        calls = harness.worker.calls_for(HarnessStates.EXECUTE)
        assert calls == [
            (HarnessStates.EXECUTE, 1, 1, 0),
            (HarnessStates.EXECUTE, 1, 1, 1),
            (HarnessStates.EXECUTE, 1, 2, 0),
            (HarnessStates.EXECUTE, 1, 2, 1),
        ]
        assert result.final_context[ContextKeys.LAST_GATE_SLUG] == GateSlug.LEASH_CAP

    def test_reset_on_a_granted_leash_continue(self, make_harness) -> None:
        """Explicit user direction is not a third unattended attempt.

        Grant the first leash-continue and deny the second: the reset must
        buy exactly two more executor dispatches.  With the reset dropped, the
        pre-step gate would refuse and the count would stay at 2.
        """
        harness = make_harness(
            _failing_execute_script(),
            approvals=ApprovalRecorder({APPROVAL_LEASH: lambda index: index == 1}),
        )
        result = harness.run()

        assert harness.approvals.count(APPROVAL_LEASH) == 2
        assert harness.worker.count_for(HarnessStates.EXECUTE) == 4
        assert result.final_context[ContextKeys.FIX_ATTEMPTS] == 2
        assert result.final_context[ContextKeys.LAST_GATE_SLUG] == GateSlug.LEASH_CAP

    def test_reset_on_pivot(self, make_harness) -> None:
        """The leash never carries across a pivot."""
        reflect_calls: list[int] = []

        def reflect(request: RoleRequest) -> dict[str, Any]:
            reflect_calls.append(request.fix_attempts)
            if len(reflect_calls) == 1:
                return {"ctx": {ContextKeys.NEEDS_PIVOT: True}}
            return {"ctx": {ContextKeys.ALL_CRITERIA_PASS: True}}

        def executor(request: RoleRequest) -> dict[str, Any]:
            return {"success": request.iteration > 1}

        script = _traverse_script()
        script[HarnessStates.EXECUTE] = executor
        script[HarnessStates.REFLECT] = reflect

        harness = make_harness(script)
        harness.run()

        assert reflect_calls[0] == 1, "the verifier should see the spent attempt"
        pivot_calls = harness.worker.calls_for(HarnessStates.PIVOT)
        assert pivot_calls, "the run never reached PIVOT"
        assert pivot_calls[0][3] == 0, "PIVOT entry must clear fix_attempts"


# ---------------------------------------------------------------------------
# Iteration accounting (D-018) and the iteration cap (D-019)
# ---------------------------------------------------------------------------


class TestIterationAccounting:
    """``iteration`` counts re-planned attempts, nothing else."""

    def test_increments_on_plan_to_execute(self, make_harness) -> None:
        harness = make_harness(_traverse_script())
        result = harness.run()
        assert result.final_context[ContextKeys.ITERATION] == 1

    def test_completion_fix_does_not_increment(self, make_harness) -> None:
        """REFLECT -> EXECUTE is the same iteration by definition."""
        seen: dict[str, int] = {}

        def executor(request: RoleRequest) -> dict[str, Any]:
            seen["n"] = seen.get("n", 0) + 1
            return {"success": seen["n"] > 1}

        script = _traverse_script()
        script[HarnessStates.EXECUTE] = executor
        script[HarnessStates.REFLECT] = lambda r: (
            {"ctx": {ContextKeys.ALL_CRITERIA_PASS: True}}
            if seen.get("n", 0) > 1
            else {"ctx": {}}
        )

        harness = make_harness(script)
        result = harness.run()

        assert harness.worker.count_for(HarnessStates.EXECUTE) == 2
        assert result.final_context[ContextKeys.ITERATION] == 1

    def test_pivot_loop_back_increments_on_the_next_plan_edge(
        self, make_harness
    ) -> None:
        """A pivot itself does not count; the re-plan that follows it does."""
        reflect_calls: list[int] = []

        def reflect(request: RoleRequest) -> dict[str, Any]:
            reflect_calls.append(request.iteration)
            if len(reflect_calls) == 1:
                return {"ctx": {ContextKeys.NEEDS_PIVOT: True}}
            return {"ctx": {ContextKeys.ALL_CRITERIA_PASS: True}}

        script = _traverse_script()
        script[HarnessStates.REFLECT] = reflect

        harness = make_harness(script)
        result = harness.run()

        pivot_calls = harness.worker.calls_for(HarnessStates.PIVOT)
        assert pivot_calls[0][1] == 1, "PIVOT must not increment the iteration"
        assert result.final_context[ContextKeys.ITERATION] == 2

    def test_iteration_cap_halts_without_a_leash_presentation(
        self, make_harness
    ) -> None:
        """The cap is a different halt from the leash (decisions.md D-019).

        A run can reach the cap with zero fix attempts; reporting a leash there
        would point the user at a failing step that does not exist.  The
        *absence* assertions below are the point of this test.
        """
        script = _traverse_script()
        script[HarnessStates.REFLECT] = {"ctx": {ContextKeys.NEEDS_PIVOT: True}}

        harness = make_harness(script, iteration_hard_cap=2)
        result = harness.run()

        assert result.success is False
        assert (
            result.final_context[ContextKeys.LAST_GATE_SLUG] == GateSlug.ITERATION_CAP
        )
        assert result.final_context[ContextKeys.FIX_ATTEMPTS] == 0
        assert "iteration cap" in result.answer.lower()

        # No leash anywhere: not the slug, not the gate, not the wording.
        assert result.final_context[ContextKeys.LAST_GATE_SLUG] != GateSlug.LEASH_CAP
        assert harness.approvals.count(APPROVAL_LEASH) == 0
        assert "leash" not in result.answer.lower()


# ---------------------------------------------------------------------------
# Approvals (invariant I6, D-015 / D-023)
# ---------------------------------------------------------------------------


class TestApprovals:
    """The callback is consulted at every gate, and the default denies.

    TIGHTENED at step 7d.  Five assertions here read
    ``final_context.get(key) is not True``, which is satisfied by an ABSENT key
    -- and an absent gate flag is precisely the pre-7a state in which the FSM's
    own Pass-1 extraction was asked to invent one (review C1).  Since D-044
    every driver-owned flag is present-and-falsy from turn 1, so the assertions
    are now ``final_context[key] is False``: a regression that deletes a flag
    instead of clearing it fails here rather than passing quietly.
    """

    def test_approval_gate_names_are_stable(self) -> None:
        """Consumer callbacks branch on these strings; a rename breaks them."""
        assert APPROVAL_PLAN == "harness.approve_plan"
        assert APPROVAL_CLOSE == "harness.confirm_close"
        assert APPROVAL_LEASH == "harness.continue_after_leash"
        assert APPROVAL_REVERT == "harness.revert_uncommitted"
        assert len(set(APPROVAL_GATES)) == 4

    def test_default_callback_denies(self) -> None:
        """An unattended run cannot approve its own plan (invariant I6)."""
        worker = RecordingWorker(_traverse_script())
        agent = HarnessAgent(worker_factory=worker, llm_interface=MockLLM2Interface())
        result = agent.run("unattended goal")

        assert worker.count_for(HarnessStates.EXECUTE) == 0
        assert result.final_context[ContextKeys.PLAN_APPROVED] is False
        assert result.success is False

    def test_plan_gate_consults_the_callback(self, make_harness) -> None:
        """Assert CONSULTATION, not merely the outcome (decisions.md D-023).

        ``HumanInTheLoop.request_approval`` currently has no auto-approve
        branch at all.  If a future core release reintroduces one, a
        deny-and-check-the-gate test would still pass while the harness
        silently auto-approved; this one fails.
        """
        harness = make_harness(_traverse_script())
        harness.run()

        assert harness.approvals.count(APPROVAL_PLAN) >= 1
        request = next(
            r for r in harness.approvals.requests if r.tool_name == APPROVAL_PLAN
        )
        assert ContextKeys.ITERATION in request.parameters
        assert request.reasoning

    def test_close_gate_consults_the_callback(self, make_harness) -> None:
        harness = make_harness(_traverse_script())
        harness.run()
        assert harness.approvals.count(APPROVAL_CLOSE) == 1

    def test_denied_plan_gate_holds_plan_and_stalls(self, make_harness) -> None:
        harness = make_harness(
            _traverse_script(), approvals=ApprovalRecorder(default=False)
        )
        result = harness.run()

        assert harness.worker.states == [HarnessStates.EXPLORE, HarnessStates.PLAN]
        assert harness.approvals.count(APPROVAL_PLAN) >= 1
        assert harness.approvals.count(APPROVAL_CLOSE) == 0
        assert result.success is False

    def test_denied_close_gate_never_reaches_close(self, make_harness) -> None:
        harness = make_harness(
            _traverse_script(),
            approvals=ApprovalRecorder({APPROVAL_CLOSE: False}),
        )
        result = harness.run()

        assert harness.approvals.count(APPROVAL_CLOSE) >= 1
        assert harness.worker.count_for(HarnessStates.CLOSE) == 0
        assert result.final_context[ContextKeys.CLOSE_CONFIRMED] is False

    def test_every_gate_is_distinguishable_by_tool_name(
        self, make_harness, roots, plan_dir
    ) -> None:
        """One callback serves all four gates and can tell them apart.

        The revert gate only exists when a ``revert_callback`` was supplied --
        without one there is nothing to approve, because nothing is executed
        (D-039) -- so it takes its own run to observe.
        """
        traverse = make_harness(_traverse_script())
        traverse.run()
        leash = make_harness(
            _failing_execute_script(),
            approvals=ApprovalRecorder({APPROVAL_LEASH: False}),
        )
        leash.run()
        _seed_findings(plan_dir, "alpha", "beta", "gamma")
        reverting = make_harness(
            _failing_execute_script(),
            approvals=ApprovalRecorder({APPROVAL_LEASH: False}),
            roots=roots,
            revert_callback=lambda directive: True,
        )
        reverting.run()

        observed = (
            set(traverse.approvals.names)
            | set(leash.approvals.names)
            | set(reverting.approvals.names)
        )
        assert observed == set(APPROVAL_GATES)

    def test_a_raising_callback_is_treated_as_denied(self, make_harness) -> None:
        """A broken approval path denies; it never opens the gate, never crashes."""
        approvals = ApprovalRecorder(raises=RuntimeError("approval UI is down"))
        harness = make_harness(_traverse_script(), approvals=approvals)
        result = harness.run()

        assert approvals.count(APPROVAL_PLAN) >= 1
        assert harness.worker.count_for(HarnessStates.EXECUTE) == 0
        assert result.final_context[ContextKeys.PLAN_APPROVED] is False

    @pytest.mark.parametrize(
        "key", [ContextKeys.PLAN_APPROVED, ContextKeys.CLOSE_CONFIRMED]
    )
    def test_worker_set_approval_flags_are_ignored(
        self, make_harness, key: str
    ) -> None:
        """A worker cannot vote itself through a human gate."""
        script = _traverse_script()
        script[HarnessStates.EXPLORE] = {
            "ctx": {ContextKeys.FINDINGS_COUNT: 3, key: True}
        }
        script[HarnessStates.PLAN] = {"ctx": {ContextKeys.TOTAL_STEPS: 1, key: True}}
        script[HarnessStates.REFLECT] = {
            "ctx": {ContextKeys.ALL_CRITERIA_PASS: True, key: True}
        }

        harness = make_harness(script, approvals=ApprovalRecorder(default=False))
        result = harness.run()

        assert result.final_context[key] is False
        assert harness.worker.count_for(HarnessStates.EXECUTE) == 0


# ---------------------------------------------------------------------------
# Re-entrancy guard (invariant I5, D-014)
# ---------------------------------------------------------------------------


class TestReentrancy:
    """A dispatched worker may not become a second coordinator."""

    def test_worker_constructing_and_running_a_second_driver_raises(self) -> None:
        """The realistic re-entry: a worker spawning its own sub-planner.

        An instance-scoped flag could not see this, which is why the guard is
        module-level ``threading.local`` (decisions.md D-014).
        """

        def reentrant(request: RoleRequest) -> AgentResult:
            inner = HarnessAgent(llm_interface=MockLLM2Interface())
            inner.run("nested goal")
            return AgentResult(answer="unreachable", success=True)

        agent = HarnessAgent(
            worker_factory=reentrant, llm_interface=MockLLM2Interface()
        )
        with pytest.raises(HarnessReentrancyError) as excinfo:
            agent.run("outer goal")

        assert excinfo.value.role == Role.EXPLORER

    @pytest.mark.parametrize("attribute", ["api", "conversation_id"])
    def test_worker_touching_the_parent_driver_raises(self, attribute: str) -> None:
        """Reading the live ``API`` or conversation id from a worker is re-entry."""
        holder: dict[str, HarnessAgent] = {}

        def toucher(request: RoleRequest) -> AgentResult:
            getattr(holder["agent"], attribute)
            return AgentResult(answer="unreachable", success=True)

        agent = HarnessAgent(worker_factory=toucher, llm_interface=MockLLM2Interface())
        holder["agent"] = agent

        with pytest.raises(HarnessReentrancyError) as excinfo:
            agent.run("touch goal")

        assert excinfo.value.details["reentered"] == attribute

    def test_reentrancy_error_is_raised_as_itself_not_wrapped(self) -> None:
        """``BaseAgent`` wraps everything in ``AgentError``; this must survive.

        A broken I5 invariant reported as a generic "Harness execution failed"
        is indistinguishable from any other run failure.
        """

        def reentrant(request: RoleRequest) -> AgentResult:
            HarnessAgent(llm_interface=MockLLM2Interface()).run("nested")
            return AgentResult(answer="unreachable", success=True)

        agent = HarnessAgent(
            worker_factory=reentrant, llm_interface=MockLLM2Interface()
        )
        with pytest.raises(HarnessReentrancyError) as excinfo:
            agent.run("outer goal")

        assert not isinstance(excinfo.value, AgentError)
        assert type(excinfo.value) is HarnessReentrancyError

    def test_reentrancy_cause_chain_is_acyclic(self) -> None:
        """``raise reentry from None`` -- never ``from exc``.

        The re-entrancy error already lives inside the wrapper's ``__cause__``
        chain, so re-raising it *from* the wrapper closes a cycle and hangs any
        naive ``while error.__cause__`` walker.
        """

        def reentrant(request: RoleRequest) -> AgentResult:
            HarnessAgent(llm_interface=MockLLM2Interface()).run("nested")
            return AgentResult(answer="unreachable", success=True)

        agent = HarnessAgent(
            worker_factory=reentrant, llm_interface=MockLLM2Interface()
        )
        with pytest.raises(HarnessReentrancyError) as excinfo:
            agent.run("outer goal")

        seen: set[int] = set()
        current: BaseException | None = excinfo.value
        hops = 0
        while current is not None:
            assert id(current) not in seen, "cause chain contains a cycle"
            assert hops < 50, "cause chain did not terminate"
            seen.add(id(current))
            current = current.__cause__
            hops += 1

        assert excinfo.value.__cause__ is None

    def test_the_guard_clears_after_a_normal_dispatch(self, make_harness) -> None:
        """A completed dispatch must not leave the thread marked as in-flight."""
        harness = make_harness(_traverse_script())
        harness.run()
        # A second run on the same thread would raise immediately if the guard
        # leaked, so simply running twice is the assertion.
        second = make_harness(_traverse_script())
        result = second.run()
        assert result.success is True


# ---------------------------------------------------------------------------
# Failure survival
# ---------------------------------------------------------------------------


class TestFailureSurvival:
    """A bad worker degrades the turn; it never takes the run down."""

    def test_raising_worker_is_recorded_as_a_failed_result(self, make_harness) -> None:
        script = _traverse_script()
        script[HarnessStates.EXPLORE] = {"raises": RuntimeError("boom from explorer")}

        harness = make_harness(script)
        result = harness.run()

        results = result.final_context[ContextKeys.ROLE_RESULTS]
        # CHANGED at step 23 (D-029): a raising EXPLORE worker is now
        # re-dispatched while its gate is BLOCKED, so there is one record per
        # dispatch rather than exactly one.  Every one of them must still be a
        # recorded FAILURE carrying the exception text -- and the count must
        # stay bounded, which is the property that replaced the literal 1.
        assert 1 <= len(results) <= 1 + harness.agent.max_explore_redispatches
        assert [entry["role"] for entry in results] == [Role.EXPLORER] * len(results)
        assert all(entry["success"] is False for entry in results)
        assert all("boom from explorer" in entry["answer"] for entry in results)
        assert isinstance(result, AgentResult)

    def test_worker_returning_a_non_agentresult_is_recorded_failed(
        self, make_harness
    ) -> None:
        def liar(request: RoleRequest) -> Any:
            return "not an AgentResult"

        harness = make_harness(worker=liar)
        result = harness.agent.run("lying worker goal")

        results = result.final_context[ContextKeys.ROLE_RESULTS]
        assert results[0]["success"] is False
        assert "str" in results[0]["answer"]

    def test_worker_factory_none_degrades_without_crashing(self) -> None:
        """No worker at all: the run BLOCKS legibly instead of crashing.

        CHANGED at step 7a (D-045).  This test used to assert
        ``findings_count is None`` and call the degrade path "the FSM's own
        extraction gets the turns" -- which was true, and was the defect: with
        the gate flags unseeded, the FSM's Pass-1 extraction was a second,
        unguarded writer that could invent every gate flag from prose.  The
        degrade path and invariants I6/I8 are mutually exclusive; I6 won.  A
        worker-less run now has no evidence source at all, so every gate stays
        shut and the run halts on the stall detector.  ``success=False`` with a
        named gate is the honest outcome, not a regression.
        """
        agent = HarnessAgent(
            worker_factory=None,
            approval_callback=ApprovalRecorder(),
            llm_interface=MockLLM2Interface(),
        )
        result = agent.run("degrade goal")

        assert isinstance(result, AgentResult)
        assert result.success is False
        assert "EXPLORE" in result.answer
        assert result.final_context.get(ContextKeys.FINDINGS_COUNT) == 0
        assert result.final_context.get(ContextKeys.ITERATION) == 0

    def test_a_failed_dispatch_does_not_advance_the_protocol(
        self, make_harness
    ) -> None:
        """An unsuccessful worker writes no gate flag (invariant I8).

        The seeded ``0`` is the assertion, not ``None``: every driver-owned gate
        flag is seeded falsy before turn 1 so the FSM's own extraction can never
        be asked for it (D-044).  A failed dispatch must leave that seed
        untouched -- the worker's ``9`` never reaches context.
        """
        script = _traverse_script()
        script[HarnessStates.EXPLORE] = {
            "success": False,
            "ctx": {ContextKeys.FINDINGS_COUNT: 9},
        }

        harness = make_harness(script)
        result = harness.run()

        assert harness.worker.count_for(HarnessStates.PLAN) == 0
        assert result.final_context.get(ContextKeys.FINDINGS_COUNT) == 0


# ---------------------------------------------------------------------------
# Fail closed (invariant I8) -- the driver allowlist
# ---------------------------------------------------------------------------


class TestFailClosed:
    """The driver's type allowlist is the real invariant-I8 enforcement.

    The JsonLogic gate layer uses soft comparison, so ``findings_count = "3"``
    would satisfy ``>= 3`` if it ever reached context (see
    ``test_fsm_definition.py::TestGateTypeGuardBoundary``).  It never does:
    ``_WORKER_WRITABLE`` drops a value whose runtime type is not exactly right.

    TIGHTENED at step 7d for the same reason as :class:`TestApprovals`: an
    ``is not True`` assertion is satisfied by an absent key, which is the one
    state the driver must never be in.
    """

    @pytest.mark.parametrize(
        ("label", "value"),
        [
            ("prose", "four"),
            ("numeric string", "3"),
            ("float", 3.0),
            ("bool", True),
            ("null", None),
            ("list", [3]),
        ],
    )
    def test_type_wrong_findings_count_never_advances_explore(
        self, make_harness, label: str, value: Any
    ) -> None:
        script = _traverse_script()
        script[HarnessStates.EXPLORE] = {"ctx": {ContextKeys.FINDINGS_COUNT: value}}

        harness = make_harness(script)
        result = harness.run()

        assert harness.worker.count_for(HarnessStates.PLAN) == 0, (
            f"a {label} findings_count advanced the protocol"
        )
        # The seeded 0 (D-044), not None: the dropped value never landed.
        assert result.final_context.get(ContextKeys.FINDINGS_COUNT) == 0

    def test_a_well_typed_counter_does_advance(self, make_harness) -> None:
        """The control case: the allowlist is not simply dropping everything."""
        harness = make_harness(_traverse_script(findings=3))
        harness.run()
        assert harness.worker.count_for(HarnessStates.PLAN) == 1

    @pytest.mark.parametrize("value", ["yes", 1, None, "True"])
    def test_type_wrong_criteria_flag_never_opens_the_close_gate(
        self, make_harness, value: Any
    ) -> None:
        script = _traverse_script()
        script[HarnessStates.REFLECT] = {"ctx": {ContextKeys.ALL_CRITERIA_PASS: value}}

        harness = make_harness(script)
        result = harness.run()

        assert harness.worker.count_for(HarnessStates.CLOSE) == 0
        assert result.final_context[ContextKeys.ALL_CRITERIA_PASS] is False

    def test_reflect_routing_flags_stay_mutually_exclusive(self, make_harness) -> None:
        """A greedy verifier setting every routing flag must pick exactly one.

        The three flags gate three different outbound edges at three
        priorities, so two true flags would resolve silently by ordering -- a
        completion fix would swallow a pivot the reviewer explicitly asked for.
        The driver keeps at most one, in ``_REFLECT_ROUTING_FLAGS`` order.
        """
        reflect_calls: list[int] = []

        def reflect(request: RoleRequest) -> dict[str, Any]:
            reflect_calls.append(request.iteration)
            if len(reflect_calls) == 1:
                return {
                    "ctx": {
                        ContextKeys.COMPLETION_FIX: True,
                        ContextKeys.NEEDS_PIVOT: True,
                        ContextKeys.NEEDS_EXPLORE: True,
                    }
                }
            return {"ctx": {ContextKeys.ALL_CRITERIA_PASS: True}}

        script = _traverse_script()
        script[HarnessStates.REFLECT] = reflect

        harness = make_harness(script)
        result = harness.run()

        # completion_fix has the highest precedence, so EXECUTE is re-entered
        # and neither of the other two edges is taken.
        assert harness.worker.count_for(HarnessStates.EXECUTE) == 2
        assert harness.worker.count_for(HarnessStates.PIVOT) == 0
        assert harness.worker.count_for(HarnessStates.EXPLORE) == 1

        live = [
            key
            for key in (
                ContextKeys.COMPLETION_FIX,
                ContextKeys.NEEDS_PIVOT,
                ContextKeys.NEEDS_EXPLORE,
            )
            if result.final_context.get(key) is True
        ]
        assert len(live) <= 1, f"multiple routing flags survived: {live}"


# ---------------------------------------------------------------------------
# Context caps (D-020)
# ---------------------------------------------------------------------------


class TestContextCaps:
    """Prompt-size caps: the reason a long run does not crash at ~turn 150.

    ``dispatch_ledger``, ``role_results`` and each recorded answer are ordinary
    context keys, so every one of them is rendered into BOTH LLM prompts on
    every turn.  Uncapped, a measured EXECUTE <-> REFLECT remediation cycle
    grew the response-generation system prompt by ~145 characters per turn and
    crashed the run on ``ResponseGenerationRequest``'s 30,000-character bound.

    This test drives a real remediation loop long enough to exceed all three
    caps, so a raised or removed cap fails here rather than in production.
    """

    #: Leash-continues granted before the run is allowed to stop.  Each grant
    #: buys two more EXECUTE <-> REFLECT cycles, i.e. ~4 dispatches and ~8
    #: ledger entries -- comfortably past the 64-entry and 20-result caps.
    #:
    #: CHANGED at step 7c (D-052).  This used to be the approval callback's
    #: verdict schedule ALONE, and it worked because an approving callback made
    #: the leash unbounded (review C3b): 30 grants were honoured because ANY
    #: number would have been.  ``max_leash_grants`` is now a per-step cap the
    #: callback cannot raise, so a run that wants 30 continuations has to be
    #: CONFIGURED for them.  A default-configured run reaches 6 executor
    #: dispatches and stops -- which is the point of the fix, and is why this
    #: test now says the number twice.
    GRANTS = 30

    def test_ledger_role_results_and_answers_are_capped(self, make_harness) -> None:
        script = _failing_execute_script()
        script[HarnessStates.EXECUTE] = {
            "success": False,
            "answer": "x" * 5000,
            "ctx": {},
        }

        harness = make_harness(
            script,
            approvals=ApprovalRecorder(
                {APPROVAL_LEASH: lambda index: index <= self.GRANTS}
            ),
            max_leash_grants=self.GRANTS,
        )
        result = harness.run()

        ledger = result.final_context[ContextKeys.DISPATCH_LEDGER]
        role_results = result.final_context[ContextKeys.ROLE_RESULTS]

        # The run really did generate more history than the caps allow.
        assert harness.worker.count_for(HarnessStates.EXECUTE) > 20
        assert len(ledger) == 64
        assert len(role_results) == 20
        assert all(len(entry["answer"]) <= 400 for entry in role_results)

        executor_entries = [
            entry for entry in role_results if entry["role"] == Role.EXECUTOR
        ]
        assert executor_entries, "no executor result survived the cap"
        assert all(entry["answer"] == "x" * 400 for entry in executor_entries)


# ---------------------------------------------------------------------------
# Filesystem roots carried in context (forward wiring for step 11)
# ---------------------------------------------------------------------------


class TestFilesystemRootsInContext:
    """The plan-dir / workspace roots ride in context untouched by the driver.

    Both tests SEED the findings directory.  Since D-032 the driver derives
    ``findings_count`` from that directory, so a run whose script merely CLAIMS
    three findings halts at the exploration cap -- which would leave these two
    asserting that the roots survive a halt, not a full run.  The seeding keeps
    them measuring what their names say.
    """

    def test_roots_survive_a_full_run(self, make_harness, plan_dir, workspace) -> None:
        _seed_findings(plan_dir, "alpha", "beta", "gamma")
        harness = make_harness(_traverse_script())
        result = harness.run(
            initial_context={
                ContextKeys.PLAN_DIR: str(plan_dir),
                ContextKeys.WORKSPACE_ROOT: str(workspace),
            }
        )

        assert HarnessStates.CLOSE in harness.worker.states, (
            "this test is meant to cover a FULL traverse; if the run now halts "
            "early it is no longer measuring what its name says"
        )
        assert result.final_context[ContextKeys.PLAN_DIR] == str(plan_dir)
        assert result.final_context[ContextKeys.WORKSPACE_ROOT] == str(workspace)

    def test_roots_are_visible_to_every_worker(
        self, make_harness, plan_dir, workspace
    ) -> None:
        _seed_findings(plan_dir, "alpha", "beta", "gamma")
        harness = make_harness(_traverse_script())
        harness.run(
            initial_context={
                ContextKeys.PLAN_DIR: str(plan_dir),
                ContextKeys.WORKSPACE_ROOT: str(workspace),
            }
        )

        assert HarnessStates.CLOSE in harness.worker.states
        for request in harness.worker.requests:
            assert request.context[ContextKeys.PLAN_DIR] == str(plan_dir)
            assert request.context[ContextKeys.WORKSPACE_ROOT] == str(workspace)


# ---------------------------------------------------------------------------
# Writer provenance (success criterion 18, D-044) -- review C1 and C4
# ---------------------------------------------------------------------------


class TestExtractionCannotOpenAGate:
    """The FSM's own Pass-1 extraction may not write ANY driver-owned key.

    This class exists because 139 green tests coexisted with a run that
    traversed EXPLORE -> PLAN -> EXECUTE -> REFLECT -> CLOSE on hallucinated
    gate flags while every worker failed and a DENYING approval callback was
    never consulted once (findings/review-iter-1.md C1).  The suite could not
    see it: every fixture built ``MockLLM2Interface()`` with no extraction data,
    so the second writer into gate context was inert in 100% of the suite (C4).

    Parametrisation is over the driver-owned TABLES, not over a list of key
    names typed out here.  Review C1 enumerated nine writable flags; step 7a's
    fix covers twenty-one keys; a hand-written list would have gone stale
    between those two sentences.
    """

    @pytest.mark.parametrize("key", sorted(DRIVER_OWNED_SEEDS))
    def test_a_seeded_key_keeps_its_seed(self, make_harness, key: str) -> None:
        """A fabricated value for a seeded key never reaches ``final_context``.

        The mechanism is PRESENCE: a key that is already non-``None`` in
        context is skipped by ``_execute_data_extraction`` entirely, so it
        never enters ``extracted_data`` and therefore never reaches the
        transition evaluator through the payload copy no handler can clean.
        """
        harness = make_harness(
            _all_failing_script(),
            extraction_data={key: FABRICATED_DRIVER_OWNED[key]},
            approvals=ApprovalRecorder(default=False),
        )
        result = harness.run()

        assert _is_exactly(result.final_context.get(key), DRIVER_OWNED_SEEDS[key]), (
            f"the LLM moved driver-owned '{key}' off its seed: "
            f"{result.final_context.get(key)!r}"
        )

    @pytest.mark.parametrize("key", sorted(DRIVER_OWNED_UNSET))
    def test_an_unset_key_never_takes_the_fabricated_value(
        self, make_harness, key: str
    ) -> None:
        """A key whose default is ABSENCE cannot be conjured out of prose.

        ``halt_reason`` is the one the user is shown as the run's outcome
        (review N4), and ``plan_dir`` selects the directory a role's write
        tools are confined to -- a fabricated one re-points the protocol's own
        memory.
        """
        harness = make_harness(
            _all_failing_script(),
            extraction_data={key: FABRICATED_DRIVER_OWNED[key]},
            approvals=ApprovalRecorder(default=False),
        )
        result = harness.run()

        assert result.final_context.get(key) != FABRICATED_DRIVER_OWNED[key]

    def test_every_driver_owned_key_at_once_leaves_the_run_in_explore(
        self, make_harness
    ) -> None:
        """Success criterion 18, verbatim, as one end-to-end run.

        A fabricating LLM + a worker that fails every dispatch + a callback
        that denies everything: the run must not leave EXPLORE, the two human
        gate flags must stay falsy, and the callback must be consulted ZERO
        times -- because a denied gate and an un-asked gate are the two shapes
        review C1 found indistinguishable in the shipped driver.
        """
        harness = make_harness(
            _all_failing_script(),
            extraction_data=FABRICATED_DRIVER_OWNED,
            approvals=ApprovalRecorder(default=False),
        )
        result = harness.run()

        # CHANGED at step 23 (D-029): EXPLORE is re-dispatched while its gate is
        # BLOCKED, so the sequence is one-or-more EXPLORE dispatches rather than
        # exactly one.  The assertion this class exists for is unchanged and
        # gains a bound: the run must never LEAVE explore, however many
        # explorers it spends.
        assert set(harness.worker.states) == {HarnessStates.EXPLORE}
        assert len(harness.worker.states) <= 1 + harness.agent.max_explore_redispatches
        assert result.final_context[ContextKeys.PLAN_APPROVED] is False
        assert result.final_context[ContextKeys.CLOSE_CONFIRMED] is False
        assert result.final_context[ContextKeys.ITERATION] == 0
        assert result.final_context[ContextKeys.FINDINGS_COUNT] == 0
        assert harness.approvals.requests == []
        assert result.success is False

    # DECISION plan-2026-07-21T125237-191b2eb2/D-056
    # A MUTATION test, and it is what keeps the class above from being vacuous.
    # Do NOT delete it as "a test that asserts broken behaviour": every other
    # test here passes just as well against the DEGENERATE fixture review C4
    # found (`MockLLM2Interface()` with no extraction data, where the second
    # writer is silent), and the only way to tell the two apart is to remove
    # the mechanism and watch the gate open. Do NOT weaken it to monkeypatch
    # the CONTEXT_UPDATE guard instead: step 7a measured that the guard alone
    # does not hold at all, so patching it out changes nothing and the test
    # would silently stop proving anything. See decisions.md D-056.
    def test_seeding_is_what_holds_the_gate(self, make_harness, monkeypatch) -> None:
        """Anti-vacuity: with seeding removed, the SAME fixture opens the gate.

        This is the RED half of the test above, committed rather than run once
        by hand, because C4's whole lesson is that a green assertion proves
        nothing until you have seen it fail.  It also pins WHICH mechanism
        holds: step 7a measured that the review's proposed ``CONTEXT_UPDATE``
        guard does not hold on its own (``MessagePipeline`` hands the
        transition evaluator a second copy of the extraction payload that no
        handler at any timing can reach), so the guard stays registered here
        and the gate opens anyway.
        """
        monkeypatch.setattr(harness_module, "DRIVER_OWNED_SEEDS", {})

        harness = make_harness(
            _all_failing_script(),
            extraction_data=FABRICATED_DRIVER_OWNED,
            approvals=ApprovalRecorder(default=False),
        )
        result = harness.run()

        assert HarnessStates.PLAN in harness.worker.states, (
            "unseeded, a fabricated findings_count should still open "
            "EXPLORE -> PLAN; if it no longer does, this test is asserting "
            "the wrong mechanism and the one above may be vacuous"
        )
        assert (
            result.final_context[ContextKeys.FINDINGS_COUNT]
            == (FABRICATED_DRIVER_OWNED[ContextKeys.FINDINGS_COUNT])
        )

    def test_the_filesystem_roots_stay_the_callers(self, make_harness, roots) -> None:
        """A fabricated root cannot re-point the protocol's own memory."""
        harness = make_harness(
            _all_failing_script(),
            extraction_data=FABRICATED_DRIVER_OWNED,
            approvals=ApprovalRecorder(default=False),
            roots=roots,
        )
        result = harness.run()

        assert result.final_context[ContextKeys.PLAN_DIR] == roots[ContextKeys.PLAN_DIR]
        assert (
            result.final_context[ContextKeys.WORKSPACE_ROOT]
            == roots[ContextKeys.WORKSPACE_ROOT]
        )
        assert harness.worker.requests
        for request in harness.worker.requests:
            assert request.plan_dir == roots[ContextKeys.PLAN_DIR]
            assert request.workspace_root == roots[ContextKeys.WORKSPACE_ROOT]

    def test_a_fabricated_root_does_not_appear_when_none_was_supplied(
        self, make_harness
    ) -> None:
        """No plan directory means NO plan directory, not one the model chose.

        ``RoleRequest.plan_dir is None`` is what tells a worker factory to hand
        the role no plan-file tools at all; a fabricated string there would
        confine those tools to a path the caller never authorised.
        """
        harness = make_harness(
            _all_failing_script(),
            extraction_data=FABRICATED_DRIVER_OWNED,
            approvals=ApprovalRecorder(default=False),
        )
        result = harness.run()

        assert ContextKeys.PLAN_DIR not in result.final_context
        assert ContextKeys.WORKSPACE_ROOT not in result.final_context
        assert all(request.plan_dir is None for request in harness.worker.requests)

    def test_a_worker_written_halt_reason_still_survives(self, make_harness) -> None:
        """The control case for review N4: the DRIVER's writer is not blocked.

        ``halt_reason`` is driver-owned, but CLOSE's worker allowlist grants it
        -- so a value that arrives through the allowlist must reach the answer,
        while the identical value arriving from prose must not.  Without this
        control, the test above would pass just as well if the guard deleted
        ``halt_reason`` unconditionally.
        """
        script = _traverse_script()
        script[HarnessStates.CLOSE] = {
            "ctx": {ContextKeys.HALT_REASON: "archivist closed the plan"}
        }

        harness = make_harness(script, extraction_data=FABRICATED_DRIVER_OWNED)
        result = harness.run()

        assert (
            result.final_context[ContextKeys.HALT_REASON] == "archivist closed the plan"
        )
        assert result.answer == "archivist closed the plan"


class TestDriverOwnedTable:
    """The three mechanisms that keep a driver-owned key present and correct.

    Seeding is enforcement; these are what keep the seeds in place
    (decisions.md D-044) and what makes the two dispatch call sites agree about
    deletion (D-053, review W3).
    """

    @pytest.fixture
    def halted(self, make_harness) -> Any:
        """A driver whose run halted in EXPLORE, so every SEEDED key is at its seed."""
        harness = make_harness(
            _all_failing_script(), approvals=ApprovalRecorder(default=False)
        )
        harness.run()
        return harness

    @pytest.fixture
    def halted_owning_no_unset_key(self, make_harness) -> Any:
        """A halted run in which the driver owns NO value for any UNSET key.

        The D-045 diagnostic mode: with no worker, nothing is dispatched, so
        nothing records a halt reason or a gate slug and every
        ``DRIVER_OWNED_UNSET`` key keeps its default -- absence.  ``halted``
        above stopped having that property at step 23: an all-failing EXPLORE
        worker spends the D-029 re-dispatch bound and the driver then OWNS
        ``halt_reason`` and ``last_gate_slug`` (the ``explore-cap`` halt), which
        is the driver writing them, not the LLM inventing them.
        """
        harness = make_harness(worker=None)
        harness.run()
        return harness

    def test_the_guard_is_registered_at_both_timings(self, halted) -> None:
        """PRE_PROCESSING is the enforcement; CONTEXT_UPDATE is the cleanup.

        Dropping the PRE_PROCESSING half as "redundant" is exactly the fix the
        review proposed and step 7a measured to be insufficient.
        """
        handlers = halted.agent.api.handler_system.handlers
        guard = next(h for h in handlers if h.name == HandlerNames.EXTRACTION_GUARD)

        assert guard.timings == {
            HandlerTiming.PRE_PROCESSING,
            HandlerTiming.CONTEXT_UPDATE,
        }
        assert guard.priority < min(
            h.priority for h in handlers if h.name != HandlerNames.EXTRACTION_GUARD
        )

    @pytest.mark.parametrize("key", sorted(DRIVER_OWNED_SEEDS))
    def test_the_guard_restores_a_seeded_key(self, halted, key: str) -> None:
        """Whatever the LLM wrote, the driver's value wins."""
        delta = halted.agent._reassert_driver_owned({key: FABRICATED_DRIVER_OWNED[key]})
        assert _is_exactly(delta[key], DRIVER_OWNED_SEEDS[key])

    @pytest.mark.parametrize("key", sorted(DRIVER_OWNED_UNSET))
    def test_the_guard_deletes_an_unset_key(
        self, halted_owning_no_unset_key, key: str
    ) -> None:
        """``None`` in a handler delta means DELETE; absence is the default."""
        agent = halted_owning_no_unset_key.agent
        assert agent._driver_owned[key] is None, (
            "precondition: the driver must own no value for this key, or the "
            "guard is correctly RESTORING rather than deleting"
        )
        delta = agent._reassert_driver_owned({key: FABRICATED_DRIVER_OWNED[key]})
        assert delta[key] is None

    @pytest.mark.parametrize("key", sorted(DRIVER_OWNED_UNSET))
    def test_the_guard_restores_a_driver_written_unset_key(
        self, halted, key: str
    ) -> None:
        """The other half: a value the DRIVER wrote wins over the LLM's.

        The control for the test above.  Without it, the guard could satisfy
        that assertion by deleting an unset key unconditionally -- which would
        silently erase the ``explore-cap`` halt reason the run is reporting.
        """
        owned = halted.agent._driver_owned[key]
        delta = halted.agent._reassert_driver_owned({key: FABRICATED_DRIVER_OWNED[key]})
        assert delta[key] == owned
        assert delta[key] != FABRICATED_DRIVER_OWNED[key]

    def test_the_guard_is_silent_when_the_llm_wrote_nothing(self, halted) -> None:
        """The normal path costs nothing: no delta, no log, no churn."""
        context = {
            key: value
            for key, value in halted.agent._driver_owned.items()
            if value is not None
        }
        assert halted.agent._reassert_driver_owned(context) == {}

    @pytest.mark.parametrize("key", sorted(DRIVER_OWNED_SEEDS))
    def test_apply_coerces_a_none_delta_on_a_seeded_key_to_its_seed(
        self, halted, key: str
    ) -> None:
        """A seeded key can be RESET, never deleted -- deleted means extractable.

        Reproduced at step 7a: with the pre-7a ``plan_approved: None`` clear on
        PLAN entry restored, a fabricating LLM opened PLAN -> EXECUTE even with
        the guard handler registered.
        """
        updates: dict[str, Any] = {}
        working: dict[str, Any] = {key: FABRICATED_DRIVER_OWNED[key]}

        halted.agent._apply(updates, working, {key: None})

        assert _is_exactly(updates[key], DRIVER_OWNED_SEEDS[key])
        assert _is_exactly(working[key], DRIVER_OWNED_SEEDS[key])

    @pytest.mark.parametrize("key", sorted(DRIVER_OWNED_UNSET))
    def test_apply_records_an_unset_key_as_absent(self, halted, key: str) -> None:
        """The other half of the one deletion convention (D-053)."""
        updates: dict[str, Any] = {}
        working: dict[str, Any] = {key: "something"}

        halted.agent._apply(updates, working, {key: None})

        assert updates[key] is None
        assert key not in working
        assert halted.agent._driver_owned[key] is None

    @pytest.mark.parametrize(
        ("key", "expected_writable"),
        [
            (ContextKeys.PLAN_APPROVED, True),  # seeded: coerced, so writable
            (ContextKeys.HALT_REASON, False),  # unset: dropped, deleted by the guard
        ],
    )
    def test_both_dispatch_call_sites_agree_about_a_cleared_key(
        self, halted, key: str, expected_writable: bool
    ) -> None:
        """W3: the entry path and the loop path clear a flag the same way.

        The entry path returns its delta as a handler result, where ``None``
        deletes.  The loop path goes through ``API.update_context``, which
        cannot delete -- so the two agree only because ``_apply`` coerces a
        seeded key first and records an unset one as absent for the
        PRE_PROCESSING guard to remove.
        """
        updates: dict[str, Any] = {}
        halted.agent._apply(updates, {}, {key: None})
        writable = halted.agent._writable_updates(updates)

        assert (key in writable) is expected_writable
        assert halted.agent._driver_owned[key] == updates[key]

    def test_writable_updates_warns_about_a_foreign_key_it_cannot_clear(
        self, halted, captured_logs
    ) -> None:
        """A non-driver-owned deletion is dropped LOUDLY, not silently.

        Silently dropping a ``None`` for ANY key with no record that it had
        done so is exactly the review W3 defect; the warning is what makes a
        future producer of such a delta notice.
        """
        writable = halted.agent._writable_updates({"some_foreign_key": None})

        assert writable == {}
        assert any(
            "some_foreign_key" in line and "cannot delete" in line
            for line in captured_logs
        ), captured_logs


class TestEntryBookkeeping:
    """A routing flag is cleared where it is CONSUMED (D-044, review W4).

    Without this, one hallucinated ``needs_explore=True`` survived for the rest
    of the run and silently won REFLECT routing at priority 600 on every later
    visit.  Every clear writes ``False``, never ``None``: an absent flag is an
    extractable flag.
    """

    @pytest.mark.parametrize(
        ("state", "cleared"),
        [
            (HarnessStates.EXPLORE, (ContextKeys.NEEDS_EXPLORE,)),
            (
                HarnessStates.PLAN,
                (ContextKeys.PLAN_APPROVED, ContextKeys.PIVOT_RESOLVED),
            ),
            (
                HarnessStates.EXECUTE,
                (ContextKeys.EXECUTE_COMPLETE, ContextKeys.COMPLETION_FIX),
            ),
            (HarnessStates.REFLECT, (ContextKeys.EXECUTE_COMPLETE,)),
            (
                HarnessStates.PIVOT,
                (
                    ContextKeys.COMPLETION_FIX,
                    ContextKeys.NEEDS_PIVOT,
                    ContextKeys.FIX_ATTEMPTS,
                    ContextKeys.LEASH_GRANTS,
                ),
            ),
        ],
    )
    def test_entry_clears_the_flags_that_state_consumes(
        self, make_harness, state: str, cleared: tuple[str, ...]
    ) -> None:
        harness = make_harness()
        poisoned = dict.fromkeys(cleared, True)

        updates = harness.agent._entry_bookkeeping(state, poisoned)

        for key in cleared:
            assert key in updates, f"entering {state} must clear {key}"
            assert updates[key] in (False, 0)
            assert updates[key] is not None, (
                f"{key} cleared with None on {state} entry -- an absent flag "
                "is an extractable flag (D-044)"
            )

    def test_both_leash_counters_reset_together_on_pivot(self, make_harness) -> None:
        """D-052: ``fix_attempts`` alone is the escape review C3(b) reproduced."""
        harness = make_harness()
        updates = harness.agent._entry_bookkeeping(
            HarnessStates.PIVOT,
            {ContextKeys.FIX_ATTEMPTS: 2, ContextKeys.LEASH_GRANTS: 2},
        )
        assert updates[ContextKeys.FIX_ATTEMPTS] == 0
        assert updates[ContextKeys.LEASH_GRANTS] == 0


class TestRoutingExclusivity:
    """Two precedence layers, both load-bearing (D-044)."""

    @staticmethod
    def _rank(written: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        return HarnessAgent._enforce_routing_exclusivity(written, context)

    def test_nothing_true_anywhere_is_a_no_op(self) -> None:
        assert self._rank({}, {}) == {}

    def test_a_fresh_verdict_outranks_a_flag_lying_in_context(self) -> None:
        """The trap step 7a hit: ``completion_fix`` outranks ``needs_pivot``.

        The executor sets ``completion_fix`` on its way OUT of a failed step,
        so at every REFLECT entry after a failure it is ALREADY true.  A merged
        scan ranked by flag order therefore makes a verifier's explicit PIVOT
        verdict unreachable -- which is the exact swallowing this function
        exists to prevent.
        """
        delta = self._rank(
            {ContextKeys.NEEDS_PIVOT: True},
            {ContextKeys.COMPLETION_FIX: True, ContextKeys.NEEDS_PIVOT: True},
        )
        assert delta[ContextKeys.COMPLETION_FIX] is False
        assert ContextKeys.NEEDS_PIVOT not in delta

    def test_a_stale_flag_still_loses_to_flag_order_within_its_layer(self) -> None:
        delta = self._rank(
            {},
            {ContextKeys.NEEDS_PIVOT: True, ContextKeys.COMPLETION_FIX: True},
        )
        assert delta[ContextKeys.NEEDS_PIVOT] is False
        assert ContextKeys.COMPLETION_FIX not in delta

    def test_a_stale_flag_is_ranked_at_all_when_the_worker_sets_none(self) -> None:
        """W4: exclusivity runs over the MERGED view, not the worker delta.

        Before step 7a this ran only inside ``_apply_role_result``, i.e. only
        for worker-written flags, so a stale flag was never even examined.
        """
        delta = self._rank({}, {ContextKeys.NEEDS_EXPLORE: True})
        assert delta[ContextKeys.COMPLETION_FIX] is False
        assert delta[ContextKeys.NEEDS_PIVOT] is False

    def test_losers_are_cleared_to_false_never_to_none(self) -> None:
        delta = self._rank({ContextKeys.COMPLETION_FIX: True}, {})
        assert delta
        assert all(value is False for value in delta.values())

    def test_it_runs_on_a_failed_reflect_dispatch_too(self, make_harness) -> None:
        """End-to-end: a stale flag is ranked even when the verifier fails."""
        script = _traverse_script()
        script[HarnessStates.EXECUTE] = {"success": False, "ctx": {}}
        script[HarnessStates.REFLECT] = {"success": False, "ctx": {}}

        harness = make_harness(
            script, approvals=ApprovalRecorder({APPROVAL_LEASH: False})
        )
        result = harness.run()

        live = [
            key
            for key in (
                ContextKeys.COMPLETION_FIX,
                ContextKeys.NEEDS_PIVOT,
                ContextKeys.NEEDS_EXPLORE,
            )
            if result.final_context.get(key) is True
        ]
        assert len(live) <= 1, f"multiple routing flags survived: {live}"


# ---------------------------------------------------------------------------
# The leash is bounded in both directions (D-051 / D-052, review C3)
# ---------------------------------------------------------------------------


class TestLeashIsBounded:
    """Neither a raising worker nor an approving human can escape the leash."""

    @staticmethod
    def _script(*, raises: bool) -> dict[str, Any]:
        """A traverse whose executor fails the same way twice over."""
        script = _traverse_script()
        script[HarnessStates.EXECUTE] = (
            {"raises": RuntimeError("executor exploded")}
            if raises
            else {"success": False, "ctx": {}}
        )
        script[HarnessStates.REFLECT] = {"ctx": {}}
        return script

    def test_a_raising_executor_spends_attempts_exactly_like_a_failing_one(
        self, make_harness
    ) -> None:
        """C3(a): ``result is None`` collapsed two DIFFERENT events into one.

        Measured before the fix: an executor that raised on every dispatch
        spent ZERO attempts, emitted no ``leash-cap`` slug and held EXECUTE
        until the stall halt -- the leash did not engage at all on the loudest
        possible failure.  ``_record_role_result`` had always recorded the same
        event as ``success=False``, so two writers read one condition in
        opposite directions.
        """
        outcomes = {}
        for label in ("raises", "returns-false"):
            harness = make_harness(
                self._script(raises=label == "raises"),
                approvals=ApprovalRecorder({APPROVAL_LEASH: False}),
            )
            result = harness.run()
            outcomes[label] = (
                harness.worker.count_for(HarnessStates.EXECUTE),
                result.final_context[ContextKeys.FIX_ATTEMPTS],
                result.final_context[ContextKeys.LAST_GATE_SLUG],
            )

        assert outcomes["raises"] == outcomes["returns-false"]
        assert outcomes["raises"] == (2, 2, GateSlug.LEASH_CAP)

    def test_no_worker_configured_spends_nothing(self, make_harness) -> None:
        """The other side of D-051: nothing attempted, nothing spent.

        A raising worker is a genuine failed attempt; ``worker_factory=None``
        is the D-045 diagnostic mode and must advance nothing.
        """
        harness = make_harness(worker=None)
        result = harness.run()
        assert result.final_context[ContextKeys.FIX_ATTEMPTS] == 0

    @pytest.mark.parametrize(
        ("max_fix_attempts", "max_leash_grants"),
        [(2, 0), (2, 2), (2, 3), (1, 1), (3, 1)],
    )
    def test_an_always_approving_callback_still_terminates_on_the_bound(
        self, make_harness, max_fix_attempts: int, max_leash_grants: int
    ) -> None:
        """C3(b): the property the counter buys, over several configurations.

        With an always-approving callback -- which is this suite's OWN default
        -- and an always-failing executor, resetting ``fix_attempts`` on every
        grant cycled EXECUTE <-> REFLECT until ``BudgetExhaustedError``: the
        cap the callback saw was infinite.  Executor dispatches on ONE plan
        step are now bounded by ``max_fix_attempts * (1 + max_leash_grants)``
        for ANY sequence of approvals.
        """
        harness = make_harness(
            self._script(raises=False),
            approvals=ApprovalRecorder(default=True),
            max_fix_attempts=max_fix_attempts,
            max_leash_grants=max_leash_grants,
        )
        result = harness.run()

        assert harness.worker.count_for(HarnessStates.EXECUTE) == (
            max_fix_attempts * (1 + max_leash_grants)
        )
        assert harness.approvals.count(APPROVAL_LEASH) == max_leash_grants
        assert result.final_context[ContextKeys.LAST_GATE_SLUG] == GateSlug.LEASH_CAP
        assert result.success is False

    def test_a_grant_resets_fix_attempts_but_never_leash_grants(
        self, make_harness
    ) -> None:
        """The two halves of the per-step budget are NOT one fact here.

        ``fix_attempts`` must reset (explicit user direction buys another pair
        of attempts); ``leash_grants`` must not (it is what bounds the asking).
        """
        harness = make_harness(
            self._script(raises=False), approvals=ApprovalRecorder(default=True)
        )
        result = harness.run()

        assert result.final_context[ContextKeys.LEASH_GRANTS] == (
            harness.agent.max_leash_grants
        )
        assert result.final_context[ContextKeys.FIX_ATTEMPTS] == (
            harness.agent.max_fix_attempts
        )

    def test_the_driver_stops_asking_once_the_grant_budget_is_spent(
        self, make_harness
    ) -> None:
        """An approval whose answer would be discarded is theatre."""
        harness = make_harness(
            self._script(raises=False),
            approvals=ApprovalRecorder(default=True),
            max_leash_grants=1,
        )
        result = harness.run()

        assert harness.approvals.count(APPROVAL_LEASH) == 1
        assert "already been continued" in result.answer

    def test_the_callback_cannot_reach_the_grant_counter(self, make_harness) -> None:
        """The callback gets a SUMMARY of the context and returns a bool.

        A callback that mutates everything it is handed must not be able to buy
        itself more grants -- the counter is driver-owned, and the FSM's own
        extraction cannot invent it either (pinned separately by
        ``TestExtractionCannotOpenAGate``).  ``ApprovalRequest`` carries
        ``parameters`` and ``context_summary``, both of which are built
        per-gate and never merged back, so this asserts a structural property
        rather than a promise.
        """

        class Saboteur(ApprovalRecorder):
            def __call__(self, request: Any) -> bool:
                for target in (request.parameters, request.context_summary):
                    target[ContextKeys.LEASH_GRANTS] = 0
                    target[ContextKeys.FIX_ATTEMPTS] = 0
                return super().__call__(request)

        harness = make_harness(
            self._script(raises=False), approvals=Saboteur(default=True)
        )
        result = harness.run()

        assert harness.worker.count_for(HarnessStates.EXECUTE) == (
            harness.agent.max_fix_attempts * (1 + harness.agent.max_leash_grants)
        )
        assert result.final_context[ContextKeys.LEASH_GRANTS] == (
            harness.agent.max_leash_grants
        )


# ---------------------------------------------------------------------------
# One run per instance (D-055, review W7)
# ---------------------------------------------------------------------------


class TestSingleRunPerInstance:
    """Concurrent runs on ONE instance corrupt each other, so they are refused.

    ``_api``, ``_conversation_id``, ``_stall_turns``, ``_stall_signature`` and
    ``_driver_owned`` are plain instance attributes rewritten by every run.
    D-014's ``threading.local`` covers the re-entrancy FLAG, not this state --
    and being thread-scoped it cannot see the corrupting case at all, which is
    why the second run below is started on a SECOND thread.
    """

    def test_a_second_concurrent_run_raises(self, make_harness) -> None:
        first_dispatch = threading.Event()
        second_finished = threading.Event()
        failures: list[BaseException] = []

        def blocking_worker(request: RoleRequest) -> AgentResult:
            first_dispatch.set()
            second_finished.wait(timeout=10)
            return AgentResult(answer="ok", success=False, final_context={})

        harness = make_harness(worker=blocking_worker)

        def second_run() -> None:
            try:
                first_dispatch.wait(timeout=10)
                harness.agent.run("concurrent goal")
            except BaseException as exc:
                failures.append(exc)
            finally:
                second_finished.set()

        thread = threading.Thread(target=second_run, daemon=True)
        thread.start()
        result = harness.run()
        thread.join(timeout=10)

        assert not thread.is_alive()
        assert len(failures) == 1
        assert isinstance(failures[0], HarnessError)
        assert "already in flight" in str(failures[0])
        assert result.success is False

    def test_the_lock_is_released_so_the_instance_is_reusable(
        self, make_harness
    ) -> None:
        """Sequential reuse must still work; the lock is per-run, not per-life."""
        harness = make_harness(_traverse_script())
        assert harness.run().success is True
        assert harness.run("second goal").success is True


# ---------------------------------------------------------------------------
# The two gate-flag channels (review C1's blind spot, D-015)
# ---------------------------------------------------------------------------


class TestBothGateChannelsAreRead:
    """``_apply_role_result`` reads ``structured_output`` AND ``final_context``.

    Review C1 found the fail-open here and the reviewer named the blind spot
    exactly: no test drove a ``concluded=False`` dispatch carrying a VALID
    ``structured_output`` through ``_dispatch_if_needed``.  Step 2's repair turn
    made that shape routine -- the loop exhausts its budget, writes nothing, and
    a schema-valid payload is still extracted afterwards.

    These tests pin the merge so ``roles.py``'s filesystem-derived correction
    cannot be undone by an ordering change, and prove the driver is not simply
    ignoring one channel.
    """

    @staticmethod
    def _explore_delta(
        make_harness, *, structured: Any, final_context: dict[str, Any]
    ) -> dict[str, Any]:
        """Drive ONE EXPLORE dispatch and return the context delta it produced."""

        class _Payload:
            """A stand-in for the role's pydantic payload (duck-typed dump)."""

            def __init__(self, data: dict[str, Any]) -> None:
                self._data = data

            def model_dump(self) -> dict[str, Any]:
                return dict(self._data)

        def worker(request: RoleRequest) -> AgentResult:
            return AgentResult(
                # `success=True` is the shape `roles.py` returns for a parseable
                # payload; the AGENT's own `concluded` was False.
                answer="I reviewed three areas.",
                success=True,
                final_context=dict(final_context),
                structured_output=_Payload(structured) if structured else None,
            )

        harness = make_harness(worker=worker)
        return harness.agent._dispatch_if_needed(
            HarnessStates.EXPLORE,
            {ContextKeys.GOAL: "g", ContextKeys.ITERATION: 1},
        )

    def test_a_structured_only_claim_is_read(self, make_harness) -> None:
        """Not a recommendation -- the fact that makes the next test matter.

        This is review C1 in one line: with nothing in ``final_context``, a
        payload claiming three findings reaches the EXPLORE gate.  The default
        worker factory is what stops it, by correcting BOTH channels before the
        driver ever sees them (``roles.py``, D-015).
        """
        delta = self._explore_delta(
            make_harness,
            structured={ContextKeys.FINDINGS_COUNT: 3},
            final_context={},
        )

        assert delta[ContextKeys.FINDINGS_COUNT] == 3

    def test_final_context_overrides_a_contradicting_structured_claim(
        self, make_harness
    ) -> None:
        """The shape the corrected factory returns: claim 3, disk 0."""
        delta = self._explore_delta(
            make_harness,
            structured={ContextKeys.FINDINGS_COUNT: 3},
            final_context={ContextKeys.FINDINGS_COUNT: 0},
        )

        assert delta[ContextKeys.FINDINGS_COUNT] == 0

    def test_the_control_a_verified_count_still_opens_the_gate(
        self, make_harness
    ) -> None:
        """Otherwise the test above would pass on a driver that dropped the key."""
        delta = self._explore_delta(
            make_harness,
            structured={ContextKeys.FINDINGS_COUNT: 0},
            final_context={ContextKeys.FINDINGS_COUNT: 3},
        )

        assert delta[ContextKeys.FINDINGS_COUNT] == 3

    def test_a_dropped_key_leaves_the_gate_shut(self, make_harness) -> None:
        """No plan directory means nothing to count: the key must not appear."""
        delta = self._explore_delta(make_harness, structured=None, final_context={})

        assert ContextKeys.FINDINGS_COUNT not in delta


# ---------------------------------------------------------------------------
# Bounded EXPLORE re-dispatch (D-028 / D-029)
# ---------------------------------------------------------------------------


def _explore_script(counts: list[int]) -> dict[str, Any]:
    """A traverse script whose Nth EXPLORE dispatch reports ``counts[N-1]``.

    The last value repeats forever, so a script can say "blocked from here on"
    without knowing how many dispatches the bound allows.
    """
    script = _traverse_script()
    seen: dict[str, int] = {"n": 0}

    def explorer(request: RoleRequest) -> dict[str, Any]:
        seen["n"] += 1
        index = min(seen["n"], len(counts)) - 1
        return {"ctx": {ContextKeys.FINDINGS_COUNT: counts[index]}}

    script[HarnessStates.EXPLORE] = explorer
    return script


class TestBoundedExploreRedispatch:
    """One explorer per topic, re-dispatched while the gate is BLOCKED (D-028).

    Four mechanisms were measured against "make ONE dispatch produce three
    findings" and all four failed at 0 runs in 10 (decisions.md D-022, D-027).
    The source protocol never asks for that -- ``agents/ip-orchestrator.md``'s
    EXPLORE step 7 is "if gate fails: spawn additional explorers for gaps" --
    so the driver now re-dispatches instead.

    The BOUND is the safety-critical half.  These tests pin it against the
    escape shape ``plans/LESSONS.md`` [I:5] records: "a safety cap the caller
    can reset from inside its own callback is not a cap."
    """

    def test_a_blocked_gate_redispatches_the_explorer(self, make_harness) -> None:
        """The mechanism, at its simplest: 1 entry + N re-dispatches."""
        harness = make_harness(_explore_script([0]), max_explore_redispatches=3)
        harness.run()

        assert harness.worker.count_for(HarnessStates.EXPLORE) == 4

    def test_redispatch_stops_the_moment_the_gate_is_satisfied(
        self, make_harness
    ) -> None:
        """The measured shape: one topic per dispatch until the gate opens.

        The re-dispatch condition is the GATE's own variable, so the loop ends
        on the dispatch that satisfies it -- not one later, and not at the cap.
        """
        harness = make_harness(_explore_script([1, 2, 3]), max_explore_redispatches=5)
        result = harness.run()

        assert harness.worker.count_for(HarnessStates.EXPLORE) == 3
        assert harness.worker.count_for(HarnessStates.PLAN) == 1
        assert result.final_context[ContextKeys.FINDINGS_COUNT] == 3
        assert result.final_context.get(ContextKeys.LAST_GATE_SLUG) != (
            GateSlug.EXPLORE_CAP
        )

    def test_a_satisfied_gate_never_redispatches(self, make_harness) -> None:
        """The control: an explorer that clears the gate is dispatched once."""
        harness = make_harness(_traverse_script(findings=3))
        harness.run()

        assert harness.worker.count_for(HarnessStates.EXPLORE) == 1
        assert harness.agent._explore_redispatches == 0

    @pytest.mark.parametrize("cap", [0, 1, 2, 5])
    def test_the_bound_is_exact(self, make_harness, cap: int) -> None:
        """``cap`` EXTRA dispatches, never ``cap + 1`` -- including ``cap = 0``."""
        harness = make_harness(_explore_script([0]), max_explore_redispatches=cap)
        harness.run()

        assert harness.worker.count_for(HarnessStates.EXPLORE) == 1 + cap
        assert harness.agent._explore_redispatches == cap

    def test_spending_the_bound_halts_with_its_own_slug(self, make_harness) -> None:
        """A distinct, honest reason -- and the gate stays exactly as shut."""
        harness = make_harness(_explore_script([0]), max_explore_redispatches=2)
        result = harness.run()

        assert result.success is False
        assert result.final_context[ContextKeys.LAST_GATE_SLUG] == GateSlug.EXPLORE_CAP
        assert "Exploration cap" in result.answer
        # ...and NOT by pretending the exploration succeeded.
        assert result.final_context[ContextKeys.FINDINGS_COUNT] == 0
        assert harness.worker.count_for(HarnessStates.PLAN) == 0
        assert harness.approvals.count(APPROVAL_PLAN) == 0

    def test_the_halt_names_the_numbers_a_reader_needs(
        self, make_harness, captured_logs
    ) -> None:
        """Spent, cap, reached and required -- an unactionable halt is a bug."""
        harness = make_harness(_explore_script([1]), max_explore_redispatches=2)
        result = harness.run()

        reason = result.final_context[ContextKeys.HALT_REASON]
        assert "2 extra explorer(s) dispatched (cap 2)" in reason
        assert "holds 1 of the 3 findings" in reason
        assert any(GateSlug.EXPLORE_CAP in line for line in captured_logs)

    def test_a_worker_cannot_reset_the_bound(self, make_harness) -> None:
        """Drive a worker that TRIES, through every channel it has.

        A worker's reply reaches context through ``final_context`` and
        ``structured_output``, both filtered by ``_WORKER_WRITABLE``; the FSM's
        own Pass-1 extraction is the second writer.  The bound is reachable from
        none of them, because it is not a context key at all.
        """

        class _Payload:
            def __init__(self, data: dict[str, Any]) -> None:
                self._data = data

            def model_dump(self) -> dict[str, Any]:
                return dict(self._data)

        greedy = {
            "explore_redispatches": 0,
            "_explore_redispatches": 0,
            "max_explore_redispatches": 99,
            ContextKeys.FIX_ATTEMPTS: 0,
            ContextKeys.LEASH_GRANTS: 0,
            ContextKeys.NEEDS_EXPLORE: True,
        }

        def worker(request: RoleRequest) -> AgentResult:
            return AgentResult(
                answer="reset your counters",
                success=True,
                final_context={**greedy, ContextKeys.FINDINGS_COUNT: 0},
                structured_output=_Payload(greedy),
            )

        harness = make_harness(
            worker=worker, extraction_data=greedy, max_explore_redispatches=2
        )
        result = harness.run()

        assert harness.agent._explore_redispatches == 2
        assert result.final_context[ContextKeys.LAST_GATE_SLUG] == GateSlug.EXPLORE_CAP
        assert harness.agent.max_explore_redispatches == 2

    def test_a_worker_cannot_refill_the_bound_by_routing_back_into_explore(
        self, make_harness
    ) -> None:
        """The [I:5] escape shape, aimed at THIS cap.

        ``needs_explore`` IS in the verifier's writable set, so a worker can
        route REFLECT -> EXPLORE as often as it likes.  If the bound reset on
        EXPLORE entry, that would refill it every time -- the predecessor's 7c
        defect with a different counter.  It is per RUN, so it does not.

        Dispatch 1 blocks (spends 1), dispatch 2 opens the gate, the plan runs,
        REFLECT routes back, and the second occupancy has ONE re-dispatch left,
        not a fresh budget of two.
        """
        script = _explore_script([0, 3, 0])
        script[HarnessStates.REFLECT] = {"ctx": {ContextKeys.NEEDS_EXPLORE: True}}

        harness = make_harness(script, max_explore_redispatches=2)
        result = harness.run()

        # 2 (first occupancy) + 2 (second: its entry + the ONE remaining
        # re-dispatch).  A per-entry reset would make the second occupancy 3.
        assert harness.worker.count_for(HarnessStates.EXPLORE) == 4
        assert harness.agent._explore_redispatches == 2
        assert result.final_context[ContextKeys.LAST_GATE_SLUG] == GateSlug.EXPLORE_CAP

    def test_the_bound_resets_between_runs_on_one_instance(self, make_harness) -> None:
        """Per RUN, not per LIFETIME: a reused instance is not pre-exhausted."""
        harness = make_harness(_explore_script([0]), max_explore_redispatches=2)
        harness.run()
        assert harness.agent._explore_redispatches == 2

        harness.run("a second goal")

        assert harness.agent._explore_redispatches == 2
        assert harness.worker.count_for(HarnessStates.EXPLORE) == 6

    def test_a_raising_worker_still_counts_against_the_bound(
        self, make_harness
    ) -> None:
        """A dispatch that failed is a dispatch spent; it is not free."""
        script = _traverse_script()
        script[HarnessStates.EXPLORE] = {"raises": RuntimeError("boom")}

        harness = make_harness(script, max_explore_redispatches=2)
        harness.run()

        assert harness.worker.count_for(HarnessStates.EXPLORE) == 3

    def test_no_worker_at_all_does_not_redispatch(self, make_harness) -> None:
        """D-045's diagnostic mode is untouched: nothing attempted, nothing spent.

        Re-dispatching a factory that does not exist would burn turns and would
        relabel the honest "no worker" stall as an exploration cap.
        """
        harness = make_harness(worker=None, max_explore_redispatches=3)
        result = harness.run()

        assert harness.agent._explore_redispatches == 0
        assert result.final_context.get(ContextKeys.LAST_GATE_SLUG) != (
            GateSlug.EXPLORE_CAP
        )
        assert "EXPLORE" in result.answer

    def test_redispatch_reopens_only_the_explore_key(self, make_harness) -> None:
        """The ledger's other entries survive; the entry marker is untouched.

        The re-authorisation is EXPLICIT and one key wide (D-017).  A delta that
        cleared the ledger, or dropped the ``entry:`` marker, would re-open every
        suppressed dispatch in the window and make a duplicate handler fire
        dispatch again.
        """
        harness = make_harness(_explore_script([0]), max_explore_redispatches=1)
        context = {
            ContextKeys.GOAL: "g",
            ContextKeys.ITERATION: 0,
            ContextKeys.STEP_NUMBER: 0,
            ContextKeys.DISPATCH_LEDGER: ["dispatch:execute:1:1", "entry:explore:'x'"],
            ContextKeys.ROLE_RESULTS: [],
        }

        delta = harness.agent._dispatch_if_needed(HarnessStates.EXPLORE, context)

        ledger = delta[ContextKeys.DISPATCH_LEDGER]
        assert "dispatch:explore:0:0" not in ledger
        assert "dispatch:execute:1:1" in ledger
        assert "entry:explore:'x'" in ledger

    def test_a_blocked_plan_still_dispatches_once(self, make_harness) -> None:
        """No regression to the ledger's original purpose (invariant I4).

        PLAN holds its gate BLOCKED for exactly the reason EXPLORE does above --
        the gate flag it needs is not there -- and it must NOT re-dispatch.
        PLAN's own re-dispatch budget covers a FAILED plan-writer reply only
        (``TestBoundedPlanRedispatch``): a SUCCESSFUL dispatch whose plan a
        human declined to approve is a shut gate, not a retryable failure.
        """
        harness = make_harness(
            _traverse_script(), approvals=ApprovalRecorder(default=False)
        )
        harness.run()

        assert harness.worker.count_for(HarnessStates.PLAN) == 1
        assert harness.agent._explore_redispatches == 0

    def test_a_blocked_reflect_still_dispatches_once_per_authorisation(
        self, make_harness
    ) -> None:
        """The same, one state further on: one dispatch per authorisation.

        Since Step 4b (plan-2026-07-23T173454-2c22e5f6 D-003) an UNROUTABLE
        verdict is a retryable failure under ``max_reflect_redispatches`` --
        the S4b slugless stall this script used to reproduce.  The invariant
        this test pins is I4's, unchanged: with the budget at 0, the held
        REFLECT dispatches EXACTLY once, and the run now ends on the honest
        ``reflect-cap`` slug instead of the stall detector's ``slug=None``.
        """
        script = _traverse_script()
        script[HarnessStates.REFLECT] = {"ctx": {}}

        harness = make_harness(script, max_reflect_redispatches=0)
        result = harness.run()

        assert harness.worker.count_for(HarnessStates.REFLECT) == 1
        assert result.final_context[ContextKeys.LAST_GATE_SLUG] == (
            GateSlug.REFLECT_CAP
        )


# ---------------------------------------------------------------------------
# The bound's SIZE, and the arithmetic it came from (D-031)
# ---------------------------------------------------------------------------

#: Dispatch index by which EVERY measured run that reached the findings
#: threshold had reached it, on `ollama_chat/qwen3.5:4b`, n=10, at step 24: the
#: four successful runs took 9, 8, 6 and 5 dispatches.
MEASURED_YIELD_HORIZON = 9

#: Dispatches measured BEYOND that horizon, and the distinct files they added:
#: six runs ran to 18 dispatches and added nothing at all after the horizon.
MEASURED_BEYOND_HORIZON = (60, 0)


class TestExploreBoundIsSizedFromTheMeasuredHorizon:
    """The bound must cover the measured horizon -- and must not exceed it far.

    Two sizings have now been measured and both were wrong in a way this class
    exists to stop recurring:
      * 5 (step 23) came from "one topic per dispatch", which step 23's own
        data falsified;
      * 17 (step 24, first attempt) came from extrapolating the pooled rate
        0.35 files/dispatch linearly, and step 24's own data falsified THAT:
        at 18 dispatches the pooled rate fell to 0.14 and runs reaching 3 went
        4/10, because the yield is not linear -- it stops.
    What the traces support is a horizon, so that is what is pinned, on both
    sides: big enough to contain every measured success, small enough not to
    spend minutes of wall clock past the point where the yield was measured at
    exactly zero.
    """

    def test_the_bound_covers_every_measured_success(self) -> None:
        total_dispatches = 1 + Defaults.MAX_EXPLORE_REDISPATCHES

        assert total_dispatches >= MEASURED_YIELD_HORIZON, (
            f"{total_dispatches} total dispatches cannot contain the slowest "
            f"measured success, which took {MEASURED_YIELD_HORIZON}. Re-size "
            "the bound from the traces, or re-measure the horizon -- do not "
            "lower the threshold."
        )

    def test_the_bound_does_not_run_far_past_the_horizon(self) -> None:
        """Dispatches past the horizon are measured waste, not headroom."""
        total_dispatches = 1 + Defaults.MAX_EXPLORE_REDISPATCHES
        spent, gained = MEASURED_BEYOND_HORIZON

        assert gained == 0, "re-derive this test: the horizon moved"
        assert total_dispatches <= MEASURED_YIELD_HORIZON + 3, (
            f"{total_dispatches} total dispatches runs "
            f"{total_dispatches - MEASURED_YIELD_HORIZON} past the measured "
            f"horizon, where {spent} measured dispatches produced {gained} new "
            "findings files. That is wall clock, not headroom."
        )

    @pytest.mark.parametrize("superseded", [0, 5, 17])
    def test_the_superseded_bounds_would_fail_these_tests(
        self, superseded: int
    ) -> None:
        """Not vacuous: every value this bound has ever held is now excluded.

        Without this control a later edit could set the bound to anything and
        the two tests above would still be green for the wrong reason.
        """
        total = 1 + superseded

        assert total < MEASURED_YIELD_HORIZON or total > MEASURED_YIELD_HORIZON + 3, (
            f"{superseded} is no longer excluded; the horizon moved"
        )

    def test_the_raised_bound_is_still_exactly_enforced(self, make_harness) -> None:
        """Bigger is not looser: the default bound is spent, then it HALTS."""
        harness = make_harness(_explore_script([0]))
        result = harness.run()

        assert harness.worker.count_for(HarnessStates.EXPLORE) == (
            1 + Defaults.MAX_EXPLORE_REDISPATCHES
        )
        assert harness.agent._explore_redispatches == (
            Defaults.MAX_EXPLORE_REDISPATCHES
        )
        assert result.final_context[ContextKeys.LAST_GATE_SLUG] == GateSlug.EXPLORE_CAP
        assert result.final_context[ContextKeys.FINDINGS_COUNT] == 0
        assert harness.worker.count_for(HarnessStates.PLAN) == 0


# ---------------------------------------------------------------------------
# Bounded PLAN re-dispatch (plan-2026-07-22T184813-6549c7cb D-001)
# ---------------------------------------------------------------------------


def _failing_plan_script() -> dict[str, Any]:
    """EXPLORE clears its gate; the plan-writer FAILS every dispatch."""
    script = _traverse_script()
    script[HarnessStates.PLAN] = {"success": False, "ctx": {}}
    return script


class TestBoundedPlanRedispatch:
    """A failed plan-writer is retried under a bound, then halted with a slug.

    Guards the defect L6 B0 measured (run 3,
    ``scripts/bench_data/l6-e2e/rows.jsonl``): PLAN dispatched exactly once per
    ``(iteration, step)``, so ONE empty plan-writer reply was terminal, and the
    eventual halt was the stall detector's ``slug=None`` raise -- the run
    recorded ``plan_md_bytes: 0`` with ``halt_slug: null``, the slugless stall
    shape the L6 floor exists to catch.

    The mechanism mirrors ``TestBoundedExploreRedispatch`` (D-028/D-029): the
    SAME dispatch key is re-opened by ledger removal (never a widened key --
    D-017), the bound lives on ``self`` where no worker or approval callback
    can reach it, and spending it pre-writes ``plan-cap`` so the halt is never
    slugless again.
    """

    def test_an_always_failing_plan_worker_halts_at_the_cap_with_its_own_slug(
        self, make_harness
    ) -> None:
        """The L6 B0 run-3 shape, fixed: 1 + MAX dispatches, then ``plan-cap``.

        Measured RED against the unpatched driver (2026-07-22): PLAN
        dispatches == 1 and ``LAST_GATE_SLUG`` was ``None``.  The default
        approval callback here APPROVES every gate, which is the
        unreachability half: an approving callback is never even consulted,
        because a plan that was never written is never put to it.
        """
        harness = make_harness(_failing_plan_script())
        result = harness.run()

        assert harness.worker.count_for(HarnessStates.PLAN) == (
            1 + Defaults.MAX_PLAN_REDISPATCHES
        )
        assert result.success is False
        assert result.final_context[ContextKeys.LAST_GATE_SLUG] == GateSlug.PLAN_CAP
        assert "Plan cap" in result.answer
        # ...and NOT by pretending a plan exists.
        assert harness.worker.count_for(HarnessStates.EXECUTE) == 0
        assert harness.approvals.count(APPROVAL_PLAN) == 0

    @pytest.mark.parametrize("cap", [0, 1, 2])
    def test_the_bound_is_exact(self, make_harness, cap: int) -> None:
        """``cap`` EXTRA dispatches, never ``cap + 1`` -- including ``cap = 0``."""
        harness = make_harness(_failing_plan_script(), max_plan_redispatches=cap)
        harness.run()

        assert harness.worker.count_for(HarnessStates.PLAN) == 1 + cap
        assert harness.agent._plan_redispatches == cap

    def test_the_halt_names_the_numbers_a_reader_needs(
        self, make_harness, captured_logs
    ) -> None:
        """Spent and cap, in the reason -- an unactionable halt is a bug."""
        harness = make_harness(_failing_plan_script(), max_plan_redispatches=2)
        result = harness.run()

        reason = result.final_context[ContextKeys.HALT_REASON]
        assert "2 extra plan-writer dispatch(es) spent (cap 2)" in reason
        assert any(GateSlug.PLAN_CAP in line for line in captured_logs)

    def test_a_plan_writer_that_recovers_proceeds_without_the_slug(
        self, make_harness
    ) -> None:
        """Fails twice, succeeds on the third: the run proceeds past PLAN.

        This is the outcome the budget exists to buy -- "fails once cold" is
        distinguished from "persistently fails" -- and the recovered run must
        carry no ``plan-cap`` residue.
        """
        script = _traverse_script()
        seen = {"n": 0}

        def plan_writer(request: RoleRequest) -> dict[str, Any]:
            seen["n"] += 1
            if seen["n"] <= 2:
                return {"success": False, "ctx": {}}
            return {"ctx": {ContextKeys.TOTAL_STEPS: 1}}

        script[HarnessStates.PLAN] = plan_writer
        harness = make_harness(script)
        result = harness.run()

        assert harness.worker.count_for(HarnessStates.PLAN) == 3
        assert harness.agent._plan_redispatches == 2
        assert harness.worker.count_for(HarnessStates.EXECUTE) == 1
        assert result.final_context.get(ContextKeys.LAST_GATE_SLUG) != (
            GateSlug.PLAN_CAP
        )

    def test_the_bound_resets_between_runs_on_one_instance(self, make_harness) -> None:
        """Per RUN, not per LIFETIME: a reused instance is not pre-exhausted."""
        harness = make_harness(_failing_plan_script(), max_plan_redispatches=1)
        harness.run()
        assert harness.agent._plan_redispatches == 1

        harness.run("a second goal")

        assert harness.agent._plan_redispatches == 1
        assert harness.worker.count_for(HarnessStates.PLAN) == 4

    def test_a_raising_plan_worker_still_counts_against_the_bound(
        self, make_harness
    ) -> None:
        """A dispatch that RAISED is a dispatch spent; it is not free."""
        script = _traverse_script()
        script[HarnessStates.PLAN] = {"raises": RuntimeError("boom")}

        harness = make_harness(script, max_plan_redispatches=2)
        result = harness.run()

        assert harness.worker.count_for(HarnessStates.PLAN) == 3
        assert result.final_context[ContextKeys.LAST_GATE_SLUG] == GateSlug.PLAN_CAP

    def test_a_worker_cannot_reach_the_counter(self, make_harness) -> None:
        """The [I:5] escape shape, aimed at THIS cap.

        A worker's reply reaches context through ``final_context`` and
        ``structured_output`` (both filtered by ``_WORKER_WRITABLE``) and the
        FSM's Pass-1 extraction is the second writer.  The bound is reachable
        from none of them, because it is not a context key at all -- the L6 B0
        defect must not be "fixed" into a cap a worker can refill.
        """
        greedy = {
            "plan_redispatches": 0,
            "_plan_redispatches": 0,
            "max_plan_redispatches": 99,
        }
        script = _failing_plan_script()
        script[HarnessStates.PLAN] = {"success": False, "ctx": dict(greedy)}

        harness = make_harness(script, extraction_data=greedy, max_plan_redispatches=2)
        result = harness.run()

        assert harness.agent._plan_redispatches == 2
        assert harness.agent.max_plan_redispatches == 2
        assert result.final_context[ContextKeys.LAST_GATE_SLUG] == GateSlug.PLAN_CAP

    def test_the_counter_is_not_a_context_key(self, make_harness) -> None:
        """Structural pin: the counter exists nowhere a worker can write.

        The counter must never grow a ``ContextKeys`` entry or a
        ``DRIVER_OWNED_SEEDS`` seed -- that surface is rendered into both LLM
        prompts every turn and policed by ``_WORKER_WRITABLE``, which is
        exactly where a refillable cap would be born.
        """
        context_key_values = {
            value
            for name, value in vars(ContextKeys).items()
            if isinstance(value, str) and not name.startswith("__")
        }
        for spelling in ("plan_redispatches", "plan-redispatches"):
            assert spelling not in context_key_values
            assert spelling not in DRIVER_OWNED_SEEDS

        harness = make_harness(_failing_plan_script(), max_plan_redispatches=1)
        result = harness.run()

        assert "plan_redispatches" not in result.final_context
        assert "_plan_redispatches" not in result.final_context

    def test_plan_cap_is_not_a_pre_step_gate_slug(self) -> None:
        """``plan-cap`` sits NEXT TO ``explore-cap``, outside ``ORDER``.

        ``plan_validator.pre_step_gate`` iterates ``ORDER`` to decide what it
        may report; a driver halt on the PLAN -> EXECUTE edge does not belong
        in a pre-EXECUTE-step check's vocabulary.
        """
        assert GateSlug.PLAN_CAP == "plan-cap"
        assert GateSlug.PLAN_CAP not in GateSlug.ORDER
        assert GateSlug.EXPLORE_CAP not in GateSlug.ORDER

    def test_no_worker_at_all_spends_nothing(self, make_harness) -> None:
        """D-045's diagnostic mode is untouched: nothing attempted, nothing spent."""
        harness = make_harness(worker=None)

        delta = harness.agent._after_plan_dispatch(None, None, {})

        assert delta == {}
        assert harness.agent._plan_redispatches == 0

    def test_redispatch_reopens_only_the_plan_key(self, make_harness) -> None:
        """The re-authorisation is EXPLICIT and one key wide (D-017).

        The ledger's other entries survive and the ``entry:`` marker is
        untouched -- the dispatch key keeps its exact ``(state, iteration,
        step)`` shape, re-opened by removal, never widened by a retry count.
        """
        harness = make_harness(_failing_plan_script(), max_plan_redispatches=1)
        context = {
            ContextKeys.GOAL: "g",
            ContextKeys.ITERATION: 0,
            ContextKeys.STEP_NUMBER: 0,
            ContextKeys.DISPATCH_LEDGER: ["dispatch:execute:1:1", "entry:plan:'x'"],
            ContextKeys.ROLE_RESULTS: [],
        }

        delta = harness.agent._dispatch_if_needed(HarnessStates.PLAN, context)

        ledger = delta[ContextKeys.DISPATCH_LEDGER]
        assert "dispatch:plan:0:0" not in ledger
        assert "dispatch:execute:1:1" in ledger
        assert "entry:plan:'x'" in ledger

    def test_explore_budget_is_not_shared_with_plan(self, make_harness) -> None:
        """Two states, two counters: spending EXPLORE's leaves PLAN's whole."""
        script = _explore_script([2, 3])  # dispatch 1 blocked, dispatch 2 clears
        script[HarnessStates.PLAN] = {"success": False, "ctx": {}}

        harness = make_harness(script, max_plan_redispatches=1)
        result = harness.run()

        assert harness.agent._explore_redispatches == 1
        assert harness.agent._plan_redispatches == 1
        assert harness.worker.count_for(HarnessStates.PLAN) == 2
        assert result.final_context[ContextKeys.LAST_GATE_SLUG] == GateSlug.PLAN_CAP

    # -- Defect B: the disk-derived empty-plan.md check (D-005) --------------

    def test_a_success_reply_with_an_empty_plan_md_halts_with_the_slug(
        self, make_harness, roots, plan_dir
    ) -> None:
        """The residual slugless PLAN stall, closed structurally (D-005).

        Measured shape (CLAUDE 'Known gaps after B1'): a plan-writer that
        reports ``success=True`` while leaving ``plan.md`` EMPTY used to SKIP
        the worker-failure budget (which keyed on ``result.success`` alone),
        fall through to a DENIED approval -- DENY is the unattended default,
        and an empty plan is denied -- and stall SLUGLESSLY, because
        ``_check_stall`` always raises ``slug=None``.  The disk-derived
        ``_plan_has_content`` folds that shape into the budget condition, so the
        empty-plan reply now consumes the budget exactly like a worker failure
        and, at exhaustion, halts on the honest ``plan-cap`` slug -- never
        ``slug=None``.
        """
        _seed_findings(plan_dir, "alpha", "beta", "gamma")
        script = _traverse_script()
        # A FAITHLESS plan-writer: claims success but writes NO plan.md, so the
        # on-disk plan stays empty (overrides the faithful default plan entry).
        script[HarnessStates.PLAN] = {"ctx": {ContextKeys.TOTAL_STEPS: 1}}
        harness = make_harness(
            script, roots=roots, approvals=ApprovalRecorder(default=False)
        )
        result = harness.run()

        assert harness.worker.count_for(HarnessStates.PLAN) == (
            1 + Defaults.MAX_PLAN_REDISPATCHES
        )
        assert harness.agent._plan_redispatches == Defaults.MAX_PLAN_REDISPATCHES
        slug = result.final_context[ContextKeys.LAST_GATE_SLUG]
        assert slug == GateSlug.PLAN_CAP
        assert slug is not None
        # the empty plan was never advanced to EXECUTE
        assert harness.worker.count_for(HarnessStates.EXECUTE) == 0

    def test_a_success_reply_with_a_real_plan_md_is_not_second_guessed(
        self, make_harness, roots, plan_dir
    ) -> None:
        """The happy path is UNTOUCHED: a real plan.md still reaches approval.

        A plan-writer that reports success AND writes a non-empty ``plan.md``
        (the faithful default ``_traverse_script`` writer) consumes NONE of the
        redispatch budget and is put to approval, exactly as before Defect B's
        fix.  Only the empty-plan case is newly caught.
        """
        _seed_findings(plan_dir, "alpha", "beta", "gamma")
        harness = make_harness(_traverse_script(), roots=roots)
        result = harness.run()

        assert harness.agent._plan_redispatches == 0
        assert harness.approvals.count(APPROVAL_PLAN) == 1
        assert result.final_context.get(ContextKeys.LAST_GATE_SLUG) != (
            GateSlug.PLAN_CAP
        )
        # the default approval APPROVES, so the run proceeds past PLAN
        assert harness.worker.count_for(HarnessStates.EXECUTE) >= 1

    def test_a_zero_byte_seeded_plan_md_reads_as_empty(
        self, make_harness, plan_dir
    ) -> None:
        """The step-1/step-5 interaction (A4): a seeded plan.md is EMPTY.

        ``PlanDirectory.seed_protocol_skeleton()`` creates ``plan.md`` as a
        ZERO-BYTE placeholder.  ``_plan_has_content`` is SUBSTANTIVE (D-001): a
        zero-byte file fails ``PlanDoc`` parse, so ``_artifact`` returns None and
        the check reads False -- a plan that was never written must NOT pass as
        real.
        """
        storage_module.PlanDirectory(plan_dir).seed_protocol_skeleton()
        plan_path = plan_dir / ArtifactNames.PLAN
        assert plan_path.exists()
        assert plan_path.stat().st_size == 0

        agent = make_harness().agent
        context = {ContextKeys.PLAN_DIR: str(plan_dir)}
        assert agent._plan_has_content(context) is False

    def test_the_empty_plan_check_leaks_no_key_under_wrong_spellings(
        self, make_harness, roots, plan_dir
    ) -> None:
        """The [I:5] escape shape, aimed at Defect B's check.

        The empty-plan budget consumption is driven off DISK
        (``_plan_has_content``), not off any context key, so no
        ``plan_has_content`` spelling -- however a worker misspells it in its
        reply or the FSM's Pass-1 extraction "finds" it -- appears in the final
        context.  Mirrors ``test_a_worker_cannot_reach_the_counter``.
        """
        _seed_findings(plan_dir, "alpha", "beta", "gamma")
        greedy = {
            "plan_has_content": True,
            "_plan_has_content": True,
            "plan-has-content": True,
        }
        script = _traverse_script()
        script[HarnessStates.PLAN] = {"success": True, "ctx": dict(greedy)}
        harness = make_harness(
            script,
            roots=roots,
            extraction_data=greedy,
            approvals=ApprovalRecorder(default=False),
        )
        result = harness.run()

        # the faithless plan-writer left plan.md empty, so the budget caught it
        assert harness.agent._plan_redispatches == Defaults.MAX_PLAN_REDISPATCHES
        assert result.final_context[ContextKeys.LAST_GATE_SLUG] == GateSlug.PLAN_CAP
        for spelling in ("plan_has_content", "_plan_has_content", "plan-has-content"):
            assert spelling not in result.final_context


# ---------------------------------------------------------------------------
# Bounded REFLECT re-dispatch (plan-2026-07-23T173454-2c22e5f6 D-003)
# ---------------------------------------------------------------------------


def _stuck_reflect_script() -> dict[str, Any]:
    """EXPLORE/PLAN/EXECUTE are well-behaved; the verifier is UNROUTABLE.

    NOT a strawman: this is the S5 probe's retained REFLECT record shape
    (``scripts/bench_data/l6-e2e/probe-s5-mechanism/
    probe-run-1-observations.json``) -- ``success=True``, a non-empty answer,
    and a coerced delta that sets NO routing key (no ``all_criteria_pass``, no
    ``completion_fix``/``needs_pivot``/``needs_explore``) after a SUCCESSFUL
    EXECUTE step.  The same shape ended L6 B5 run 1 and B6 run 2.
    """
    script = _traverse_script()
    script[HarnessStates.REFLECT] = {"answer": "criteria reviewed", "ctx": {}}
    return script


class TestBoundedReflectRedispatch:
    """An unroutable verifier is retried under a bound, then halted with a slug.

    Guards the S4b defect the S5 probe reproduced (run 1,
    ``scripts/bench_data/l6-e2e/probe-s5-mechanism/probe-rows.jsonl``): the
    REFLECT dispatch returned ``success=True`` with no routing key, the
    dispatch key stayed in the ledger, all four REFLECT edges stayed BLOCKED,
    and the run recorded ``halt_slug: null`` / ``honest_halt: false`` off the
    stall detector's ``slug=None`` raise -- the slugless stall shape the L6
    floor's honest-halt clause exists to catch.

    The mechanism mirrors ``TestBoundedPlanRedispatch`` (D-001 of
    plan-2026-07-22T184813-6549c7cb) exactly: the SAME dispatch key is
    re-opened by ledger removal (never a widened key -- D-017), the bound lives
    on ``self`` where no worker or approval callback can reach it, and spending
    it pre-writes ``reflect-cap`` so the halt is never slugless again.
    """

    def test_the_probe_shape_now_halts_at_the_cap_with_its_own_slug(
        self, make_harness
    ) -> None:
        """The S5-probe run-1 shape, fixed: 1 + MAX dispatches, ``reflect-cap``.

        Measured RED against the unpatched driver (2026-07-23): REFLECT
        dispatches == 1 and ``LAST_GATE_SLUG`` was ``None`` (the probe row's
        ``halt_slug: null``).  The default approval callback here APPROVES
        every gate, which is the unreachability half: an approving callback is
        never even consulted, because a verdict that routes nowhere never
        reaches the close gate.
        """
        harness = make_harness(_stuck_reflect_script())
        result = harness.run()

        assert harness.worker.count_for(HarnessStates.REFLECT) == (
            1 + Defaults.MAX_REFLECT_REDISPATCHES
        )
        assert result.success is False
        assert result.final_context[ContextKeys.LAST_GATE_SLUG] == (
            GateSlug.REFLECT_CAP
        )
        assert "Reflect cap" in result.answer
        # ...and NOT by pretending a verdict exists.
        assert harness.worker.count_for(HarnessStates.CLOSE) == 0
        assert harness.approvals.count(APPROVAL_CLOSE) == 0

    @pytest.mark.parametrize("cap", [0, 1, 2])
    def test_the_bound_is_exact(self, make_harness, cap: int) -> None:
        """``cap`` EXTRA dispatches, never ``cap + 1`` -- including ``cap = 0``."""
        harness = make_harness(_stuck_reflect_script(), max_reflect_redispatches=cap)
        harness.run()

        assert harness.worker.count_for(HarnessStates.REFLECT) == 1 + cap
        assert harness.agent._reflect_redispatches == cap

    def test_the_halt_names_the_numbers_a_reader_needs(
        self, make_harness, captured_logs
    ) -> None:
        """Spent and cap, in the reason -- an unactionable halt is a bug."""
        harness = make_harness(_stuck_reflect_script(), max_reflect_redispatches=2)
        result = harness.run()

        reason = result.final_context[ContextKeys.HALT_REASON]
        assert "2 extra verifier dispatch(es) spent (cap 2)" in reason
        assert any(GateSlug.REFLECT_CAP in line for line in captured_logs)

    def test_a_verifier_that_recovers_proceeds_without_the_slug(
        self, make_harness
    ) -> None:
        """Unroutable twice, routable on the third: the run reaches the close.

        This is the outcome the budget exists to buy -- "fails once cold" is
        distinguished from "persistently fails" -- and the recovered run must
        carry no ``reflect-cap`` residue.
        """
        script = _traverse_script()
        seen = {"n": 0}

        def verifier(request: RoleRequest) -> dict[str, Any]:
            seen["n"] += 1
            if seen["n"] <= 2:
                return {"ctx": {}}
            return {"ctx": {ContextKeys.ALL_CRITERIA_PASS: True}}

        script[HarnessStates.REFLECT] = verifier
        harness = make_harness(script)
        result = harness.run()

        assert harness.worker.count_for(HarnessStates.REFLECT) == 3
        assert harness.agent._reflect_redispatches == 2
        assert harness.approvals.count(APPROVAL_CLOSE) == 1
        assert result.final_context.get(ContextKeys.LAST_GATE_SLUG) != (
            GateSlug.REFLECT_CAP
        )

    def test_a_healthy_verifier_consumes_none_of_the_budget(self, make_harness) -> None:
        """The happy path is UNTOUCHED: one dispatch, zero budget, a close."""
        harness = make_harness(_traverse_script())
        result = harness.run()

        assert harness.agent._reflect_redispatches == 0
        assert harness.worker.count_for(HarnessStates.REFLECT) == 1
        assert harness.worker.count_for(HarnessStates.CLOSE) == 1
        assert result.final_context.get(ContextKeys.LAST_GATE_SLUG) != (
            GateSlug.REFLECT_CAP
        )

    def test_a_routable_verdict_consumes_none_of_the_budget(self, make_harness) -> None:
        """A routing flag IS a verdict: the pivot edge fires, budget untouched."""
        reflect_calls: list[int] = []

        def verifier(request: RoleRequest) -> dict[str, Any]:
            reflect_calls.append(request.iteration)
            if len(reflect_calls) == 1:
                return {"ctx": {ContextKeys.NEEDS_PIVOT: True}}
            return {"ctx": {ContextKeys.ALL_CRITERIA_PASS: True}}

        script = _traverse_script()
        script[HarnessStates.REFLECT] = verifier
        harness = make_harness(script)
        harness.run()

        assert harness.worker.count_for(HarnessStates.PIVOT) == 1
        assert harness.agent._reflect_redispatches == 0

    def test_the_leash_path_is_untouched_and_consumes_none_of_the_budget(
        self, make_harness
    ) -> None:
        """A leash-cap REFLECT entry routes to the leash offer, not the budget.

        ``_failing_execute_script``'s verifier writes no routing flag either,
        but the driver's own ``completion_fix`` (attempt 1) and ``leash-cap``
        slug (attempt 2) route every REFLECT entry, so the reflect budget must
        never fire -- the two counters answer different failures.
        """
        harness = make_harness(
            _failing_execute_script(),
            approvals=ApprovalRecorder({APPROVAL_LEASH: False}),
        )
        result = harness.run()

        assert harness.agent._reflect_redispatches == 0
        assert harness.approvals.count(APPROVAL_LEASH) == 1
        assert result.final_context[ContextKeys.LAST_GATE_SLUG] == GateSlug.LEASH_CAP

    def test_the_bound_resets_between_runs_on_one_instance(self, make_harness) -> None:
        """Per RUN, not per LIFETIME: a reused instance is not pre-exhausted."""
        harness = make_harness(_stuck_reflect_script(), max_reflect_redispatches=1)
        harness.run()
        assert harness.agent._reflect_redispatches == 1

        harness.run("a second goal")

        assert harness.agent._reflect_redispatches == 1
        assert harness.worker.count_for(HarnessStates.REFLECT) == 4

    def test_a_raising_verifier_still_counts_against_the_bound(
        self, make_harness
    ) -> None:
        """A dispatch that RAISED is a dispatch spent; it is not free."""
        script = _traverse_script()
        script[HarnessStates.REFLECT] = {"raises": RuntimeError("boom")}

        harness = make_harness(script, max_reflect_redispatches=2)
        result = harness.run()

        assert harness.worker.count_for(HarnessStates.REFLECT) == 3
        assert result.final_context[ContextKeys.LAST_GATE_SLUG] == (
            GateSlug.REFLECT_CAP
        )

    def test_a_worker_cannot_reach_the_counter(self, make_harness) -> None:
        """The [I:5] escape shape, aimed at THIS cap.

        A worker's reply reaches context through ``final_context`` and
        ``structured_output`` (both filtered by ``_WORKER_WRITABLE``) and the
        FSM's Pass-1 extraction is the second writer.  The bound is reachable
        from none of them, because it is not a context key at all -- the S4b
        defect must not be "fixed" into a cap a worker can refill.
        """
        greedy = {
            "reflect_redispatches": 0,
            "_reflect_redispatches": 0,
            "max_reflect_redispatches": 99,
        }
        script = _stuck_reflect_script()
        script[HarnessStates.REFLECT] = {"ctx": dict(greedy)}

        harness = make_harness(
            script, extraction_data=greedy, max_reflect_redispatches=2
        )
        result = harness.run()

        assert harness.agent._reflect_redispatches == 2
        assert harness.agent.max_reflect_redispatches == 2
        assert result.final_context[ContextKeys.LAST_GATE_SLUG] == (
            GateSlug.REFLECT_CAP
        )

    def test_an_approving_callback_cannot_refill_the_counter(
        self, make_harness
    ) -> None:
        """The cap is unreachable from the approval seam, by construction.

        The callback receives an ``ApprovalRequest`` (a context COPY plus
        parameters) and returns a bool -- it holds no driver reference.  With a
        stuck verifier it is never even consulted, and an always-approving
        callback changes neither the spend nor the slug (the LESSONS [I:5]
        property, proven here for THIS counter the way D-052 proved it for the
        leash).
        """
        approvals = ApprovalRecorder(default=True)
        harness = make_harness(
            _stuck_reflect_script(), approvals=approvals, max_reflect_redispatches=1
        )
        result = harness.run()

        # The plan approval on the way in is legitimate; DURING the stuck
        # REFLECT the callback is never consulted at all.
        assert approvals.names == [APPROVAL_PLAN]
        assert approvals.count(APPROVAL_CLOSE) == 0
        assert approvals.count(APPROVAL_LEASH) == 0
        assert harness.agent._reflect_redispatches == 1
        assert result.final_context[ContextKeys.LAST_GATE_SLUG] == (
            GateSlug.REFLECT_CAP
        )

    def test_the_counter_is_not_a_context_key(self, make_harness) -> None:
        """Structural pin: the counter exists nowhere a worker can write."""
        context_key_values = {
            value
            for name, value in vars(ContextKeys).items()
            if isinstance(value, str) and not name.startswith("__")
        }
        for spelling in ("reflect_redispatches", "reflect-redispatches"):
            assert spelling not in context_key_values
            assert spelling not in DRIVER_OWNED_SEEDS

        harness = make_harness(_stuck_reflect_script(), max_reflect_redispatches=1)
        result = harness.run()

        assert "reflect_redispatches" not in result.final_context
        assert "_reflect_redispatches" not in result.final_context

    def test_reflect_cap_is_not_a_pre_step_gate_slug(self) -> None:
        """``reflect-cap`` sits NEXT TO ``plan-cap``, outside ``ORDER``."""
        assert GateSlug.REFLECT_CAP == "reflect-cap"
        assert GateSlug.REFLECT_CAP not in GateSlug.ORDER

    def test_no_worker_at_all_spends_nothing(self, make_harness) -> None:
        """D-045's diagnostic mode is untouched: nothing attempted, nothing spent."""
        harness = make_harness(worker=None)

        delta = harness.agent._after_reflect_dispatch(None, None, {})

        assert delta == {}
        assert harness.agent._reflect_redispatches == 0

    def test_redispatch_reopens_only_the_reflect_key(self, make_harness) -> None:
        """The re-authorisation is EXPLICIT and one key wide (D-017)."""
        harness = make_harness(_stuck_reflect_script(), max_reflect_redispatches=1)
        context = {
            ContextKeys.GOAL: "g",
            ContextKeys.ITERATION: 0,
            ContextKeys.STEP_NUMBER: 0,
            ContextKeys.DISPATCH_LEDGER: ["dispatch:execute:1:1", "entry:reflect:'x'"],
            ContextKeys.ROLE_RESULTS: [],
        }

        delta = harness.agent._dispatch_if_needed(HarnessStates.REFLECT, context)

        ledger = delta[ContextKeys.DISPATCH_LEDGER]
        assert "dispatch:reflect:0:0" not in ledger
        assert "dispatch:execute:1:1" in ledger
        assert "entry:reflect:'x'" in ledger

    def test_plan_budget_is_not_shared_with_reflect(self, make_harness) -> None:
        """Two states, two counters: spending REFLECT's leaves PLAN's whole."""
        harness = make_harness(_stuck_reflect_script(), max_reflect_redispatches=1)
        result = harness.run()

        assert harness.agent._plan_redispatches == 0
        assert harness.agent._reflect_redispatches == 1
        assert result.final_context[ContextKeys.LAST_GATE_SLUG] == (
            GateSlug.REFLECT_CAP
        )


# ---------------------------------------------------------------------------
# substantive _plan_has_content (D-001)
# ---------------------------------------------------------------------------


class TestSubstantivePlanHasContent:
    """``_plan_has_content`` is APPROVABLE: valid PlanDoc AND ALL-non-placeholder.

    ALIGNED to the honest approval gate's bar (`_plan_is_approvable`, D-001):
    since approval DENIAL does not redispatch, the budget gate MUST match the
    approval gate or a substantive-but-unapprovable plan reopens the slugless
    stall.  So an all-placeholder plan, a PARTIALLY-filled
    plan (some placeholder), and a non-empty-but-INVALID plan (bad headers) all
    read False -- each consumes the redispatch budget and halts on the honest
    ``plan-cap`` slug.  The no-plan-directory degrade path (return True) is
    PRESERVED (D-005's measured 59-test trap).
    """

    def test_a_filled_valid_plan_is_substantive(self, make_harness, plan_dir) -> None:
        (plan_dir / ArtifactNames.PLAN).write_text(_PLAN_MD)
        agent = make_harness().agent
        assert agent._plan_has_content({ContextKeys.PLAN_DIR: str(plan_dir)}) is True

    def test_an_all_placeholder_plan_is_not_substantive(
        self, make_harness, plan_dir
    ) -> None:
        """A valid PlanDoc with all 11 sections present but every body EMPTY is
        all-placeholder, so the gate DENIES it (the ethos bar, F4)."""
        (plan_dir / ArtifactNames.PLAN).write_text(_hollow_plan_doc().to_markdown())
        agent = make_harness().agent
        context = {ContextKeys.PLAN_DIR: str(plan_dir)}
        assert agent._plan_has_content(context) is False

    def test_a_non_empty_but_invalid_plan_is_not_substantive(
        self, make_harness, plan_dir
    ) -> None:
        """Predecessor S2: a non-empty plan missing the 11 headers fails
        ``PlanDoc`` parse, so ``_artifact`` returns None -> False (closing the
        slugless invalid-plan stall)."""
        (plan_dir / ArtifactNames.PLAN).write_text(
            "# Not a plan\n\nJust some prose with no section headers at all.\n"
        )
        agent = make_harness().agent
        assert agent._plan_has_content({ContextKeys.PLAN_DIR: str(plan_dir)}) is False

    def test_a_partially_filled_plan_is_not_substantive(
        self, make_harness, plan_dir
    ) -> None:
        """ALL-non-placeholder is the ALIGNED bar (D-001): a valid PlanDoc with
        ONE real section but the rest still placeholder is NOT approvable, so it
        reads False -- it consumes the budget and the model is redispatched to
        fill the rest.  Aligning with the approval gate (which would DENY this
        partial plan) closes the slugless-stall gap: approval denial does not
        redispatch."""
        partial = _hollow_plan_doc()
        partial.sections[0].body = "Ship the retry with capped backoff."
        (plan_dir / ArtifactNames.PLAN).write_text(partial.to_markdown())
        agent = make_harness().agent
        assert agent._plan_has_content({ContextKeys.PLAN_DIR: str(plan_dir)}) is False

    def test_no_plan_directory_returns_true_degrade_path_preserved(
        self, make_harness
    ) -> None:
        """D-005's trap: no plan dir -> True (defer to result.success), NOT a
        reclassification of every degrade-path plan reply as a failure."""
        agent = make_harness().agent
        assert agent._plan_has_content({}) is True


# ---------------------------------------------------------------------------
# structured_output -> plan.md renderer (D-001 of this plan)
# ---------------------------------------------------------------------------


def _plan_payload(**overrides: Any) -> Any:
    """A real PLAN ``output_schema`` instance with 11 DISTINCT non-empty bodies.

    Built through ``get_role_spec(PLAN).output_schema`` so the test exercises
    the same schema the live repair turn produces, not a hand-rolled stand-in;
    ``overrides`` replaces individual slug bodies (e.g. an empty ``steps``).
    Each default body is distinct and non-placeholder, so an all-defaults
    payload renders an APPROVABLE plan (`_plan_is_approvable` True).
    """
    spec = get_role_spec(HarnessStates.PLAN)
    bodies = {
        slug: f"Real content for the {section} section, item {index}."
        for index, (section, slug) in enumerate(
            zip(PlanSchema.SECTIONS, PlanSchema.SECTION_SLUGS, strict=True)
        )
    }
    bodies.update(overrides)
    return spec.output_schema(
        **{
            ContextKeys.TOTAL_STEPS: 3,
            ContextKeys.NEEDS_EXPLORE: False,
            "message": "plan drafted",
            **bodies,
        }
    )


def _render(agent: HarnessAgent, plan_dir: Path, payload: Any) -> str:
    """Render *payload* to ``plan.md`` via the driver and return the file text."""
    result = AgentResult(answer="", success=True, structured_output=payload)
    agent._render_plan_from_structured(result, {ContextKeys.PLAN_DIR: str(plan_dir)})
    return (plan_dir / ArtifactNames.PLAN).read_text()


class TestRenderPlanFromStructured:
    """``_render_plan_from_structured`` distributes the 11 fields BY CONSTRUCTION.

    Success Criteria 2 & 3 (D-001).  The renderer maps each ``response_format``
    field under ITS ``PlanSchema.SECTIONS`` heading, so the result round-trips
    exactly and every field lands in its own section -- the property the refuted
    iter-6 append-to-end mechanism could not hold (content concentrated in one
    section / duplicate headers, floor 0/3 across B0-B3).  It NEVER invents
    filler: an empty field renders an empty section the UNCHANGED
    ``_plan_is_approvable`` denies.  Offline, deterministic, no live model.
    """

    def test_eleven_non_empty_fields_render_and_roundtrip(
        self, make_harness, plan_dir
    ) -> None:
        """Exactly 11 sections, titles in SECTIONS order, each body == its field,
        and an EXACT ``to_markdown(from_markdown(text)) == text`` round-trip."""
        agent = make_harness().agent
        payload = _plan_payload()

        text = _render(agent, plan_dir, payload)

        plan = PlanDoc.from_markdown(text)
        assert len(plan.sections) == 11
        assert [section.name for section in plan.sections] == list(PlanSchema.SECTIONS)
        dumped = payload.model_dump()
        for section, slug in zip(
            PlanSchema.SECTIONS, PlanSchema.SECTION_SLUGS, strict=True
        ):
            assert plan.body_of(section) == dumped[slug]
        # exact text round-trip: the rendered file re-serialises to itself
        assert PlanDoc.from_markdown(text).to_markdown() == text
        # by construction every section is real, so the plan is approvable
        assert harness_module._plan_is_approvable(plan) is True

    def test_a_heading_injection_body_still_roundtrips_and_is_preserved(
        self, make_harness, plan_dir
    ) -> None:
        """A field embedding ``## `` / ``# `` lines is DEMOTED, not lost: still
        exactly 11 sections, exact round-trip, and the injected text survives in
        the body (the ``_demote_heading_lines`` guard, D-001)."""
        agent = make_harness().agent
        injected = "Real goal line.\n## Injected H2 header\n# Injected H1 header\ntail"
        payload = _plan_payload(**{PlanSchema.SECTION_SLUGS[0]: injected})

        text = _render(agent, plan_dir, payload)

        plan = PlanDoc.from_markdown(text)
        # the injected headers did NOT create spurious sections
        assert len(plan.sections) == 11
        assert [section.name for section in plan.sections] == list(PlanSchema.SECTIONS)
        assert PlanDoc.from_markdown(text).to_markdown() == text
        goal_body = plan.body_of(PlanSchema.SECTIONS[0])
        # the text is preserved (escaped/demoted), not dropped
        assert "## Injected H2 header" in goal_body
        assert "# Injected H1 header" in goal_body

    def test_an_empty_field_denied_by_the_unchanged_gate_no_filler(
        self, make_harness, plan_dir
    ) -> None:
        """Ethos (Success Criterion 3): ONE empty field renders an empty section
        that the UNCHANGED ``_plan_is_approvable`` DENIES, and NO filler prose is
        substituted for the hole -- the section body stays empty."""
        agent = make_harness().agent
        # `steps` is SECTIONS[4]; author every other field, leave this one blank
        payload = _plan_payload(**{PlanSchema.SECTION_SLUGS[4]: "   "})

        text = _render(agent, plan_dir, payload)

        plan = PlanDoc.from_markdown(text)
        # the empty section is a placeholder the unchanged gate denies
        assert harness_module._plan_is_approvable(plan) is False
        # ... and NO filler was invented: the body is empty/whitespace, not prose
        empty_body = plan.body_of(PlanSchema.SECTIONS[4])
        assert empty_body.strip() == ""
        assert _is_placeholder(empty_body) is True

    def test_success_false_but_valid_structured_renders_approvable(
        self, make_harness, plan_dir
    ) -> None:
        """The L6 B4 "config could-not-succeed" regression, fixed (D-002).

        Under ``response_format`` the PLAN model authors every field but calls
        NO write tool, so the D-016 write-obligation forces ``result.success``
        to False even on a perfectly VALID 14-field reply.  The OLD renderer
        guard (``not result.success -> return None``) DISCARDED that valid plan,
        leaving ``plan.md`` at 0 bytes -> ``_plan_has_content`` False ->
        redispatch -> honest ``plan-cap`` -- the exact B4 floor-0/3 shape, which
        measured our own wiring gap, not 4b's plan authoring.  With the render
        keyed on ``structured_output`` presence (not success), the same reply now
        WRITES an approvable ``plan.md`` and ``_plan_has_content`` is True, so
        ``_after_plan_dispatch`` would NOT redispatch and would proceed to
        approval.  This is the deterministic offline test that RED-flags the B4
        config before any live run.
        """
        agent = make_harness().agent
        context = {ContextKeys.PLAN_DIR: str(plan_dir)}
        # a valid 14-field reply, but success=False -- exactly the D-016
        # unverified-write outcome for a driver-rendered PLAN deliverable
        result = AgentResult(
            answer="", success=False, structured_output=_plan_payload()
        )

        path = agent._render_plan_from_structured(result, context)

        # the valid plan WAS rendered despite success=False
        assert path is not None
        plan = PlanDoc.from_markdown((plan_dir / ArtifactNames.PLAN).read_text())
        assert harness_module._plan_is_approvable(plan) is True
        # ... and the disk gate the redispatch/approval condition keys on is True
        assert agent._plan_has_content(context) is True

    def test_success_true_but_no_structured_still_fails_closed(
        self, make_harness, plan_dir
    ) -> None:
        """Fail-closed preserved (D-002 ethos): no structured -> nothing written.

        A reply with ``success=True`` but ``structured_output=None`` renders
        NOTHING (the model authored no fields), so ``_plan_has_content`` reads
        False off disk and ``_after_plan_dispatch`` still takes the redispatch
        path.  Dropping ``not result.success`` from the gate did not let a
        content-free reply through -- the disk gate catches it.
        """
        agent = make_harness().agent
        context = {ContextKeys.PLAN_DIR: str(plan_dir)}
        result = AgentResult(answer="", success=True, structured_output=None)

        path = agent._render_plan_from_structured(result, context)

        assert path is None
        assert not (plan_dir / ArtifactNames.PLAN).exists()
        assert agent._plan_has_content(context) is False

    def test_success_false_with_empty_field_still_denied(
        self, make_harness, plan_dir
    ) -> None:
        """Fail-closed on a hollow field even when success=False (D-002 ethos).

        A ``success=False`` reply whose structured payload has an EMPTY section
        renders that section empty; the UNCHANGED ``_plan_is_approvable`` denies
        it, so ``_plan_has_content`` is False and the redispatch path is taken.
        Decoupling the render from ``result.success`` never weakens the honest
        gate: a hollow plan is still denied.
        """
        agent = make_harness().agent
        context = {ContextKeys.PLAN_DIR: str(plan_dir)}
        payload = _plan_payload(**{PlanSchema.SECTION_SLUGS[4]: "   "})
        result = AgentResult(answer="", success=False, structured_output=payload)

        agent._render_plan_from_structured(result, context)

        assert agent._plan_has_content(context) is False


# ---------------------------------------------------------------------------
# Evidence vs testimony: the driver's own filesystem read (D-032)
# ---------------------------------------------------------------------------


def _seed_findings(plan_dir: Path, *names: str) -> None:
    """Write one non-empty ``findings/<name>.md`` per name."""
    directory = plan_dir / "findings"
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        (directory / f"{name}.md").write_text(f"# {name}\n\nreal content\n")


class TestDiskDerivedGateCounts:
    """A count the DRIVER reads off disk is evidence; a worker's is testimony.

    Invariant I8 discards a failed dispatch's gate keys, which is right for a
    value the worker CLAIMED and wrong for one the driver DERIVED: the files
    exist whether or not the dispatch that wrote them reported success.  Step 23
    measured a run holding a real findings file and a gate value of 0 because
    all six of its dispatches failed.

    Both directions are pinned here on purpose.  Widening this to worker-claimed
    keys would re-open review C1's fail-open gate (a dispatch that writes zero
    bytes and claims three findings), so every test that proves the disk is read
    has a partner proving the claim still is not.
    """

    # -- the correction -------------------------------------------------

    def test_a_failed_dispatch_still_has_its_disk_read(
        self, make_harness, roots, plan_dir
    ) -> None:
        """The step-23 shape: real files, every dispatch FAILED, gate value 0."""
        _seed_findings(plan_dir, "alpha", "beta", "gamma")
        script = _traverse_script()
        script[HarnessStates.EXPLORE] = {"success": False, "ctx": {}}

        harness = make_harness(script, roots=roots)
        result = harness.run()

        assert result.final_context[ContextKeys.FINDINGS_COUNT] == 3
        assert harness.worker.count_for(HarnessStates.EXPLORE) == 1

    def test_a_raising_dispatch_still_has_its_disk_read(
        self, make_harness, roots, plan_dir
    ) -> None:
        """A worker that RAISED wrote its files before it raised."""
        _seed_findings(plan_dir, "alpha", "beta", "gamma")
        script = _traverse_script()
        script[HarnessStates.EXPLORE] = {"raises": RuntimeError("boom")}

        harness = make_harness(script, roots=roots)
        result = harness.run()

        assert result.final_context[ContextKeys.FINDINGS_COUNT] == 3

    def test_the_disk_outranks_a_successful_worker_claim(
        self, make_harness, roots, plan_dir
    ) -> None:
        """One derivation, and it is the filesystem's -- not the reply's."""
        _seed_findings(plan_dir, "alpha")
        harness = make_harness(_traverse_script(findings=3), roots=roots)
        result = harness.run()

        assert result.final_context[ContextKeys.FINDINGS_COUNT] == 1

    def test_an_empty_file_is_not_a_finding(
        self, make_harness, roots, plan_dir
    ) -> None:
        """The derivation counts BYTES, exactly as ``gate_files`` defines it."""
        _seed_findings(plan_dir, "alpha", "beta")
        (plan_dir / "findings" / "empty.md").write_text("   \n")
        (plan_dir / "findings" / "notes.txt").write_text("not markdown")

        harness = make_harness(_traverse_script(), roots=roots)
        result = harness.run()

        assert result.final_context[ContextKeys.FINDINGS_COUNT] == 2

    def test_the_traverse_runs_on_evidence(self, make_harness, roots, plan_dir) -> None:
        """The positive control: three real files open EXPLORE -> PLAN."""
        _seed_findings(plan_dir, "alpha", "beta", "gamma")
        harness = make_harness(_traverse_script(), roots=roots)
        harness.run()

        assert harness.worker.count_for(HarnessStates.EXPLORE) == 1
        assert HarnessStates.PLAN in harness.worker.states

    # -- what must NOT move ---------------------------------------------

    def test_zero_bytes_and_a_claim_of_three_keeps_the_gate_shut(
        self, make_harness, roots, plan_dir
    ) -> None:
        """Review C1's fail-open reproduction, re-run against this change.

        A dispatch that writes NOTHING and reports ``findings_count: 3`` is the
        exact shape step 20 closed.  It must stay closed on the SUCCESS path
        (where I8 would let the claim through) as well as on the failure path.
        """
        assert not list((plan_dir / "findings").glob("*.md"))
        harness = make_harness(_traverse_script(findings=3), roots=roots)
        result = harness.run()

        assert result.final_context[ContextKeys.FINDINGS_COUNT] == 0
        assert harness.worker.count_for(HarnessStates.PLAN) == 0
        assert result.final_context[ContextKeys.LAST_GATE_SLUG] == GateSlug.EXPLORE_CAP
        assert harness.approvals.count(APPROVAL_PLAN) == 0

    def test_a_failed_dispatch_claiming_a_non_derived_key_is_still_discarded(
        self, make_harness, roots, plan_dir
    ) -> None:
        """The other half of the pair: I8 is untouched for a CLAIMED key.

        ``needs_explore`` is in EXPLORE's writable set and is NOT disk-derived,
        so a failed dispatch's value for it must still be dropped -- while the
        disk-derived count from the SAME reply is read.
        """
        _seed_findings(plan_dir, "alpha", "beta", "gamma")
        script = _traverse_script()
        script[HarnessStates.EXPLORE] = {
            "success": False,
            "ctx": {
                ContextKeys.NEEDS_EXPLORE: True,
                ContextKeys.FINDINGS_COUNT: 99,
            },
        }

        harness = make_harness(script, roots=roots)
        result = harness.run()

        assert result.final_context[ContextKeys.NEEDS_EXPLORE] is False
        assert result.final_context[ContextKeys.FINDINGS_COUNT] == 3

    def test_a_failed_dispatch_is_still_recorded_as_failed(
        self, make_harness, roots, plan_dir
    ) -> None:
        """Reading the disk is not counting the dispatch as work done."""
        _seed_findings(plan_dir, "alpha", "beta", "gamma")
        script = _traverse_script()
        script[HarnessStates.EXPLORE] = {"success": False, "ctx": {}}

        harness = make_harness(script, roots=roots)
        result = harness.run()

        explore_results = [
            entry
            for entry in result.final_context[ContextKeys.ROLE_RESULTS]
            if entry["state"] == HarnessStates.EXPLORE
        ]
        assert explore_results
        assert all(entry["success"] is False for entry in explore_results)

    def test_a_failed_executor_still_spends_its_leash_attempt(
        self, make_harness, roots, plan_dir
    ) -> None:
        """EXECUTE owns no disk-derived key, so its accounting cannot move."""
        _seed_findings(plan_dir, "alpha", "beta", "gamma")
        harness = make_harness(_failing_execute_script(), roots=roots)
        result = harness.run()

        assert result.final_context[ContextKeys.FIX_ATTEMPTS] == 2
        assert result.final_context[ContextKeys.LAST_GATE_SLUG] == GateSlug.LEASH_CAP
        assert harness.worker.count_for(HarnessStates.EXECUTE) >= 2

    def test_a_state_that_owns_no_disk_derived_key_derives_nothing(
        self, make_harness, roots, plan_dir
    ) -> None:
        """The derivation is scoped by ``_WORKER_WRITABLE``, like the allowlist.

        EXECUTE does not own ``findings_count``, so an executor dispatch must
        not write one -- a count re-asserted from a state that does not own it
        is a driver writing a gate value on a turn nobody asked it to.
        """
        _seed_findings(plan_dir, "alpha", "beta")
        harness = make_harness(_traverse_script(), roots=roots)
        context = {ContextKeys.PLAN_DIR: roots[ContextKeys.PLAN_DIR]}

        assert harness.agent._derive_gate_counts(HarnessStates.EXECUTE, context) == {}
        assert harness.agent._derive_gate_counts(HarnessStates.REFLECT, context) == {}
        assert harness.agent._derive_gate_counts(HarnessStates.EXPLORE, context) == {
            ContextKeys.FINDINGS_COUNT: 2
        }

    def test_no_plan_directory_leaves_the_worker_reply_alone(
        self, make_harness
    ) -> None:
        """No directory is no evidence: the pre-D-032 path, unchanged.

        This is also what keeps the change narrow -- with nothing to read, the
        driver falls back to exactly the allowlist behaviour it had before.
        """
        harness = make_harness(_traverse_script(findings=3))
        harness.run()

        assert harness.worker.count_for(HarnessStates.EXPLORE) == 1
        assert HarnessStates.PLAN in harness.worker.states

    def test_no_worker_at_all_derives_nothing(
        self, make_harness, roots, plan_dir
    ) -> None:
        """D-045's diagnostic mode stays byte-identical.

        Three real files sit on disk, but nothing was dispatched, so no gate
        opens: a directory left behind by an earlier run is not this run's
        evidence.
        """
        _seed_findings(plan_dir, "alpha", "beta", "gamma")
        harness = make_harness(worker=None, roots=roots)
        result = harness.run()

        assert result.final_context[ContextKeys.FINDINGS_COUNT] == 0
        assert result.final_context.get(ContextKeys.LAST_GATE_SLUG) != (
            GateSlug.EXPLORE_CAP
        )

    @pytest.mark.parametrize("shape", ["file", "absent"])
    def test_an_unreadable_plan_directory_changes_nothing(
        self, make_harness, tmp_path, workspace, shape: str
    ) -> None:
        """A root that cannot be read is not evidence of zero findings."""
        blocker = tmp_path / "not-a-directory"
        if shape == "file":
            blocker.write_text("this is a file, not a plan directory")
        roots = {
            ContextKeys.PLAN_DIR: str(blocker),
            ContextKeys.WORKSPACE_ROOT: str(workspace),
        }

        harness = make_harness(_traverse_script(findings=3), roots=roots)
        result = harness.run()

        # The claim survives, because the driver derived nothing to replace it.
        assert result.final_context[ContextKeys.FINDINGS_COUNT] == 3
        # ...and the driver created no protocol memory just by counting it.
        assert not blocker.is_dir()

    def test_a_raising_filesystem_read_does_not_lose_the_turn(
        self, make_harness, roots, plan_dir, captured_logs, monkeypatch
    ) -> None:
        """The turn survives an I/O error, and still derives nothing from it.

        Every filesystem-shaped failure this could meet in practice (a missing
        directory, a path that is a file, an unreadable directory) is answered
        before or inside ``gate_files`` without raising, so the guard's
        ``except`` is reached only by a genuine OS error.  It is pinned by
        forcing one: a driver that let it escape would lose the whole protocol
        turn to a filesystem hiccup, and one that answered 0 would report a
        confident count it never read.
        """
        _seed_findings(plan_dir, "alpha", "beta", "gamma")

        def boom(*args: Any, **kwargs: Any) -> None:
            raise OSError("input/output error")

        monkeypatch.setattr(harness_module, "PlanMemory", boom)
        harness = make_harness(_traverse_script(findings=3), roots=roots)
        result = harness.run()

        assert result.final_context[ContextKeys.FINDINGS_COUNT] == 3
        # The construction is now the SHARED read-only helper's (step 25), so
        # the counting caller and the topic-assigning caller fail identically.
        assert any(
            "Could not open plan directory" in line and "unobserved" in line
            for line in captured_logs
        )

    def test_the_derived_count_is_driver_owned_provenance(
        self, make_harness, roots, plan_dir
    ) -> None:
        """The derivation goes through ``_apply``, so extraction cannot undo it.

        ``findings_count`` is a driver-owned key; a write that bypassed
        ``_apply`` would be reverted by ``_reassert_driver_owned`` on the next
        turn, which is the silent failure this asserts against.
        """
        _seed_findings(plan_dir, "alpha", "beta")
        harness = make_harness(
            _traverse_script(findings=3),
            roots=roots,
            extraction_data=FABRICATED_DRIVER_OWNED,
        )
        result = harness.run()

        assert result.final_context[ContextKeys.FINDINGS_COUNT] == 2
        assert harness.agent._driver_owned[ContextKeys.FINDINGS_COUNT] == 2

    def test_redispatch_stops_when_the_DISK_satisfies_the_gate(
        self, make_harness, roots, plan_dir
    ) -> None:
        """The loop condition still reads the gate's own variable.

        The explorer claims 0 forever; the files appear on disk during the
        second dispatch.  The re-dispatch loop must end there -- it tests the
        same number the gate does, and that number is now the disk's.
        """
        script = _traverse_script()
        seen = {"n": 0}

        def explorer(request: RoleRequest) -> dict[str, Any]:
            seen["n"] += 1
            if seen["n"] == 2:
                _seed_findings(plan_dir, "alpha", "beta", "gamma")
            return {"ctx": {ContextKeys.FINDINGS_COUNT: 0}}

        script[HarnessStates.EXPLORE] = explorer
        harness = make_harness(script, roots=roots)
        harness.run()

        assert harness.worker.count_for(HarnessStates.EXPLORE) == 2
        assert harness.worker.count_for(HarnessStates.PLAN) == 1


# ---------------------------------------------------------------------------
# Driver-assigned topic slugs (D-035)
# ---------------------------------------------------------------------------


def _explore_topics_seen(harness: Any) -> list[str | None]:
    """The slug each EXPLORE dispatch was actually handed, in order."""
    return [
        request.assigned_topic
        for request in harness.worker.requests
        if request.state == HarnessStates.EXPLORE
    ]


class TestDriverAssignedExploreTopics:
    """The driver names ONE ``findings/<slug>.md`` per EXPLORE dispatch.

    The source protocol assigns each explorer "a distinct kebab-case
    ``findings/{topic-slug}.md`` slug ... first check ``findings/`` for an
    existing file with that name -- no two live explorers may share a slug".
    This harness never did: it let the model choose, and then measured four
    mechanisms trying to make one model persist toward a COUNT of three
    (decisions.md D-027, D-031).  These pin the assignment AND the two ways it
    could quietly become a fail-open:

    * assigning a slug must leave the filesystem untouched -- a driver that
      pre-created its topics would move the gate without any role writing a
      byte, which is review C1's defect in a new costume;
    * a failed observation (no plan directory, an unreadable one) must assign
      NOTHING rather than a guessed topic.
    """

    # -- distinctness and the collision rule -----------------------------

    def test_each_dispatch_gets_a_distinct_slug(self, make_harness, roots) -> None:
        """Three blocked dispatches, three different topics -- never a repeat."""
        harness = make_harness(
            _explore_script([0]), roots=roots, max_explore_redispatches=2
        )
        harness.run()

        slugs = _explore_topics_seen(harness)

        assert len(slugs) == 3
        assert len(set(slugs)) == 3
        assert all(slug for slug in slugs)

    def test_a_redispatch_gets_the_next_slug_not_a_repeat(
        self, make_harness, roots, plan_dir
    ) -> None:
        """The re-dispatch loop (D-028) and the assignment agree on progress.

        Each dispatch writes its own assigned file, so the next one must be
        handed the NEXT topic -- not sent back at a file that now exists.
        """
        written: list[str] = []

        def explorer(request: RoleRequest) -> dict[str, Any]:
            _seed_findings(plan_dir, request.assigned_topic or "unassigned")
            written.append(request.assigned_topic or "unassigned")
            return {"ctx": {}}

        script = _traverse_script()
        script[HarnessStates.EXPLORE] = explorer

        harness = make_harness(script, roots=roots, max_explore_redispatches=5)
        result = harness.run()

        assert written == [topic.slug for topic in EXPLORE_TOPICS]
        assert len(set(written)) == 3
        # ...and the gate opened off the DISK, on the third file.
        assert result.final_context[ContextKeys.FINDINGS_COUNT] == 3
        assert harness.worker.count_for(HarnessStates.PLAN) == 1

    def test_a_slug_already_on_disk_is_never_assigned(
        self, make_harness, roots, plan_dir
    ) -> None:
        """The source's collision rule, read off the SAME derivation as the gate."""
        _seed_findings(plan_dir, "problem-scope")

        harness = make_harness(
            _explore_script([0]), roots=roots, max_explore_redispatches=3
        )
        harness.run()

        assert "problem-scope" not in _explore_topics_seen(harness)

    def test_every_topic_is_tried_before_any_is_repeated(
        self, make_harness, roots
    ) -> None:
        """A topic the model never writes must not monopolise the whole bound.

        The selection is "least-assigned free topic", which is BOTH rules at
        once: distinct while any topic is untried, then round-robin.  A plain
        "first free topic" rule would re-send every dispatch at the same slug
        and the other two would never be explored at all.
        """
        harness = make_harness(
            _explore_script([0]), roots=roots, max_explore_redispatches=5
        )
        harness.run()

        slugs = _explore_topics_seen(harness)

        assert len(slugs) == 6
        assert set(slugs) == {topic.slug for topic in EXPLORE_TOPICS}
        assert slugs[:3] == [topic.slug for topic in EXPLORE_TOPICS]
        assert sorted(slugs) == sorted(2 * [t.slug for t in EXPLORE_TOPICS])

    # -- assignment writes NOTHING ---------------------------------------

    def test_assigning_a_slug_creates_no_file(
        self, make_harness, roots, plan_dir
    ) -> None:
        """The fail-open this could become: reserving topics as empty files.

        A count that moves because the DRIVER touched the filesystem is not
        evidence that a role did any work.  Assignment is a read.
        """
        harness = make_harness(
            _explore_script([0]), roots=roots, max_explore_redispatches=4
        )
        result = harness.run()

        assert list((plan_dir / "findings").iterdir()) == []
        assert result.final_context[ContextKeys.FINDINGS_COUNT] == 0
        assert result.final_context[ContextKeys.LAST_GATE_SLUG] == GateSlug.EXPLORE_CAP

    def test_assigning_does_not_create_the_plan_directory(
        self, make_harness, tmp_path, workspace
    ) -> None:
        """``PlanMemory`` CREATES its root; the assignment must not call it."""
        absent = tmp_path / "plans" / "never-created"
        roots = {
            ContextKeys.PLAN_DIR: str(absent),
            ContextKeys.WORKSPACE_ROOT: str(workspace),
        }

        harness = make_harness(
            _explore_script([0]), roots=roots, max_explore_redispatches=2
        )
        harness.run()

        assert not absent.exists()
        assert _explore_topics_seen(harness) == [None, None, None]

    # -- failing closed ---------------------------------------------------

    def test_no_plan_directory_assigns_nothing(self, make_harness) -> None:
        """No directory to check for collisions means no assignment at all."""
        harness = make_harness(_explore_script([0]), max_explore_redispatches=2)
        harness.run()

        assert _explore_topics_seen(harness) == [None, None, None]
        assert harness.agent._assigned_topics == []

    def test_an_unreadable_plan_directory_assigns_nothing_and_moves_no_gate(
        self, make_harness, roots, plan_dir, monkeypatch
    ) -> None:
        """A failed observation fabricates neither a topic nor a count."""
        _seed_findings(plan_dir, "alpha")

        def boom(*args: Any, **kwargs: Any) -> None:
            raise OSError("input/output error")

        monkeypatch.setattr(harness_module, "PlanMemory", boom)
        harness = make_harness(
            _explore_script([0]), roots=roots, max_explore_redispatches=1
        )
        result = harness.run()

        assert _explore_topics_seen(harness) == [None, None]
        assert harness.agent._assigned_topics == []
        assert result.final_context[ContextKeys.FINDINGS_COUNT] == 0

    def test_the_diagnostic_mode_assigns_nothing(self, make_harness, roots) -> None:
        """``worker_factory=None`` attempts no dispatch, so it spends no topic."""
        harness = make_harness(worker=None, roots=roots)
        harness.run()

        assert harness.agent._assigned_topics == []

    # -- scope and lifetime ------------------------------------------------

    def test_only_explore_dispatches_are_assigned_a_topic(
        self, make_harness, roots, plan_dir
    ) -> None:
        """Every other role owns no ``findings/`` file and must not be told to."""
        _seed_findings(plan_dir, "alpha", "beta", "gamma")
        harness = make_harness(_traverse_script(findings=3), roots=roots)
        harness.run()

        assigned = {
            request.state: request.assigned_topic
            for request in harness.worker.requests
            if request.state != HarnessStates.EXPLORE
        }

        assert assigned
        assert set(assigned.values()) == {None}

    def test_the_assignment_ledger_resets_between_runs(
        self, make_harness, roots
    ) -> None:
        """Per RUN, like ``_explore_redispatches`` -- not per EXPLORE entry.

        A per-entry reset would be refillable by a worker: the verifier may set
        ``needs_explore``, which routes REFLECT -> EXPLORE (D-029).  A per-run
        reset means the second run starts from the protocol's first topic
        again, against whatever that run's directory holds.
        """
        harness = make_harness(
            _explore_script([0]), roots=roots, max_explore_redispatches=1
        )
        harness.run()
        first = list(harness.agent._assigned_topics)
        harness.run()

        assert first == [topic.slug for topic in EXPLORE_TOPICS[:2]]
        assert harness.agent._assigned_topics == first

    def test_a_worker_cannot_reach_the_assignment_ledger(
        self, make_harness, roots
    ) -> None:
        """It is driver run state, not context -- so no worker can rewrite it."""
        harness = make_harness(
            _explore_script([0]), roots=roots, max_explore_redispatches=1
        )
        result = harness.run()

        assert not any("assigned" in str(key).lower() for key in result.final_context)
        assert all(
            "assigned_topic" not in dict(request.context)
            for request in harness.worker.requests
        )

    # -- the C1 fail-open, re-run with the assignment in place -------------

    def test_a_dispatch_that_writes_nothing_and_claims_three_leaves_the_gate_shut(
        self, make_harness, roots, plan_dir
    ) -> None:
        """Review C1's reproduction, under the new mechanism.

        An assignment tells a dispatch WHERE to write; it never tells the gate
        that it did.  The count is still the driver's own read of the files
        really on disk.
        """
        harness = make_harness(
            _explore_script([3]), roots=roots, max_explore_redispatches=2
        )
        result = harness.run()

        assert list((plan_dir / "findings").iterdir()) == []
        assert result.final_context[ContextKeys.FINDINGS_COUNT] == 0
        assert result.final_context[ContextKeys.LAST_GATE_SLUG] == GateSlug.EXPLORE_CAP
        assert harness.worker.count_for(HarnessStates.PLAN) == 0


def _plan_doc_with(files_rows: str, steps: str | None = None) -> PlanDoc:
    """``_PLAN_MD`` with its Files-To-Modify rows (and optionally Steps) swapped.

    A clone of the REAL 11-section fixture rather than a hand-stub, because
    ``PlanDoc.from_markdown`` rejects anything less and the derivation under
    test only ever sees a plan that parsed.
    """
    text = _PLAN_MD.replace("| `harness.py` | wire the seam | step 10 |", files_rows)
    if steps is not None:
        text = text.replace(
            "1. [x] **Wire the pre-step gate onto the on-disk validator.**"
            " [RISK: high] [deps: none]\n"
            "2. [ ] **Emit the presentation contracts from the artifacts.**"
            " [RISK: low] [deps: 1]",
            steps,
        )
    return PlanDoc.from_markdown(text)


class TestDriverAssignedExecuteTarget:
    """The driver names the ONE workspace file an EXECUTE step is to edit.

    B0 (scripts/bench_data/l4-execute-write/B0, n=40) measured why: the native
    executor issued a write tool 15/40 but landed the requested edit on the
    assigned workspace file only 2/40 -- 13 of 15 issued writes missed the
    target, in the wrong-root shapes ``write_file('<plan-id>/uploader.py')`` /
    ``read_plan_file('<plan-id>/uploader.py')``.  D-010 extends the D-035
    driver-assigned-target pattern to EXECUTE: the driver reads plan.md's
    Files To Modify (the plan-writer's existing obligation, rules.py) and
    assigns the target; the model never infers the root.  The prompt half is
    pinned in ``test_roles_and_tools.py``.

    Fail-open contract (plan.md edge case, verbatim): never guess a root --
    unparseable means NO assignment, so the dispatch prompt falls back to the
    pre-D-010 text, never to a wrong assignment.
    """

    # -- the pure derivation ------------------------------------------------

    def test_parses_the_seeded_plan_files_to_modify(self) -> None:
        """The fixture table's one row is the assignment."""
        assert derive_execute_target(PlanDoc.from_markdown(_PLAN_MD), 1) == (
            "harness.py"
        )

    def test_prefers_the_candidate_named_in_the_current_step_text(self) -> None:
        """The ONE heuristic beyond "first": a per-step mapping, when it exists.

        Without it, every step of a multi-file plan would be pointed at the
        same first file -- a wrong assignment, which is the one failure shape
        the fail-open rule forbids.
        """
        plan = _plan_doc_with(
            "| `alpha.py` | a | r |\n| `beta.py` | b | r |",
            steps=(
                "1. [x] **Edit `alpha.py`.** [RISK: low] [deps: none]\n"
                "2. [ ] **Edit `beta.py`.** [RISK: low] [deps: 1]"
            ),
        )

        assert derive_execute_target(plan, 1) == "alpha.py"
        assert derive_execute_target(plan, 2) == "beta.py"

    def test_multiple_files_and_no_step_mapping_assigns_the_first(self) -> None:
        """The plan.md edge case verbatim: ambiguous -> FIRST target, stated."""
        plan = _plan_doc_with("| `alpha.py` | a | r |\n| `beta.py` | b | r |")

        assert derive_execute_target(plan, 2) == "alpha.py"

    def test_only_the_file_column_supplies_candidates(self) -> None:
        """First path-shaped token per LINE -- prose columns must not compete.

        ``beta.py`` here appears only in a Change cell and in the step text;
        promoting it would assign a file the plan never listed for editing.
        """
        plan = _plan_doc_with(
            "| `alpha.py` | touch `beta.py` too | r |",
            steps="1. [ ] **Edit `beta.py`.** [RISK: low] [deps: none]",
        )

        assert derive_execute_target(plan, 1) == "alpha.py"

    @pytest.mark.parametrize(
        "rows",
        [
            "| nothing quoted here | x | y |",
            "| `/etc/passwd` | absolute | y |",
            "| `../evil.py` | traversal | y |",
            "| `scripts/bench_data/**` | a glob, not a file | y |",
            "| `NEW` | a bare word, not a path | y |",
            "| `upload()` | a call, not a path | y |",
        ],
    )
    def test_an_unparseable_section_assigns_nothing(self, rows: str) -> None:
        """Never guess: no path-shaped candidate -> ``None``, not a repair.

        Each row is a shape the derivation must REFUSE -- guessing any of them
        into an assignment would point a live executor at a file outside the
        workspace contract (the B0 defect, driver-made).
        """
        assert derive_execute_target(_plan_doc_with(rows), 1) is None

    def test_a_step_number_the_plan_does_not_contain_falls_back_to_first(
        self,
    ) -> None:
        """An out-of-range cursor skips the heuristic, not the assignment."""
        assert derive_execute_target(PlanDoc.from_markdown(_PLAN_MD), 99) == (
            "harness.py"
        )

    # -- the driver wrapper fails closed, and says WHY (D-005) --------------
    #
    # Until D-005 the wrapper collapsed three distinct no-assignment causes
    # (no plan dir / plan.md unusable / no _TARGET_RE token) into one silent
    # None, so a live run whose plan-writer wrote a Files-To-Modify table in a
    # rejected shape was indistinguishable from a run with no plan.md at all.

    def test_the_wrapper_reads_the_seeded_plan_directory(self, plan_dir: Path) -> None:
        (plan_dir / ArtifactNames.PLAN).write_text(_PLAN_MD)
        context = {ContextKeys.PLAN_DIR: str(plan_dir), ContextKeys.STEP_NUMBER: 1}

        assert HarnessAgent._assign_execute_target(context) == (
            "harness.py",
            EXECUTE_TARGET_ASSIGNED,
        )

    def test_no_plan_md_assigns_nothing(self, plan_dir: Path) -> None:
        """An unreadable plan is a failed observation, not a licence to guess."""
        context = {ContextKeys.PLAN_DIR: str(plan_dir), ContextKeys.STEP_NUMBER: 1}

        assert HarnessAgent._assign_execute_target(context) == (
            None,
            EXECUTE_TARGET_NO_PLAN_DOC,
        )

    def test_a_garbled_plan_md_is_a_no_plan_doc_not_a_no_token(
        self, plan_dir: Path
    ) -> None:
        """A plan.md that fails ``PlanDoc`` parsing is the DOCUMENT's failure.

        Distinguishing it from ``no-target-token`` is the point of the tag: a
        garbled document and a valid document with an unparseable file table
        are different defects with different fixes.
        """
        (plan_dir / ArtifactNames.PLAN).write_text("not a plan at all\n")
        context = {ContextKeys.PLAN_DIR: str(plan_dir), ContextKeys.STEP_NUMBER: 1}

        assert HarnessAgent._assign_execute_target(context) == (
            None,
            EXECUTE_TARGET_NO_PLAN_DOC,
        )

    def test_a_valid_plan_with_no_path_token_is_a_no_token(
        self, plan_dir: Path
    ) -> None:
        """The plan PARSED; only its Files-To-Modify had no path-shaped token."""
        text = _PLAN_MD.replace(
            "| `harness.py` | wire the seam | step 10 |",
            "| nothing quoted here | x | y |",
        )
        (plan_dir / ArtifactNames.PLAN).write_text(text)
        context = {ContextKeys.PLAN_DIR: str(plan_dir), ContextKeys.STEP_NUMBER: 1}

        assert HarnessAgent._assign_execute_target(context) == (
            None,
            EXECUTE_TARGET_NO_TOKEN,
        )

    def test_no_plan_directory_key_assigns_nothing(self) -> None:
        assert HarnessAgent._assign_execute_target({}) == (
            None,
            EXECUTE_TARGET_NO_PLAN_DIR,
        )

    def test_an_absent_plan_directory_is_not_created(self, tmp_path: Path) -> None:
        """Assignment is a READ (D-035 property 2): no mkdir side effect."""
        absent = tmp_path / "plans" / "never-created"
        context = {ContextKeys.PLAN_DIR: str(absent), ContextKeys.STEP_NUMBER: 1}

        assert HarnessAgent._assign_execute_target(context) == (
            None,
            EXECUTE_TARGET_NO_PLAN_DIR,
        )
        assert not absent.exists()

    # -- the assigned value reaches the dispatch request --------------------

    def test_the_assigned_target_reaches_the_execute_dispatch(
        self, make_harness, roots, plan_dir
    ) -> None:
        """End of the seam: the ``RoleRequest`` a real EXECUTE worker receives."""
        _seed_plan_directory(plan_dir)
        harness = make_harness(_traverse_script(), roots=roots)
        harness.run()

        targets = [
            request.assigned_write_target
            for request in harness.worker.requests
            if request.state == HarnessStates.EXECUTE
        ]
        assert targets == ["harness.py"]

    def test_only_execute_dispatches_are_assigned_a_target(
        self, make_harness, roots, plan_dir
    ) -> None:
        """No other role is told to edit the workspace; no other state drifts."""
        _seed_plan_directory(plan_dir)
        harness = make_harness(_traverse_script(), roots=roots)
        harness.run()

        assigned = {
            request.state: request.assigned_write_target
            for request in harness.worker.requests
            if request.state != HarnessStates.EXECUTE
        }
        assert assigned
        assert set(assigned.values()) == {None}

    # -- the D-005 reason tag reaches the dispatch, and ONLY the dispatch ----

    def test_the_execute_dispatch_carries_the_assigned_reason(
        self, make_harness, roots, plan_dir
    ) -> None:
        """A successful assignment is tagged, on the same ``RoleRequest``."""
        _seed_plan_directory(plan_dir)
        harness = make_harness(_traverse_script(), roots=roots)
        harness.run()

        reasons = [
            request.execute_target_reason
            for request in harness.worker.requests
            if request.state == HarnessStates.EXECUTE
        ]
        assert reasons == [EXECUTE_TARGET_ASSIGNED]

    def test_non_execute_dispatches_carry_no_reason(
        self, make_harness, roots, plan_dir
    ) -> None:
        """The tag is EXECUTE diagnostics, not a new per-state channel."""
        _seed_plan_directory(plan_dir)
        harness = make_harness(_traverse_script(), roots=roots)
        harness.run()

        reasons = {
            request.execute_target_reason
            for request in harness.worker.requests
            if request.state != HarnessStates.EXECUTE
        }
        assert reasons == {None}

    def test_the_reason_never_changes_the_unassigned_prompt_bytes(self) -> None:
        """D-010 fail-open under the new field: the prompt NEVER reads it.

        An unassigned EXECUTE dispatch must render byte-identically whatever
        diagnostic reason rides along -- the tag is for the bench rubric, not
        for the model.
        """
        from dataclasses import replace

        from fsm_llm_harness.roles import build_role_prompt, get_role_spec

        rules = get_rules(HarnessStates.EXECUTE)
        base = RoleRequest(
            role=ROLE_BY_STATE[HarnessStates.EXECUTE],
            state=HarnessStates.EXECUTE,
            goal="exercise the harness protocol",
            operative_rules=rules.operative_rules,
            gate_summary=rules.gate_summary,
            iteration=1,
            step_number=1,
            total_steps=1,
            fix_attempts=0,
        )
        spec = get_role_spec(HarnessStates.EXECUTE)
        baseline = build_role_prompt(base, spec)

        for reason in (
            EXECUTE_TARGET_NO_PLAN_DIR,
            EXECUTE_TARGET_NO_PLAN_DOC,
            EXECUTE_TARGET_NO_TOKEN,
        ):
            tagged = replace(base, execute_target_reason=reason)
            assert build_role_prompt(tagged, spec) == baseline


# ---------------------------------------------------------------------------
# The integration seam: artifacts + storage + validator in the driver
# (D-037 the driver read path, D-038 the state.md sync + resume,
#  D-039 the revert seam, D-040 the four-slug action table)
# ---------------------------------------------------------------------------


def _state_md(
    *,
    state: str = HarnessStates.EXECUTE,
    iteration: int = 1,
    step: str = "1 of 1",
    attempts: int = 0,
    history: tuple[str, ...] = (),
) -> str:
    """A ``state.md`` the real serializer produces, so it round-trips exactly."""
    return StateDoc(
        state=state,
        iteration=iteration,
        current_step=step,
        fix_attempts=[f"Step 1, attempt {n}" for n in range(1, attempts + 1)],
        transition_history=list(history),
    ).to_markdown()


def _gate_context(**overrides: Any) -> dict[str, Any]:
    """A merged-view context of the shape ``_pre_step_gate`` is handed."""
    context: dict[str, Any] = {
        ContextKeys.ITERATION: 1,
        ContextKeys.STEP_NUMBER: 1,
        ContextKeys.TOTAL_STEPS: 1,
        ContextKeys.FIX_ATTEMPTS: 0,
        ContextKeys.ROLE_RESULTS: [],
    }
    context.update(overrides)
    return context


def _names(agent: HarnessAgent) -> list[str]:
    return [presentation.name for presentation in agent.presentations]


class TestProseExecuteTargetFallback:
    """D-001: an existence-gated PROSE fallback for the EXECUTE target.

    A real 4b plan writes its Files-To-Modify as prose (no backticks), so the
    strict ``derive_execute_target`` (backtick-only) returns ``None``, no target
    is assigned, and the model's self-directed EXECUTE write produces a label the
    HASH-FROZEN L6 floor normalizer cannot credit (B5 run-1: verified_write=False).
    The fallback extracts a path-shaped, extension-bearing prose token that names
    an EXISTING workspace file; the existence gate keeps D-010's real invariant
    (never point a live executor somewhere the plan never said) and makes
    selection order-independent.  ``derive_execute_target`` stays byte-unchanged.
    """

    _REAL_B5_PLAN = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "bench_data"
        / "l6-e2e"
        / "B5"
        / "artifacts"
        / "run-1"
        / "plan.md"
    )
    _SEED = frozenset({"uploader.py", "config.py", "README.md"})

    @staticmethod
    def _prose_plan_text(files_body: str) -> str:
        return _PLAN_MD.replace(
            "| `harness.py` | wire the seam | step 10 |", files_body
        )

    # -- the pure prose derivation -----------------------------------------

    def test_prose_target_on_the_real_b5_plan(self) -> None:
        """The retained real plan whose prose broke B5 run-1 -> ``uploader.py``."""
        text = self._REAL_B5_PLAN.read_text()
        plan = PlanDoc.from_markdown(text)
        # The strict path finds nothing (this is WHY the fallback exists).
        assert derive_execute_target(plan, 1) is None
        assert _derive_prose_target(plan, 1, self._SEED) == "uploader.py"

    def test_the_existence_gate_is_order_independent(self) -> None:
        """A method-call token BEFORE the filename does not win: only a real
        workspace file can be assigned, wherever it sits in the line."""
        plan = _plan_doc_with("Wrap requests.post inside uploader.py to add retry")
        assert _derive_prose_target(plan, 1, self._SEED) == "uploader.py"

    def test_a_method_call_token_alone_assigns_nothing(self) -> None:
        """``requests.post`` is path-shaped and extension-bearing but is not a
        workspace file, so the gate refuses it (no guess)."""
        plan = _plan_doc_with("modify the requests.post call in the uploader")
        assert _derive_prose_target(plan, 1, self._SEED) is None

    def test_a_token_naming_no_existing_file_assigns_nothing(self) -> None:
        """A new-file step names a file that does not exist yet -> fail-open."""
        plan = _plan_doc_with("create retry_helper.py with the backoff logic")
        assert _derive_prose_target(plan, 1, self._SEED) is None

    def test_an_empty_existence_set_assigns_nothing(self) -> None:
        """No known workspace files (unset/unreadable root) -> fail-open None."""
        plan = _plan_doc_with("uploader.py: add retry")
        assert _derive_prose_target(plan, 1, frozenset()) is None

    def test_first_existing_file_per_line_wins_across_lines(self) -> None:
        """Multi-line prose: the first existing-file token, line by line."""
        plan = _plan_doc_with("uploader.py: add retry\nconfig.py: set RETRIES=3")
        assert _derive_prose_target(plan, 1, self._SEED) == "uploader.py"

    def test_prefers_the_file_named_in_the_current_step_text(self) -> None:
        """The same per-step preference the strict derivation uses."""
        plan = _plan_doc_with(
            "uploader.py: add retry\nconfig.py: set RETRIES=3",
            steps=(
                "1. [ ] **Edit config.py first.** [RISK: low] [deps: none]\n"
                "2. [ ] **Then uploader.py.** [RISK: low] [deps: 1]"
            ),
        )
        assert _derive_prose_target(plan, 1, self._SEED) == "config.py"

    def test_a_backticked_prose_line_is_still_stripped_and_gated(self) -> None:
        """Inline backticks the strict path already handles are also fine here
        (the fallback only RUNS on the strict None, but must not choke on them)."""
        plan = _plan_doc_with("edit `uploader.py` for the retry loop")
        assert _derive_prose_target(plan, 1, self._SEED) == "uploader.py"

    # -- the driver wrapper, and the new reason tag ------------------------

    def test_assign_execute_target_uses_the_prose_fallback(
        self, plan_dir: Path, tmp_path: Path
    ) -> None:
        """End to end through ``_assign_execute_target`` with a real workspace."""
        (plan_dir / ArtifactNames.PLAN).write_text(
            self._prose_plan_text("uploader.py: wrap upload() with retry backoff")
        )
        workspace = tmp_path / "ws"
        workspace.mkdir()
        for name in self._SEED:
            (workspace / name).write_text("x")
        context = {
            ContextKeys.PLAN_DIR: str(plan_dir),
            ContextKeys.WORKSPACE_ROOT: str(workspace),
            ContextKeys.STEP_NUMBER: 1,
        }
        assert HarnessAgent._assign_execute_target(context) == (
            "uploader.py",
            EXECUTE_TARGET_ASSIGNED_PROSE,
        )

    def test_a_backticked_plan_never_reaches_the_prose_fallback(
        self, plan_dir: Path, tmp_path: Path
    ) -> None:
        """Regression: when the strict path finds a target, the reason is the
        plain ``assigned`` tag, not the prose one -- behaviour unchanged."""
        (plan_dir / ArtifactNames.PLAN).write_text(_PLAN_MD)
        workspace = tmp_path / "ws"
        workspace.mkdir()
        (workspace / "harness.py").write_text("x")
        context = {
            ContextKeys.PLAN_DIR: str(plan_dir),
            ContextKeys.WORKSPACE_ROOT: str(workspace),
            ContextKeys.STEP_NUMBER: 1,
        }
        assert HarnessAgent._assign_execute_target(context) == (
            "harness.py",
            EXECUTE_TARGET_ASSIGNED,
        )

    def test_no_workspace_root_leaves_the_prose_plan_a_no_token(
        self, plan_dir: Path
    ) -> None:
        """Fail-open: no workspace root -> empty existence set -> the prose plan
        still resolves to ``no-target-token`` (byte-identical prompt)."""
        (plan_dir / ArtifactNames.PLAN).write_text(
            self._prose_plan_text("uploader.py: wrap upload() with retry backoff")
        )
        context = {ContextKeys.PLAN_DIR: str(plan_dir), ContextKeys.STEP_NUMBER: 1}
        assert HarnessAgent._assign_execute_target(context) == (
            None,
            EXECUTE_TARGET_NO_TOKEN,
        )


class TestFourSlugsFourActions:
    """D-040: the four pre-step-gate slugs are four DIFFERENT events."""

    def test_the_table_is_complete_and_injective(self) -> None:
        """Four slugs, four distinct actions -- not one halt wearing four names."""
        assert set(harness_module._GATE_ACTIONS) == set(GateSlug.ORDER)
        assert len(set(harness_module._GATE_ACTIONS.values())) == 4

    def test_leash_cap_emits_the_leash_block_and_routes_to_reflect(
        self, make_harness, roots, plan_dir
    ) -> None:
        harness = make_harness(_failing_execute_script(), roots=roots)
        agent = harness.agent
        (plan_dir / ArtifactNames.STATE).write_text(_state_md(attempts=2))

        delta = agent._pre_step_gate(
            _gate_context(**roots, **{ContextKeys.FIX_ATTEMPTS: 2})
        )

        assert delta is not None
        assert delta[ContextKeys.LAST_GATE_SLUG] == GateSlug.LEASH_CAP
        # It CONTINUES -- EXECUTE -> REFLECT -- rather than ending the run.
        assert delta[ContextKeys.EXECUTE_COMPLETE] is True
        assert _names(agent) == ["PC-EXECUTE-LEASH"]
        assert agent._halt_request is None

    def test_iteration_cap_hard_stops_with_no_leash_block(
        self, make_harness, roots, plan_dir
    ) -> None:
        """A run can hit the cap with ZERO attempts; a leash block would lie."""
        harness = make_harness(_traverse_script(), roots=roots, iteration_hard_cap=2)
        agent = harness.agent
        (plan_dir / ArtifactNames.STATE).write_text(_state_md(iteration=2))

        delta = agent._pre_step_gate(
            _gate_context(**roots, **{ContextKeys.ITERATION: 2})
        )

        assert delta is not None
        assert delta[ContextKeys.LAST_GATE_SLUG] == GateSlug.ITERATION_CAP
        assert _names(agent) == []
        assert agent._reverts == []
        # It ENDS the run: a halt is pending, and it does not route onward.
        assert agent._halt_request is not None
        assert agent._halt_request.slug == GateSlug.ITERATION_CAP
        assert ContextKeys.EXECUTE_COMPLETE not in delta

    def test_wrong_state_recovers_and_does_not_write_state_md(
        self, make_harness, roots, plan_dir
    ) -> None:
        """The driver must not silence its own alarm by overwriting state.md."""
        harness = make_harness(_traverse_script(), roots=roots)
        stale = _state_md(state=HarnessStates.REFLECT)
        (plan_dir / ArtifactNames.STATE).write_text(stale)

        delta = harness.agent._pre_step_gate(_gate_context(**roots))

        assert delta is not None
        assert delta[ContextKeys.LAST_GATE_SLUG] == GateSlug.WRONG_STATE
        assert (plan_dir / ArtifactNames.STATE).read_text() == stale
        assert _names(harness.agent) == []
        assert ContextKeys.EXECUTE_COMPLETE not in delta

    def test_no_plan_recovers_and_does_not_create_state_md(
        self, make_harness, roots, plan_dir
    ) -> None:
        """``no-plan`` fires BECAUSE state.md is unreadable; writing it is the bug."""
        harness = make_harness(_traverse_script(), roots=roots)

        delta = harness.agent._pre_step_gate(_gate_context(**roots))

        assert delta is not None
        assert delta[ContextKeys.LAST_GATE_SLUG] == GateSlug.NO_PLAN
        assert not (plan_dir / ArtifactNames.STATE).exists()
        assert _names(harness.agent) == []
        assert ContextKeys.EXECUTE_COMPLETE not in delta

    def test_each_capability_belongs_to_exactly_one_slug(
        self, make_harness, roots, plan_dir
    ) -> None:
        """Stated once, as a whole, over every slug at once.

        Three behavioural shapes, not four: ``wrong-state`` and ``no-plan``
        deliberately SHARE the recovery shape (both record and write nothing)
        and differ in the slug and reason they record -- which is asserted
        separately below rather than folded in here, where including the slug
        would make the comparison trivially true.
        """
        harness = make_harness(_traverse_script(), roots=roots, iteration_hard_cap=2)
        agent = harness.agent
        seen: dict[str, tuple[Any, ...]] = {}
        reasons: dict[str, str] = {}
        cases = {
            GateSlug.NO_PLAN: (None, _gate_context(**roots)),
            GateSlug.WRONG_STATE: (
                _state_md(state=HarnessStates.PLAN),
                _gate_context(**roots),
            ),
            GateSlug.LEASH_CAP: (
                _state_md(attempts=2),
                _gate_context(**roots, **{ContextKeys.FIX_ATTEMPTS: 2}),
            ),
            GateSlug.ITERATION_CAP: (
                _state_md(iteration=2),
                _gate_context(**roots, **{ContextKeys.ITERATION: 2}),
            ),
        }
        for slug, (content, context) in cases.items():
            agent._presentations = []
            agent._halt_request = None
            if content is None:
                (plan_dir / ArtifactNames.STATE).unlink(missing_ok=True)
            else:
                (plan_dir / ArtifactNames.STATE).write_text(content)
            delta = agent._pre_step_gate(context)
            assert delta is not None and delta[ContextKeys.LAST_GATE_SLUG] == slug
            seen[slug] = (
                delta.get(ContextKeys.EXECUTE_COMPLETE),
                tuple(_names(agent)),
                agent._halt_request is not None,
            )
            reasons[slug] = str(delta[ContextKeys.HALT_REASON])

        # Exactly one slug emits the leash block.
        assert [s for s, (_, blocks, _) in seen.items() if blocks] == [
            GateSlug.LEASH_CAP
        ]
        # Exactly one slug arms a halt.
        assert [s for s, (_, _, halt) in seen.items() if halt] == [
            GateSlug.ITERATION_CAP
        ]
        # Exactly one slug routes the protocol onward.
        assert [s for s, (done, _, _) in seen.items() if done] == [GateSlug.LEASH_CAP]
        # The two recovery slugs share a shape and differ in what they say.
        assert seen[GateSlug.NO_PLAN] == seen[GateSlug.WRONG_STATE]
        assert len(set(reasons.values())) == 4, reasons
        assert GateSlug.NO_PLAN in reasons[GateSlug.NO_PLAN]
        assert GateSlug.WRONG_STATE in reasons[GateSlug.WRONG_STATE]

    def test_an_earlier_slug_short_circuits_the_later_checks(
        self, make_harness, roots, plan_dir
    ) -> None:
        """``no-plan`` wins over three simultaneous later failures."""
        harness = make_harness(_traverse_script(), roots=roots, iteration_hard_cap=1)
        (plan_dir / ArtifactNames.STATE).write_text("not a state document at all\n")

        delta = harness.agent._pre_step_gate(
            _gate_context(
                **roots, **{ContextKeys.ITERATION: 9, ContextKeys.FIX_ATTEMPTS: 0}
            )
        )

        assert delta is not None
        assert delta[ContextKeys.LAST_GATE_SLUG] == GateSlug.NO_PLAN

    def test_wrong_state_wins_over_a_simultaneous_leash_cap(
        self, make_harness, roots, plan_dir
    ) -> None:
        harness = make_harness(_traverse_script(), roots=roots)
        (plan_dir / ArtifactNames.STATE).write_text(
            _state_md(state=HarnessStates.PIVOT, attempts=2)
        )

        delta = harness.agent._pre_step_gate(_gate_context(**roots))

        assert delta is not None
        assert delta[ContextKeys.LAST_GATE_SLUG] == GateSlug.WRONG_STATE

    def test_no_plan_directory_at_all_leaves_the_in_memory_leash_intact(
        self, make_harness
    ) -> None:
        """The four slugs are statements about an on-disk protocol.

        A run without one is not FAILING them -- but its leash must still bite,
        which is what the in-memory channel is for.
        """
        harness = make_harness(_traverse_script())
        agent = harness.agent

        assert agent._pre_step_gate(_gate_context()) is None
        blocked = agent._pre_step_gate(_gate_context(**{ContextKeys.FIX_ATTEMPTS: 2}))
        assert blocked is not None
        assert blocked[ContextKeys.LAST_GATE_SLUG] == GateSlug.LEASH_CAP

    def test_the_shut_channel_wins_when_the_two_disagree(
        self, make_harness, roots, plan_dir
    ) -> None:
        """Disk says fine, the driver's own counter says spent: the leash bites.

        ``fix_attempts`` in context is derived from ``AgentResult.success``,
        which no worker can understate; state.md's attempt lines are a file a
        torn write or a hand edit can shorten.
        """
        harness = make_harness(_traverse_script(), roots=roots)
        (plan_dir / ArtifactNames.STATE).write_text(_state_md(attempts=0))

        delta = harness.agent._pre_step_gate(
            _gate_context(**roots, **{ContextKeys.FIX_ATTEMPTS: 2})
        )

        assert delta is not None
        assert delta[ContextKeys.LAST_GATE_SLUG] == GateSlug.LEASH_CAP

    def test_a_raising_gate_fails_closed_as_no_plan(
        self, make_harness, roots, monkeypatch
    ) -> None:
        monkeypatch.setattr(
            harness_module,
            "pre_step_gate",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("disk on fire")),
        )
        harness = make_harness(_traverse_script(), roots=roots)

        delta = harness.agent._pre_step_gate(_gate_context(**roots))

        assert delta is not None
        assert delta[ContextKeys.LAST_GATE_SLUG] == GateSlug.NO_PLAN

    def test_the_leash_cap_halt_reaches_the_run_end_to_end(
        self, make_harness, roots, plan_dir
    ) -> None:
        """The real protocol, not a hand-built context."""
        _seed_findings(plan_dir, "alpha", "beta", "gamma")
        harness = make_harness(
            _failing_execute_script(),
            approvals=ApprovalRecorder({APPROVAL_LEASH: False}),
            roots=roots,
        )
        result = harness.run()

        assert result.final_context[ContextKeys.LAST_GATE_SLUG] == GateSlug.LEASH_CAP
        assert harness.worker.count_for(HarnessStates.EXECUTE) == 2
        assert "PC-EXECUTE-LEASH" in _names(harness.agent)

    def test_the_iteration_cap_halt_reaches_the_run_end_to_end(
        self, make_harness, roots, plan_dir
    ) -> None:
        """The cap ends the run through a real halt, with no leash block."""
        _seed_findings(plan_dir, "alpha", "beta", "gamma")
        harness = make_harness(_traverse_script(), roots=roots, iteration_hard_cap=1)
        result = harness.run()

        assert result.success is False
        assert (
            result.final_context[ContextKeys.LAST_GATE_SLUG] == GateSlug.ITERATION_CAP
        )
        assert "PC-EXECUTE-LEASH" not in _names(harness.agent)
        assert harness.worker.count_for(HarnessStates.EXECUTE) == 0
        # The cap ENDS the run through its own halt.  Without the deferred
        # raise the protocol would merely sit there until the stall detector
        # gave up, which reports the same slug for a different reason -- so the
        # ANSWER, not the slug, is what discriminates the two.
        assert result.answer.startswith("Iteration cap:")
        assert "Stalled" not in result.answer


class TestStateDocSync:
    """D-038: the driver writes exactly one artifact, and writes it atomically."""

    def test_state_md_records_the_drivers_position(
        self, make_harness, roots, plan_dir
    ) -> None:
        _seed_findings(plan_dir, "alpha", "beta", "gamma")
        harness = make_harness(_traverse_script(total_steps=3), roots=roots)
        harness.run()

        doc = StateDoc.from_markdown((plan_dir / ArtifactNames.STATE).read_text())
        assert doc.state in HarnessStates.ALL
        assert doc.iteration == 1
        assert doc.current_step.startswith("3 of 3")

    def test_the_attempt_lines_use_the_protocols_own_grammar(
        self, make_harness, roots, plan_dir
    ) -> None:
        """The gate counts these lines; a private format would count zero."""
        _seed_findings(plan_dir, "alpha", "beta", "gamma")
        harness = make_harness(
            _failing_execute_script(),
            approvals=ApprovalRecorder({APPROVAL_LEASH: False}),
            roots=roots,
        )
        harness.run()

        doc = StateDoc.from_markdown((plan_dir / ArtifactNames.STATE).read_text())
        assert doc.fix_attempt_count == 2

    def test_transition_history_is_preserved_and_never_appended_to(
        self, make_harness, roots, plan_dir
    ) -> None:
        """D-038: a line per FSM entry would make the derived iteration outrun
        the real one and fire ``iteration-cap`` on a run that never re-planned."""
        history = ("INIT → EXPLORE (task started)",)
        _seed_findings(plan_dir, "alpha", "beta", "gamma")
        (plan_dir / ArtifactNames.STATE).write_text(_state_md(history=history))
        harness = make_harness(_failing_execute_script(), roots=roots)
        harness.run()

        doc = StateDoc.from_markdown((plan_dir / ArtifactNames.STATE).read_text())
        assert doc.transition_history == list(history)
        assert not any(
            "EXECUTE" in line and "REFLECT" in line for line in doc.transition_history
        )

    def test_the_driver_does_not_create_the_plan_directory(
        self, make_harness, tmp_path
    ) -> None:
        """A directory the driver made is one whose emptiness it would misread."""
        absent = tmp_path / "plans" / "never-created"
        harness = make_harness(
            _traverse_script(),
            roots={
                ContextKeys.PLAN_DIR: str(absent),
                ContextKeys.WORKSPACE_ROOT: str(tmp_path),
            },
        )
        harness.run()

        assert not absent.exists()

    def test_a_write_that_cannot_complete_leaves_the_old_bytes(
        self, make_harness, roots, plan_dir, monkeypatch
    ) -> None:
        """Atomicity at the seam: old-or-new, never a blend, never a temp file."""
        original = _state_md(state=HarnessStates.PLAN, iteration=7)
        (plan_dir / ArtifactNames.STATE).write_text(original)
        monkeypatch.setattr(
            storage_module.os,
            "replace",
            lambda *a, **k: (_ for _ in ()).throw(OSError("no space left on device")),
        )
        harness = make_harness(_traverse_script(), roots=roots)

        assert harness.agent._sync_state_doc(HarnessStates.EXPLORE, dict(roots)) is None
        assert (plan_dir / ArtifactNames.STATE).read_text() == original
        assert [p.name for p in plan_dir.iterdir() if p.name.startswith(".state")] == []

    def test_a_failed_write_does_not_lose_the_handler_delta(
        self, make_harness, roots, plan_dir
    ) -> None:
        """Protocol memory is a record of the run, not a precondition for it."""
        (plan_dir / ArtifactNames.STATE).mkdir()  # unwritable as a file
        harness = make_harness(_traverse_script(), roots=roots)
        result = harness.run()

        assert harness.worker.count_for(HarnessStates.EXPLORE) >= 1
        assert result.final_context[ContextKeys.FINDINGS_COUNT] == 0

    def test_the_driver_may_not_write_an_artifact_it_does_not_own(
        self, make_harness, roots, plan_dir
    ) -> None:
        """Ownership is composed in, not restated (rules.py D-048)."""
        directory = HarnessAgent._plan_directory(dict(roots))
        assert directory is not None
        assert directory.role == Role.ORCHESTRATOR

        directory.write_text(ArtifactNames.STATE, "# Current State: EXPLORE\n")
        for unowned in (
            "findings/whatever.md",
            "checkpoints/cp-000-iter1.md",
            "summary.md",
        ):
            with pytest.raises(HarnessOwnershipError):
                directory.write_text(unowned, "not mine")

    def test_the_gate_never_reads_a_document_written_in_its_own_call(
        self, make_harness, roots, plan_dir, monkeypatch
    ) -> None:
        """D-038's ordering rule, stated as a mechanism.

        A sync inside ``_dispatch_if_needed`` would make ``wrong-state`` and
        ``no-plan`` structurally unreachable: the driver would always have just
        written the answer it then checks.
        """
        events: list[str] = []
        for attribute, marker in (
            ("_sync_state_doc", "sync"),
            ("_pre_step_gate", "gate"),
            ("_dispatch_if_needed", "dispatch-start"),
        ):
            real = getattr(HarnessAgent, attribute)

            def wrapper(self, *args, _real=real, _marker=marker, **kwargs):
                events.append(_marker)
                return _real(self, *args, **kwargs)

            monkeypatch.setattr(HarnessAgent, attribute, wrapper)
        _seed_findings(plan_dir, "alpha", "beta", "gamma")
        harness = make_harness(_traverse_script(), roots=roots)
        harness.run()

        # The rule, stated exactly: between the dispatch call opening and the
        # gate inside it, NOTHING writes state.md.  A sync there would make
        # `wrong-state` structurally unreachable -- the driver would always
        # have just written the answer it then checks.
        assert "gate" in events
        for index, event in enumerate(events):
            if event != "gate":
                continue
            opened = max(
                position
                for position, earlier in enumerate(events[:index])
                if earlier == "dispatch-start"
            )
            assert "sync" not in events[opened:index], events[opened : index + 1]


class TestResumeFromStateMd:
    """D-038: an interrupted run picks up where its ``state.md`` left off."""

    def test_counters_and_state_are_resumed(
        self, make_harness, roots, plan_dir
    ) -> None:
        _seed_findings(plan_dir, "alpha", "beta", "gamma")
        (plan_dir / ArtifactNames.STATE).write_text(
            _state_md(
                state=HarnessStates.EXECUTE, iteration=2, step="3 of 5", attempts=1
            )
        )
        harness = make_harness(_traverse_script(total_steps=5), roots=roots)
        harness.run()

        first = harness.worker.requests[0]
        assert first.state == HarnessStates.EXECUTE
        assert first.role == ROLE_BY_STATE[HarnessStates.EXECUTE]
        assert (first.iteration, first.step_number, first.fix_attempts) == (2, 3, 1)

    def test_total_steps_is_recovered_from_plan_md(
        self, make_harness, roots, plan_dir
    ) -> None:
        """``plan.md`` is read, not guessed: the cursor needs its own bound."""
        _seed_findings(plan_dir, "alpha", "beta", "gamma")
        (plan_dir / ArtifactNames.STATE).write_text(
            _state_md(state=HarnessStates.EXECUTE, step="1 of 1")
        )
        (plan_dir / ArtifactNames.PLAN).write_text(_PLAN_MD)
        harness = make_harness(_traverse_script(), roots=roots)
        harness.run()

        assert harness.worker.requests[0].total_steps == 2

    def test_a_run_with_no_state_md_starts_fresh_in_explore(
        self, make_harness, roots
    ) -> None:
        harness = make_harness(_traverse_script(), roots=roots)
        harness.run()

        assert harness.worker.requests[0].state == HarnessStates.EXPLORE
        assert harness.agent._initial_state == HarnessStates.INITIAL

    def test_an_unparseable_state_md_starts_fresh_rather_than_guessing(
        self, make_harness, roots, plan_dir
    ) -> None:
        (plan_dir / ArtifactNames.STATE).write_text("# not a state document\n")
        harness = make_harness(_traverse_script(), roots=roots)
        harness.run()

        assert harness.worker.requests[0].state == HarnessStates.EXPLORE

    def test_a_resumed_run_dispatches_the_resumed_states_role(
        self, make_harness, roots, plan_dir
    ) -> None:
        """The START handler names the RESUMED state, not the EXPLORE literal.

        Naming ``HarnessStates.INITIAL`` there would dispatch an EXPLORER while
        the FSM sits in REFLECT.
        """
        (plan_dir / ArtifactNames.STATE).write_text(
            _state_md(state=HarnessStates.REFLECT)
        )
        harness = make_harness(_traverse_script(), roots=roots)
        harness.run()

        assert harness.worker.requests[0].state == HarnessStates.REFLECT
        assert harness.worker.count_for(HarnessStates.EXPLORE) == 0

    def test_a_closed_plan_resumes_its_counters_but_not_its_state(
        self, make_harness, roots, plan_dir
    ) -> None:
        """CLOSE is terminal, so it orphans every other state as an initial one.

        ``FSMDefinition`` rejects the whole definition in that case, so this is
        a structural limit rather than a policy call -- and a run that started
        in CLOSE could do nothing anyway.
        """
        (plan_dir / ArtifactNames.STATE).write_text(
            _state_md(state=HarnessStates.CLOSE, iteration=4)
        )
        harness = make_harness(_traverse_script(), roots=roots)
        harness.run()

        assert harness.agent._initial_state == HarnessStates.INITIAL
        assert harness.worker.requests[0].state == HarnessStates.EXPLORE
        assert harness.worker.requests[0].iteration == 4


class TestPresentationContracts:
    """Every emitted block carries its floor fields, from disk not testimony."""

    def test_the_leash_block_carries_all_five_floor_fields(
        self, make_harness, roots, plan_dir
    ) -> None:
        _seed_plan_directory(plan_dir)
        harness = make_harness(
            _failing_execute_script(),
            approvals=ApprovalRecorder({APPROVAL_LEASH: False}),
            roots=roots,
        )
        harness.run()

        blocks = [
            p for p in harness.agent.presentations if p.name == "PC-EXECUTE-LEASH"
        ]
        assert blocks, _names(harness.agent)
        for block in blocks:
            assert block.missing_floor == ()
            assert "cp-000-iter1.md" in block.fields["checkpoints"]
            assert (
                "attempt" in block.fields["attempts"].lower()
                or "FAIL" in block.fields["attempts"]
            )

    def test_every_emitted_block_names_a_real_contract(
        self, make_harness, roots, plan_dir
    ) -> None:
        _seed_plan_directory(plan_dir)
        harness = make_harness(_traverse_script(), roots=roots)
        harness.run()

        assert _names(harness.agent)
        for presentation in harness.agent.presentations:
            assert presentation.name in PRESENTATION_CONTRACTS
            assert presentation.block.startswith(f"### {presentation.name}")

    def test_a_readable_plan_directory_fills_every_floor_field(
        self, make_harness, roots, plan_dir
    ) -> None:
        _seed_plan_directory(plan_dir)
        harness = make_harness(_traverse_script(), roots=roots)
        harness.run()

        starved = {
            p.name: p.missing_floor
            for p in harness.agent.presentations
            if p.missing_floor
        }
        assert starved == {}, starved

    def test_an_empty_floor_section_shows_up_as_a_missing_floor_field(
        self, make_harness, roots, plan_dir
    ) -> None:
        """The honest failure: report the gap, do not invent the section.

        Since plan-2026-07-23 both a NON-EMPTY-but-INVALID plan.md AND a
        PARTIALLY-filled valid plan (some placeholder sections) consume the PLAN
        redispatch budget (ALIGNED ``_plan_has_content``, D-001: valid PlanDoc
        AND ALL sections non-placeholder) and NEVER reach PC-PLAN via the run --
        the gate now stops exactly the shape this test used to walk through.  So
        the emission's honest-gap behaviour is exercised at the SEAM directly:
        ``_emit_plan`` over a plan whose floor sections are EMPTY must report
        each one missing rather than invent it.  (The complementary
        placeholder-cannot-satisfy-a-floor case is covered by the sibling test
        just below.)
        """
        # Valid PlanDoc, Goal filled, every PC-PLAN floor section
        # (steps/success/verification/failure/assumptions) left EMPTY.
        plan = _hollow_plan_doc()
        plan.sections[0].body = "Wire the seam the run exercises."
        (plan_dir / ArtifactNames.PLAN).write_text(plan.to_markdown())
        agent = make_harness(_traverse_script(), roots=roots).agent
        agent._emit_plan({ContextKeys.PLAN_DIR: str(plan_dir)})

        plan_blocks = [p for p in agent.presentations if p.name == "PC-PLAN"]
        assert plan_blocks
        assert set(plan_blocks[0].missing_floor) == set(
            PRESENTATION_CONTRACTS["PC-PLAN"].floor
        )

    def test_a_placeholder_cannot_satisfy_a_floor_field(
        self, make_harness, roots
    ) -> None:
        agent = make_harness(_traverse_script(), roots=roots).agent
        emitted = agent._emit_contract(
            "PC-EXECUTE-STEP",
            {
                field: "   "
                for field in PRESENTATION_CONTRACTS["PC-EXECUTE-STEP"].required
            },
        )

        assert set(emitted.missing_floor) == set(
            PRESENTATION_CONTRACTS["PC-EXECUTE-STEP"].floor
        )
        assert harness_module._CONTRACT_ABSENT in emitted.block

    def test_the_step_block_reads_the_changelog_not_the_worker(
        self, make_harness, roots, plan_dir
    ) -> None:
        """Evidence over testimony (D-032's rule, applied to a report)."""
        _seed_plan_directory(plan_dir)
        harness = make_harness(
            {
                **_traverse_script(),
                HarnessStates.EXECUTE: {
                    "answer": "I changed src/invented.py and committed deadbee",
                    "ctx": {},
                },
            },
            roots=roots,
        )
        harness.run()

        step_blocks = [
            p for p in harness.agent.presentations if p.name == "PC-EXECUTE-STEP"
        ]
        assert step_blocks
        assert step_blocks[0].fields["files"] == "src/fsm_llm_harness/harness.py"
        assert step_blocks[0].fields["commit"] == "abc1234"
        assert "invented.py" not in step_blocks[0].fields["files"]

    def test_the_plan_block_maps_all_eleven_sections(self) -> None:
        """The PC-PLAN field names are DERIVED from the 11 plan sections."""
        derived = {harness_module._contract_field(s) for s in PlanSchema.SECTIONS}
        assert derived <= set(PRESENTATION_CONTRACTS["PC-PLAN"].required)
        assert (
            harness_module._contract_field("Pre-Mortem & Falsification Signals")
            == "pre-mortem"
        )

    def test_contract_blocks_stay_out_of_the_fsm_context(
        self, make_harness, roots, plan_dir
    ) -> None:
        """D-020: context is rendered into BOTH prompts on EVERY turn."""
        _seed_plan_directory(plan_dir)
        harness = make_harness(_traverse_script(), roots=roots)
        result = harness.run()

        rendered = str(result.final_context)
        assert "PC-EXECUTE-STEP" not in rendered
        assert "PC-PLAN" not in rendered

    def test_presentations_are_not_reachable_from_inside_a_dispatch(
        self, make_harness, roots
    ) -> None:
        seen: list[type[BaseException]] = []

        def probe(request: RoleRequest) -> AgentResult:
            for name in ("presentations", "reverts", "audit_issues"):
                try:
                    getattr(agent, name)
                except HarnessReentrancyError as exc:
                    seen.append(type(exc))
            return AgentResult(answer="done", success=True, final_context={})

        harness = make_harness(_traverse_script(), roots=roots, worker=probe)
        agent = harness.agent
        harness.run()

        assert seen and set(seen) == {HarnessReentrancyError}


class TestLeashRevertDirective:
    """D-009 scope, D-039 execution: computed always, executed never by default."""

    def test_the_directive_spares_the_plan_directory(
        self, make_harness, tmp_path
    ) -> None:
        """``plans/`` is gitignored, so ``git clean -fd`` would delete it."""
        workspace = tmp_path / "repo"
        plan_dir = workspace / "plans" / "plan-2026-07-22T000000-abcdef12"
        plan_dir.mkdir(parents=True)
        _seed_findings(plan_dir, "alpha", "beta", "gamma")
        harness = make_harness(
            _failing_execute_script(),
            approvals=ApprovalRecorder({APPROVAL_LEASH: False}),
            roots={
                ContextKeys.PLAN_DIR: str(plan_dir),
                ContextKeys.WORKSPACE_ROOT: str(workspace),
            },
        )
        harness.run()

        assert harness.agent.reverts
        for directive in harness.agent.reverts:
            assert directive.root == str(workspace)
            assert directive.exclude == ("plans/plan-2026-07-22T000000-abcdef12",)
            assert str(plan_dir) not in " ".join(directive.commands)

    def test_a_plan_directory_outside_the_workspace_needs_no_exclusion(
        self, make_harness, roots, plan_dir
    ) -> None:
        _seed_findings(plan_dir, "alpha", "beta", "gamma")
        harness = make_harness(
            _failing_execute_script(),
            approvals=ApprovalRecorder({APPROVAL_LEASH: False}),
            roots=roots,
        )
        harness.run()

        assert harness.agent.reverts
        assert all(d.exclude == () for d in harness.agent.reverts)

    def test_nothing_is_executed_without_a_revert_callback(
        self, make_harness, roots, plan_dir
    ) -> None:
        _seed_findings(plan_dir, "alpha", "beta", "gamma")
        harness = make_harness(
            _failing_execute_script(),
            approvals=ApprovalRecorder({APPROVAL_LEASH: False}),
            roots=roots,
        )
        harness.run()

        assert harness.agent.reverts
        assert harness.approvals.count(APPROVAL_REVERT) == 0

    def test_a_supplied_callback_is_still_gated_by_the_human(
        self, make_harness, roots, plan_dir
    ) -> None:
        _seed_findings(plan_dir, "alpha", "beta", "gamma")
        executed: list[Any] = []
        harness = make_harness(
            _failing_execute_script(),
            approvals=ApprovalRecorder({APPROVAL_LEASH: False, APPROVAL_REVERT: False}),
            roots=roots,
            revert_callback=lambda directive: bool(executed.append(directive)),
        )
        harness.run()

        assert harness.approvals.count(APPROVAL_REVERT) >= 1
        assert executed == []

    def test_an_approved_callback_receives_the_scoped_directive(
        self, make_harness, roots, plan_dir
    ) -> None:
        _seed_findings(plan_dir, "alpha", "beta", "gamma")
        executed: list[Any] = []

        def revert(directive: Any) -> bool:
            executed.append(directive)
            return True

        harness = make_harness(
            _failing_execute_script(),
            approvals=ApprovalRecorder({APPROVAL_LEASH: False}),
            roots=roots,
            revert_callback=revert,
        )
        harness.run()

        assert executed
        assert executed[0].commands == harness_module._REVERT_COMMANDS

    def test_a_raising_callback_does_not_crash_the_turn(
        self, make_harness, roots, plan_dir
    ) -> None:
        _seed_findings(plan_dir, "alpha", "beta", "gamma")
        harness = make_harness(
            _failing_execute_script(),
            approvals=ApprovalRecorder({APPROVAL_LEASH: False}),
            roots=roots,
            revert_callback=lambda directive: (_ for _ in ()).throw(
                RuntimeError("git is not installed")
            ),
        )
        result = harness.run()

        assert result.final_context[ContextKeys.LAST_GATE_SLUG] == GateSlug.LEASH_CAP

    def test_a_run_with_no_workspace_root_computes_no_directive(
        self, make_harness, plan_dir
    ) -> None:
        _seed_findings(plan_dir, "alpha", "beta", "gamma")
        harness = make_harness(
            _failing_execute_script(),
            approvals=ApprovalRecorder({APPROVAL_LEASH: False}),
            roots={ContextKeys.PLAN_DIR: str(plan_dir)},
        )
        harness.run()

        assert harness.agent.reverts == ()


class TestCloseAudit:
    """``audit()``'s second call site: advisory, never a blocker."""

    def test_close_records_the_audit(self, make_harness, roots, plan_dir) -> None:
        _seed_plan_directory(plan_dir)
        harness = make_harness(_traverse_script(), roots=roots)
        harness.run()

        assert harness.agent.audit_issues is not None
        assert all(isinstance(issue, Issue) for issue in harness.agent.audit_issues)

    def test_audit_findings_do_not_block_the_close(
        self, make_harness, roots, plan_dir
    ) -> None:
        """The CLOSE gate is a HUMAN decision (invariant I6), not an audit."""
        _seed_findings(plan_dir, "alpha", "beta", "gamma")
        harness = make_harness(_traverse_script(), roots=roots)
        result = harness.run()

        assert harness.worker.count_for(HarnessStates.CLOSE) == 1
        assert result.final_context[ContextKeys.CLOSE_CONFIRMED] is True
        assert harness.agent.audit_issues
        assert any(issue.is_error for issue in harness.agent.audit_issues)

    def test_a_run_that_never_closes_records_no_audit(
        self, make_harness, roots, plan_dir
    ) -> None:
        _seed_findings(plan_dir, "alpha", "beta", "gamma")
        harness = make_harness(
            _failing_execute_script(),
            approvals=ApprovalRecorder({APPROVAL_LEASH: False}),
            roots=roots,
        )
        harness.run()

        assert harness.agent.audit_issues is None


class TestTheC1FailOpenIsStillShut:
    """The seam must not re-open what step 20 closed."""

    def test_a_fabricating_llm_still_cannot_open_a_gate_with_artifacts_wired(
        self, make_harness, roots, plan_dir
    ) -> None:
        _seed_plan_directory(plan_dir)
        harness = make_harness(
            _all_failing_script(),
            approvals=ApprovalRecorder(default=False),
            roots=roots,
            extraction_data=FABRICATED_DRIVER_OWNED,
        )
        result = harness.run()

        assert result.final_context[ContextKeys.PLAN_APPROVED] is False
        assert result.final_context[ContextKeys.CLOSE_CONFIRMED] is False
        assert harness.worker.count_for(HarnessStates.CLOSE) == 0
        assert harness.agent.audit_issues is None

    def test_state_md_records_the_drivers_counters_not_the_models(
        self, make_harness, roots, plan_dir
    ) -> None:
        """The one artifact the driver writes is written from driver-owned state."""
        harness = make_harness(
            _all_failing_script(),
            approvals=ApprovalRecorder(default=False),
            roots=roots,
            extraction_data=FABRICATED_DRIVER_OWNED,
        )
        harness.run()

        doc = StateDoc.from_markdown((plan_dir / ArtifactNames.STATE).read_text())
        assert doc.iteration == 0
        assert doc.current_step.startswith("0 of 1")
