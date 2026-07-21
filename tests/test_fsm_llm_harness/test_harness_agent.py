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
from typing import Any

import pytest

from fsm_llm.handlers import HandlerTiming
from fsm_llm_agents.definitions import AgentResult
from fsm_llm_agents.exceptions import AgentError
from fsm_llm_harness import harness as harness_module
from fsm_llm_harness.constants import (
    DRIVER_OWNED_SEEDS,
    DRIVER_OWNED_UNSET,
    ContextKeys,
    GateSlug,
    HandlerNames,
    HarnessStates,
    Role,
)
from fsm_llm_harness.exceptions import HarnessError, HarnessReentrancyError
from fsm_llm_harness.harness import HarnessAgent, RoleRequest
from tests.conftest import MockLLM2Interface
from tests.test_fsm_llm_harness.conftest import (
    APPROVAL_CLOSE,
    APPROVAL_GATES,
    APPROVAL_LEASH,
    APPROVAL_PLAN,
    FABRICATED_DRIVER_OWNED,
    ApprovalRecorder,
    RecordingWorker,
)

Call = tuple[str, int, int, int]


# ---------------------------------------------------------------------------
# Script builders
# ---------------------------------------------------------------------------


def _traverse_script(*, total_steps: int = 1, findings: int = 3) -> dict[str, Any]:
    """A script that walks EXPLORE -> PLAN -> EXECUTE -> REFLECT -> CLOSE."""
    return {
        HarnessStates.EXPLORE: {"ctx": {ContextKeys.FINDINGS_COUNT: findings}},
        HarnessStates.PLAN: {"ctx": {ContextKeys.TOTAL_STEPS: total_steps}},
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
        assert len(set(APPROVAL_GATES)) == 3

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

    def test_the_three_gates_are_distinguishable_by_tool_name(
        self, make_harness
    ) -> None:
        """One callback serves all three gates and can tell them apart."""
        traverse = make_harness(_traverse_script())
        traverse.run()
        leash = make_harness(
            _failing_execute_script(),
            approvals=ApprovalRecorder({APPROVAL_LEASH: False}),
        )
        leash.run()

        observed = set(traverse.approvals.names) | set(leash.approvals.names)
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
        assert len(results) == 1
        assert results[0]["role"] == Role.EXPLORER
        assert results[0]["success"] is False
        assert "boom from explorer" in results[0]["answer"]
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
    """The plan-dir / workspace roots ride in context untouched by the driver."""

    def test_roots_survive_a_full_run(self, make_harness, plan_dir, workspace) -> None:
        harness = make_harness(_traverse_script())
        result = harness.run(
            initial_context={
                ContextKeys.PLAN_DIR: str(plan_dir),
                ContextKeys.WORKSPACE_ROOT: str(workspace),
            }
        )

        assert result.final_context[ContextKeys.PLAN_DIR] == str(plan_dir)
        assert result.final_context[ContextKeys.WORKSPACE_ROOT] == str(workspace)

    def test_roots_are_visible_to_every_worker(
        self, make_harness, plan_dir, workspace
    ) -> None:
        harness = make_harness(_traverse_script())
        harness.run(
            initial_context={
                ContextKeys.PLAN_DIR: str(plan_dir),
                ContextKeys.WORKSPACE_ROOT: str(workspace),
            }
        )

        assert harness.worker.requests
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

        assert harness.worker.states == [HarnessStates.EXPLORE]
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
        """A driver whose run halted in EXPLORE, so every key is at its seed."""
        harness = make_harness(
            _all_failing_script(), approvals=ApprovalRecorder(default=False)
        )
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
    def test_the_guard_deletes_an_unset_key(self, halted, key: str) -> None:
        """``None`` in a handler delta means DELETE; absence is the default."""
        delta = halted.agent._reassert_driver_owned({key: FABRICATED_DRIVER_OWNED[key]})
        assert delta[key] is None

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
