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
===============================  ==========================================

Everything runs against ``MockLLM2Interface``: no network, no sleeps, no
ollama.  The mock extracts nothing (every field comes back ``None``), so the
protocol advances only through worker replies and driver bookkeeping -- which
is precisely the surface under test.
"""

from __future__ import annotations

from typing import Any

import pytest

from fsm_llm_agents.definitions import AgentResult
from fsm_llm_agents.exceptions import AgentError
from fsm_llm_harness.constants import ContextKeys, GateSlug, HarnessStates, Role
from fsm_llm_harness.exceptions import HarnessReentrancyError
from fsm_llm_harness.harness import HarnessAgent, RoleRequest
from tests.conftest import MockLLM2Interface
from tests.test_fsm_llm_harness.conftest import (
    APPROVAL_CLOSE,
    APPROVAL_GATES,
    APPROVAL_LEASH,
    APPROVAL_PLAN,
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
    """The callback is consulted at every gate, and the default denies."""

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
        assert result.final_context.get(ContextKeys.PLAN_APPROVED) is not True
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
        assert result.final_context.get(ContextKeys.CLOSE_CONFIRMED) is not True

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
        assert result.final_context.get(ContextKeys.PLAN_APPROVED) is not True

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

        assert result.final_context.get(key) is not True
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
        assert result.final_context.get(ContextKeys.ALL_CRITERIA_PASS) is not True

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
