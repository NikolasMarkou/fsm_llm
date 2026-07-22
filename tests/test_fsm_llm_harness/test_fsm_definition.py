"""Falsifying tests for ``fsm_llm_harness.fsm_definition`` and ``rules``.

Every gate assertion below is driven through the **real**
``fsm_llm.transition_evaluator.TransitionEvaluator``.  A bespoke evaluator
would only prove that the test author and the FSM author agree; the point of
these tests is that the engine that actually decides transitions at runtime
agrees.

Success criteria covered: 1 (FSM validity), 2 (graph shape), 3 (gate
mechanicity + allowlist conformance).
"""

from __future__ import annotations

from itertools import pairwise
from typing import Any

import pytest

from fsm_llm.constants import ALLOWED_JSONLOGIC_OPERATIONS
from fsm_llm.definitions import (
    FSMContext,
    FSMDefinition,
    TransitionEvaluationResult,
    _walk_logic_operators,
)
from fsm_llm.transition_evaluator import TransitionEvaluator
from fsm_llm.validator import FSMValidator
from fsm_llm_harness import build_harness_fsm
from fsm_llm_harness.constants import ArtifactNames, ContextKeys, HarnessStates, Role
from fsm_llm_harness.rules import OWNERSHIP, ROLE_BY_STATE, RULES

BLOCKED = TransitionEvaluationResult.BLOCKED
DETERMINISTIC = TransitionEvaluationResult.DETERMINISTIC
AMBIGUOUS = TransitionEvaluationResult.AMBIGUOUS

#: The complete, authoritative edge set (decisions.md D-013).  Asserted as a
#: literal so an added or removed edge fails loudly instead of shifting a count.
EXPECTED_EDGES = frozenset(
    {
        (HarnessStates.EXPLORE, HarnessStates.PLAN),
        (HarnessStates.PLAN, HarnessStates.EXECUTE),
        (HarnessStates.PLAN, HarnessStates.EXPLORE),
        (HarnessStates.EXECUTE, HarnessStates.REFLECT),
        (HarnessStates.REFLECT, HarnessStates.CLOSE),
        (HarnessStates.REFLECT, HarnessStates.EXECUTE),
        (HarnessStates.REFLECT, HarnessStates.PIVOT),
        (HarnessStates.REFLECT, HarnessStates.EXPLORE),
        (HarnessStates.PIVOT, HarnessStates.PLAN),
    }
)


def _evaluate(
    fsm: FSMDefinition,
    evaluator: TransitionEvaluator,
    state: str,
    context: dict[str, Any],
) -> tuple[TransitionEvaluationResult, str | None]:
    """Evaluate *state*'s outbound edges against *context*, through core.

    Returns:
        ``(result_type, deterministic_target)``; the target is ``None`` unless
        the result is DETERMINISTIC.
    """
    evaluation = evaluator.evaluate_transitions(
        fsm.states[state], FSMContext(data=dict(context))
    )
    return evaluation.result_type, evaluation.deterministic_transition


def _all_conditions(fsm: FSMDefinition):
    """Yield every ``TransitionCondition`` in the FSM."""
    for state in fsm.states.values():
        for transition in state.transitions:
            yield from transition.conditions or []


# ---------------------------------------------------------------------------
# Criterion 1 -- validity
# ---------------------------------------------------------------------------


class TestFSMValidity:
    """The built FSM is schema-valid and validator-clean."""

    def test_fsm_validates(self, harness_fsm_dict: dict[str, Any]) -> None:
        """Zero pydantic errors, zero validator errors AND zero warnings.

        Warnings are asserted too: step 2 shipped a warning-free FSM, and a
        regression to "valid but warns" is a real quality loss that an
        errors-only assertion would wave through.
        """
        FSMDefinition(**harness_fsm_dict)

        result = FSMValidator(harness_fsm_dict).validate()

        assert result.errors == [], f"validator errors: {result.errors}"
        assert result.warnings == [], f"validator warnings: {result.warnings}"
        assert result.is_valid is True

    def test_custom_thresholds_still_validate(self) -> None:
        """Non-default gate thresholds do not break schema or validator."""
        fsm_dict = build_harness_fsm(
            "custom thresholds",
            findings_threshold=1,
            max_fix_attempts=5,
            iteration_hard_cap=2,
        )
        FSMDefinition(**fsm_dict)
        result = FSMValidator(fsm_dict).validate()
        assert result.errors == []
        assert result.warnings == []


# ---------------------------------------------------------------------------
# Criterion 2 -- graph shape
# ---------------------------------------------------------------------------


class TestGraphShape:
    """The state and edge sets are exactly what the protocol specifies."""

    def test_state_graph_matches_spec(self, harness_fsm: FSMDefinition) -> None:
        """Exactly the 6 states and exactly the 9 documented edges (D-013)."""
        assert set(harness_fsm.states) == set(HarnessStates.ALL)
        assert len(HarnessStates.ALL) == 6

        edges = frozenset(
            (state_id, transition.target_state)
            for state_id, state in harness_fsm.states.items()
            for transition in state.transitions
        )

        assert edges == EXPECTED_EDGES
        assert len(edges) == 9

    def test_cyclic_edges_exist(self, harness_fsm: FSMDefinition) -> None:
        """The three loop-back edges are present.

        These are why the protocol cannot be an ``AgentGraph`` or a workflow
        ``DependencyResolver`` -- both reject cycles structurally.
        """
        edges = frozenset(
            (state_id, transition.target_state)
            for state_id, state in harness_fsm.states.items()
            for transition in state.transitions
        )
        assert (HarnessStates.PIVOT, HarnessStates.PLAN) in edges
        assert (HarnessStates.PLAN, HarnessStates.EXPLORE) in edges
        assert (HarnessStates.REFLECT, HarnessStates.EXECUTE) in edges

    def test_no_self_loops(self, harness_fsm: FSMDefinition) -> None:
        """No state transitions to itself (decisions.md D-012).

        A self-loop would be a second always-passing edge out of every state,
        racing the real gated edge into an AMBIGUOUS result.
        """
        self_loops = [
            state_id
            for state_id, state in harness_fsm.states.items()
            for transition in state.transitions
            if transition.target_state == state_id
        ]
        assert self_loops == []

    def test_close_is_the_only_terminal_state(self, harness_fsm: FSMDefinition) -> None:
        """CLOSE has no outbound edges; every other state has at least one."""
        terminal = [
            state_id
            for state_id, state in harness_fsm.states.items()
            if not state.transitions
        ]
        assert terminal == [HarnessStates.CLOSE]
        assert HarnessStates.TERMINAL == HarnessStates.CLOSE

    def test_every_transition_target_exists(self, harness_fsm: FSMDefinition) -> None:
        """No edge points at a state the FSM does not define."""
        for state_id, state in harness_fsm.states.items():
            for transition in state.transitions:
                assert transition.target_state in harness_fsm.states, (
                    f"{state_id} -> {transition.target_state} is dangling"
                )


# ---------------------------------------------------------------------------
# Criterion 3 -- gate mechanicity, one class per hard gate
# ---------------------------------------------------------------------------


class TestFindingsGate:
    """EXPLORE -> PLAN: ``findings_count >= 3``."""

    @pytest.mark.parametrize(
        ("label", "context"),
        [
            ("missing key entirely", {}),
            ("zero findings", {ContextKeys.FINDINGS_COUNT: 0}),
            ("one below threshold", {ContextKeys.FINDINGS_COUNT: 2}),
            ("fractionally below", {ContextKeys.FINDINGS_COUNT: 2.9}),
            ("null value", {ContextKeys.FINDINGS_COUNT: None}),
            ("non-numeric string", {ContextKeys.FINDINGS_COUNT: "four"}),
            ("list value", {ContextKeys.FINDINGS_COUNT: []}),
            ("dict value", {ContextKeys.FINDINGS_COUNT: {"n": 3}}),
        ],
    )
    def test_gate_blocked(
        self,
        harness_fsm: FSMDefinition,
        evaluator: TransitionEvaluator,
        label: str,
        context: dict[str, Any],
    ) -> None:
        """Below threshold, missing, or unusable -> BLOCKED (invariant I8)."""
        result, target = _evaluate(
            harness_fsm, evaluator, HarnessStates.EXPLORE, context
        )
        assert result is BLOCKED, f"{label} should block, got {result} -> {target}"

    @pytest.mark.parametrize("count", [3, 4, 99])
    def test_gate_deterministic_at_or_above_threshold(
        self,
        harness_fsm: FSMDefinition,
        evaluator: TransitionEvaluator,
        count: int,
    ) -> None:
        """At or above the threshold the edge fires deterministically."""
        result, target = _evaluate(
            harness_fsm,
            evaluator,
            HarnessStates.EXPLORE,
            {ContextKeys.FINDINGS_COUNT: count},
        )
        assert result is DETERMINISTIC
        assert target == HarnessStates.PLAN

    def test_threshold_is_configurable(self, evaluator: TransitionEvaluator) -> None:
        """A raised threshold moves the gate, proving it is not hardcoded."""
        fsm = FSMDefinition(**build_harness_fsm("g", findings_threshold=7))
        blocked, _ = _evaluate(
            fsm, evaluator, HarnessStates.EXPLORE, {ContextKeys.FINDINGS_COUNT: 6}
        )
        opened, target = _evaluate(
            fsm, evaluator, HarnessStates.EXPLORE, {ContextKeys.FINDINGS_COUNT: 7}
        )
        assert blocked is BLOCKED
        assert opened is DETERMINISTIC
        assert target == HarnessStates.PLAN


class TestPlanApprovalGate:
    """PLAN -> EXECUTE: ``plan_approved == true AND iteration < 6``."""

    @pytest.mark.parametrize(
        ("label", "context"),
        [
            ("both keys missing", {}),
            ("approval only", {ContextKeys.PLAN_APPROVED: True}),
            ("iteration only", {ContextKeys.ITERATION: 0}),
            (
                "not approved",
                {ContextKeys.PLAN_APPROVED: False, ContextKeys.ITERATION: 0},
            ),
            (
                "approval is null",
                {ContextKeys.PLAN_APPROVED: None, ContextKeys.ITERATION: 0},
            ),
            (
                "approval is prose",
                {ContextKeys.PLAN_APPROVED: "maybe", ContextKeys.ITERATION: 0},
            ),
            (
                "iteration is null",
                {ContextKeys.PLAN_APPROVED: True, ContextKeys.ITERATION: None},
            ),
            (
                "at the iteration cap",
                {ContextKeys.PLAN_APPROVED: True, ContextKeys.ITERATION: 6},
            ),
            (
                "above the iteration cap",
                {ContextKeys.PLAN_APPROVED: True, ContextKeys.ITERATION: 7},
            ),
        ],
    )
    def test_gate_blocked(
        self,
        harness_fsm: FSMDefinition,
        evaluator: TransitionEvaluator,
        label: str,
        context: dict[str, Any],
    ) -> None:
        """Either conjunct unsatisfied, missing or unusable -> BLOCKED."""
        result, target = _evaluate(harness_fsm, evaluator, HarnessStates.PLAN, context)
        assert result is BLOCKED, f"{label} should block, got {result} -> {target}"

    @pytest.mark.parametrize("iteration", [0, 1, 5])
    def test_gate_deterministic_when_approved_below_cap(
        self,
        harness_fsm: FSMDefinition,
        evaluator: TransitionEvaluator,
        iteration: int,
    ) -> None:
        """Approved and under the cap fires EXECUTE deterministically."""
        result, target = _evaluate(
            harness_fsm,
            evaluator,
            HarnessStates.PLAN,
            {ContextKeys.PLAN_APPROVED: True, ContextKeys.ITERATION: iteration},
        )
        assert result is DETERMINISTIC
        assert target == HarnessStates.EXECUTE

    def test_needs_explore_edge_is_independent(
        self, harness_fsm: FSMDefinition, evaluator: TransitionEvaluator
    ) -> None:
        """An unapproved plan can still bounce back to EXPLORE."""
        result, target = _evaluate(
            harness_fsm,
            evaluator,
            HarnessStates.PLAN,
            {
                ContextKeys.PLAN_APPROVED: False,
                ContextKeys.ITERATION: 0,
                ContextKeys.NEEDS_EXPLORE: True,
            },
        )
        assert result is DETERMINISTIC
        assert target == HarnessStates.EXPLORE


class TestCloseGate:
    """REFLECT -> CLOSE: ``close_confirmed AND all_criteria_pass``."""

    @pytest.mark.parametrize(
        ("label", "context"),
        [
            ("both keys missing", {}),
            ("confirmation only", {ContextKeys.CLOSE_CONFIRMED: True}),
            ("criteria only", {ContextKeys.ALL_CRITERIA_PASS: True}),
            (
                "criteria pass but unconfirmed",
                {
                    ContextKeys.CLOSE_CONFIRMED: False,
                    ContextKeys.ALL_CRITERIA_PASS: True,
                },
            ),
            (
                "confirmed but criteria fail",
                {
                    ContextKeys.CLOSE_CONFIRMED: True,
                    ContextKeys.ALL_CRITERIA_PASS: False,
                },
            ),
            (
                "criteria value is null",
                {
                    ContextKeys.CLOSE_CONFIRMED: True,
                    ContextKeys.ALL_CRITERIA_PASS: None,
                },
            ),
            (
                "criteria value is prose",
                {
                    ContextKeys.CLOSE_CONFIRMED: True,
                    ContextKeys.ALL_CRITERIA_PASS: "mostly",
                },
            ),
        ],
    )
    def test_gate_blocked(
        self,
        harness_fsm: FSMDefinition,
        evaluator: TransitionEvaluator,
        label: str,
        context: dict[str, Any],
    ) -> None:
        """A run can never close on one half of the gate (invariant I1)."""
        result, target = _evaluate(
            harness_fsm, evaluator, HarnessStates.REFLECT, context
        )
        assert result is BLOCKED, f"{label} should block, got {result} -> {target}"

    def test_gate_deterministic_when_both_true(
        self, harness_fsm: FSMDefinition, evaluator: TransitionEvaluator
    ) -> None:
        """Both conjuncts true fires CLOSE deterministically."""
        result, target = _evaluate(
            harness_fsm,
            evaluator,
            HarnessStates.REFLECT,
            {
                ContextKeys.CLOSE_CONFIRMED: True,
                ContextKeys.ALL_CRITERIA_PASS: True,
            },
        )
        assert result is DETERMINISTIC
        assert target == HarnessStates.CLOSE


class TestLeashGate:
    """REFLECT -> EXECUTE: ``completion_fix AND fix_attempts < 2``."""

    @pytest.mark.parametrize(
        ("label", "context"),
        [
            ("attempts key missing", {ContextKeys.COMPLETION_FIX: True}),
            ("completion flag missing", {ContextKeys.FIX_ATTEMPTS: 0}),
            (
                "no completion fix wanted",
                {ContextKeys.COMPLETION_FIX: False, ContextKeys.FIX_ATTEMPTS: 0},
            ),
            (
                "completion flag is null",
                {ContextKeys.COMPLETION_FIX: None, ContextKeys.FIX_ATTEMPTS: 0},
            ),
            (
                "attempts value is null",
                {ContextKeys.COMPLETION_FIX: True, ContextKeys.FIX_ATTEMPTS: None},
            ),
            (
                "attempts at the cap",
                {ContextKeys.COMPLETION_FIX: True, ContextKeys.FIX_ATTEMPTS: 2},
            ),
            (
                "attempts above the cap",
                {ContextKeys.COMPLETION_FIX: True, ContextKeys.FIX_ATTEMPTS: 3},
            ),
        ],
    )
    def test_gate_blocked(
        self,
        harness_fsm: FSMDefinition,
        evaluator: TransitionEvaluator,
        label: str,
        context: dict[str, Any],
    ) -> None:
        """The leash is an FSM edge condition, not advice (invariant I1)."""
        result, target = _evaluate(
            harness_fsm, evaluator, HarnessStates.REFLECT, context
        )
        assert result is BLOCKED, f"{label} should block, got {result} -> {target}"

    @pytest.mark.parametrize("attempts", [0, 1])
    def test_gate_deterministic_below_cap(
        self,
        harness_fsm: FSMDefinition,
        evaluator: TransitionEvaluator,
        attempts: int,
    ) -> None:
        """Under the leash a completion fix routes straight back to EXECUTE."""
        result, target = _evaluate(
            harness_fsm,
            evaluator,
            HarnessStates.REFLECT,
            {
                ContextKeys.COMPLETION_FIX: True,
                ContextKeys.FIX_ATTEMPTS: attempts,
            },
        )
        assert result is DETERMINISTIC
        assert target == HarnessStates.EXECUTE

    def test_exhausted_leash_lets_pivot_win_on_its_own_merits(
        self, harness_fsm: FSMDefinition, evaluator: TransitionEvaluator
    ) -> None:
        """At the cap the fix edge shuts, so PIVOT wins by condition, not order."""
        result, target = _evaluate(
            harness_fsm,
            evaluator,
            HarnessStates.REFLECT,
            {
                ContextKeys.COMPLETION_FIX: True,
                ContextKeys.FIX_ATTEMPTS: 2,
                ContextKeys.NEEDS_PIVOT: True,
            },
        )
        assert result is DETERMINISTIC
        assert target == HarnessStates.PIVOT


class TestUngatedEdges:
    """The two single-condition edges also fail closed."""

    @pytest.mark.parametrize(
        ("state", "key", "target"),
        [
            (
                HarnessStates.EXECUTE,
                ContextKeys.EXECUTE_COMPLETE,
                HarnessStates.REFLECT,
            ),
            (HarnessStates.PIVOT, ContextKeys.PIVOT_RESOLVED, HarnessStates.PLAN),
        ],
    )
    @pytest.mark.parametrize("bad", [None, False, "yes", 0])
    def test_blocked_without_a_true_flag(
        self,
        harness_fsm: FSMDefinition,
        evaluator: TransitionEvaluator,
        state: str,
        key: str,
        target: str,
        bad: Any,
    ) -> None:
        result, _ = _evaluate(harness_fsm, evaluator, state, {key: bad})
        assert result is BLOCKED

    @pytest.mark.parametrize(
        ("state", "key", "target"),
        [
            (
                HarnessStates.EXECUTE,
                ContextKeys.EXECUTE_COMPLETE,
                HarnessStates.REFLECT,
            ),
            (HarnessStates.PIVOT, ContextKeys.PIVOT_RESOLVED, HarnessStates.PLAN),
        ],
    )
    def test_missing_key_blocks(
        self,
        harness_fsm: FSMDefinition,
        evaluator: TransitionEvaluator,
        state: str,
        key: str,
        target: str,
    ) -> None:
        result, _ = _evaluate(harness_fsm, evaluator, state, {})
        assert result is BLOCKED

    @pytest.mark.parametrize(
        ("state", "key", "target"),
        [
            (
                HarnessStates.EXECUTE,
                ContextKeys.EXECUTE_COMPLETE,
                HarnessStates.REFLECT,
            ),
            (HarnessStates.PIVOT, ContextKeys.PIVOT_RESOLVED, HarnessStates.PLAN),
        ],
    )
    def test_true_flag_fires(
        self,
        harness_fsm: FSMDefinition,
        evaluator: TransitionEvaluator,
        state: str,
        key: str,
        target: str,
    ) -> None:
        result, actual = _evaluate(harness_fsm, evaluator, state, {key: True})
        assert result is DETERMINISTIC
        assert actual == target

    def test_close_never_transitions(
        self, harness_fsm: FSMDefinition, evaluator: TransitionEvaluator
    ) -> None:
        """CLOSE is terminal even with a fully populated context."""
        result, _ = _evaluate(
            harness_fsm,
            evaluator,
            HarnessStates.CLOSE,
            {
                ContextKeys.HALT_REASON: "done",
                ContextKeys.ALL_CRITERIA_PASS: True,
                ContextKeys.CLOSE_CONFIRMED: True,
            },
        )
        assert result is BLOCKED


# DECISION plan-2026-07-21T125237-191b2eb2/D-025
# These tests assert that `findings_count = "3"` (and `3.0`, and `True`) DO
# open a hard gate. That is NOT a bug being enshrined and it is NOT a weakened
# invariant -- it is the measured boundary of `fsm_llm.expressions`' soft
# JsonLogic comparison, which this package may not change. Do NOT "fix" these
# to expect BLOCKED: they would fail, and papering over that by adding an
# `is_int` guard would mean re-implementing a second gate evaluator (a named
# Complexity-Budget breach). Invariant I8's real enforcement point is the
# DRIVER's `_WORKER_WRITABLE` type allowlist, pinned by
# test_harness_agent.py::TestFailClosed, which drops a type-wrong worker value
# before it can reach context at all. See decisions.md D-025.
class TestGateTypeGuardBoundary:
    """Where the JsonLogic layer's fail-closed guarantee actually stops.

    ``fsm_llm.expressions`` implements JsonLogic with *soft* comparison:
    ``less()`` coerces both operands with ``float()`` first, and
    ``soft_equals()`` compares a bool against the other operand's truthiness.
    So ``"3" >= 3``, ``3.0 >= 3`` and even ``True >= 3`` all evaluate true.

    That is core behaviour this package may not change, so the gate layer is
    **not** a type guard: it fails closed on an absent key, on ``None`` and on
    any value ``float()`` cannot digest, and nothing more.  The primary
    invariant-I8 enforcement is the driver's ``_WORKER_WRITABLE`` allowlist,
    which drops a type-wrong worker value before it can ever reach context --
    see ``test_harness_agent.py::TestFailClosed``.

    These tests exist so that a future change to core's comparison semantics
    surfaces here, deliberately, instead of silently widening or narrowing
    every harness gate.
    """

    @pytest.mark.parametrize("value", ["3", 3.0, True])
    def test_soft_comparison_admits_coercible_values(
        self,
        harness_fsm: FSMDefinition,
        evaluator: TransitionEvaluator,
        value: Any,
    ) -> None:
        result, target = _evaluate(
            harness_fsm,
            evaluator,
            HarnessStates.EXPLORE,
            {ContextKeys.FINDINGS_COUNT: value},
        )
        assert result is DETERMINISTIC
        assert target == HarnessStates.PLAN

    @pytest.mark.parametrize("value", [None, "four", [], {}])
    def test_soft_comparison_still_fails_closed_on_junk(
        self,
        harness_fsm: FSMDefinition,
        evaluator: TransitionEvaluator,
        value: Any,
    ) -> None:
        result, _ = _evaluate(
            harness_fsm,
            evaluator,
            HarnessStates.EXPLORE,
            {ContextKeys.FINDINGS_COUNT: value},
        )
        assert result is BLOCKED


class TestOperatorAllowlist:
    """Every gate stays inside the security-reviewed operator allowlist."""

    def test_gate_logic_operators_within_allowlist(
        self, harness_fsm: FSMDefinition
    ) -> None:
        """No condition may need an operator core refuses to evaluate.

        ``ALLOWED_JSONLOGIC_OPERATIONS`` is enforced at load time too, so a
        violation would raise -- but this test names the offending operator
        instead of leaving a pydantic error to be decoded, and it is the test
        that fires when a *future* gate reaches for something forbidden.
        """
        used: set[str] = set()
        for condition in _all_conditions(harness_fsm):
            used |= set(_walk_logic_operators(condition.logic or {}))

        assert used, "no gate carries any JsonLogic at all"
        assert used <= ALLOWED_JSONLOGIC_OPERATIONS, (
            f"operators outside the allowlist: "
            f"{sorted(used - ALLOWED_JSONLOGIC_OPERATIONS)}"
        )
        assert used == {"and", "==", "<", ">=", "var"}

    def test_every_condition_declares_its_required_keys(
        self, harness_fsm: FSMDefinition
    ) -> None:
        """``requires_context_keys`` is what makes I8 mechanical.

        The evaluator fails a condition whose required key is absent *before*
        evaluating the logic, so a condition that forgot to declare its keys
        would silently evaluate against missing data.
        """
        for condition in _all_conditions(harness_fsm):
            assert condition.requires_context_keys, (
                f"condition '{condition.description}' declares no required keys"
            )
            referenced = set(_iter_vars(condition.logic))
            assert referenced <= set(condition.requires_context_keys), (
                f"condition '{condition.description}' reads "
                f"{sorted(referenced - set(condition.requires_context_keys))} "
                "without requiring it"
            )


def _iter_vars(node: Any) -> list[str]:
    """Collect every ``{"var": name}`` reference inside a JsonLogic term."""
    found: list[str] = []
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "var" and isinstance(value, str):
                found.append(value)
            else:
                found.extend(_iter_vars(value))
    elif isinstance(node, list):
        for element in node:
            found.extend(_iter_vars(element))
    return found


# ---------------------------------------------------------------------------
# Priority spacing (the invariant step 3 discovered the hard way)
# ---------------------------------------------------------------------------


class TestPrioritySpacing:
    """Simultaneously passing edges must resolve DETERMINISTIC, never AMBIGUOUS.

    ``TransitionEvaluator`` derives base confidence from priority as
    ``max(0.1, 1.0 - priority / 1000)`` and only calls a multi-candidate race
    DETERMINISTIC when the leader clears the runner-up by
    ``ambiguity_threshold`` (0.1).  Priority slots closer than ~150 apart would
    therefore hand a HARD gate decision to the LLM classifier -- exactly what
    invariant I1 forbids.  These tests pin the spacing.
    """

    @pytest.mark.parametrize(
        ("label", "context", "expected"),
        [
            (
                "pivot outranks explore",
                {ContextKeys.NEEDS_PIVOT: True, ContextKeys.NEEDS_EXPLORE: True},
                HarnessStates.PIVOT,
            ),
            (
                "completion fix outranks pivot",
                {
                    ContextKeys.COMPLETION_FIX: True,
                    ContextKeys.FIX_ATTEMPTS: 0,
                    ContextKeys.NEEDS_PIVOT: True,
                },
                HarnessStates.EXECUTE,
            ),
            (
                "completion fix outranks explore",
                {
                    ContextKeys.COMPLETION_FIX: True,
                    ContextKeys.FIX_ATTEMPTS: 0,
                    ContextKeys.NEEDS_EXPLORE: True,
                },
                HarnessStates.EXECUTE,
            ),
            (
                "close outranks everything",
                {
                    ContextKeys.CLOSE_CONFIRMED: True,
                    ContextKeys.ALL_CRITERIA_PASS: True,
                    ContextKeys.COMPLETION_FIX: True,
                    ContextKeys.FIX_ATTEMPTS: 0,
                    ContextKeys.NEEDS_PIVOT: True,
                    ContextKeys.NEEDS_EXPLORE: True,
                },
                HarnessStates.CLOSE,
            ),
        ],
    )
    def test_reflect_races_resolve_deterministically(
        self,
        harness_fsm: FSMDefinition,
        evaluator: TransitionEvaluator,
        label: str,
        context: dict[str, Any],
        expected: str,
    ) -> None:
        result, target = _evaluate(
            harness_fsm, evaluator, HarnessStates.REFLECT, context
        )
        assert result is not AMBIGUOUS, (
            f"{label}: priority spacing collapsed -- a HARD gate decision "
            "would be handed to the LLM classifier"
        )
        assert result is DETERMINISTIC
        assert target == expected

    def test_plan_race_resolves_to_the_gated_edge(
        self, harness_fsm: FSMDefinition, evaluator: TransitionEvaluator
    ) -> None:
        """Approval (evidence) beats ``needs_explore`` (judgement)."""
        result, target = _evaluate(
            harness_fsm,
            evaluator,
            HarnessStates.PLAN,
            {
                ContextKeys.PLAN_APPROVED: True,
                ContextKeys.ITERATION: 0,
                ContextKeys.NEEDS_EXPLORE: True,
            },
        )
        assert result is not AMBIGUOUS
        assert result is DETERMINISTIC
        assert target == HarnessStates.EXECUTE

    def test_priority_slots_are_spaced_wide_enough(
        self, harness_fsm: FSMDefinition, evaluator: TransitionEvaluator
    ) -> None:
        """Adjacent priorities in one state differ by more than the threshold.

        Stated as a property of the graph so a new edge squeezed in at, say,
        priority 250 fails here rather than in a live run months later.
        """
        threshold = evaluator.config.ambiguity_threshold
        for state_id, state in harness_fsm.states.items():
            priorities = sorted(t.priority for t in state.transitions)
            for lower, upper in pairwise(priorities):
                confidence_gap = (upper - lower) / 1000.0
                assert confidence_gap > threshold, (
                    f"{state_id}: priorities {lower} and {upper} are "
                    f"{confidence_gap:.3f} apart, under the "
                    f"{threshold} ambiguity threshold"
                )


# ---------------------------------------------------------------------------
# rules.py wiring
# ---------------------------------------------------------------------------


class TestStateRules:
    """Every state's prose and role wiring comes from ``rules.RULES``."""

    def test_states_carry_rules(self, harness_fsm: FSMDefinition) -> None:
        """No state ships empty instructions, and none is hand-written."""
        for state_id in HarnessStates.ALL:
            state = harness_fsm.states[state_id]
            rules = RULES[state_id]

            assert state.description == rules.description
            assert state.purpose == rules.purpose
            assert state.response_instructions == rules.response_instructions

            assert state.description.strip()
            assert state.purpose.strip()
            assert state.response_instructions.strip()
            assert rules.operative_rules
            assert all(rule.strip() for rule in rules.operative_rules)
            assert rules.gate_summary.strip()

    def test_no_state_carries_extraction_instructions(
        self, harness_fsm: FSMDefinition
    ) -> None:
        """D-041: the field is absent, not empty, on every state.

        ``TestBulkExtractionIsUnreachable`` proves what that absence BUYS (core
        issues no extraction call).  This one pins the shape it buys it with,
        so a placeholder string reintroduced "for documentation" fails here
        with a message naming the cost.
        """
        for state_id in HarnessStates.ALL:
            assert harness_fsm.states[state_id].extraction_instructions is None, (
                f"{state_id} carries extraction_instructions; that costs one "
                "LLM call per turn (decisions.md D-041)"
            )
            assert not hasattr(RULES[state_id], "extraction_instructions")

    def test_required_context_keys_match_the_gated_keys(
        self, harness_fsm: FSMDefinition
    ) -> None:
        """Each state requires exactly the keys its own edges gate on.

        Derived rather than declared in ``fsm_definition._build_state``; this
        pins the derivation, since a hand-maintained second copy is what the
        validator's unknown-key warning exists to catch.
        """
        for state_id, state in harness_fsm.states.items():
            gated = {
                key
                for transition in state.transitions
                for condition in transition.conditions or []
                for key in condition.requires_context_keys
            }
            assert set(state.required_context_keys or []) == gated, state_id

    def test_role_by_state_covers_every_state_exactly_once(self) -> None:
        """All 6 states map to 6 distinct dispatchable worker roles."""
        assert set(ROLE_BY_STATE) == set(HarnessStates.ALL)
        roles = list(ROLE_BY_STATE.values())
        assert len(set(roles)) == len(roles) == 6
        assert set(roles) == set(Role.WORKERS)
        assert Role.ORCHESTRATOR not in roles

    def test_rules_cover_every_state(self) -> None:
        assert set(RULES) == set(HarnessStates.ALL)
        for state_id, rules in RULES.items():
            assert rules.state == state_id
            assert rules.role == ROLE_BY_STATE[state_id]


class TestOwnershipModel:
    """Invariant I7: exactly one writing role per artifact, verifier writes none."""

    @staticmethod
    def _all_artifacts() -> set[str]:
        return {
            value
            for name, value in vars(ArtifactNames).items()
            if not name.startswith("_") and isinstance(value, str)
        }

    def test_ownership_map_is_consistent(self) -> None:
        """Every artifact is owned, every owner is a real role."""
        artifacts = self._all_artifacts()
        assert set(OWNERSHIP) == artifacts, (
            f"unowned artifacts: {sorted(artifacts - set(OWNERSHIP))}; "
            f"phantom entries: {sorted(set(OWNERSHIP) - artifacts)}"
        )

        known_roles = set(Role.WORKERS) | {Role.ORCHESTRATOR}
        for artifact, owners in OWNERSHIP.items():
            assert owners, f"{artifact} has no owner"
            assert len(set(owners)) == len(owners), (
                f"{artifact} lists a duplicate owner"
            )
            assert set(owners) <= known_roles, (
                f"{artifact} names unknown roles: {sorted(set(owners) - known_roles)}"
            )

    def test_verifier_writes_nothing(self) -> None:
        """A verifier RETURNS results; the driver merges them (invariant I7)."""
        writers = {role for owners in OWNERSHIP.values() for role in owners}
        assert Role.VERIFIER not in writers

    def test_owned_artifacts_are_a_projection_of_the_ownership_table(self) -> None:
        """A state's ``owned_artifacts`` is derived, never a second copy."""
        for state_id, rules in RULES.items():
            expected = tuple(
                artifact
                for artifact, owners in OWNERSHIP.items()
                if rules.role in owners
            )
            assert rules.owned_artifacts == expected, state_id

    def test_reflect_owns_no_artifact(self) -> None:
        """REFLECT dispatches the verifier, so it owns nothing to write."""
        assert RULES[HarnessStates.REFLECT].owned_artifacts == ()
