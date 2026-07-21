"""
FSM definition factory for the iterative-planner harness protocol.

``build_harness_fsm()`` returns an FSM-JSON ``dict`` consumable by
``fsm_llm.API.from_definition``, following the ``build_*_fsm()`` factory
convention already used 12 times in ``fsm_llm_agents/fsm_definitions.py``.
All prose (descriptions, purposes, extraction/response instructions) is sourced
from :mod:`fsm_llm_harness.rules`; this module contributes only graph shape and
gate logic.

Graph shape -- 6 states, 9 transitions
--------------------------------------
======================  ==========  =====================================
Edge                    Priority    Gate
======================  ==========  =====================================
EXPLORE -> PLAN         10          HARD: findings_count >= threshold
PLAN    -> EXECUTE      10          HARD: plan_approved AND iteration < cap
PLAN    -> EXPLORE      200         needs_explore
EXECUTE -> REFLECT      10          execute_complete
REFLECT -> CLOSE        10          HARD: close_confirmed AND all_criteria_pass
REFLECT -> EXECUTE      200         HARD: completion_fix AND fix_attempts < cap
REFLECT -> PIVOT        400         needs_pivot
REFLECT -> EXPLORE      600         needs_explore
PIVOT   -> PLAN         10          pivot_resolved
======================  ==========  =====================================

The source skill's transition table lists 12 rows, but three of them are not
FSM edges: ``[*] -> EXPLORE`` is bootstrap, ``CLOSE -> [*]`` is termination,
and ``any -> CLOSE`` is the administrative ``bootstrap.mjs close`` shortcut
(classified INCIDENTAL by ``findings/iterative-planner-source.md``).  Nine is
the real edge count; do not add three placeholder edges to reach twelve.

Priority semantics
------------------
In this codebase a **lower** ``priority`` value wins.  ``TransitionEvaluator``
derives base confidence as ``max(0.1, 1.0 - priority / 1000)``
(``transition_evaluator.py:250``), so priority 10 outranks priority 600.  The
slots below are spaced >= 150 apart because two passing edges are resolved
DETERMINISTICally only when their confidence gap clears the 0.1
``ambiguity_threshold``; a narrower spacing would route a gate decision to the
LLM classifier, which is exactly what invariant I1 forbids.

Ordering rationale, per state:

* **PLAN** -- the mechanically gated approval edge (-> EXECUTE) is checked
  before the soft "I cannot state the problem" edge (-> EXPLORE).  Approval is
  evidence; ``needs_explore`` is judgement.
* **REFLECT** -- ordered by "cheapest legal corrective action first, escalating
  only when a HARD gate refuses":  CLOSE (the run is done and verified) beats
  EXECUTE (a same-iteration completion fix) beats PIVOT (abandon the approach)
  beats EXPLORE (widest loop-back).  Putting EXECUTE above PIVOT is deliberate:
  a small remediation must not silently become a pivot, and when the leash is
  exhausted the ``fix_attempts`` conjunct blocks that edge so PIVOT wins on its
  own merits rather than by ordering.
* **EXPLORE / EXECUTE / PIVOT** -- single outbound edge each; the slot is
  cosmetic but stated explicitly so no edge relies on the field default.

Gate logic
----------
Each gate is ONE ``TransitionCondition`` carrying one JsonLogic term (an
``and`` term where the gate is a conjunction), using only operators inside
``fsm_llm.constants.ALLOWED_JSONLOGIC_OPERATIONS``: ``>=``, ``<``, ``==``,
``and``, ``var``.  Keeping it to one condition per edge also keeps the
evidence-weight boost uniform across edges, so the priority spacing above is
the only thing that decides a multi-edge race.

Every condition also declares ``requires_context_keys``.  That is what makes
invariant I8 (fail closed) mechanical: ``TransitionEvaluator`` fails a
condition whose required key is absent from context *before* evaluating the
logic, so a garbled or missing worker reply leaves the edge BLOCKED rather
than accidentally satisfied.
"""

from __future__ import annotations

from typing import Any

from .constants import ContextKeys, Defaults, HarnessStates
from .rules import RULES, StateRules

# ---------------------------------------------------------------------------
# Priority slots (lower value = evaluated first; see module docstring)
# ---------------------------------------------------------------------------

_PRIORITY_1ST = 10
_PRIORITY_2ND = 200
_PRIORITY_3RD = 400
_PRIORITY_4TH = 600

#: How much of the goal is echoed into the FSM ``description`` field.  The
#: full goal lives in context under ``ContextKeys.GOAL``; this is a preview for
#: humans reading a dumped FSM, not a data channel.
_GOAL_PREVIEW_CHARS = 200

#: Multi-pass extraction refinement is disabled on every state.  The gate flags
#: below are set by the driver's handlers from worker results, not extracted
#: from user prose, so a refinement pass triggered by "required key missing"
#: would spend an extra LLM call per turn to re-ask a question the model was
#: never the source of.  On a 4B model that cost is the whole turn budget.
_EXTRACTION_RETRIES = 0

#: Terse operator persona. Deliberately mechanical: the harness reports
#: counters and evidence, and a chatty persona on a small model tends to
#: narrate a gate as satisfied when it is not.
DEFAULT_PERSONA = (
    "You operate a 6-state iterative planning protocol (EXPLORE, PLAN, "
    "EXECUTE, REFLECT, PIVOT, CLOSE). You are terse, factual and mechanical: "
    "you report state, counters, files and evidence, never impressions. You "
    "never describe a gate as satisfied without naming the evidence that "
    "satisfies it, and you never claim work you did not do."
)


def _build_state(
    rules: StateRules, transitions: list[dict[str, Any]]
) -> dict[str, Any]:
    """Assemble one FSM-JSON state from its rule set and outbound edges.

    ``required_context_keys`` is derived from the edges rather than declared,
    so it can never name a key that no condition gates on (which
    ``FSMValidator._validate_required_context_keys`` warns about) and can never
    omit one that does.

    Args:
        rules: The frozen rule set supplying all prose for this state.
        transitions: Outbound edges in FSM-JSON form; empty for a terminal state.

    Returns:
        An FSM-JSON state dict.
    """
    gated_keys: list[str] = []
    for transition in transitions:
        for condition in transition["conditions"]:
            for key in condition["requires_context_keys"]:
                if key not in gated_keys:
                    gated_keys.append(key)

    state: dict[str, Any] = {
        "id": rules.state,
        "description": rules.description,
        "purpose": rules.purpose,
        "extraction_instructions": rules.extraction_instructions,
        "response_instructions": rules.response_instructions,
        "extraction_retries": _EXTRACTION_RETRIES,
        "transitions": transitions,
    }
    if gated_keys:
        state["required_context_keys"] = gated_keys
    return state


def build_harness_fsm(
    goal: str,
    *,
    persona: str | None = None,
    findings_threshold: int = Defaults.FINDINGS_THRESHOLD,
    max_fix_attempts: int = Defaults.MAX_FIX_ATTEMPTS,
    iteration_hard_cap: int = Defaults.ITERATION_HARD_CAP,
) -> dict[str, Any]:
    """Build the 6-state iterative-planner protocol FSM.

    Args:
        goal: The run's goal. Echoed (truncated) into the FSM description; the
            authoritative copy belongs in context under ``ContextKeys.GOAL``.
        persona: Response persona. Defaults to a terse operator persona.
        findings_threshold: Minimum indexed findings for EXPLORE -> PLAN.
        max_fix_attempts: Autonomy leash cap for the REFLECT -> EXECUTE
            completion-fix edge.
        iteration_hard_cap: Iteration count at which PLAN -> EXECUTE stops
            opening.

    Returns:
        An FSM-JSON dict accepted by ``FSMDefinition(**result)`` and
        ``fsm_llm.API.from_definition``.
    """
    # DECISION plan-2026-07-21T125237-191b2eb2/D-012
    # There are deliberately NO self-loop edges (EXPLORE -> EXPLORE,
    # PLAN -> PLAN, EXECUTE -> EXECUTE, ...) even though the source skill's
    # table lists a PLAN -> PLAN revise cycle. Do NOT "fix" a state that seems
    # to have no way to stay put by adding one: `TransitionEvaluator` returns
    # BLOCKED when no outbound edge passes, and BLOCKED natively holds the
    # current state with no exception. An explicit self-loop would be a second
    # ALWAYS-passing edge out of every state, which would (a) race the real
    # gated edge into an AMBIGUOUS result and hand a HARD gate decision to the
    # LLM classifier -- breaking invariant I1 -- and (b) make "the protocol did
    # not advance" indistinguishable from "the protocol advanced to itself" in
    # state.md's transition history. See decisions.md D-012.
    explore_transitions: list[dict[str, Any]] = [
        {
            "target_state": HarnessStates.PLAN,
            "description": (
                "Enough grounded context: the minimum number of findings is indexed."
            ),
            "priority": _PRIORITY_1ST,
            "conditions": [
                {
                    "description": (
                        f"HARD gate: at least {findings_threshold} findings are indexed"
                    ),
                    "requires_context_keys": [ContextKeys.FINDINGS_COUNT],
                    "logic": {
                        ">=": [
                            {"var": ContextKeys.FINDINGS_COUNT},
                            findings_threshold,
                        ]
                    },
                }
            ],
        },
    ]

    plan_transitions: list[dict[str, Any]] = [
        {
            "target_state": HarnessStates.EXECUTE,
            "description": "The user approved the plan and the iteration cap is clear.",
            "priority": _PRIORITY_1ST,
            "conditions": [
                {
                    "description": (
                        "HARD gate: the user approved the plan AND iteration "
                        f"< {iteration_hard_cap}"
                    ),
                    "requires_context_keys": [
                        ContextKeys.PLAN_APPROVED,
                        ContextKeys.ITERATION,
                    ],
                    "logic": {
                        "and": [
                            {"==": [{"var": ContextKeys.PLAN_APPROVED}, True]},
                            {"<": [{"var": ContextKeys.ITERATION}, iteration_hard_cap]},
                        ]
                    },
                }
            ],
        },
        {
            "target_state": HarnessStates.EXPLORE,
            "description": (
                "The problem cannot be stated or the files cannot be listed: "
                "more context first."
            ),
            "priority": _PRIORITY_2ND,
            "conditions": [
                {
                    "description": "More exploration is required before planning",
                    "requires_context_keys": [ContextKeys.NEEDS_EXPLORE],
                    "logic": {"==": [{"var": ContextKeys.NEEDS_EXPLORE}, True]},
                }
            ],
        },
    ]

    # One edge covers every way a step can end -- all steps done, a failure, a
    # surprise discovery, or a leash hit. The driver decides which of those
    # occurred and when to set the flag; splitting it into four edges would put
    # that classification inside the FSM, where it cannot see the failure.
    execute_transitions: list[dict[str, Any]] = [
        {
            "target_state": HarnessStates.REFLECT,
            "description": (
                "The step ended: completed, failed, surprising, or leash-capped."
            ),
            "priority": _PRIORITY_1ST,
            "conditions": [
                {
                    "description": "Execution of the current step is finished",
                    "requires_context_keys": [ContextKeys.EXECUTE_COMPLETE],
                    "logic": {"==": [{"var": ContextKeys.EXECUTE_COMPLETE}, True]},
                }
            ],
        },
    ]

    reflect_transitions: list[dict[str, Any]] = [
        {
            "target_state": HarnessStates.CLOSE,
            "description": "Every criterion passed and the user confirmed closing.",
            "priority": _PRIORITY_1ST,
            "conditions": [
                {
                    "description": (
                        "HARD gate: the user confirmed closing AND every "
                        "success criterion passed"
                    ),
                    "requires_context_keys": [
                        ContextKeys.CLOSE_CONFIRMED,
                        ContextKeys.ALL_CRITERIA_PASS,
                    ],
                    "logic": {
                        "and": [
                            {"==": [{"var": ContextKeys.CLOSE_CONFIRMED}, True]},
                            {"==": [{"var": ContextKeys.ALL_CRITERIA_PASS}, True]},
                        ]
                    },
                }
            ],
        },
        {
            "target_state": HarnessStates.EXECUTE,
            "description": (
                "Small same-iteration fixes finish the work; the iteration "
                "does not increment."
            ),
            "priority": _PRIORITY_2ND,
            "conditions": [
                {
                    "description": (
                        "HARD gate: a completion fix is needed AND fix "
                        f"attempts < {max_fix_attempts}"
                    ),
                    "requires_context_keys": [
                        ContextKeys.COMPLETION_FIX,
                        ContextKeys.FIX_ATTEMPTS,
                    ],
                    "logic": {
                        "and": [
                            {"==": [{"var": ContextKeys.COMPLETION_FIX}, True]},
                            {
                                "<": [
                                    {"var": ContextKeys.FIX_ATTEMPTS},
                                    max_fix_attempts,
                                ]
                            },
                        ]
                    },
                }
            ],
        },
        {
            "target_state": HarnessStates.PIVOT,
            "description": "The approach itself failed; a new direction is needed.",
            "priority": _PRIORITY_3RD,
            "conditions": [
                {
                    "description": "A pivot away from the current approach is needed",
                    "requires_context_keys": [ContextKeys.NEEDS_PIVOT],
                    "logic": {"==": [{"var": ContextKeys.NEEDS_PIVOT}, True]},
                }
            ],
        },
        {
            "target_state": HarnessStates.EXPLORE,
            "description": "Unknowns must be investigated before deciding anything.",
            "priority": _PRIORITY_4TH,
            "conditions": [
                {
                    "description": "More context is required before routing",
                    "requires_context_keys": [ContextKeys.NEEDS_EXPLORE],
                    "logic": {"==": [{"var": ContextKeys.NEEDS_EXPLORE}, True]},
                }
            ],
        },
    ]

    pivot_transitions: list[dict[str, Any]] = [
        {
            "target_state": HarnessStates.PLAN,
            "description": "A new direction was chosen and the decision is logged.",
            "priority": _PRIORITY_1ST,
            "conditions": [
                {
                    "description": "The pivot is resolved into a chosen direction",
                    "requires_context_keys": [ContextKeys.PIVOT_RESOLVED],
                    "logic": {"==": [{"var": ContextKeys.PIVOT_RESOLVED}, True]},
                }
            ],
        },
    ]

    transitions_by_state: dict[str, list[dict[str, Any]]] = {
        HarnessStates.EXPLORE: explore_transitions,
        HarnessStates.PLAN: plan_transitions,
        HarnessStates.EXECUTE: execute_transitions,
        HarnessStates.REFLECT: reflect_transitions,
        HarnessStates.PIVOT: pivot_transitions,
        # CLOSE is the FSM's only terminal state: no outbound edges.
        HarnessStates.CLOSE: [],
    }

    states = {
        state_id: _build_state(RULES[state_id], transitions_by_state[state_id])
        for state_id in HarnessStates.ALL
    }

    return {
        "name": "iterative_planner_harness",
        "description": (
            "Iterative-planner protocol harness. Goal: "
            f"{goal[:_GOAL_PREVIEW_CHARS] or 'unspecified'}"
        ),
        "initial_state": HarnessStates.INITIAL,
        "persona": persona or DEFAULT_PERSONA,
        "states": states,
    }
