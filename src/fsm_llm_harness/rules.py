"""
Per-state operative rules for the harness protocol.

This module is the Python analogue of the source skill's ``emit-state.mjs``
router, which serves ``scripts/modules/state-<state>.md`` byte-verbatim.  Here
the same content is **frozen Python data**, not Markdown files citing each
other by path (decisions.md D-007): the rules become importable, type-checked
and testable, and the whole "does this prose citation resolve" class of
validation disappears with them.

Two authoring rules apply to everything below:

1. **Operative rules stay short and imperative** -- at most 8 bullets per
   state, one sentence each.  These strings end up inside LLM prompts that a
   4B-parameter model has to follow; verbose, nested or hedged rules measurably
   degrade ``:4b`` output, so density here is a small-model hardening decision,
   not a stylistic preference.
2. **No duplicated string literals.**  Every state id, role name, artifact
   name, context key and threshold is read from :mod:`fsm_llm_harness.constants`.

``CLOSE`` has no ``state-close.md`` module in the source skill (SKILL.md:237 --
"CLOSE has no module").  Its rules below are distilled from SKILL.md's State
Machine / Transitions table and from the ``ip-archivist`` agent definition
instead.

The File Ownership Model (SKILL.md:381-402) is encoded once, in
:data:`OWNERSHIP` (artifact -> roles permitted to write it).  Each state's
``owned_artifacts`` is *derived* from that table rather than restated, so the
two can never drift apart.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from fsm_llm.definitions import StateNotFoundError

from .constants import (
    ArtifactNames,
    ContextKeys,
    Defaults,
    HarnessStates,
    PlanSchema,
    Role,
)

# ---------------------------------------------------------------------------
# Role dispatch
# ---------------------------------------------------------------------------

#: state id -> the worker role the driver dispatches on entry to that state.
#:
#: Each of ``Role.WORKERS`` appears exactly once.  Two mappings are worth
#: naming explicitly because they are choices rather than transcriptions:
#:
#: * ``REFLECT -> VERIFIER`` -- REFLECT's dispatched worker runs the
#:   verification strategy and RETURNS results.  The reviewer is an
#:   iteration-2+ *additional* pass in the source skill, not REFLECT's owner.
#: * ``PIVOT -> REVIEWER`` -- PIVOT is orchestrator-owned in the source skill
#:   (no sub-agent).  The reviewer is the closest analogue: its job is the
#:   adversarial "was this approach wrong, and what did we not test" analysis
#:   that a pivot needs.
ROLE_BY_STATE: Mapping[str, str] = MappingProxyType(
    {
        HarnessStates.EXPLORE: Role.EXPLORER,
        HarnessStates.PLAN: Role.PLAN_WRITER,
        HarnessStates.EXECUTE: Role.EXECUTOR,
        HarnessStates.REFLECT: Role.VERIFIER,
        HarnessStates.PIVOT: Role.REVIEWER,
        HarnessStates.CLOSE: Role.ARCHIVIST,
    }
)


# ---------------------------------------------------------------------------
# File Ownership Model
# ---------------------------------------------------------------------------

#: artifact name -> the roles permitted to WRITE it (invariant I7).
#:
#: Transcribed from SKILL.md:381-402.  Co-ownership is permitted only where the
#: writes are disjoint and sequenced by the driver; the driver
#: (``Role.ORCHESTRATOR``) is the non-authoring co-writer almost everywhere,
#: with ``decisions.md`` the documented inversion.
#:
#: ``Role.VERIFIER`` appears nowhere on purpose: a verifier RETURNS results and
#: never writes an artifact -- the driver merges them into ``verification.md``.
#: Step 9 enforces this table; do not encode ownership anywhere else.
OWNERSHIP: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        ArtifactNames.STATE: (Role.ORCHESTRATOR,),
        ArtifactNames.PLAN: (Role.PLAN_WRITER, Role.ORCHESTRATOR),
        ArtifactNames.DECISIONS: (
            Role.ORCHESTRATOR,
            Role.PLAN_WRITER,
            Role.EXECUTOR,
            Role.ARCHIVIST,
        ),
        ArtifactNames.FINDINGS_INDEX: (Role.ORCHESTRATOR,),
        ArtifactNames.FINDINGS_DIR: (Role.EXPLORER, Role.REVIEWER),
        ArtifactNames.PROGRESS: (Role.ORCHESTRATOR,),
        ArtifactNames.VERIFICATION: (Role.PLAN_WRITER, Role.ORCHESTRATOR),
        ArtifactNames.CHANGELOG: (Role.EXECUTOR, Role.ORCHESTRATOR),
        ArtifactNames.CHECKPOINTS_DIR: (Role.EXECUTOR,),
        ArtifactNames.SUMMARY: (Role.ARCHIVIST,),
        ArtifactNames.CROSS_FINDINGS: (Role.ARCHIVIST,),
        ArtifactNames.CROSS_DECISIONS: (Role.ARCHIVIST,),
        ArtifactNames.LESSONS: (Role.ARCHIVIST,),
        ArtifactNames.LESSONS_ARCHIVE: (Role.ARCHIVIST,),
        ArtifactNames.SYSTEM: (Role.ARCHIVIST,),
        ArtifactNames.INDEX: (Role.ARCHIVIST,),
    }
)


def _artifacts_writable_by(role: str) -> tuple[str, ...]:
    """Return every artifact ``role`` may write, in ``OWNERSHIP`` order.

    Sole reader of :data:`OWNERSHIP` at import time, so a state's
    ``owned_artifacts`` is always a projection of the ownership table rather
    than a second hand-maintained copy of it.
    """
    return tuple(name for name, owners in OWNERSHIP.items() if role in owners)


# ---------------------------------------------------------------------------
# Per-state rules
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StateRules:
    """The frozen protocol rules for one harness state.

    Instances are module-level constants built once at import time and shared
    by the FSM builder (which reads ``description`` / ``purpose`` /
    ``*_instructions``), the driver (``role``, ``operative_rules``) and the
    ownership enforcement in ``storage.py`` (``owned_artifacts``).

    Attributes:
        state: The FSM state id this rule set governs.
        role: The worker role dispatched on entry to ``state``.
        description: Short human-readable state description (FSM ``description``).
        purpose: What must be accomplished here (FSM ``purpose``).
        operative_rules: The protocol rules a worker must follow, at most 8
            single-sentence imperatives.
        extraction_instructions: Pass-1 instructions naming the context keys
            this state must produce.
        response_instructions: Pass-2 instructions for the user-facing report.
        owned_artifacts: Artifacts ``role`` is permitted to write.
        gate_summary: One line describing the exit gate, at default thresholds.
    """

    state: str
    role: str
    description: str
    purpose: str
    operative_rules: tuple[str, ...]
    extraction_instructions: str
    response_instructions: str
    owned_artifacts: tuple[str, ...]
    gate_summary: str


_EXPLORE = StateRules(
    state=HarnessStates.EXPLORE,
    role=ROLE_BY_STATE[HarnessStates.EXPLORE],
    description="Gather context: read code, index findings, classify constraints.",
    purpose=(
        "Build enough grounded context to plan: at least "
        f"{Defaults.FINDINGS_THRESHOLD} indexed findings covering problem "
        "scope, affected files, and the existing patterns or constraints that "
        "any solution must respect."
    ),
    operative_rules=(
        "Read the current state and the cross-plan memory files before the "
        "first search.",
        "Ask one focused question at a time and answer it by reading, "
        "grepping and globbing real files.",
        f"Write each finding to {ArtifactNames.FINDINGS_DIR}/<topic>.md and "
        f"never touch the {ArtifactNames.FINDINGS_INDEX} index.",
        "Record file paths and call traces as evidence instead of summaries.",
        "Classify every constraint as hard, soft, or ghost, and say which.",
        f"Do not leave EXPLORE below {Defaults.FINDINGS_THRESHOLD} indexed "
        "findings, however obvious the answer looks.",
        "Self-assess problem scope, solution space and risk visibility before "
        "handing off to PLAN.",
        "On re-entry from REFLECT, append corrections marked "
        "[CORRECTED iter-N]; never overwrite an earlier finding.",
    ),
    extraction_instructions=(
        "Report exploration progress as JSON with exactly these fields:\n"
        f"- {ContextKeys.FINDINGS_COUNT} (integer): how many findings are "
        f"indexed in {ArtifactNames.FINDINGS_INDEX} right now.\n"
        f"- {ContextKeys.NEEDS_EXPLORE} (boolean): true if more research is "
        "still required.\n"
        "Count only findings already written to disk. The exit gate is "
        "mechanical: an absent or non-numeric count keeps it closed."
    ),
    response_instructions=(
        "State the current findings count, the topics covered so far, and the "
        "single next question you will investigate. Be terse and factual."
    ),
    owned_artifacts=_artifacts_writable_by(ROLE_BY_STATE[HarnessStates.EXPLORE]),
    gate_summary=(
        f"HARD: leave for PLAN only when {ContextKeys.FINDINGS_COUNT} >= "
        f"{Defaults.FINDINGS_THRESHOLD}."
    ),
)


_PLAN = StateRules(
    state=HarnessStates.PLAN,
    role=ROLE_BY_STATE[HarnessStates.PLAN],
    description="Write the plan and the verification template; get approval.",
    purpose=(
        f"Turn findings into a {len(PlanSchema.SECTIONS)}-section "
        f"{ArtifactNames.PLAN} with annotated steps, assumptions, failure "
        "modes, falsification signals, success criteria and a complexity "
        "budget, then wait for explicit user approval."
    ),
    operative_rules=(
        f"Re-read {ArtifactNames.FINDINGS_INDEX} and every findings file "
        "before writing anything.",
        f"Go back to EXPLORE if fewer than {Defaults.FINDINGS_THRESHOLD} "
        "findings are indexed.",
        "Write the Problem Statement first: expected behavior, invariants, edge cases.",
        f"Write all {len(PlanSchema.SECTIONS)} required sections of "
        f"{ArtifactNames.PLAN}, in order: "
        f"{', '.join(PlanSchema.SECTIONS)}.",
        "List every file to modify; if you cannot list them, go back to EXPLORE.",
        "Tag each step with its risk and its dependencies, and start with the "
        "riskiest one.",
        f"Phrase every entry in {ArtifactNames.DECISIONS} as 'X at the cost "
        "of Y'; alternatives belong there, not in the plan.",
        f"Seed {ArtifactNames.VERIFICATION} with one row per success "
        "criterion, then present the plan and wait for approval.",
    ),
    extraction_instructions=(
        "Report planning progress as JSON with exactly these fields:\n"
        f"- {ContextKeys.PLAN_APPROVED} (boolean): true ONLY after the user "
        "explicitly approved this plan.\n"
        f"- {ContextKeys.NEEDS_EXPLORE} (boolean): true if the problem cannot "
        "be stated or the files cannot be listed.\n"
        f"Never set {ContextKeys.PLAN_APPROVED} on your own judgement -- it "
        "records a user decision. An absent field keeps the gate closed."
    ),
    response_instructions=(
        "Present the plan for approval: the steps, the success criteria, the "
        "verification strategy, the failure modes and the assumptions. End by "
        "asking for an explicit approve or revise decision."
    ),
    owned_artifacts=_artifacts_writable_by(ROLE_BY_STATE[HarnessStates.PLAN]),
    gate_summary=(
        f"HARD: leave for EXECUTE only when {ContextKeys.PLAN_APPROVED} is "
        f"true AND {ContextKeys.ITERATION} < {Defaults.ITERATION_HARD_CAP}."
    ),
)


_EXECUTE = StateRules(
    state=HarnessStates.EXECUTE,
    role=ROLE_BY_STATE[HarnessStates.EXECUTE],
    description="Implement exactly one plan step, then log and commit it.",
    purpose=(
        "Carry out a single plan step under the "
        f"{Defaults.MAX_FIX_ATTEMPTS}-attempt autonomy leash, recording every "
        "edit in the changelog and every non-obvious choice as an anchored "
        "decision."
    ),
    operative_rules=(
        f"Re-read {ArtifactNames.STATE}, {ArtifactNames.PLAN}, "
        f"{ArtifactNames.PROGRESS} and {ArtifactNames.DECISIONS} before "
        "editing anything.",
        "Implement exactly one plan step and never look ahead to the next.",
        "Before the first edit of iteration 1, create the cp-000 checkpoint "
        "recording the pre-change commit hash.",
        "Create a checkpoint before any change touching three or more files.",
        f"Append one {ArtifactNames.CHANGELOG} line per file written, with "
        "path, operation, blast radius and a one-clause reason.",
        "On breakage, stop and revert first; a fix needing more than 10 new "
        "lines is not a fix.",
        f"Take at most {Defaults.MAX_FIX_ATTEMPTS} fix attempts, then report "
        "the failure instead of trying again.",
        "Anchor each non-obvious choice with a DECISION comment and back-link "
        "it from the decision entry in the same commit.",
    ),
    extraction_instructions=(
        "Report step execution as JSON with exactly these fields:\n"
        f"- {ContextKeys.EXECUTE_COMPLETE} (boolean): true when this step "
        "finished, failed, hit the fix leash, or produced a surprise "
        "discovery.\n"
        f"- {ContextKeys.STEP_NUMBER} (integer): the plan step just "
        "attempted.\n"
        f"- {ContextKeys.FIX_ATTEMPTS} (integer): fix attempts used on this "
        "step so far.\n"
        "Report the real attempt count; understating it bypasses the leash."
    ),
    response_instructions=(
        "Report the step number and description, the files created or "
        "modified, the commit, any surprises, and the next step in one line "
        "each. No narrative."
    ),
    owned_artifacts=_artifacts_writable_by(ROLE_BY_STATE[HarnessStates.EXECUTE]),
    gate_summary=(
        f"Leave for REFLECT when {ContextKeys.EXECUTE_COMPLETE} is true "
        "(step done, failed, surprising, or leash-capped)."
    ),
)


_REFLECT = StateRules(
    state=HarnessStates.REFLECT,
    role=ROLE_BY_STATE[HarnessStates.REFLECT],
    description="Verify the work, analyse failures, and route the next move.",
    purpose=(
        "Run the verification strategy, check for regressions and scope "
        "drift, record root causes and simplification results, then recommend "
        "close, pivot, explore or execute and wait for the user."
    ),
    operative_rules=(
        f"Read {ArtifactNames.PLAN}, {ArtifactNames.PROGRESS}, "
        f"{ArtifactNames.VERIFICATION}, {ArtifactNames.FINDINGS_INDEX}, the "
        f"checkpoints, {ArtifactNames.DECISIONS} and "
        f"{ArtifactNames.CHANGELOG} before evaluating anything.",
        "Run every check in the Verification Strategy and record PASS or FAIL "
        "with the evidence that produced it.",
        "Re-run tests that passed before this iteration; any regression "
        "blocks closing.",
        "Compare the files actually changed against the planned file list and "
        "justify or revert the drift.",
        "Write down what you did not verify and why; absence of evidence is "
        "not evidence of absence.",
        "On failure, record the immediate cause, the contributing factor, the "
        "defense that should have caught it, and the prevention.",
        "Run the simplification checks against the written criteria, not "
        "against memory.",
        "Present completed work, remaining work, verification results, issues "
        "and a recommendation -- and never close without user confirmation.",
    ),
    extraction_instructions=(
        "Report the evaluation as JSON with exactly these fields:\n"
        f"- {ContextKeys.ALL_CRITERIA_PASS} (boolean): true only if every "
        "success criterion is verified PASS with evidence.\n"
        f"- {ContextKeys.CLOSE_CONFIRMED} (boolean): true ONLY after the user "
        "confirmed closing.\n"
        f"- {ContextKeys.NEEDS_PIVOT} (boolean): true if the approach itself "
        "failed and a new one is needed.\n"
        f"- {ContextKeys.COMPLETION_FIX} (boolean): true if small fixes in "
        "this same iteration finish the work.\n"
        f"- {ContextKeys.NEEDS_EXPLORE} (boolean): true if unknowns must be "
        "investigated before deciding.\n"
        "Set exactly one routing flag, and never set "
        f"{ContextKeys.CLOSE_CONFIRMED} on your own judgement."
    ),
    response_instructions=(
        "Report five things in order: what was completed, what remains, the "
        "verification results, the issues found, and one recommendation of "
        "close, pivot, explore or execute. Then wait for confirmation."
    ),
    owned_artifacts=_artifacts_writable_by(ROLE_BY_STATE[HarnessStates.REFLECT]),
    gate_summary=(
        f"HARD to CLOSE: {ContextKeys.CLOSE_CONFIRMED} AND "
        f"{ContextKeys.ALL_CRITERIA_PASS}. HARD back to EXECUTE: "
        f"{ContextKeys.COMPLETION_FIX} AND {ContextKeys.FIX_ATTEMPTS} < "
        f"{Defaults.MAX_FIX_ATTEMPTS}. Soft: "
        f"{ContextKeys.NEEDS_PIVOT}, {ContextKeys.NEEDS_EXPLORE}."
    ),
)


_PIVOT = StateRules(
    state=HarnessStates.PIVOT,
    role=ROLE_BY_STATE[HarnessStates.PIVOT],
    description="Abandon the failed approach and formulate a new direction.",
    purpose=(
        "Analyse why the approach failed, surface ghost constraints, decide "
        "keep-versus-revert against the checkpoints, and log a new direction "
        "before returning to PLAN."
    ),
    operative_rules=(
        f"Read {ArtifactNames.DECISIONS}, {ArtifactNames.FINDINGS_INDEX}, "
        f"{ArtifactNames.PLAN}, {ArtifactNames.VERIFICATION} and the "
        "checkpoints first.",
        "Decide keep-versus-revert per checkpoint; when unsure, revert to the "
        "latest one.",
        "Ask which constraint that forced the failed approach no longer "
        "holds, and log every ghost constraint found.",
        "Correct stale findings in place with [CORRECTED iter-N]; append, "
        "never delete.",
        f"Log the pivot in {ArtifactNames.DECISIONS} with a complexity "
        "assessment and an 'X at the cost of Y' trade-off.",
        "Reset the fix-attempt counter so the leash does not carry into the "
        "next EXECUTE.",
        "Offer one to three candidate directions, each framed as a trade-off, "
        "and name the checkpoints available.",
        "Get an explicit direction from the user before returning to PLAN.",
    ),
    extraction_instructions=(
        "Report the pivot as JSON with exactly these fields:\n"
        f"- {ContextKeys.PIVOT_RESOLVED} (boolean): true once the user picked "
        "a direction and the decision is logged.\n"
        f"- {ContextKeys.PIVOT_REASON} (string): one sentence naming what "
        "failed and why.\n"
        "An absent field keeps the gate closed."
    ),
    response_instructions=(
        "Report the pivot reason, the available checkpoints, the ghost "
        "constraints surfaced, and one to three candidate directions framed "
        "as trade-offs. Ask which direction to take."
    ),
    owned_artifacts=_artifacts_writable_by(ROLE_BY_STATE[HarnessStates.PIVOT]),
    gate_summary=(
        f"Return to PLAN when {ContextKeys.PIVOT_RESOLVED} is true "
        "(direction chosen and decision logged)."
    ),
)


_CLOSE = StateRules(
    state=HarnessStates.CLOSE,
    role=ROLE_BY_STATE[HarnessStates.CLOSE],
    description="Finalize: summary, anchor audit, cross-plan memory, release.",
    purpose=(
        "Complete closing housekeeping: audit decision anchors, write the "
        "summary, merge the cross-plan files, rewrite the lessons and system "
        "atlas under their line caps, and release the plan."
    ),
    operative_rules=(
        "Audit decision anchors both ways: entries missing anchors, and "
        "anchors with no backing entry.",
        f"Write {ArtifactNames.SUMMARY} with outcome, iterations, key "
        "decisions, files changed and the anchor registry.",
        f"Merge this plan's findings and decisions into "
        f"{ArtifactNames.CROSS_FINDINGS} and {ArtifactNames.CROSS_DECISIONS}.",
        f"Rewrite {ArtifactNames.LESSONS} whole, at most "
        f"{Defaults.LESSONS_LINE_CAP} lines, trimming by importance then "
        "recency.",
        f"Never drop an importance-{Defaults.LESSONS_PROTECTED_IMPORTANCE} "
        f"lesson, and append every dropped line to "
        f"{ArtifactNames.LESSONS_ARCHIVE} before removing it.",
        f"Rewrite {ArtifactNames.SYSTEM} whole, at most "
        f"{Defaults.SYSTEM_LINE_CAP} lines, demoting by staleness rather "
        "than truncating.",
        f"Append this plan to {ArtifactNames.INDEX} and release the plan "
        "pointer exactly once.",
        f"Compress any consolidated file over "
        f"{Defaults.CONSOLIDATED_COMPRESS_LINES} lines behind the "
        "compressed-summary marker.",
    ),
    extraction_instructions=(
        "Report closing housekeeping as JSON with exactly this field:\n"
        f"- {ContextKeys.HALT_REASON} (string): one sentence recording why "
        "the run ended.\n"
        "CLOSE is terminal; no further transition is evaluated."
    ),
    response_instructions=(
        "Report the outcome, the iterations used, the key decisions, the "
        "files changed and the lessons recorded. One line each."
    ),
    owned_artifacts=_artifacts_writable_by(ROLE_BY_STATE[HarnessStates.CLOSE]),
    gate_summary="Terminal: CLOSE has no outbound transition.",
)


#: state id -> its frozen rule set.  Covers all of ``HarnessStates.ALL``.
RULES: Mapping[str, StateRules] = MappingProxyType(
    {
        HarnessStates.EXPLORE: _EXPLORE,
        HarnessStates.PLAN: _PLAN,
        HarnessStates.EXECUTE: _EXECUTE,
        HarnessStates.REFLECT: _REFLECT,
        HarnessStates.PIVOT: _PIVOT,
        HarnessStates.CLOSE: _CLOSE,
    }
)


def get_rules(state: str) -> StateRules:
    """Return the frozen rule set for ``state``.

    Args:
        state: A state id from :class:`~fsm_llm_harness.constants.HarnessStates`.

    Returns:
        The :class:`StateRules` governing that state.

    Raises:
        StateNotFoundError: If ``state`` is not one of the 6 protocol states.
            Carries ``state_id`` so callers can branch without parsing text.
    """
    try:
        return RULES[state]
    except KeyError:
        raise StateNotFoundError(
            f"Unknown harness state '{state}'; expected one of "
            f"{', '.join(HarnessStates.ALL)}",
            state_id=state,
        ) from None
