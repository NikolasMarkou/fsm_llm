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
#: Do not encode ownership anywhere else.
#
# DECISION plan-2026-07-21T125237-191b2eb2/D-048
# THIS TABLE IS THE OWNERSHIP MODEL; plan.md's invariant I7 is a summary of it,
# and where the two disagreed (review N5) the table won and the prose was
# corrected -- not the other way round. Two entries look like transcription
# slips and are not:
#   1. FINDINGS_DIR is (EXPLORER, REVIEWER), not (EXPLORER,). Do NOT drop
#      REVIEWER "to match I7's wording": PIVOT's own operative rules order the
#      reviewer to correct stale findings in place with [CORRECTED iter-N].
#      Removing it recreates exactly the review-C2 defect this table now closes
#      -- a role ordered to write a file it holds no tool for.
#   2. DECISIONS has FOUR owners. Do NOT reduce it to one: the writes are
#      disjoint and sequenced by the driver (one appended entry per phase), and
#      the strict `## D-NNN | PHASE | date` header records which phase wrote
#      each one, so a single-writer rule here would silence three of the four
#      phases that must log a trade-off.
# `PlanMemory.authorise` reads this table directly, so an edit here changes what
# a live role can write. See decisions.md D-048.
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


def artifacts_writable_by(role: str) -> tuple[str, ...]:
    """Return every artifact ``role`` may write, in ``OWNERSHIP`` order.

    Interface contract (shared helper, 3 call sites: each state's
    ``owned_artifacts`` below, ``roles.py``'s plan tool-scope derivation, and
    ``roles.py``'s prompt text):
        - Parameter: a ``Role`` member.  An unknown role yields ``()``.
        - Returns artifact names in :data:`OWNERSHIP` order.
        - Never raises; performs no I/O.

    Sole projection of :data:`OWNERSHIP`, so tool scope, prompt text and a
    state's ``owned_artifacts`` are the same fact read three times rather than
    three hand-maintained copies of it.
    """
    return tuple(name for name, owners in OWNERSHIP.items() if role in owners)


# ---------------------------------------------------------------------------
# EXPLORE topic decomposition
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExploreTopic:
    """One EXPLORE topic: a findings-file slug plus the two ways it is phrased.

    Attributes:
        slug: kebab-case, and the STEM of the ``findings/<slug>.md`` file the
            driver assigns.  Distinct across :data:`EXPLORE_TOPICS`.
        label: a noun phrase naming the topic, used both in the EXPLORE
            purpose sentence and at the head of the assignment prompt.
        brief: one clause saying what belongs in the file.
    """

    slug: str
    label: str
    brief: str


# DECISION plan-2026-07-21T191807-bf7ffe24/D-035
# The decomposition is a FIXED TABLE, not an LLM call, and the three topics are
# not invented here: they are the three coverage axes EXPLORE's own `purpose`
# has always named, which is why that sentence is now DERIVED from this table
# (`_topic_phrase`) instead of restating it. Two alternatives were rejected.
#   * A driver-side LLM decomposition of the goal. It would put the SAME 4B
#     model that has failed four times to produce three topics in charge of
#     naming them, add a call that must fail closed, and make the assignment
#     non-reproducible between two runs of the same goal. If it ever becomes
#     worth it, it belongs behind this function, not instead of it.
#   * Deriving slugs from the goal TEXT. Goal wording is arbitrary; two
#     dispatches of "add a retry with backoff to the uploader" would get
#     `retry`, `backoff`, `uploader` -- three restatements of one topic, which
#     is the exact degeneracy step 22 measured and D-028 exists to stop.
# The slugs are goal-INDEPENDENT and the topics are goal-RELATIVE: "the problem
# scope OF THIS GOAL" is a different finding for every run, and the goal is in
# the same prompt two blocks above the assignment.
# Do NOT make this list longer "for coverage": every extra topic is an extra
# dispatch the run must spend before its gate can open, and the measured
# yield horizon is ~6-9 dispatches (decisions.md D-031).
# See decisions.md D-035.
EXPLORE_TOPICS: tuple[ExploreTopic, ...] = (
    ExploreTopic(
        slug="problem-scope",
        label="problem scope",
        brief="what the goal actually requires, and what is out of scope",
    ),
    ExploreTopic(
        slug="affected-files",
        label="affected files",
        brief="which real files and symbols the change touches, with paths",
    ),
    ExploreTopic(
        slug="constraints-and-patterns",
        label=("the existing patterns or constraints that any solution must respect"),
        brief=(
            "the conventions, invariants and hard constraints the code already imposes"
        ),
    ),
)

#: Slug used to extend :data:`EXPLORE_TOPICS` when a caller raises the findings
#: threshold above the number of protocol axes.  Numbered from 1 so the first
#: extension reads ``open-question-1``.
_EXTRA_TOPIC_SLUG = "open-question"


def explore_topics(
    threshold: int = Defaults.FINDINGS_THRESHOLD,
) -> tuple[ExploreTopic, ...]:
    """The ordered EXPLORE topics, at least *threshold* of them.

    Interface contract (2 call sites: :data:`_EXPLORE`'s purpose sentence and
    ``harness.HarnessAgent._assign_explore_topic``):
        - ``threshold``: the findings gate the run is configured with.  A
          threshold above the number of protocol axes is EXTENDED with
          numbered ``open-question-N`` topics rather than truncated, because a
          gate that needs more distinct files than the driver can ever assign
          slugs for is a gate that cannot open.
        - Returns at least ``max(threshold, len(EXPLORE_TOPICS))`` topics, in
          assignment order, with distinct slugs.
        - Never raises; performs no I/O.
    """
    extra = threshold - len(EXPLORE_TOPICS)
    if extra <= 0:
        return EXPLORE_TOPICS
    return (
        *EXPLORE_TOPICS,
        *(
            ExploreTopic(
                slug=f"{_EXTRA_TOPIC_SLUG}-{n}",
                label=f"open question {n}",
                brief=(
                    "a question the topics above left unanswered, and the "
                    "evidence that answers it"
                ),
            )
            for n in range(1, extra + 1)
        ),
    )


def explore_topic(slug: str) -> ExploreTopic:
    """The topic for *slug*, or a synthesised one for a slug not in the table.

    Interface contract (1 call site, ``roles.py``'s assignment prompt block):
        - ``slug``: whatever the driver assigned.  A custom driver may assign a
          slug this module has never heard of; that must render, not raise.
        - Returns the table entry when there is one, else a topic whose label
          is the slug with its hyphens spelled out.
        - Never raises; performs no I/O.
    """
    for topic in EXPLORE_TOPICS:
        if topic.slug == slug:
            return topic
    readable = slug.replace("-", " ").strip() or slug
    return ExploreTopic(slug=slug, label=readable, brief=f"the {readable} of the goal")


def _topic_phrase(threshold: int = Defaults.FINDINGS_THRESHOLD) -> str:
    """Render the topic labels as one English list, for EXPLORE's purpose."""
    labels = [topic.label for topic in explore_topics(threshold)]
    if len(labels) == 1:
        return labels[0]
    return f"{', '.join(labels[:-1])}, and {labels[-1]}"


# ---------------------------------------------------------------------------
# Per-state rules
# ---------------------------------------------------------------------------


# DECISION plan-2026-07-21T191807-bf7ffe24/D-041
# THERE IS NO `extraction_instructions` FIELD HERE, AND ADDING ONE BACK COSTS ONE
# LLM CALL PER TURN. This resolves the predecessor's D-046, which kept six such
# blocks and deferred the "should harness states extract at all" question to a
# live re-run; the re-run happened (step 11) and the answer is no.
#
# The mechanism, from `fsm_llm/pipeline.py::_execute_data_extraction` as it
# stands -- re-read it before changing this, and do not trust the paraphrase:
#   * `has_extraction_instructions = bool(state.extraction_instructions)`;
#   * if there are NO field/classification configs, a truthy value takes the
#     early bulk-fallback branch and issues one `_make_llm_call`;
#   * if there ARE configs, a truthy value takes the ADDITIVE bulk branch after
#     the per-field passes and issues one `_make_llm_call` -- UNCONDITIONALLY.
# The additive branch is gated on `has_extraction_instructions and (configs)`
# ALONE. It is NOT gated on anything being missing, so "make the field configs
# exhaustive so the fallback never triggers" (this plan's own D-008) does not
# work and was measured not to: 2.000 core calls per turn, `data_extraction` on
# 15/15 turns. The ONLY package-local way to make the trigger unreachable is for
# `State.extraction_instructions` to be falsy, which is what the absence of this
# field guarantees for all six states at once.
#
# Nothing is lost by the removal, which is why it is a deletion rather than an
# empty string:
#   * The strings were never rendered into any prompt the harness issues except
#     the bulk-extraction prompt itself. `DataExtractionPromptBuilder` (the only
#     other reader) is constructed but never called on the per-field path, and
#     Pass 2 reads `response_instructions`, not this.
#   * They described a JSON payload the harness does not want from Pass-1 prose
#     anyway: all nine gate flags are driver-owned, seeded before turn 1 and
#     re-asserted by `HarnessAgent._reassert_driver_owned` (D-044), so every
#     value the blocks asked for was inert by construction.
#   * The worker's payload shape is `roles.py`'s `output_schema`; the exit
#     condition is `gate_summary`. Both outlive this field, so restating it here
#     was duplication as well as cost.
# The bulk prompt also told the model "Use descriptive snake_case key names" and
# merged whatever it invented into context, so removing it also closes an
# unbounded key-injection channel into every later prompt.
# `TestBulkExtractionIsUnreachable` derives the trigger from core's own predicate
# over all six states and fails the moment this comes back. See decisions.md
# D-041.
@dataclass(frozen=True)
class StateRules:
    """The frozen protocol rules for one harness state.

    Instances are module-level constants built once at import time and shared
    by the FSM builder (which reads ``description`` / ``purpose`` /
    ``response_instructions``), the driver (``role``, ``operative_rules``) and
    the ownership enforcement in ``storage.py`` (``owned_artifacts``).

    Attributes:
        state: The FSM state id this rule set governs.
        role: The worker role dispatched on entry to ``state``.
        description: Short human-readable state description (FSM ``description``).
        purpose: What must be accomplished here (FSM ``purpose``).
        operative_rules: The protocol rules a worker must follow, at most 8
            single-sentence imperatives.
        response_instructions: Pass-2 instructions for the user-facing report.
        owned_artifacts: Artifacts ``role`` is permitted to write.
        gate_summary: One line describing the exit gate, at default thresholds.
    """

    state: str
    role: str
    description: str
    purpose: str
    operative_rules: tuple[str, ...]
    response_instructions: str
    owned_artifacts: tuple[str, ...]
    gate_summary: str


_EXPLORE = StateRules(
    state=HarnessStates.EXPLORE,
    role=ROLE_BY_STATE[HarnessStates.EXPLORE],
    description="Gather context: read code, index findings, classify constraints.",
    # The coverage axes are READ from `EXPLORE_TOPICS`, not restated here: the
    # driver assigns one `findings/<slug>.md` per dispatch from that same table
    # (D-035), so the topics this sentence promises and the topics actually
    # handed out are one fact.  The rendered string is unchanged from the
    # hand-written one it replaces, and a test pins it.
    purpose=(
        "Build enough grounded context to plan: at least "
        f"{Defaults.FINDINGS_THRESHOLD} indexed findings covering "
        f"{_topic_phrase()}."
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
    response_instructions=(
        "State the current findings count, the topics covered so far, and the "
        "single next question you will investigate. Be terse and factual."
    ),
    owned_artifacts=artifacts_writable_by(ROLE_BY_STATE[HarnessStates.EXPLORE]),
    # DECISION plan-2026-07-21T191807-bf7ffe24/D-015
    # THIS LINE, not an `extraction_instructions` block, is where the count is
    # defined -- and it is defined against the `findings/` files, the ones
    # `OWNERSHIP` grants the explorer, NOT against `findings.md`, the index the
    # operative rule above forbids it to touch. Review C3: while the definition
    # named the index, a rule-compliant explorer had to report 0 forever, so the
    # only way the gate ever opened was a model reporting a number it could not
    # know. Do NOT "restore" the index wording for consistency with the count's
    # NAME: the count is DERIVED from the files on disk
    # (`tools.derive_disk_counts`), and a worker's own integer is advisory.
    # This string is verbatim in the EXPLORE role prompt and was measured there
    # (step 25, 10/10). Do not reword it for tidiness -- an unmeasured prompt
    # edit is not free, whatever it costs to read. See decisions.md D-015.
    gate_summary=(
        f"HARD: leave for PLAN only when {ContextKeys.FINDINGS_COUNT} >= "
        f"{Defaults.FINDINGS_THRESHOLD}, counted from the "
        f"{ArtifactNames.FINDINGS_DIR}/ files actually on disk."
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
    response_instructions=(
        "Present the plan for approval: the steps, the success criteria, the "
        "verification strategy, the failure modes and the assumptions. End by "
        "asking for an explicit approve or revise decision."
    ),
    owned_artifacts=artifacts_writable_by(ROLE_BY_STATE[HarnessStates.PLAN]),
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
    response_instructions=(
        "Report the step number and description, the files created or "
        "modified, the commit, any surprises, and the next step in one line "
        "each. No narrative."
    ),
    owned_artifacts=artifacts_writable_by(ROLE_BY_STATE[HarnessStates.EXECUTE]),
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
        "Run every check in the Verification Strategy and REPORT PASS or FAIL "
        f"with the evidence; you write nothing -- the driver merges your reply "
        f"into {ArtifactNames.VERIFICATION}.",
        "Re-run tests that passed before this iteration; any regression "
        "blocks closing.",
        "Compare the files actually changed against the planned file list and "
        "justify or revert the drift.",
        "Report what you did not verify and why; absence of evidence is "
        "not evidence of absence.",
        "On failure, report the immediate cause, the contributing factor, the "
        "defense that should have caught it, and the prevention.",
        "Run the simplification checks against the written criteria, not "
        "against memory.",
        "Present completed work, remaining work, verification results, issues "
        "and a recommendation -- and never close without user confirmation.",
    ),
    response_instructions=(
        "Report five things in order: what was completed, what remains, the "
        "verification results, the issues found, and one recommendation of "
        "close, pivot, explore or execute. Then wait for confirmation."
    ),
    owned_artifacts=artifacts_writable_by(ROLE_BY_STATE[HarnessStates.REFLECT]),
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
        f"Correct stale {ArtifactNames.FINDINGS_DIR}/ files in place with "
        "[CORRECTED iter-N]; append, never delete.",
        f"REPORT the pivot for {ArtifactNames.DECISIONS} with a complexity "
        "assessment and an 'X at the cost of Y' trade-off; the driver logs it.",
        "Do not touch the fix-attempt counter; the driver resets it so the "
        "leash does not carry into the next EXECUTE.",
        "Offer one to three candidate directions, each framed as a trade-off, "
        "and name the checkpoints available.",
        "Get an explicit direction from the user before returning to PLAN.",
    ),
    response_instructions=(
        "Report the pivot reason, the available checkpoints, the ghost "
        "constraints surfaced, and one to three candidate directions framed "
        "as trade-offs. Ask which direction to take."
    ),
    owned_artifacts=artifacts_writable_by(ROLE_BY_STATE[HarnessStates.PIVOT]),
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
    response_instructions=(
        "Report the outcome, the iterations used, the key decisions, the "
        "files changed and the lessons recorded. One line each."
    ),
    owned_artifacts=artifacts_writable_by(ROLE_BY_STATE[HarnessStates.CLOSE]),
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
