"""
Constants for the fsm_llm_harness package.

Every string literal, threshold and magic value used by the harness protocol
is consolidated here.  Values are frozen data only -- no dynamic construction,
no runtime mutation -- so that the FSM definition, the validator and the
artifact serializers all read the protocol's numbers from a single place.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, ClassVar

from fsm_llm.constants import DEFAULT_LLM_MODEL, ENV_LLM_MODEL

# ---------------------------------------------------------------------------
# Protocol states
# ---------------------------------------------------------------------------


class HarnessStates:
    """The 6 states of the iterative-planner protocol FSM.

    Attribute names are the protocol's uppercase phase names; values are the
    lowercase snake ids used as FSM state ids.
    """

    EXPLORE = "explore"
    PLAN = "plan"
    EXECUTE = "execute"
    REFLECT = "reflect"
    PIVOT = "pivot"
    CLOSE = "close"

    #: Every state id, in protocol order.  Consumed by the FSM builder and by
    #: the state.md serializer; keeps the state set defined exactly once.
    ALL: tuple[str, ...] = (EXPLORE, PLAN, EXECUTE, REFLECT, PIVOT, CLOSE)

    #: The state a fresh run starts in.
    INITIAL = EXPLORE

    #: The only terminal state.
    TERMINAL = CLOSE


# ---------------------------------------------------------------------------
# Roles
# ---------------------------------------------------------------------------


class Role:
    """The 7 worker roles the driver can dispatch.

    ``ORCHESTRATOR`` is the driver itself (it is never dispatched as a worker);
    the other 6 are dispatched through the ``worker_factory`` seam.
    """

    ORCHESTRATOR = "orchestrator"
    EXPLORER = "explorer"
    PLAN_WRITER = "plan-writer"
    EXECUTOR = "executor"
    VERIFIER = "verifier"
    REVIEWER = "reviewer"
    ARCHIVIST = "archivist"

    #: The 6 dispatchable worker roles (everything except the driver itself).
    WORKERS: tuple[str, ...] = (
        EXPLORER,
        PLAN_WRITER,
        EXECUTOR,
        VERIFIER,
        REVIEWER,
        ARCHIVIST,
    )


# ---------------------------------------------------------------------------
# Context keys
# ---------------------------------------------------------------------------


class ContextKeys:
    """FSM context keys used by the harness.

    Every key referenced by a JsonLogic ``TransitionCondition.logic`` term
    lives here, so a gate can never be written against an undeclared variable.
    No key uses an internal prefix (``_``, ``system_``, ``internal_``, ``__``)
    -- those are stripped by ``fsm_llm.context.clean_context_keys``.
    """

    # --- Task ---------------------------------------------------------
    GOAL = "goal"

    # --- Gate variables (referenced by TransitionCondition.logic) ------
    # EXPLORE -> PLAN
    FINDINGS_COUNT = "findings_count"
    # PLAN -> EXECUTE
    PLAN_APPROVED = "plan_approved"
    ITERATION = "iteration"
    # REFLECT -> CLOSE
    CLOSE_CONFIRMED = "close_confirmed"
    ALL_CRITERIA_PASS = "all_criteria_pass"
    # EXECUTE continue edge
    FIX_ATTEMPTS = "fix_attempts"
    #: Human leash-continues already granted on the CURRENT plan step.  Not a
    #: gate variable -- it bounds how many times the human may be asked, which
    #: is what stops an approving callback from making the leash unbounded.
    LEASH_GRANTS = "leash_grants"

    # --- Non-gated edge selectors --------------------------------------
    NEEDS_EXPLORE = "needs_explore"
    NEEDS_PIVOT = "needs_pivot"
    COMPLETION_FIX = "completion_fix"
    EXECUTE_COMPLETE = "execute_complete"
    PIVOT_RESOLVED = "pivot_resolved"

    # --- Step counters --------------------------------------------------
    STEP_NUMBER = "step_number"
    TOTAL_STEPS = "total_steps"

    # --- Verification rollup --------------------------------------------
    CRITERIA_PASS_COUNT = "criteria_pass_count"
    CRITERIA_TOTAL = "criteria_total"

    # --- Role dispatch ---------------------------------------------------
    CURRENT_ROLE = "current_role"
    CURRENT_ROLE_RESULT = "current_role_result"
    ROLE_RESULTS = "role_results"
    DISPATCH_LEDGER = "dispatch_ledger"

    # --- Protocol bookkeeping --------------------------------------------
    PIVOT_REASON = "pivot_reason"
    LAST_GATE_SLUG = "last_gate_slug"
    HALT_REASON = "halt_reason"

    # --- Filesystem-as-memory roots ---------------------------------------
    PLAN_DIR = "plan_dir"
    WORKSPACE_ROOT = "workspace_root"


# ---------------------------------------------------------------------------
# Driver-owned context (invariant I6 / I8 -- writer provenance)
# ---------------------------------------------------------------------------

# DECISION plan-2026-07-21T125237-191b2eb2/D-044
# These two tables are the ENTIRE writer-provenance boundary. Every key named
# here is written by the DRIVER (or, where `harness._WORKER_WRITABLE` allows it,
# by a worker through the driver's exact-type allowlist) and by NOTHING ELSE.
# In particular the FSM's own Pass-1 field extraction must never write one.
#
# Do NOT "simplify" this by deleting a key because "the LLM would never say
# that". Measured 2026-07-21 (findings/review-iter-1.md C1): with the harness
# FSM as shipped before this table existed, an LLM that emitted
# {"plan_approved": true, "close_confirmed": true, ...} drove a full
# EXPLORE -> PLAN -> EXECUTE -> REFLECT -> CLOSE traverse while EVERY worker
# dispatch failed and a DENYING approval callback was never consulted once.
# The mechanism is core's, not ours: `_build_field_configs_from_state`
# (pipeline.py:837-890) mints a required FieldExtractionConfig for every key
# named in a transition condition's `requires_context_keys`, and
# `_execute_data_extraction` skips only fields already non-None
# (pipeline.py:953-956). A key missing from SEEDS below is therefore a key the
# LLM is asked to invent on every turn.
#
# Enforcement is THREE mechanisms, all in harness.py. PRESENCE is what enforces;
# the other two exist to keep these keys present:
#   1. SEEDS is written into context before turn 1, so extraction skips every one
#      of these fields and they never enter `extracted_data` at all. This is the
#      enforcement, and it is also free: it removes ~9 extraction calls per turn.
#   2. `HarnessAgent._apply` coerces a `None` (delete) delta on a SEEDED key back
#      to its seed. A deleted key is an extractable key.
#   3. `HarnessAgent._reassert_driver_owned` runs at PRE_PROCESSING (before
#      extraction -- restores a key some foreign writer removed) and at
#      CONTEXT_UPDATE (after the commit -- keeps a fabricated value out of later
#      turns' prompts and out of `final_context`).
# Do NOT reduce this to "just the CONTEXT_UPDATE handler", which is what the
# review proposed as an equal-coverage alternative to seeding. MEASURED
# 2026-07-21: `MessagePipeline` hands the transition evaluator a SECOND copy of
# the extraction payload (`evaluate_transitions(state, context,
# extraction_response.extracted_data)`, pipeline.py:1388) and
# `_prepare_working_context` merges that dict OVER the live context
# (transition_evaluator.py:146). A handler cleans `instance.context` only, so a
# value extracted on the SAME turn still opens the gate -- reproduced: the guard
# deleted `plan_approved` and PLAN -> EXECUTE fired regardless. No handler at any
# timing can reach that dict; the only defence is never being asked.
# See decisions.md D-044.

#: Driver-owned key -> the falsy value seeded before turn 1.
#:
#: The first nine are the gate FLAGS of review C1; the rest are the counters
#: and rollups. Every value is falsy, so seeding cannot open a gate: the
#: JsonLogic terms all test `== True` or a `>=` / `<` bound.
DRIVER_OWNED_SEEDS: Mapping[str, Any] = MappingProxyType(
    {
        ContextKeys.FINDINGS_COUNT: 0,
        ContextKeys.PLAN_APPROVED: False,
        ContextKeys.CLOSE_CONFIRMED: False,
        ContextKeys.ALL_CRITERIA_PASS: False,
        ContextKeys.EXECUTE_COMPLETE: False,
        ContextKeys.COMPLETION_FIX: False,
        ContextKeys.NEEDS_PIVOT: False,
        ContextKeys.NEEDS_EXPLORE: False,
        ContextKeys.PIVOT_RESOLVED: False,
        ContextKeys.ITERATION: 0,
        ContextKeys.STEP_NUMBER: 0,
        ContextKeys.TOTAL_STEPS: 1,
        ContextKeys.FIX_ATTEMPTS: 0,
        ContextKeys.LEASH_GRANTS: 0,
        ContextKeys.CRITERIA_PASS_COUNT: 0,
        ContextKeys.CRITERIA_TOTAL: 0,
    }
)

#: Driver-owned keys whose default is ABSENT rather than falsy.
#:
#: The first three are free prose the run REPORTS (``halt_reason`` is the string
#: the user is shown as the outcome), not gate variables, and "no halt reason
#: yet" is meaningfully different from an empty one. They are guarded anyway:
#: CLOSE has no transitions, so its extraction takes the bulk-fallback branch
#: and an unguarded ``halt_reason`` would be LLM-authored in production
#: (review N4).
#:
#: The last two are the filesystem-as-memory ROOTS.  They have no fixed falsy
#: value (they are per-run paths supplied by the caller through
#: ``run(initial_context=...)``), so they cannot live in ``DRIVER_OWNED_SEEDS``
#: -- but they must be driver-owned all the same: ``plan_dir`` selects the
#: directory a role's write tools are confined to, and an LLM-invented value
#: would point the protocol's own memory somewhere else.  ``run()`` adopts the
#: caller's values into the driver-owned table; absent means "no plan tools".
DRIVER_OWNED_UNSET: tuple[str, ...] = (
    ContextKeys.PIVOT_REASON,
    ContextKeys.HALT_REASON,
    ContextKeys.LAST_GATE_SLUG,
    ContextKeys.PLAN_DIR,
    ContextKeys.WORKSPACE_ROOT,
)


# ---------------------------------------------------------------------------
# Gate slugs and issue severities
# ---------------------------------------------------------------------------


class GateSlug:
    """Slugs recorded in ``ContextKeys.LAST_GATE_SLUG``.

    ``ORDER`` is the 4 PRE-STEP-GATE slugs, in their authoritative evaluation
    order -- ``pre_step_gate()`` checks each in turn and short-circuits on the
    first failure.  :data:`EXPLORE_CAP` is deliberately NOT in ``ORDER``: it is
    a driver halt on the EXPLORE -> PLAN edge, not a pre-EXECUTE-step check, and
    ``plan_validator`` iterates ``ORDER`` to decide what it may report.
    """

    NO_PLAN = "no-plan"
    WRONG_STATE = "wrong-state"
    LEASH_CAP = "leash-cap"
    ITERATION_CAP = "iteration-cap"

    ORDER: tuple[str, ...] = (NO_PLAN, WRONG_STATE, LEASH_CAP, ITERATION_CAP)

    #: The bounded EXPLORE re-dispatch budget is spent and the findings gate is
    #: still BLOCKED (D-029).  Never means "proceed anyway".
    EXPLORE_CAP = "explore-cap"

    #: The bounded PLAN re-dispatch budget is spent and the plan-writer has
    #: still not returned a successful reply (D-001 of
    #: plan-2026-07-22T184813-6549c7cb).  Never means "proceed anyway".  Like
    #: :data:`EXPLORE_CAP`, deliberately NOT in ``ORDER``: it is a driver halt
    #: on the PLAN -> EXECUTE edge, not a pre-EXECUTE-step check.
    PLAN_CAP = "plan-cap"


class Severity:
    """Severity levels for advisory ``audit()`` issues.

    ``pre_step_gate()`` failures are never issues -- they carry ``hard=True``
    and are structurally distinct from anything tagged here.
    """

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"

    ORDER: tuple[str, ...] = (ERROR, WARNING, INFO)


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


class HandlerPriorities:
    """Explicit priorities for harness handler execution order.

    Lower numbers execute first.  When multiple handlers share the same timing
    hook, priority determines their order.
    """

    EXTRACTION_GUARD = 5  # Revert driver-owned keys the LLM's extraction wrote
    START_DISPATCH = 10  # Dispatch into the initial state before anything else
    PRE_STEP_GATE = 50  # Cheap HARD gate, runs before an EXECUTE dispatch
    STATE_DISPATCH = 100  # One worker dispatch per state entry
    END_CONVERSATION = 200  # Finalize artifacts on conversation end
    ERROR = 200  # Record errors


class HandlerNames:
    """Handler names for registration."""

    EXTRACTION_GUARD = "HarnessExtractionGuard"
    START_DISPATCH = "HarnessStartDispatch"
    EXPLORE_DISPATCH = "HarnessExploreDispatch"
    PLAN_DISPATCH = "HarnessPlanDispatch"
    EXECUTE_DISPATCH = "HarnessExecuteDispatch"
    REFLECT_DISPATCH = "HarnessReflectDispatch"
    PIVOT_DISPATCH = "HarnessPivotDispatch"
    CLOSE_DISPATCH = "HarnessCloseDispatch"
    PRE_STEP_GATE = "HarnessPreStepGate"
    END_CONVERSATION = "HarnessEndConversation"
    ERROR = "HarnessErrorHandler"

    #: state id -> dispatch handler name.  The driver registers exactly one
    #: dispatch handler per state; this map is the single source of that pairing.
    BY_STATE: ClassVar[Mapping[str, str]] = MappingProxyType(
        {
            HarnessStates.EXPLORE: EXPLORE_DISPATCH,
            HarnessStates.PLAN: PLAN_DISPATCH,
            HarnessStates.EXECUTE: EXECUTE_DISPATCH,
            HarnessStates.REFLECT: REFLECT_DISPATCH,
            HarnessStates.PIVOT: PIVOT_DISPATCH,
            HarnessStates.CLOSE: CLOSE_DISPATCH,
        }
    )


# ---------------------------------------------------------------------------
# On-disk artifact names
# ---------------------------------------------------------------------------


class ArtifactNames:
    """Filenames and directory names of the filesystem-as-memory tier.

    Per-plan artifacts live directly under the plan directory; ``FINDINGS_DIR``
    and ``CHECKPOINTS_DIR`` are its two subdirectories.  The cross-plan tier
    lives one level up, beside the plan directories.
    """

    # Per-plan artifacts
    STATE = "state.md"
    PLAN = "plan.md"
    DECISIONS = "decisions.md"
    FINDINGS_INDEX = "findings.md"
    PROGRESS = "progress.md"
    VERIFICATION = "verification.md"
    CHANGELOG = "changelog.md"
    SUMMARY = "summary.md"

    # Per-plan subdirectories
    FINDINGS_DIR = "findings"
    CHECKPOINTS_DIR = "checkpoints"

    # Cross-plan tier (one level above the plan directory)
    CROSS_FINDINGS = "FINDINGS.md"
    CROSS_DECISIONS = "DECISIONS.md"
    LESSONS = "LESSONS.md"
    LESSONS_ARCHIVE = "LESSONS-archive.md"
    SYSTEM = "SYSTEM.md"
    INDEX = "INDEX.md"

    #: The cross-plan files the protocol reads and rewrites at CLOSE.
    #: ``LESSONS_ARCHIVE`` is deliberately excluded -- it is append-only and
    #: never read back by the protocol.
    CROSS_PLAN: tuple[str, ...] = (
        CROSS_FINDINGS,
        CROSS_DECISIONS,
        LESSONS,
        SYSTEM,
        INDEX,
    )

    #: Every per-plan artifact the driver bootstraps for a fresh plan directory.
    PER_PLAN: tuple[str, ...] = (
        STATE,
        PLAN,
        DECISIONS,
        FINDINGS_INDEX,
        PROGRESS,
        VERIFICATION,
        CHANGELOG,
    )


def _plan_section_slug(title: str) -> str:
    """Derive a stable pydantic-identifier slug from a ``plan.md`` section title.

    Deterministic and TOTAL over :data:`PlanSchema.SECTIONS`: lowercase, ``&`` ->
    ``and``, every run of non-alphanumeric characters -> one ``_``, then strip
    leading/trailing ``_``.  The result is a valid Python identifier used as a
    ``response_format`` field name for the PLAN role's structured plan schema;
    :data:`PlanSchema.SECTION_BY_SLUG` recovers the exact title for rendering.

    Interface contract (2 call sites: this module's ``PlanSchema`` map builders,
    and ``roles._schema_fields`` via ``PlanSchema.SECTION_SLUGS``):
        - Parameter: a section title string.
        - Returns a lowercase snake_case identifier; never empty for the 11
          real titles (a title with no alphanumerics would return ``""`` -- not
          possible for SECTIONS, and the uniqueness invariant below would catch
          a regression).
        - Pure and deterministic; never raises.
    """
    lowered = title.lower().replace("&", " and ")
    return re.sub(r"[^a-z0-9]+", "_", lowered).strip("_")


class PlanSchema:
    """Structural rules for ``plan.md``."""

    #: The 11 required ``## `` headings, in the exact order the validator
    #: expects them.  Order is load-bearing: the section check is positional.
    SECTIONS: tuple[str, ...] = (
        "Goal",
        "Problem Statement",
        "Context",
        "Files To Modify",
        "Steps",
        "Assumptions",
        "Failure Modes",
        "Pre-Mortem & Falsification Signals",
        "Success Criteria",
        "Verification Strategy",
        "Complexity Budget",
    )

    #: The 11 section slugs, in SECTIONS order.  DERIVED via
    #: :func:`_plan_section_slug` (never hand-written) so the PLAN
    #: ``response_format`` field set and the renderer's section order are one
    #: fact.  Order is load-bearing, mirroring SECTIONS.
    SECTION_SLUGS: ClassVar[tuple[str, ...]] = tuple(
        _plan_section_slug(title) for title in SECTIONS
    )

    #: Section title -> its snake_case field slug.  Bidirectional with
    #: :data:`SECTION_BY_SLUG`.  Invariant (pinned by unit test): keys ==
    #: SECTIONS and all slugs are unique.
    SLUG_BY_SECTION: ClassVar[Mapping[str, str]] = MappingProxyType(
        {title: _plan_section_slug(title) for title in SECTIONS}
    )

    #: Field slug -> its section title.  The renderer (Step 2) reindexes the 11
    #: structured fields back into SECTIONS order/titles through this map.
    SECTION_BY_SLUG: ClassVar[Mapping[str, str]] = MappingProxyType(
        {_plan_section_slug(title): title for title in SECTIONS}
    )


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


class Defaults:
    """Default configuration values and protocol thresholds."""

    # --- LLM / agent loop ------------------------------------------------
    MODEL = DEFAULT_LLM_MODEL
    TEMPERATURE = 0.3
    MAX_TOKENS = 2000
    #: Hard bound on driver turns; passed as ``BaseAgent(max_iterations=...)``.
    #: Sized for 6 protocol states x the 6-iteration cap plus slack.
    MAX_TURNS = 60
    #: Whole-run budget for the driver.
    TIMEOUT_SECONDS = 1800.0
    #: Per-LLM-call budget inside a role dispatch.
    LLM_TIMEOUT_SECONDS = 120.0
    #: Message sent to advance the FSM conversation loop.
    CONTINUE_MESSAGE = "Continue."

    # --- Harness-level retry (LiteLLMInterface(retries=) is a no-op for
    #     ollama_chat/*, so retry lives here instead) ----------------------
    RETRY_ATTEMPTS = 3
    RETRY_BASE_DELAY = 1.0
    RETRY_MAX_DELAY = 30.0
    RETRY_BACKOFF_FACTOR = 2.0

    # --- Protocol gates ---------------------------------------------------
    #: EXPLORE -> PLAN requires at least this many indexed findings.
    FINDINGS_THRESHOLD = 3
    # DECISION plan-2026-07-21T191807-bf7ffe24/D-031
    #: EXTRA explorer dispatches the driver may spend, per RUN, while the
    #: findings gate is still BLOCKED.  Total EXPLORE dispatches per run are
    #: bounded by (genuine state entries) + this number.
    #:
    #: This number is sized from a MEASURED YIELD HORIZON, not from a rate, and
    #: NOT by extrapolating one.  The distinction is the whole content of this
    #: comment, because the extrapolation was tried and measured wrong:
    #:   * step 23 pooled 9 distinct files over 26 re-dispatches (0.35
    #:     files/dispatch) and this bound was set to 17 on that average, i.e.
    #:     18 total dispatches for an expected 6.3 files against a threshold
    #:     of 3.  MEASURED at that bound (step 24, n=10, `:4b`): the pooled rate
    #:     fell to 0.14 (19 files / 136 dispatches) and runs reaching 3 went
    #:     4/10.  Yield is NOT linear in dispatches.
    #:   * What the per-dispatch traces show instead is a HORIZON.  Every run
    #:     that ever reached 3 did so by its **9th** dispatch (measured: 9, 8,
    #:     6, 5).  Every run that had not got going by then stayed exactly where
    #:     it was: six runs spent dispatches 10-18 -- about 60 dispatches -- and
    #:     added ZERO new files between them.
    #:   * So the bound covers the horizon plus one: 9 + 1 = 10 total dispatches
    #:     = 1 entry dispatch + **9** extra.
    #:
    #: Do NOT raise it past ~12 total.  It is not a dial that trades wall clock
    #: for findings: measured, dispatches beyond the horizon cost ~30-40 s each
    #: on `:4b` and return nothing.  17 was measured and is strictly worse than
    #: this value -- same outcome, ~470 s per blocked run instead of ~300 s.
    #: Do NOT put it back to 5 either (6 total dispatches would have missed two
    #: of the four measured successes, which took 8 and 9), and do NOT set it to
    #: 0: that restores the one-dispatch shape four steps of this plan measured
    #: at 0/10 on criterion (b) (decisions.md D-022, D-027).
    #: See decisions.md D-031.
    MAX_EXPLORE_REDISPATCHES = 9
    # DECISION plan-2026-07-22T184813-6549c7cb/D-001
    #: EXTRA plan-writer dispatches the driver may spend, per RUN, after a
    #: FAILED (or empty) plan-writer reply.  Total PLAN dispatches per
    #: (iteration, step) are bounded by 1 + this number.
    #:
    #: Before this bound existed, PLAN dispatched exactly ONCE per
    #: (iteration, step): `_after_plan_dispatch` returned `{}` on a failed
    #: reply, nothing re-opened the dispatch key, and the eventual halt was the
    #: stall detector's -- which always raises `slug=None`.  MEASURED (L6 B0
    #: run 3, `scripts/bench_data/l6-e2e/rows.jsonl`, `:4b`): one empty
    #: plan-writer reply was terminal, and the run recorded `plan_md_bytes: 0`
    #: with `halt_slug: null` -- the slugless stall shape the L6 floor exists
    #: to catch.
    #:
    #: 3 is an UNMEASURED PLACEHOLDER (n=1 stall observation), NOT a measured
    #: horizon like MAX_EXPLORE_REDISPATCHES above.  It was chosen because
    #: PLAN's task is ONE structured write, not EXPLORE's multi-topic
    #: discovery (whose 9 is a yield horizon for a different task shape), and
    #: 4 total dispatches at the observed ~100-170 s each is enough to
    #: separate "fails once cold" from "persistently fails" without
    #: quadrupling worst-case PLAN wall clock.  Do NOT tune it without a
    #: dedicated bench: `plan-cap` rows with zero plan.md bytes across all 4
    #: dispatches would mean the budget is not the lever at all.  See
    #: decisions.md D-001 (plan-2026-07-22T184813-6549c7cb).
    MAX_PLAN_REDISPATCHES = 3
    #: The autonomy leash: the 3rd fix attempt is HARD-blocked.
    MAX_FIX_ATTEMPTS = 2
    # DECISION plan-2026-07-21T125237-191b2eb2/D-052
    #: Human leash-continues granted on ONE plan step before the driver stops
    #: asking.  Do NOT raise this to "let the user decide when to stop": the
    #: approval callback is the thing being bounded.  Measured 2026-07-21
    #: (findings/review-iter-1.md C3b): with an always-approving callback and an
    #: always-failing executor, `_after_reflect_dispatch` reset `fix_attempts`
    #: to 0 on EVERY grant, so the run cycled EXECUTE <-> REFLECT until
    #: `BudgetExhaustedError` -- the leash was decorative. With this cap the
    #: executor dispatches on one plan step are bounded by
    #: MAX_FIX_ATTEMPTS * (1 + MAX_LEASH_GRANTS) = 6 for ANY sequence of
    #: approvals.  See decisions.md D-052.
    MAX_LEASH_GRANTS = 2
    #: PLAN -> EXECUTE is blocked at or above this iteration count.
    ITERATION_HARD_CAP = 6
    #: Iteration at which a decomposition analysis is advised.
    ITERATION_WARN = 5

    # --- Retrospective audit tiers (deliberately NOT the gate thresholds:
    #     2 attempts is legal, 3 means the gate was passed, 4+ bypassed) ----
    LEASH_AUDIT_WARN_ATTEMPTS = 3
    LEASH_AUDIT_ERROR_ATTEMPTS = 4

    # --- Line caps and compression thresholds ------------------------------
    LESSONS_LINE_CAP = 200
    SYSTEM_LINE_CAP = 300
    DECISIONS_COMPRESS_LINES = 300
    CHANGELOG_COMPRESS_LINES = 200
    CONSOLIDATED_COMPRESS_LINES = 500
    #: Number of most-recent plans kept in the cross-plan sliding window.
    SLIDING_WINDOW_PLANS = 4
    #: Marker delimiting a compressed region in a cross-plan file.
    COMPRESSED_SUMMARY_MARKER = "<!-- COMPRESSED-SUMMARY -->"

    # --- LESSONS.md importance tags (`[I:N]`) -------------------------------
    LESSONS_IMPORTANCE_MIN = 1
    LESSONS_IMPORTANCE_MAX = 5
    #: Lines at this importance are never evicted, even over the line cap.
    LESSONS_PROTECTED_IMPORTANCE = 5

    # --- Environment variables ---------------------------------------------
    ENV_MODEL = ENV_LLM_MODEL
    ENV_LIVE_TESTS = "FSM_LLM_HARNESS_LIVE"
