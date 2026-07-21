"""
Constants for the fsm_llm_harness package.

Every string literal, threshold and magic value used by the harness protocol
is consolidated here.  Values are frozen data only -- no dynamic construction,
no runtime mutation -- so that the FSM definition, the validator and the
artifact serializers all read the protocol's numbers from a single place.
"""

from __future__ import annotations

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
        ContextKeys.CRITERIA_PASS_COUNT: 0,
        ContextKeys.CRITERIA_TOTAL: 0,
    }
)

#: Driver-owned keys whose default is ABSENT rather than falsy.
#:
#: These three are free prose the run REPORTS (``halt_reason`` is the string the
#: user is shown as the outcome), not gate variables, and "no halt reason yet"
#: is meaningfully different from an empty one. They are guarded anyway: CLOSE
#: has no transitions, so its extraction takes the bulk-fallback branch and an
#: unguarded ``halt_reason`` would be LLM-authored in production (review N4).
DRIVER_OWNED_UNSET: tuple[str, ...] = (
    ContextKeys.PIVOT_REASON,
    ContextKeys.HALT_REASON,
    ContextKeys.LAST_GATE_SLUG,
)


# ---------------------------------------------------------------------------
# Gate slugs and issue severities
# ---------------------------------------------------------------------------


class GateSlug:
    """The 4 pre-step-gate HARD failure slugs.

    ``ORDER`` is the authoritative evaluation order -- ``pre_step_gate()``
    checks each in turn and short-circuits on the first failure.
    """

    NO_PLAN = "no-plan"
    WRONG_STATE = "wrong-state"
    LEASH_CAP = "leash-cap"
    ITERATION_CAP = "iteration-cap"

    ORDER: tuple[str, ...] = (NO_PLAN, WRONG_STATE, LEASH_CAP, ITERATION_CAP)


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
    #: The autonomy leash: the 3rd fix attempt is HARD-blocked.
    MAX_FIX_ATTEMPTS = 2
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
