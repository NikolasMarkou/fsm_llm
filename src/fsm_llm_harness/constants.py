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
from typing import ClassVar

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

    START_DISPATCH = 10  # Dispatch into the initial state before anything else
    PRE_STEP_GATE = 50  # Cheap HARD gate, runs before an EXECUTE dispatch
    STATE_DISPATCH = 100  # One worker dispatch per state entry
    END_CONVERSATION = 200  # Finalize artifacts on conversation end
    ERROR = 200  # Record errors


class HandlerNames:
    """Handler names for registration."""

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
