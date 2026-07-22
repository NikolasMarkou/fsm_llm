"""
FSM-LLM Harness
===============

An FSM-LLM-native emulation of the iterative-planner protocol: a 6-state
EXPLORE / PLAN / EXECUTE / REFLECT / PIVOT / CLOSE machine with mechanically
enforced gates, filesystem-as-memory artifacts, a 2-attempt autonomy leash and
a small-model hardening layer.

The protocol's hard gates are JsonLogic ``TransitionCondition`` terms evaluated
by the core ``TransitionEvaluator``, so a transition is either DETERMINISTIC or
BLOCKED -- never an LLM judgement call on a gated edge.

Install:
    pip install fsm-llm[harness]
"""

from __future__ import annotations

# Version info — imported via __version__.py to stay in sync
from .__version__ import __version__
from .artifacts import (
    ARTIFACT_MODELS,
    DECISION_ENTRY_SCHEMAS,
    MANDATORY_ADDITIONAL_CHECKS,
    PRESENTATION_CONTRACTS,
    REJECTED_EVIDENCE,
    VERDICT_BULLETS,
    VERDICT_RECOMMENDATIONS,
    Artifact,
    ChangelogDoc,
    ChangelogEntry,
    ChecklistItem,
    CheckpointDoc,
    ConsolidatedDoc,
    CriterionRow,
    DecisionEntry,
    DecisionsDoc,
    FindingsIndexDoc,
    FindingsTopicDoc,
    IndexDoc,
    IndexRow,
    LessonsDoc,
    PlanDoc,
    PlanStep,
    PresentationContract,
    ProgressDoc,
    Section,
    SectionedArtifact,
    StateDoc,
    SummaryDoc,
    SystemAtlasDoc,
    VerificationDoc,
    compression_marker_issues,
    evidence_is_acceptable,
    lesson_importance,
    missing_entry_fields,
    missing_floor_fields,
    parse_changelog_line,
    parse_markdown_table,
)
from .constants import (
    ArtifactNames,
    ContextKeys,
    Defaults,
    GateSlug,
    HandlerNames,
    HandlerPriorities,
    HarnessStates,
    PlanSchema,
    Role,
    Severity,
)
from .exceptions import (
    HarnessArtifactError,
    HarnessConfinementError,
    HarnessError,
    HarnessOwnershipError,
    HarnessReentrancyError,
)
from .fsm_definition import DEFAULT_PERSONA, build_harness_fsm
from .hardening import (
    RETRYABLE_EXCEPTIONS,
    RoleOutput,
    as_int,
    coerce_worker_output,
    parse_json_payload,
    parse_role_output,
    retry,
    strip_model_noise,
    type_matches,
)
from .harness import (
    HarnessAgent,
    Presentation,
    RevertCallback,
    RevertDirective,
    RoleRequest,
    WorkerFactory,
)
from .plan_validator import CHECKS, GateResult, Issue, audit, pre_step_gate
from .roles import (
    ROLE_SPECS,
    AgentBuilder,
    RoleSpec,
    build_default_worker_factory,
    build_role_prompt,
    build_role_system_prompt,
    build_role_task_prompt,
    count_top_level_json_objects,
    get_role_spec,
    held_tools,
)
from .rules import (
    OWNERSHIP,
    ROLE_BY_STATE,
    RULES,
    StateRules,
    artifacts_writable_by,
    get_rules,
)
from .storage import (
    COMPRESSED_SUMMARY_CLOSE,
    COMPRESSED_SUMMARY_OPEN,
    COMPRESSED_SUMMARY_SECTION,
    DRIVER_READ_MAX_BYTES,
    PLAN_ID_RE,
    CapReport,
    PlanDirectory,
    RunState,
    WindowReport,
    apply_sliding_window,
    check_system_cap,
    evict_lessons,
    mint_plan_id,
)
from .tools import (
    COMMAND_ALLOWLIST,
    PLAN_READ_TOOLS,
    PLAN_WRITE_TOOLS,
    READ_ONLY_TOOLS,
    SHELL_TOOLS,
    VERIFICATION_COMMANDS,
    WRITE_TOOLS,
    PlanMemory,
    PlanTools,
    Workspace,
    WorkspaceTools,
    build_plan_tools,
    build_workspace_tools,
)

__all__ = [
    # Version
    "__version__",
    # Constants
    "ArtifactNames",
    "ContextKeys",
    "Defaults",
    "GateSlug",
    "HandlerNames",
    "HandlerPriorities",
    "HarnessStates",
    "PlanSchema",
    "Role",
    "Severity",
    # Exceptions
    "HarnessArtifactError",
    "HarnessConfinementError",
    "HarnessError",
    "HarnessOwnershipError",
    "HarnessReentrancyError",
    # FSM definition
    "DEFAULT_PERSONA",
    "build_harness_fsm",
    # Protocol artifacts: models, serializers and the contract tables
    "ARTIFACT_MODELS",
    "DECISION_ENTRY_SCHEMAS",
    "MANDATORY_ADDITIONAL_CHECKS",
    "PRESENTATION_CONTRACTS",
    "REJECTED_EVIDENCE",
    "VERDICT_BULLETS",
    "VERDICT_RECOMMENDATIONS",
    "Artifact",
    "ChangelogDoc",
    "ChangelogEntry",
    "CheckpointDoc",
    "ChecklistItem",
    "ConsolidatedDoc",
    "CriterionRow",
    "DecisionEntry",
    "DecisionsDoc",
    "FindingsIndexDoc",
    "FindingsTopicDoc",
    "IndexDoc",
    "IndexRow",
    "LessonsDoc",
    "PlanDoc",
    "PlanStep",
    "PresentationContract",
    "ProgressDoc",
    "Section",
    "SectionedArtifact",
    "StateDoc",
    "SummaryDoc",
    "SystemAtlasDoc",
    "VerificationDoc",
    "compression_marker_issues",
    "evidence_is_acceptable",
    "lesson_importance",
    "missing_entry_fields",
    "missing_floor_fields",
    "parse_changelog_line",
    "parse_markdown_table",
    # The plan directory: layout, atomic writes, caps and the sliding window
    "COMPRESSED_SUMMARY_CLOSE",
    "COMPRESSED_SUMMARY_OPEN",
    "COMPRESSED_SUMMARY_SECTION",
    "DRIVER_READ_MAX_BYTES",
    "PLAN_ID_RE",
    "CapReport",
    "PlanDirectory",
    "RunState",
    "WindowReport",
    "apply_sliding_window",
    "check_system_cap",
    "evict_lessons",
    "mint_plan_id",
    # Pre-step gate and retrospective audit
    "CHECKS",
    "GateResult",
    "Issue",
    "audit",
    "pre_step_gate",
    # Driver
    "HarnessAgent",
    "Presentation",
    "RevertCallback",
    "RevertDirective",
    "RoleRequest",
    "WorkerFactory",
    # Small-model hardening
    "RETRYABLE_EXCEPTIONS",
    "RoleOutput",
    "as_int",
    "coerce_worker_output",
    "parse_json_payload",
    "parse_role_output",
    "retry",
    "strip_model_noise",
    "type_matches",
    # Per-state rules
    "OWNERSHIP",
    "ROLE_BY_STATE",
    "RULES",
    "StateRules",
    "artifacts_writable_by",
    "get_rules",
    # Roles and the default worker factory
    "ROLE_SPECS",
    "AgentBuilder",
    "RoleSpec",
    "build_default_worker_factory",
    "build_role_prompt",
    "build_role_system_prompt",
    "build_role_task_prompt",
    "count_top_level_json_objects",
    "get_role_spec",
    "held_tools",
    # Confined workspace tools
    "COMMAND_ALLOWLIST",
    "READ_ONLY_TOOLS",
    "SHELL_TOOLS",
    "VERIFICATION_COMMANDS",
    "WRITE_TOOLS",
    "Workspace",
    "WorkspaceTools",
    "build_workspace_tools",
    # Confined, ownership-scoped plan directory
    "PLAN_READ_TOOLS",
    "PLAN_WRITE_TOOLS",
    "PlanMemory",
    "PlanTools",
    "build_plan_tools",
]
