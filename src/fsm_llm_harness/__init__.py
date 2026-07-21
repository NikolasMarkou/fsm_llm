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
    HarnessGateBlockedError,
    HarnessLeashError,
    HarnessOwnershipError,
    HarnessReentrancyError,
)
from .fsm_definition import DEFAULT_PERSONA, build_harness_fsm
from .hardening import (
    RETRYABLE_EXCEPTIONS,
    RoleOutput,
    as_int,
    build_response_format,
    coerce_bool,
    coerce_int,
    coerce_str,
    coerce_worker_output,
    parse_json_payload,
    parse_role_output,
    retry,
    strip_model_noise,
    type_matches,
)
from .harness import HarnessAgent, RoleRequest, WorkerFactory
from .roles import (
    ROLE_SPECS,
    AgentBuilder,
    RoleSpec,
    build_default_worker_factory,
    build_role_prompt,
    count_top_level_json_objects,
    get_role_spec,
)
from .rules import OWNERSHIP, ROLE_BY_STATE, RULES, StateRules, get_rules
from .tools import (
    COMMAND_ALLOWLIST,
    READ_ONLY_TOOLS,
    SHELL_TOOLS,
    WRITE_TOOLS,
    Workspace,
    WorkspaceTools,
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
    "HarnessGateBlockedError",
    "HarnessLeashError",
    "HarnessOwnershipError",
    "HarnessReentrancyError",
    # FSM definition
    "DEFAULT_PERSONA",
    "build_harness_fsm",
    # Driver
    "HarnessAgent",
    "RoleRequest",
    "WorkerFactory",
    # Small-model hardening
    "RETRYABLE_EXCEPTIONS",
    "RoleOutput",
    "as_int",
    "build_response_format",
    "coerce_bool",
    "coerce_int",
    "coerce_str",
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
    "get_rules",
    # Roles and the default worker factory
    "ROLE_SPECS",
    "AgentBuilder",
    "RoleSpec",
    "build_default_worker_factory",
    "build_role_prompt",
    "count_top_level_json_objects",
    "get_role_spec",
    # Confined workspace tools
    "COMMAND_ALLOWLIST",
    "READ_ONLY_TOOLS",
    "SHELL_TOOLS",
    "WRITE_TOOLS",
    "Workspace",
    "WorkspaceTools",
    "build_workspace_tools",
]
