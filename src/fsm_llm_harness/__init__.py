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
from .rules import OWNERSHIP, ROLE_BY_STATE, RULES, StateRules, get_rules

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
    # Per-state rules
    "OWNERSHIP",
    "ROLE_BY_STATE",
    "RULES",
    "StateRules",
    "get_rules",
]
