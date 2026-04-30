from __future__ import annotations

"""fsm_llm.errors — full exception hierarchy.

Canonical home for every public exception type since 0.9.0. The 0.7.0
``fsm_llm.types`` module hosted the ``FSMError`` hierarchy; in 0.9.0 the
hierarchy moved here (the request/response Pydantic models moved to the
private ``fsm_llm._models`` module).

Layout:

- **Core FSM errors**: ``FSMError`` and its dialog-side subclasses
  (``StateNotFoundError``, ``InvalidTransitionError``, ``LLMResponseError``,
  ``TransitionEvaluationError``, ``ClassificationError``,
  ``SchemaValidationError``, ``ClassificationResponseError``).
- **λ-kernel errors**: ``LambdaError`` and its subclasses
  (``ASTConstructionError``, ``TerminationError``, ``PlanningError``,
  ``OracleError``).
- **Program facade errors**: ``ProgramModeError``.
- **Handler errors**: ``HandlerSystemError``, ``HandlerExecutionError``.
- **Stdlib root errors**: ``ReasoningEngineError``, ``WorkflowError``,
  ``AgentError``, ``MonitorError`` — re-exported from their subpackages.

The full subclass tree of each stdlib root remains importable from the
subpackage (e.g. ``from fsm_llm.stdlib.workflows.exceptions import
WorkflowTimeoutError``); only the roots are surfaced here.

    from fsm_llm.errors import FSMError, LambdaError, ProgramModeError

The 0.7.0 ``fsm_llm.types`` import path is removed at 0.9.0 — update to
``fsm_llm.errors`` (for exceptions) or to the appropriate subpackage (for
request/response models).
"""

# Core FSM error hierarchy
from ._models import (
    ClassificationError,
    ClassificationResponseError,
    FSMError,
    InvalidTransitionError,
    LLMResponseError,
    SchemaValidationError,
    StateNotFoundError,
    TransitionEvaluationError,
)

# Handler errors (subclass FSMError)
from .handlers import HandlerExecutionError, HandlerSystemError

# Program facade error
from .program import ProgramModeError

# λ-kernel errors
from .runtime.errors import (
    ASTConstructionError,
    LambdaError,
    OracleError,
    PlanningError,
    TerminationError,
)

# Stdlib root errors
from .stdlib.agents.exceptions import AgentError
from .stdlib.reasoning.exceptions import ReasoningEngineError
from .stdlib.workflows.exceptions import WorkflowError

try:
    from fsm_llm_monitor.exceptions import MonitorError  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover — monitor is optional
    MonitorError = None  # type: ignore[assignment, misc]

__all__ = [
    # Core
    "FSMError",
    "StateNotFoundError",
    "InvalidTransitionError",
    "LLMResponseError",
    "TransitionEvaluationError",
    "ClassificationError",
    "SchemaValidationError",
    "ClassificationResponseError",
    # λ-kernel
    "LambdaError",
    "ASTConstructionError",
    "TerminationError",
    "PlanningError",
    "OracleError",
    # Program facade
    "ProgramModeError",
    # Handlers
    "HandlerSystemError",
    "HandlerExecutionError",
    # Stdlib roots
    "ReasoningEngineError",
    "WorkflowError",
    "AgentError",
    "MonitorError",
]
