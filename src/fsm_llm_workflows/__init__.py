"""
FSM-LLM Workflow System
=======================

An async, in-memory workflow engine built on top of FSM-LLM:

- Step graphs with automatic transitions, branching and loops through
  event/timer waits
- Event-driven waits (broadcast or targeted, with correlation keys) and timers
- Steps that call APIs, LLMs, FSM conversations and agents
- Parallel steps, retries, per-step and whole-workflow timeouts
- Lifecycle hooks (``WorkflowEngine.add_hook``)

Log output is off until ``fsm_llm.setup_logging()`` (or
``fsm_llm.enable_debug_logging()``) is called, like the core package.
"""

from __future__ import annotations

# Core models and exceptions
# Version info — imported via __version__.py to stay in sync (matches classification/reasoning pattern)
from fsm_llm.logging import is_library_logging_enabled as _logging_enabled
from fsm_llm.logging import logger as _logger

from .__version__ import __version__
from .constants import MAX_STEP_DEPTH, MAX_STEPS_PER_RUN

# Workflow definition and validation
from .definitions import (
    WorkflowDefinition,
    WorkflowValidator,
)
from .dependency_resolver import DependencyResolver

# DSL and builder functions
from .dsl import (
    WorkflowBuilder,
    agent_step,
    api_step,
    auto_step,
    condition_step,
    conditional_workflow,
    conversation_step,
    create_workflow,
    event_driven_workflow,
    linear_workflow,
    llm_step,
    parallel_step,
    retry_step,
    switch_step,
    timer_step,
    wait_event_step,
    workflow_builder,
)

# Core engine
from .engine import (
    Timer,
    WorkflowEngine,
)
from .exceptions import (
    WorkflowDefinitionError,
    WorkflowError,
    WorkflowEventError,
    WorkflowInstanceError,
    WorkflowResourceError,
    WorkflowStateError,
    WorkflowStepError,
    WorkflowTimeoutError,
    WorkflowValidationError,
)
from .models import (
    EventListener,
    WaitEventConfig,
    WorkflowEvent,
    WorkflowHistoryEntry,
    WorkflowInstance,
    WorkflowStatus,
    WorkflowStepResult,
)

# Step implementations
from .steps import (
    AgentStep,
    APICallStep,
    AutoTransitionStep,
    ConditionStep,
    ConversationStep,
    LLMProcessingStep,
    ParallelStep,
    RetryStep,
    SwitchStep,
    TimerStep,
    WaitForEventStep,
    WorkflowStep,
)

# Library logging is off by default, like the core package (whose
# `logger.disable("fsm_llm")` does not cover this top-level package).
# `fsm_llm.setup_logging()` / `enable_debug_logging()` re-enable it.
if not _logging_enabled():
    _logger.disable("fsm_llm_workflows")

__author__ = "Nikolas Markou"

# Public API
__all__ = [
    # Exceptions
    "WorkflowError",
    "WorkflowDefinitionError",
    "WorkflowStepError",
    "WorkflowInstanceError",
    "WorkflowTimeoutError",
    "WorkflowValidationError",
    "WorkflowStateError",
    "WorkflowEventError",
    "WorkflowResourceError",
    # Models
    "WorkflowStatus",
    "WorkflowEvent",
    "WorkflowStepResult",
    "WorkflowInstance",
    "WorkflowHistoryEntry",
    "EventListener",
    "WaitEventConfig",
    # Steps
    "WorkflowStep",
    "AgentStep",
    "AutoTransitionStep",
    "APICallStep",
    "ConditionStep",
    "LLMProcessingStep",
    "WaitForEventStep",
    "TimerStep",
    "ParallelStep",
    "RetryStep",
    "SwitchStep",
    "ConversationStep",
    # Definition & Validation
    "WorkflowDefinition",
    "WorkflowValidator",
    # Engine
    "WorkflowEngine",
    "Timer",
    "MAX_STEPS_PER_RUN",
    "MAX_STEP_DEPTH",
    # Dependency Resolution
    "DependencyResolver",
    # Version
    "__version__",
    # DSL
    "create_workflow",
    "workflow_builder",
    "WorkflowBuilder",
    "auto_step",
    "api_step",
    "condition_step",
    "llm_step",
    "wait_event_step",
    "timer_step",
    "parallel_step",
    "conversation_step",
    "agent_step",
    "retry_step",
    "switch_step",
    "linear_workflow",
    "conditional_workflow",
    "event_driven_workflow",
]
