"""
FSM-LLM Workflow System - Clean Architecture
===========================================

A workflow system built on top of FSM-LLM that enables:
- Automated state transitions
- Event-driven workflows
- External system integration
- Parallel workflow execution
- Monitoring and error recovery
"""

# Core models and exceptions
from .exceptions import (
    WorkflowError,
    WorkflowDefinitionError,
    WorkflowStepError,
    WorkflowInstanceError,
    WorkflowTimeoutError,
    WorkflowValidationError,
    WorkflowStateError,
    WorkflowEventError,
    WorkflowResourceError,
)

from .models import (
    WorkflowStatus,
    WorkflowEvent,
    WorkflowStepResult,
    WorkflowInstance,
    EventListener,
    WaitEventConfig,
)

# Step implementations
from .steps import (
    WorkflowStep,
    AutoTransitionStep,
    APICallStep,
    ConditionStep,
    LLMProcessingStep,
    WaitForEventStep,
    TimerStep,
    ParallelStep,
)

# Workflow definition and validation
from .definitions import (
    WorkflowDefinition,
    WorkflowValidator,
)

# Core engine
from .engine import (
    WorkflowEngine,
    Timer,
)

# Integration handlers
from .handlers import (
    AutoTransitionHandler,
    EventHandler,
    TimerHandler,
)

# DSL and builder functions
from .dsl import (
    create_workflow,
    workflow_builder,
    auto_step,
    api_step,
    condition_step,
    llm_step,
    wait_event_step,
    timer_step,
    parallel_step,
    WorkflowBuilder,
    linear_workflow,
    conditional_workflow,
    event_driven_workflow,
)

# Version info — imported from main package to stay in sync
from fsm_llm.__version__ import __version__
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
    "EventListener",
    "WaitEventConfig",

    # Steps
    "WorkflowStep",
    "AutoTransitionStep",
    "APICallStep",
    "ConditionStep",
    "LLMProcessingStep",
    "WaitForEventStep",
    "TimerStep",
    "ParallelStep",

    # Definition & Validation
    "WorkflowDefinition",
    "WorkflowValidator",

    # Engine
    "WorkflowEngine",
    "Timer",

    # Handlers
    "AutoTransitionHandler",
    "EventHandler",
    "TimerHandler",

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
    "linear_workflow",
    "conditional_workflow",
    "event_driven_workflow",
]