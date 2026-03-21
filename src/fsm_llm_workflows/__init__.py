from __future__ import annotations

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
# Version info — imported via __version__.py to stay in sync (matches classification/reasoning pattern)
from .__version__ import __version__  # noqa: F401

# Workflow definition and validation
from .definitions import (
    WorkflowDefinition,
    WorkflowValidator,
)

# DSL and builder functions
from .dsl import (
    WorkflowBuilder,
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
    WorkflowInstance,
    WorkflowStatus,
    WorkflowStepResult,
)

# Step implementations
from .steps import (
    APICallStep,
    AutoTransitionStep,
    ConditionStep,
    ConversationStep,
    LLMProcessingStep,
    ParallelStep,
    TimerStep,
    WaitForEventStep,
    WorkflowStep,
)

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
    "ConversationStep",

    # Definition & Validation
    "WorkflowDefinition",
    "WorkflowValidator",

    # Engine
    "WorkflowEngine",
    "Timer",

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
    "linear_workflow",
    "conditional_workflow",
    "event_driven_workflow",
]
