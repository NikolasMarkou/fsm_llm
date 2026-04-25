"""Backward-compat alias. Real implementation: fsm_llm.stdlib.workflows.

Module identity is preserved across both names via sys.modules aliasing,
so `fsm_llm_workflows.engine is fsm_llm.stdlib.workflows.engine` is True
(and the same for the other 6 submodules). This shim is silent — no
DeprecationWarning is raised.

The shim does NOT alias `cli` (workflows ships no CLI); imports like
`import fsm_llm_workflows.cli` continue to raise ModuleNotFoundError.
"""

from __future__ import annotations

import sys

# Import the 7 real submodules from the canonical home.
from fsm_llm.stdlib.workflows import (
    definitions,
    dependency_resolver,
    dsl,
    engine,
    exceptions,
    models,
    steps,
)

# Register sys.modules aliases BEFORE re-exporting public symbols so that
# `from fsm_llm_workflows.engine import WorkflowEngine` and
# `from fsm_llm_workflows import engine` resolve to the same module object.
sys.modules["fsm_llm_workflows.definitions"] = definitions
sys.modules["fsm_llm_workflows.dependency_resolver"] = dependency_resolver
sys.modules["fsm_llm_workflows.dsl"] = dsl
sys.modules["fsm_llm_workflows.engine"] = engine
sys.modules["fsm_llm_workflows.exceptions"] = exceptions
sys.modules["fsm_llm_workflows.models"] = models
sys.modules["fsm_llm_workflows.steps"] = steps

# Mirror the public __all__ from the canonical home, byte-for-byte.
from fsm_llm.stdlib.workflows.__version__ import __version__
from fsm_llm.stdlib.workflows.definitions import (
    WorkflowDefinition,
    WorkflowValidator,
)
from fsm_llm.stdlib.workflows.dependency_resolver import DependencyResolver
from fsm_llm.stdlib.workflows.dsl import (
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
from fsm_llm.stdlib.workflows.engine import (
    Timer,
    WorkflowEngine,
)
from fsm_llm.stdlib.workflows.exceptions import (
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
from fsm_llm.stdlib.workflows.models import (
    EventListener,
    WaitEventConfig,
    WorkflowEvent,
    WorkflowInstance,
    WorkflowStatus,
    WorkflowStepResult,
)
from fsm_llm.stdlib.workflows.steps import (
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

__author__ = "Nikolas Markou"

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
