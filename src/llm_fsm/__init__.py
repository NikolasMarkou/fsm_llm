"""LLM-FSM: Finite State Machines for Large Language Models."""

from .__version__ import __version__

from .definitions import (
    FSMDefinition,
    FSMInstance,
    State,
    Transition,
    TransitionCondition,
    FSMContext,
    StateTransition,
    LLMRequest,
    LLMResponse,
)

from .fsm import FSMManager
from .prompts import PromptBuilder
from .llm import LLMInterface, LiteLLMInterface
from .utilities import load_fsm_definition, load_fsm_from_file
from .validator import FSMValidator, validate_fsm_from_file, FSMValidationResult
from .llm_fsm import LLM_FSM  # Make sure this points to your new implementation

try:
    from .workflows import (
        WorkflowEngine,
        WorkflowDefinition,
        WorkflowStatus,
        WorkflowEvent,
        create_workflow,
        auto_step,
        api_step,
        condition_step,
        wait_event_step,
        timer_step,
        parallel_step,
    )
    _WORKFLOWS_AVAILABLE = True
except ImportError:
    _WORKFLOWS_AVAILABLE = False


__all__ = [
    "__version__",
    "FSMDefinition",
    "FSMInstance",
    "State",
    "Transition",
    "TransitionCondition",
    "FSMContext",
    "StateTransition",
    "LLMRequest",
    "LLMResponse",
    "FSMManager",
    "LLMInterface",
    "LiteLLMInterface",
    "PromptBuilder",
    "load_fsm_definition",
    "load_fsm_from_file",
    "FSMValidator",
    "validate_fsm_from_file",
    "FSMValidationResult",
    "LLM_FSM",
]

if _WORKFLOWS_AVAILABLE:
    __all__.extend([
        "WorkflowEngine",
        "WorkflowDefinition",
        "WorkflowStatus",
        "WorkflowEvent",
        "create_workflow",
        "auto_step",
        "api_step",
        "condition_step",
        "wait_event_step",
        "timer_step",
        "parallel_step",
    ])