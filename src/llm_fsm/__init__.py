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

from .api import API
from .fsm import FSMManager
from .prompts import PromptBuilder
from .llm import LLMInterface, LiteLLMInterface
from .utilities import load_fsm_definition, load_fsm_from_file
from .validator import FSMValidator, validate_fsm_from_file, FSMValidationResult


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
    "API",
]

# Optional workflows check
def has_workflows():
    """Check if workflows extension is available."""
    try:
        import llm_fsm_workflows
        return True
    except ImportError:
        return False

def get_workflows():
    """Get workflows module if available, otherwise raise ImportError."""
    try:
        import llm_fsm_workflows
        return llm_fsm_workflows
    except ImportError:
        raise ImportError(
            "Workflows functionality requires the workflows extra. "
            "Install with: pip install llm-fsm[workflows]"
        )