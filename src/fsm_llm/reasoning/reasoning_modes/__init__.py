"""
Complete FSM definitions for the reasoning engine.

One module per FSM: the orchestrator, the classifier and the 9 specialized
reasoning strategies; `registry.ALL_REASONING_FSMS` collects them by name.
All FSMs use standardized context keys from `fsm_llm.reasoning.constants`.
"""

from .abductive import abductive_fsm
from .analogical import analogical_fsm
from .analytical import analytical_fsm
from .classifier import classifier_fsm
from .creative import creative_fsm
from .critical import critical_fsm
from .deductive import deductive_fsm
from .hybrid import hybrid_fsm
from .inductive import inductive_fsm
from .orchestrator import orchestrator_fsm
from .registry import ALL_REASONING_FSMS
from .simple_calculator import simple_calculator_fsm

__all__ = [
    "orchestrator_fsm",
    "classifier_fsm",
    "simple_calculator_fsm",
    "analytical_fsm",
    "deductive_fsm",
    "inductive_fsm",
    "creative_fsm",
    "critical_fsm",
    "abductive_fsm",
    "analogical_fsm",
    "hybrid_fsm",
    "ALL_REASONING_FSMS",
]
