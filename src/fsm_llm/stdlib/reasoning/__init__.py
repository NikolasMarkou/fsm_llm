"""
FSM-LLM Reasoning Engine
========================

A structured reasoning framework for solving complex problems using
Finite State Machines to guide Large Language Models through systematic
reasoning processes.

Basic Usage:
    from fsm_llm_reasoning import ReasoningEngine

    engine = ReasoningEngine()
    solution, trace = engine.solve_problem("Your problem here")
    print(solution)

Available Reasoning Types:
    - SIMPLE_CALCULATOR: Direct arithmetic calculations
    - ANALYTICAL: Breaking down complex systems
    - DEDUCTIVE: Deriving specific from general
    - INDUCTIVE: Finding patterns from examples
    - CREATIVE: Generating novel solutions
    - CRITICAL: Evaluating arguments
    - HYBRID: Combining approaches
    - ABDUCTIVE: Finding best explanations
    - ANALOGICAL: Learning through analogies
"""

from .__version__ import __version__
from .constants import ReasoningType
from .definitions import (
    ProblemContext,
    ReasoningClassificationResult,
    ReasoningStep,
    ReasoningTrace,
    SolutionResult,
    ValidationResult,
)
from .engine import ReasoningEngine
from .exceptions import (
    ReasoningClassificationError,
    ReasoningEngineError,
    ReasoningExecutionError,
)
from .lam_factories import (
    abductive_term,
    analogical_term,
    analytical_term,
    calculator_term,
    classifier_term,
    creative_term,
    critical_term,
    deductive_term,
    hybrid_term,
    inductive_term,
    solve_term,
)
from .utilities import get_available_reasoning_types

__all__ = [
    # Main classes
    "ReasoningEngine",
    "ReasoningType",
    # Models
    "ReasoningStep",
    "ReasoningTrace",
    "ValidationResult",
    "ReasoningClassificationResult",
    "ProblemContext",
    "SolutionResult",
    # Exceptions
    "ReasoningEngineError",
    "ReasoningExecutionError",
    "ReasoningClassificationError",
    # Utilities
    "get_available_reasoning_types",
    # λ-factories (M3 slice 2)
    "analytical_term",
    "deductive_term",
    "inductive_term",
    "abductive_term",
    "analogical_term",
    "creative_term",
    "critical_term",
    "hybrid_term",
    "calculator_term",
    "classifier_term",
    "solve_term",
    # Version
    "__version__",
]
