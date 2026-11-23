"""
FSM-LLM Reasoning Engine
========================

A structured reasoning framework for solving complex problems using
Finite State Machines to guide Large Language Models through systematic
reasoning processes.

Basic Usage:
    from fsm_llm.reasoning import ReasoningEngine

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

from fsm_llm.engine.executables.namespaces import EXECUTABLE_SOURCES

from .__version__ import __version__
from .constants import ReasoningType
from .definitions import (
    ProblemContext,
    ReasoningClassificationResult,
    ReasoningStep,
    ReasoningTrace,
    ValidationResult,
)
from .engine import ReasoningEngine
from .exceptions import (
    ReasoningClassificationError,
    ReasoningError,
    ReasoningExecutionError,
    ReasoningRegistryError,
)
from .stream import SolveStream
from .utilities import get_available_reasoning_types


def _builtin_executables() -> list:
    """The ``reasoning`` source of ``executable_sources``: nine strategies and ``reasoning.solve``.

    Lazy on purpose: ``reasoning.executables`` is imported on the first call, so importing this
    package loads the ``fsm_llm.engine.executables`` package (its namespaces live there; about ten
    modules come with it) but not ``reasoning.executables`` itself (plan ace7a8c8 D-061, D-070).
    """
    from .executables import builtin_executables

    return builtin_executables()


# `override=True`: re-importing the package (a reload) replaces the factory, never collides.
EXECUTABLE_SOURCES.register("reasoning", _builtin_executables, override=True)

__all__ = [
    # Main classes
    "ReasoningEngine",
    "ReasoningType",
    "SolveStream",
    # Models
    "ReasoningStep",
    "ReasoningTrace",
    "ValidationResult",
    "ReasoningClassificationResult",
    "ProblemContext",
    # Exceptions
    "ReasoningError",
    "ReasoningExecutionError",
    "ReasoningClassificationError",
    "ReasoningRegistryError",
    # Utilities
    "get_available_reasoning_types",
    # Version
    "__version__",
]
