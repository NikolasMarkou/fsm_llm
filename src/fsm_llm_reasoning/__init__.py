"""Backward-compat alias. Real implementation: fsm_llm.stdlib.reasoning.

Module identity preserved via sys.modules aliasing: every legacy submodule path
`fsm_llm_reasoning.<name>` resolves to the *same* module object as
`fsm_llm.stdlib.reasoning.<name>`. This is intentional — tests that mock-patch
attributes on the legacy path hit the real module, not a duplicate.
"""

import sys as _sys

# Import the 7 real submodules from the new canonical home.
from fsm_llm.stdlib.reasoning import (
    constants as _constants,
    definitions as _definitions,
    engine as _engine,
    exceptions as _exceptions,
    handlers as _handlers,
    reasoning_modes as _reasoning_modes,
    utilities as _utilities,
)

# Register sys.modules aliases BEFORE mirroring public re-exports, so that both
#   `from fsm_llm_reasoning import engine`        (package-attribute access)
#   `from fsm_llm_reasoning.engine import X`      (submodule path import)
# resolve identically to the same module object.
_sys.modules["fsm_llm_reasoning.constants"] = _constants
_sys.modules["fsm_llm_reasoning.definitions"] = _definitions
_sys.modules["fsm_llm_reasoning.engine"] = _engine
_sys.modules["fsm_llm_reasoning.exceptions"] = _exceptions
_sys.modules["fsm_llm_reasoning.handlers"] = _handlers
_sys.modules["fsm_llm_reasoning.reasoning_modes"] = _reasoning_modes
_sys.modules["fsm_llm_reasoning.utilities"] = _utilities

# Bind submodules as package attributes so `from fsm_llm_reasoning import engine` works.
constants = _constants
definitions = _definitions
engine = _engine
exceptions = _exceptions
handlers = _handlers
reasoning_modes = _reasoning_modes
utilities = _utilities

# Mirror the legacy public surface — same 11 names as the pre-move __all__.
from fsm_llm.stdlib.reasoning import (  # noqa: E402, F401
    ProblemContext,
    ReasoningClassificationError,
    ReasoningClassificationResult,
    ReasoningEngine,
    ReasoningEngineError,
    ReasoningExecutionError,
    ReasoningStep,
    ReasoningTrace,
    ReasoningType,
    SolutionResult,
    ValidationResult,
    get_available_reasoning_types,
)
from fsm_llm.stdlib.reasoning.__version__ import __version__  # noqa: E402, F401

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
    # Version
    "__version__",
]
