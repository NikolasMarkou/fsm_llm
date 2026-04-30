"""Backward-compat alias. Real implementation: fsm_llm.stdlib.reasoning.

Module identity preserved via sys.modules aliasing: every legacy submodule path
`fsm_llm_reasoning.<name>` resolves to the *same* module object as
`fsm_llm.stdlib.reasoning.<name>`. This is intentional — tests that mock-patch
attributes on the legacy path hit the real module, not a duplicate.

Per ``docs/lambda_fsm_merge.md`` §3 I5 (M6c): this shim emits a
``DeprecationWarning`` at import time starting at fsm_llm 0.6.0; removal at
0.7.0. Migrate to ``from fsm_llm.stdlib.reasoning import ...``.
"""

import sys as _sys

from fsm_llm._api.deprecation import warn_deprecated as _warn_deprecated

_warn_deprecated(
    "fsm_llm_reasoning",
    since="0.6.0",
    removal="0.7.0",
    replacement="fsm_llm.stdlib.reasoning",
)

from fsm_llm.stdlib.reasoning import (
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
from fsm_llm.stdlib.reasoning import (
    constants as _constants,
)
from fsm_llm.stdlib.reasoning import (
    definitions as _definitions,
)
from fsm_llm.stdlib.reasoning import (
    engine as _engine,
)
from fsm_llm.stdlib.reasoning import (
    exceptions as _exceptions,
)
from fsm_llm.stdlib.reasoning import (
    handlers as _handlers,
)
from fsm_llm.stdlib.reasoning import (
    reasoning_modes as _reasoning_modes,
)
from fsm_llm.stdlib.reasoning import (
    utilities as _utilities,
)
from fsm_llm.stdlib.reasoning.__version__ import __version__

# Register sys.modules aliases so both
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
