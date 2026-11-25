"""The exceptions of the simulation layer."""

from __future__ import annotations

from typing import Any

from fsm_llm.core.definitions import FSMError

#: The judge is provably (or cannot be shown not to be) the executor or the compiler.
CODE_JUDGE_NOT_DISTINCT = "judge_not_distinct"


class SimulationError(FSMError):
    """A simulator was set up in a way it refuses; ``code`` is the fixed reason."""

    def __init__(
        self, message: str, *, code: str, details: dict[str, Any] | None = None
    ) -> None:
        super().__init__(message, details)
        self.code = code
