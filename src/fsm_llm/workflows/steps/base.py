"""The ``WorkflowStep`` base class.

``WorkflowStep`` carries ``model_config`` with ``extra="forbid"``: every step class inherits the refusal of an
unknown or removed keyword from this one definition, so it stays in this one module and every kind imports it.

Part of the package (plan d454fd62 step 92; the old ``workflows/steps.py`` at the same dotted path).
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ConfigDict

from ..exceptions import WorkflowStepError
from ..models import WorkflowStepResult


class WorkflowStep(BaseModel, ABC):
    """Base class for workflow steps."""

    step_id: str
    name: str
    description: str = ""
    timeout_seconds: float | None = None
    """Maximum seconds this step may run. ``None`` (the default) disables the
    timeout; set a finite value for steps that call out to slow services."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    @abstractmethod
    async def execute(self, context: dict[str, Any]) -> WorkflowStepResult:
        """Execute the step. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement execute()")

    def referenced_step_ids(self) -> set[str]:
        """The step ids this step can route to (empty markers dropped).

        Interface contract: the non-empty string values of the model fields
        whose name ends in ``_step_id`` (``next_step_id``, ``error_step_id``,
        ...). That one suffix is the whole rule: name a routing field with it.
        A step that names step ids elsewhere (a nested model, a mapping, a
        wrapped step) overrides this; ``WorkflowDefinition`` reads nothing else
        for validation, reachability and terminal detection. Never raises.
        """
        step_ids: set[str] = set()
        for name in type(self).model_fields:
            value = getattr(self, name, None)
            if name.endswith("_step_id") and isinstance(value, str) and value:
                step_ids.add(value)
        return step_ids

    async def _with_timeout(self, coro):
        """Wrap a coroutine with ``asyncio.wait_for`` if timeout is configured.

        A ``TimeoutError`` raised by the awaited code itself is not this
        step's time limit and propagates unchanged.
        """
        if self.timeout_seconds is None:
            return await coro
        raised = False

        async def run():
            nonlocal raised
            try:
                return await coro
            except TimeoutError:
                raised = True
                raise

        try:
            return await asyncio.wait_for(run(), timeout=self.timeout_seconds)
        except TimeoutError as e:
            if raised:
                raise
            raise WorkflowStepError(
                f"Step timed out after {self.timeout_seconds}s", step_id=self.step_id
            ) from e
