"""The steps that pause a workflow: ``WaitEventStep`` and ``TimerStep``.

Part of the package (plan d454fd62 step 92; the old ``workflows/steps.py`` at the same dotted path).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from pydantic import field_validator

from ..constants import CONTEXT_KEY_TIMER_INFO, CONTEXT_KEY_WAITING_INFO
from ..models import WaitEventConfig, WorkflowStepResult
from .base import WorkflowStep


class WaitEventStep(WorkflowStep):
    """A step that waits for an external event (see ``WaitEventConfig``).

    ``timeout_seconds`` on the step bounds the step's own execution;
    ``config.timeout_seconds`` is how long the wait for the event may last.
    They are different limits that share a spelling.
    """

    config: WaitEventConfig

    def referenced_step_ids(self) -> set[str]:
        """The success step id and the timeout step id of the wait."""
        return {
            state
            for state in (self.config.success_step_id, self.config.timeout_step_id)
            if state
        }

    async def execute(self, context: dict[str, Any]) -> WorkflowStepResult:
        """Set up the workflow to wait for an event."""
        waiting_info = {
            "waiting_for_event": True,
            "event_type": self.config.event_type,
            "timeout_seconds": self.config.timeout_seconds,
            "timeout_step_id": self.config.timeout_step_id,
            "success_step_id": self.config.success_step_id,
            "event_mapping": self.config.event_mapping,
            "correlation_key": self.config.correlation_key,
            "waiting_since": datetime.now(UTC).isoformat(),
        }

        return WorkflowStepResult.success_result(
            data={CONTEXT_KEY_WAITING_INFO: waiting_info},
            message=f"Waiting for event of type {self.config.event_type}",
        )


class TimerStep(WorkflowStep):
    """A step that waits for a specified time before transitioning."""

    delay_seconds: float
    next_step_id: str

    @field_validator("delay_seconds")
    @classmethod
    def _validate_delay_seconds(cls, v: float) -> float:
        if v < 0:
            raise ValueError("delay_seconds must be >= 0")
        return v

    async def execute(self, context: dict[str, Any]) -> WorkflowStepResult:
        """Set up a timer to transition after a delay."""
        timer_info = {
            "waiting_for_timer": True,
            "delay_seconds": self.delay_seconds,
            "next_step_id": self.next_step_id,
            "timer_start": datetime.now(UTC).isoformat(),
            "timer_end": (
                datetime.now(UTC) + timedelta(seconds=self.delay_seconds)
            ).isoformat(),
        }

        return WorkflowStepResult.success_result(
            data={CONTEXT_KEY_TIMER_INFO: timer_info},
            message=f"Timer set for {self.delay_seconds} seconds, will transition to {self.next_step_id}",
        )
