"""Transition evaluation results and the ``AdvanceResult`` of one message-free step."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .transition import TransitionOption


class TransitionEvaluationResult(str, Enum):
    """Results of transition evaluation."""

    DETERMINISTIC = "deterministic"  # Clear single transition
    AMBIGUOUS = "ambiguous"  # Multiple valid transitions, need LLM
    BLOCKED = "blocked"  # No valid transitions available


class TransitionEvaluation(BaseModel):
    """
    Result of evaluating possible transitions from current state.

    Used by the transition evaluator to determine next steps.
    """

    result_type: TransitionEvaluationResult = Field(
        ..., description="Type of evaluation result"
    )

    deterministic_transition: str | None = Field(
        None, description="Target state if deterministically determined"
    )

    available_options: list[TransitionOption] = Field(
        default_factory=list, description="Available transition options if ambiguous"
    )

    blocked_reason: str | None = Field(
        None, description="Reason if transitions are blocked"
    )

    @model_validator(mode="after")
    def validate_result_consistency(self) -> TransitionEvaluation:
        """Ensure populated fields match result_type."""
        if self.result_type == TransitionEvaluationResult.DETERMINISTIC:
            if (
                not self.deterministic_transition
                or not self.deterministic_transition.strip()
            ):
                raise ValueError(
                    "DETERMINISTIC result requires non-empty deterministic_transition"
                )
        elif self.result_type == TransitionEvaluationResult.AMBIGUOUS:
            if not self.available_options:
                raise ValueError(
                    "AMBIGUOUS result requires non-empty available_options"
                )
        elif self.result_type == TransitionEvaluationResult.BLOCKED:
            if self.blocked_reason is None:
                raise ValueError("BLOCKED result requires blocked_reason")
        return self


class AdvanceResult(BaseModel):
    """What one message-free step (``API.advance``) of a conversation did.

    Fields:
      - ``state_before`` / ``state_after``: the state id the step started in
        and the one it ended in; equal when the step stayed.
      - ``transition_outcome``: the transition evaluator's result for
        ``state_before``. Always set: a step never runs on a terminal state
        (``advance`` raises ``FSMError`` there), so an evaluation always ran.
        ``AMBIGUOUS`` with ``state_before == state_after`` means the classifier
        chose to stay; ``BLOCKED`` always stays.
      - ``response``: the Pass-2 reply of ``state_after``, or ``None`` when that
        state is silent (empty ``response_instructions``): no LLM call is made
        and nothing is appended to the history for it. ``API.advance`` follows it
        with the greeting of a child FSM a call state pushed at the end of the step,
        joined by one newline (plan baa37f40, D-023): the reply of the step as a
        whole, so a silent parent and a greeting child give the greeting alone.
      - ``ended``: ``True`` when ``state_after`` is terminal.

    Frozen: a result is a record of a finished step.
    """

    model_config = ConfigDict(frozen=True)

    state_before: str = Field(..., description="State the step started in")
    state_after: str = Field(..., description="State the step ended in")
    transition_outcome: TransitionEvaluationResult = Field(
        ..., description="Transition evaluation result for state_before"
    )
    response: str | None = Field(
        None, description="Pass-2 reply, or None for a silent state"
    )
    ended: bool = Field(..., description="True when state_after is terminal")
