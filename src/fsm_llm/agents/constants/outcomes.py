"""How a run ended: the stop reasons.

``ToolRunStatus`` and ``ToolObservationPrefix`` (how one tool call ended) and
``OBSERVATION_ENTRY`` (the text of a tool step) are defined in ``fsm_llm.tools.constants``
and re-exported by ``fsm_llm.agents.constants``.
"""

from __future__ import annotations

from enum import Enum

from fsm_llm.core.codes import _parse

__all__ = ["FORCED_STOP_REASONS", "StopReason"]


class StopReason(str, Enum):
    """Why an agent run ended: the value of ``AgentResult.stop_reason``.

    A ``str`` Enum with the wire strings as values: a member equals, hashes and
    dumps as its string. ``ANSWERED`` and ``EVIDENCE`` derive status ``ok``
    (``AgentResult.success``); every other value derives a status that is not
    ``ok`` (``fsm_llm.core.codes.status_of_stop_reason``). Write ``.value``,
    never ``str(member)`` or an f-string, into a message, a context dict or a
    definition: the formatting of a member is ``StopReason.X`` on every
    supported Python.
    """

    # Concluded with a real answer: an answer key or an executed tool call.
    ANSWERED = "answered"
    # Succeeded on executed work: planner evidence, subtasks.
    EVIDENCE = "evidence"
    # The iteration budget forced the stop; the last output still ships.
    MAX_ITERATIONS = "max_iterations"
    # A judge overrode a failing (or missing) verdict at a revision or budget
    # limit (EvalOpt, MakerChecker, Debate at num_rounds), or Reflexion hit
    # max_reflections without a passing evaluation. A genuine pass on the
    # limit's round is ANSWERED.
    FORCED_PASS = "forced_pass"
    # Consecutive turns with no tool selected forced the stop.
    STALLED = "stalled"
    # The answer was rejected by a verifier.
    VERIFICATION_FAILED = "verification_failed"
    # The run ended without a real result (prose fallback only, no evidence,
    # a malformed or failed turn, no successful subtask or sample).
    NO_RESULT = "no_result"
    # A prompt-chain validation gate failed.
    GATE_FAILED = "gate_failed"
    # The conversation was ended from outside the run (a hook, another
    # thread, a monitor) before it reached a final state; recorded by
    # ``BaseAgent._run_conversation_loop``, the last output still ships.
    ENDED = "ended"
    # DECISION plan-2026-11-06T093029-0068b055/D-025: the final turn's reply
    # (or an extraction it needed) was cut off by the provider at max_tokens,
    # so the answer is a fragment or the cut-off apology. Recorded by
    # ``BaseAgent._run_conversation_loop`` from core's cut-off record
    # (``CONTEXT_KEY_CUT_OFF``), and by REWOO's plan executor for a cut plan;
    # the answer text still ships. Only when no other reason forced the stop.
    # Do NOT report it as success. See decisions.md D-025.
    TRUNCATED = "truncated"

    @classmethod
    def from_string(cls, value: StopReason | str) -> StopReason:
        """The member for a member or an exact value; ``ValueError`` otherwise."""
        return _parse(cls, value, "stop_reason")


# Reasons a forcing handler may record in ``ContextKeys.FORCED_STOP_REASON``
# (the prompt-chain gate handler records ``GATE_FAILED``; the run loop records
# ``ENDED`` and ``TRUNCATED``).
FORCED_STOP_REASONS: frozenset[StopReason] = frozenset(
    {
        StopReason.MAX_ITERATIONS,
        StopReason.FORCED_PASS,
        StopReason.STALLED,
        StopReason.GATE_FAILED,
        StopReason.ENDED,
        StopReason.TRUNCATED,
    }
)
