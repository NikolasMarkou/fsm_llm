from __future__ import annotations

"""
VerifiedReactAgent — a ReAct variant with answer verification + periodic
self-reflection.

Two additive reliability features, both driven by optional ``AgentConfig``
fields (defaults reproduce stock ReactAgent behavior):

- **Verification / grounding** (``config.verification_fn``): after a run, the
  answer is checked by a caller-supplied predicate. On rejection the agent
  retries with the verifier's feedback folded into the task, up to
  ``max_verify_retries`` times. No pattern verifies its own output today; this
  closes that gap.
- **Periodic self-reflection** (``config.reflect_every_n``): every N loop
  iterations a reflection note is injected into the agent's observations
  (which the think prompt always surfaces), nudging it to re-assess progress
  and avoid repeating failed actions.

Fully additive: a thin :class:`ReactAgent` subclass. With neither config field
set it behaves exactly like :class:`ReactAgent`.

Example::

    def verify(answer, context):
        # return bool, or {"ok": bool, "feedback": "..."}
        return {"ok": "sources:" in answer, "feedback": "cite your sources"}

    agent = VerifiedReactAgent(
        tools=registry,
        config=AgentConfig(model=model, verification_fn=verify, reflect_every_n=3),
        max_verify_retries=2,
    )
"""

from typing import Any

from fsm_llm import API
from fsm_llm.logging import logger

from .definitions import AgentResult
from .react import ReactAgent

_REFLECTION_NOTE = (
    "[Reflection] Pause and assess: are recent actions making progress toward "
    "the task? If you are repeating a failed action or looping, try a different "
    "tool or conclude with your best answer."
)


class VerifiedReactAgent(ReactAgent):
    """ReAct agent with verify-and-retry and periodic self-reflection.

    Args:
        max_verify_retries: Extra attempts after the first when
            ``config.verification_fn`` rejects the answer (total attempts =
            ``max_verify_retries + 1``).
        **kwargs: Forwarded to :class:`ReactAgent`.
    """

    def __init__(self, *args: Any, max_verify_retries: int = 1, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if max_verify_retries < 0:
            from .exceptions import AgentError

            raise AgentError("max_verify_retries must be >= 0")
        self.max_verify_retries = max_verify_retries

    # --- verification ---------------------------------------------------
    def _verify(self, result: AgentResult) -> tuple[bool, str]:
        """Run config.verification_fn against a result → (ok, feedback)."""
        fn = self.config.verification_fn
        if fn is None:
            return True, ""
        try:
            verdict = fn(result.answer, result.final_context)
        except Exception as e:
            logger.warning(f"verification_fn raised, treating as pass: {e}")
            return True, ""
        if isinstance(verdict, dict):
            ok = bool(verdict.get("ok", verdict.get("passed", False)))
            return ok, str(verdict.get("feedback", ""))
        return bool(verdict), ""

    def run(
        self,
        task: str,
        initial_context: dict[str, Any] | None = None,
    ) -> AgentResult:
        if self.config.verification_fn is None:
            return super().run(task, initial_context)

        feedback: str | None = None
        result: AgentResult | None = None
        for attempt in range(self.max_verify_retries + 1):
            framed = task
            if feedback is not None:
                framed = (
                    f"{task}\n\n[A previous answer was rejected by verification: "
                    f"{feedback}\nProduce a corrected answer that addresses this.]"
                )
            result = super().run(framed, initial_context)
            ok, feedback = self._verify(result)
            if ok:
                return result
            logger.info(
                f"Verification rejected attempt {attempt + 1}/"
                f"{self.max_verify_retries + 1}: {feedback}"
            )
        assert result is not None
        return result

    # --- periodic reflection -------------------------------------------
    def _on_loop_iteration(self, api: API, conv_id: str, iteration: int) -> None:
        super()._on_loop_iteration(api, conv_id, iteration)
        n = self.config.reflect_every_n
        if n and iteration > 0 and iteration % n == 0:
            self._inject_reflection(api, conv_id)

    def _inject_reflection(self, api: API, conv_id: str) -> None:
        """Append a reflection note to the agent's observations."""
        try:
            data = api.get_data(conv_id)
            observations = data.get("observations") or []
            if not isinstance(observations, list):
                observations = []
            api.update_context(
                conv_id, {"observations": [*observations, _REFLECTION_NOTE]}
            )
        except Exception as e:  # pragma: no cover - defensive
            logger.debug(f"Reflection injection skipped: {e}")
