"""
VerifiedReactAgent — a ReAct variant with answer verification + periodic
self-reflection.

Two additive reliability features, both driven by optional ``AgentConfig``
fields (defaults reproduce stock ReactAgent behavior):

- **Verification / grounding** (``config.verification_fn``): after a run, the
  answer is checked by a caller-supplied predicate. On rejection the agent
  retries with the verifier's feedback folded into the task, up to
  ``max_verify_retries`` times. An answer still rejected after the last
  attempt ships with ``success=False, stop_reason="verification_failed"``; a
  verifier that raises counts as a rejection (fail closed).
- **Periodic self-reflection** (``config.reflect_every_n``): every N loop
  iterations a reflection note is written to ``agent_feedback`` (which the
  think prompt surfaces for one turn), nudging it to re-assess progress and
  avoid repeating failed actions.

Fully additive: a thin :class:`ReactAgent` subclass bound to the preset file
``patterns/verified_react.json``, whose hooks (``agents.verify_retry``,
``agents.periodic_reflection``, ``handlers/hooks.py``) do the work. With neither
config field set it behaves exactly like :class:`ReactAgent`.

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

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from fsm_llm.tools import require_int_limit

from .react import ReactAgent


class VerifiedReactAgent(ReactAgent):
    """ReAct agent with verify-and-retry and periodic self-reflection.

    Args:
        max_verify_retries: Extra attempts after the first when
            ``config.verification_fn`` rejects the answer (total attempts =
            ``max_verify_retries + 1``).
        **kwargs: Forwarded to :class:`ReactAgent`.
    """

    _pattern = "verified_react"

    def __init__(self, *args: Any, max_verify_retries: int = 1, **kwargs: Any) -> None:
        max_verify_retries = require_int_limit(
            "max_verify_retries", max_verify_retries, minimum=0
        )
        self._preset_options = {"max_verify_retries": max_verify_retries}
        super().__init__(*args, **kwargs)
        self.max_verify_retries = max_verify_retries

    def run_stream(
        self,
        task: str,
        initial_context: dict[str, Any] | None = None,
    ) -> Iterator[str]:
        """Stream the verified answer.

        With ``config.verification_fn`` set, the answer is only known once a
        verified (or last) attempt is done, so this runs :meth:`run` and
        yields its final answer once. Without a verifier it streams like
        :class:`ReactAgent` (periodic reflection still applies).
        """
        if self.config.verification_fn is None:
            return super().run_stream(task, initial_context)
        return self._stream_via_run(task, initial_context)
