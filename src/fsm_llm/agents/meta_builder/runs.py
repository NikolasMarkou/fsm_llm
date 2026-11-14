"""The single-shot API of the meta-builder agent: ``run`` and the build step (plan d454fd62 step 90; the old ``agents/meta_builder.py``, same dotted path).

``run`` is one type classification and one build call; ``_advance_build`` runs the ``build`` state's step
and maps a provider outage or a failed assembly handler to ``BuilderError`` (D-006). The turn-by-turn
build turn calls ``_advance_build`` too. ``RunMixin`` holds them.
"""

from __future__ import annotations

from typing import Any

from fsm_llm import API
from fsm_llm.agents.constants import MetaBuildOutcome, MetaLogMessages
from fsm_llm.agents.definitions import MetaBuilderResult
from fsm_llm.agents.exceptions import BuilderError, MetaValidationError
from fsm_llm.core.definitions import LLMResponseError
from fsm_llm.handlers import HandlerExecutionError
from fsm_llm.logging import logger

from .names import _K
from .surface import _MetaBuilderSurface


class RunMixin(_MetaBuilderSurface):
    """The single-shot API: ``run`` and the ``build`` step shared with the turn-by-turn session."""

    def run(
        self,
        task: str,
        initial_context: dict[str, Any] | None = None,
    ) -> MetaBuilderResult:
        """Build an artifact from ``task`` in one go.

        One type classification and one build call. ``initial_context`` is
        accepted for the agent call shape and not used. The run is complete
        afterwards, valid or not.

        Raises:
            BuilderError: the build call failed (chained from core's
                ``LLMResponseError``), or its reply could not be assembled.
            MetaValidationError: the model returned no artifact of the
                schema: a JSON schema, or a field of the wrong type
                (``errors`` lists each problem).
        """
        logger.info(MetaLogMessages.META_STARTED.format(model=self.meta_config.model))
        api = self._create_api()
        try:
            conversation_id, _ = api.start_conversation(
                initial_context=self._turn_hints(task, [task], build_requested=True)
            )
            # classify -> build (entry writes the request), then the build call.
            api.converse(task, conversation_id)
            self._advance_build(api, conversation_id)
            data = api.get_data(conversation_id)
        finally:
            api.close()
        if data.get(_K.BUILD_OUTCOME) == MetaBuildOutcome.MALFORMED:
            errors = list(data.get(_K.VALIDATION_ERRORS) or [])
            raise MetaValidationError("; ".join(errors), errors=errors)
        self._result = self._result_from(data)
        return self._result

    @staticmethod
    def _advance_build(api: API, conversation_id: str) -> None:
        """Run the ``build`` state's step (the build call and the assembly).

        Raises:
            BuilderError: the build call failed or the assembly handler
                raised; the step was rolled back, so the conversation is
                still in ``build`` with no result.
        """
        try:
            api.advance(conversation_id)
        except LLMResponseError as e:
            # DECISION plan-2026-07-20T040150-876e7164/D-006 [STALE]: a provider
            # outage is an error, never an empty answer. Do NOT turn it into
            # `""` or an empty spec: `run()` would then emit a stub artifact
            # as if the build had merely underperformed. `""` keeps its one
            # meaning: the model answered with nothing (an invalid build).
            raise BuilderError(f"Meta-builder LLM call failed: {e!s}") from e
        except HandlerExecutionError as e:
            raise BuilderError(
                f"Meta-builder could not assemble the build reply: {e!s}"
            ) from e
