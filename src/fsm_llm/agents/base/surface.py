"""The declarations the mixins of ``BaseAgent`` share (mypy only).

Layer ``agents`` (plan d454fd62, step 87): what a mixin of the old ``fsm_llm.agents.base.BaseAgent`` may
rely on: the attributes the constructor of the shell sets and the methods the other classes define,
declared for mypy only (a runtime no-op, like ``engine/api/surface.py``). A mixin that defines one of
these is checked against the declaration here; one that calls a method of another class is checked
against it too.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fsm_llm import API
    from fsm_llm.agents.constants import StopReason
    from fsm_llm.agents.definitions import AgentConfig, AgentTrace


class _AgentSurface:
    """What every mixin of ``BaseAgent`` may rely on: the attributes of the shell and the methods of
    the other classes, declared for mypy only (a runtime no-op).
    """

    if TYPE_CHECKING:
        config: AgentConfig
        _api_kwargs: dict[str, Any]

        # the shell (agent.py)
        def _seconds_exempt_for(self, api: API) -> frozenset[str]: ...

        def _refuse_prompt_cache_with_injected_interface(self) -> None: ...

        def _register_handlers(self, api: API) -> None: ...

        @staticmethod
        def _run_outcome(
            final_context: dict[str, Any],
            trace: AgentTrace,
            extra_answer_keys: list[str] | None,
            execution_evidence_keys: list[str] | None = None,
            judged: bool = False,
            answer_cut_off: bool = False,
        ) -> tuple[bool, StopReason]: ...

        # HitlGateMixin (hitl_gate.py)
        @property
        def _hitl_active(self) -> bool: ...

        def _handle_hitl_approval(self, api: API, conversation_id: str) -> None: ...

        # RegistrationMixin (registration.py)
        def _register_lifecycle_handlers(self, api: API, agent_type: str) -> None: ...

        # OutcomeMixin (outcome.py)
        def _extract_answer(
            self,
            final_context: dict[str, Any],
            responses: list[str],
            extra_keys: list[str] | None = None,
        ) -> str: ...

        def _answer_was_cut_off(
            self, final_context: dict[str, Any], extra_answer_keys: list[str] | None
        ) -> bool: ...

        def _build_trace(
            self, final_context: dict[str, Any], iteration: int
        ) -> AgentTrace: ...

        @staticmethod
        def _filter_context(context: dict[str, Any]) -> dict[str, Any]: ...

        def _try_parse_structured_output(
            self, answer: str, context: dict[str, Any] | None = None
        ) -> Any: ...
