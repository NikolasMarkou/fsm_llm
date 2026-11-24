"""The declarations the mixins of ``MetaBuilderAgent`` share (mypy only; plan d454fd62 step 90; the old ``agents/meta_builder.py``, same dotted path).

What a mixin of the old ``fsm_llm.agents.meta_builder.MetaBuilderAgent`` may rely on: the attributes
the constructor of the shell sets and the methods the other mixins define, declared for mypy only (a
runtime no-op, like ``agents/base/surface.py``). A mixin that defines one of these is checked against
the declaration here; one that calls a method of another mixin is checked against it too.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fsm_llm import API
    from fsm_llm.agents.definitions import (
        ArtifactType,
        MetaBuilderConfig,
        MetaBuilderResult,
    )
    from fsm_llm.agents.meta_builders import (
        AgentArtifactBuilder,
        ArtifactBuilder,
        FSMArtifactBuilder,
        WorkflowArtifactBuilder,
    )


class _MetaBuilderSurface:
    """What every mixin of ``MetaBuilderAgent`` may rely on: the attributes of the shell and the
    methods of the other mixins, declared for mypy only (a runtime no-op).
    """

    if TYPE_CHECKING:
        meta_config: MetaBuilderConfig
        _api_kwargs: dict[str, Any]
        _api: API | None
        _conversation_id: str | None
        _result: MetaBuilderResult | None
        _turn_count: int
        _agent_types: tuple[str, ...]

        # HintsMixin (hints.py)
        @staticmethod
        def _detect_type_fallback(text: str) -> ArtifactType: ...

        def _turn_hints(
            self, message: str, requirements: list[str], *, build_requested: bool | None
        ) -> dict[str, Any]: ...

        # HandlersMixin (handlers.py)
        def _create_api(self) -> API: ...

        # AssemblyMixin (assembly.py)
        @classmethod
        def _assemble_reply(
            cls,
            builder: ArtifactBuilder,
            artifact_type: ArtifactType,
            text: str,
            requirement: str,
        ) -> tuple[str, list[str]]: ...

        @staticmethod
        def _create_builder(
            artifact_type: ArtifactType, agent_types: Iterable[str] = ()
        ) -> FSMArtifactBuilder | WorkflowArtifactBuilder | AgentArtifactBuilder: ...

        # RunMixin (runs.py)
        @staticmethod
        def _advance_build(api: API, conversation_id: str) -> None: ...

        # SessionMixin (session.py)
        def start(self, initial_message: str = "") -> str: ...

        def send(self, message: str) -> str: ...

        def _session_data(self) -> dict[str, Any]: ...

        # ResultsMixin (results.py)
        def is_complete(self) -> bool: ...

        def _result_from(self, data: dict[str, Any]) -> MetaBuilderResult: ...
