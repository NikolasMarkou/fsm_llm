"""The result and state readers of the meta-builder agent (plan d454fd62 step 90; the old ``agents/meta_builder.py``, same dotted path).

``is_complete``, ``get_result``, ``get_internal_state`` (for the monitor), ``run_interactive`` (a session
on stdin and stdout) and ``_result_from`` (the result of the last build recorded in a context).
``ResultsMixin`` holds them.
"""

from __future__ import annotations

from typing import Any

from fsm_llm.agents.constants import MetaBuildOutcome
from fsm_llm.agents.definitions import ArtifactType, MetaBuilderResult
from fsm_llm.agents.exceptions import MetaBuilderError
from fsm_llm.agents.meta_output import format_artifact_json

from .names import _K, _S, _TYPE_VALUES
from .surface import _MetaBuilderSurface


class ResultsMixin(_MetaBuilderSurface):
    """Completion, the build result, the monitor's session state and the interactive session."""

    def is_complete(self) -> bool:
        """True once ``run`` finished or the session's build reached ``done``."""
        if self._result is not None:
            return True
        if self._api is None or self._conversation_id is None:
            return False
        state: str = self._api.get_current_state(self._conversation_id)
        return state == _S.DONE

    def get_result(self) -> MetaBuilderResult:
        """The build result; raises ``MetaBuilderError`` before completion."""
        if not self.is_complete():
            raise MetaBuilderError(
                "Build is not complete. Say 'build it' to trigger the build."
            )
        if self._result is not None:
            return self._result
        return self._result_from(self._session_data())

    def get_internal_state(self) -> dict[str, Any]:
        """Session state for the monitor: phase, turns, type, build progress.

        Read from the session's context (after ``run``, only the phase and
        completion are reported).
        """
        data = self._session_data()
        complete = self.is_complete()
        result: dict[str, Any] = {
            "phase": "complete" if complete else "collecting",
            "turn_count": self._turn_count,
            "is_complete": complete,
            "started": self._api is not None,
            "message_count": len(data.get(_K.REQUIREMENTS) or []),
        }
        if data.get(_K.ARTIFACT_TYPE) in _TYPE_VALUES:
            result["artifact_type"] = data[_K.ARTIFACT_TYPE]
        artifact = data.get(_K.ARTIFACT)
        built = isinstance(artifact, dict)
        result["builder_progress"] = data.get(_K.BUILD_PROGRESS) if built else None
        result["builder_summary"] = data.get(_K.BUILD_SUMMARY) if built else None
        result["artifact_preview"] = artifact if built else None
        result["validation_errors"] = (
            list(data.get(_K.VALIDATION_ERRORS) or []) if built else []
        )
        result["is_valid"] = data.get(_K.BUILD_OUTCOME) == MetaBuildOutcome.VALID
        return result

    def run_interactive(self) -> MetaBuilderResult:
        """Run a session on stdin/stdout until the build completes or EOF.

        Ctrl-C (``KeyboardInterrupt``) propagates to the caller, at the
        prompt as during a model call; EOF ends the session normally.
        """
        response = self.start()
        print(f"\n{response}\n")

        while not self.is_complete():
            try:
                user_input = input("> ")
            except EOFError:
                print("\nSession ended by user.")
                break
            if not user_input.strip():
                continue
            response = self.send(user_input)
            print(f"\n{response}\n")

        if self.is_complete():
            return self.get_result()
        return self._result_from(self._session_data())

    def _result_from(self, data: dict[str, Any]) -> MetaBuilderResult:
        """The result of the last build recorded in context ``data``."""
        value = data.get(_K.ARTIFACT_TYPE)
        artifact_type = (
            ArtifactType(value) if value in _TYPE_VALUES else ArtifactType.FSM
        )
        artifact = data.get(_K.ARTIFACT)
        if not isinstance(artifact, dict):
            return MetaBuilderResult(
                answer="Build was not completed",
                success=False,
                artifact_type=artifact_type,
                artifact={},
                artifact_json="{}",
                is_valid=False,
                validation_errors=["Builder was not initialized"],
                conversation_turns=self._turn_count,
                final_context={},
            )
        errors = list(data.get(_K.VALIDATION_ERRORS) or [])
        valid = data.get(_K.BUILD_OUTCOME) == MetaBuildOutcome.VALID
        artifact_json = format_artifact_json(artifact)
        return MetaBuilderResult(
            answer=artifact_json,
            success=valid,
            artifact_type=artifact_type,
            artifact=artifact,
            artifact_json=artifact_json,
            is_valid=valid,
            validation_errors=errors,
            conversation_turns=self._turn_count,
            final_context={
                "artifact_json": artifact,
                "artifact_type": artifact_type.value,
                "is_valid": valid,
                "validation_errors": errors,
            },
        )
