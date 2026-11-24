"""The core ``API`` of the meta-builder agent and its four handlers (plan d454fd62 step 90; the old ``agents/meta_builder.py``, same dotted path).

``_create_api`` builds the ``API`` over the meta FSM and registers the handlers: ``classify`` entry
(``_enter_classify``) and exit (``_resolve_artifact_type``), ``build`` entry (``_write_build_request``) and
the critical ``build_reply`` judge (``_judge_build_reply``, D-035). ``HandlersMixin`` holds them.
"""

from __future__ import annotations

from typing import Any

from fsm_llm import API
from fsm_llm.agents.constants import (
    META_BUILD_CALL_FAILED,
    META_BUILD_OUTPUT_KEYS,
    MetaBuildOutcome,
    MetaHandlerNames,
    MetaLogMessages,
)
from fsm_llm.agents.definitions import ArtifactType
from fsm_llm.agents.fsm_definitions.meta_builder import build_meta_builder_fsm
from fsm_llm.agents.meta_prompts import (
    build_artifact_prompt,
    build_response_format,
    build_review_presentation,
)
from fsm_llm.context import clear_keys_delta
from fsm_llm.engine.api.helpers import llm_settings_for
from fsm_llm.logging import logger

from .names import _K, _S, _TYPE_VALUES
from .surface import _MetaBuilderSurface


class HandlersMixin(_MetaBuilderSurface):
    """The core ``API`` over the meta FSM and the handlers that write the build request and judge the
    build reply."""

    def _create_api(self) -> API:
        """The core ``API`` over the meta FSM, with the build handlers."""
        # An injected interface owns its timeout: core refuses connection
        # settings beside it (D-029), so the config timeout applies only to
        # the interface core builds.
        api_kwargs: dict[str, Any] = (
            dict(self._api_kwargs)
            if self._api_kwargs.get("llm_interface") is not None
            else {"timeout": self.meta_config.timeout_seconds, **self._api_kwargs}
        )
        api = API.from_definition(
            build_meta_builder_fsm(),
            **llm_settings_for(
                api_kwargs,
                model=self.meta_config.model,
                temperature=self.meta_config.temperature,
                max_tokens=self.meta_config.max_tokens,
            ),
            **api_kwargs,
        )
        api.register_handler(
            api.create_handler(MetaHandlerNames.CLASSIFY_ENTRY)
            .on_state_entry(_S.CLASSIFY)
            .do(self._enter_classify)
        )
        api.register_handler(
            api.create_handler(MetaHandlerNames.CLASSIFY_EXIT)
            .on_state_exit(_S.CLASSIFY)
            .do(self._resolve_artifact_type)
        )
        api.register_handler(
            api.create_handler(MetaHandlerNames.BUILD_ENTRY)
            .on_state_entry(_S.BUILD)
            .do(self._write_build_request)
        )
        api.register_handler(
            api.create_handler(MetaHandlerNames.BUILD_REPLY)
            .on_state(_S.BUILD)
            .on_context_update(_K.BUILD_REPLY)
            .critical()
            .do(self._judge_build_reply)
        )
        return api

    @staticmethod
    def _enter_classify(context: dict[str, Any]) -> dict[str, Any]:
        """``classify`` entry (a type switch): stash the type, drop the build.

        The old build's outputs belong to the old type. ``artifact_type``
        itself stays set (D-023: a clear would classify twice per switch);
        its value is stashed so the exit can restore it.
        """
        return {
            **clear_keys_delta(META_BUILD_OUTPUT_KEYS, context),
            _K.PREVIOUS_ARTIFACT_TYPE: context.get(_K.ARTIFACT_TYPE),
        }

    @staticmethod
    def _resolve_artifact_type(context: dict[str, Any]) -> dict[str, Any]:
        """``classify`` exit: the classified type, else the previous, else keyword.

        A classified ``fsm``/``workflow``/``agent`` wins. Otherwise (the
        ``unknown`` fallback, an unrecognised reply, a discarded
        low-confidence intent, or a provider failure) the type stashed on
        entry is kept; with none (the first classification) the driver's
        ``keyword_type`` is used, FSM when no alias matched.
        """
        # DECISION plan-2026-10-01T093600-944e2692/D-023: a type switch
        # re-enters `classify` with `artifact_type` still set. Do NOT clear
        # `artifact_type` on entry: core classifies an unset classification
        # field of a newly entered state in the same turn (post-transition,
        # 8a03483a/D-006), so a clear plus the driver's `advance` (which the
        # collect reply needs) would classify twice per switch. The one
        # classification is the `advance`'s. A switch classification that
        # falls back (`unknown`, D-035), is below the threshold, or fails
        # keeps the previous type (stashed on entry). See decisions.md D-023.
        classified = context.get(_K.ARTIFACT_TYPE)
        if classified in _TYPE_VALUES:
            resolved = classified
        else:
            previous = context.get(_K.PREVIOUS_ARTIFACT_TYPE)
            keyword = context.get(_K.KEYWORD_TYPE)
            resolved = (
                previous
                if previous in _TYPE_VALUES
                else keyword
                if keyword in _TYPE_VALUES
                else ArtifactType.FSM.value
            )
            logger.debug(f"No classified artifact type; using '{resolved}'")
        return {_K.ARTIFACT_TYPE: resolved, _K.PREVIOUS_ARTIFACT_TYPE: None}

    @staticmethod
    def _write_build_request(context: dict[str, Any]) -> dict[str, Any]:
        """``build`` entry: the build request, and every old output cleared.

        Clearing ``build_reply`` on every entry makes a retry after a failed
        build call the model again: core skips a completion state whose
        result key is set.
        """
        artifact_type = ArtifactType(context[_K.ARTIFACT_TYPE])
        requirement = "\n".join(context.get(_K.REQUIREMENTS) or [])
        logger.info(
            MetaLogMessages.BUILD_STARTED.format(artifact_type=artifact_type.value)
        )
        return {
            **clear_keys_delta(META_BUILD_OUTPUT_KEYS, context),
            _K.BUILD_MESSAGES: [
                {
                    "role": "user",
                    "content": build_artifact_prompt(artifact_type, requirement),
                }
            ],
            _K.BUILD_RESPONSE_FORMAT: build_response_format(
                artifact_type, context.get(_K.AGENT_TYPES) or ()
            ),
        }

    @classmethod
    def _judge_build_reply(cls, context: dict[str, Any]) -> dict[str, Any]:
        """``build_reply`` committed: parse, assemble on a fresh builder, judge.

        Writes ``build_outcome``, ``artifact``, ``validation_errors``,
        ``build_progress``, ``build_summary`` and, for a valid build,
        ``review_presentation``. The reply is untrusted: a JSON-schema echo
        or a wrong-typed field is a ``malformed`` outcome whose errors name
        the problem; nothing in the reply can make this handler raise.
        """
        # DECISION plan-2026-10-01T093600-944e2692/D-035: every reply is
        # assembled on a fresh builder and every output goes into context.
        # Do NOT keep a builder, an artifact type or a completion flag on the
        # agent: such a mirror sits outside core's rollback and drifted from
        # the FSM (a failed build's states made every retry fail as orphans;
        # `done` with `is_complete()` False). Do NOT let model output raise
        # here: a raise left the session in `build`. See decisions.md D-035.
        reply = context.get(_K.BUILD_REPLY)
        if not isinstance(reply, dict) or reply.get("kind") == META_BUILD_CALL_FAILED:
            # Cleared on entry, or the driver's record of a failed call
            # (``update_context`` runs CONTEXT_UPDATE handlers too).
            return {}
        artifact_type = ArtifactType(context[_K.ARTIFACT_TYPE])
        requirement = "\n".join(context.get(_K.REQUIREMENTS) or [])
        text = reply.get("text")
        builder = cls._create_builder(artifact_type, context.get(_K.AGENT_TYPES) or ())
        outcome, errors = cls._assemble_reply(
            builder, artifact_type, text if isinstance(text, str) else "", requirement
        )
        progress = builder.get_progress()
        valid = outcome == MetaBuildOutcome.VALID
        return {
            _K.BUILD_OUTCOME: outcome,
            _K.ARTIFACT: builder.to_dict(),
            _K.VALIDATION_ERRORS: errors,
            _K.REVIEW_PRESENTATION: (
                build_review_presentation(builder, artifact_type) if valid else None
            ),
            _K.BUILD_PROGRESS: {
                "percentage": progress.percentage,
                "completed": progress.completed,
                "total_required": progress.total_required,
                "missing": builder.get_missing_fields(),
                "warnings": progress.warnings,
            },
            _K.BUILD_SUMMARY: builder.get_summary(detail_level="standard"),
        }
