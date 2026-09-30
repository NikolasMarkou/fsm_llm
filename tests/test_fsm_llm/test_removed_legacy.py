"""Absence pins for names removed by the "no legacy" cleanup.

Each test fails on the commit before the removal: the name existed there.
"""

from __future__ import annotations

import importlib
from typing import Any
from unittest.mock import patch

import pytest

import fsm_llm
from fsm_llm import API, FSMManager
from fsm_llm.definitions import ClassificationResult
from fsm_llm.handlers import HandlerSystem
from fsm_llm.transition_evaluator import TransitionEvaluatorConfig

_FSM: dict[str, Any] = {
    "name": "two_way",
    "description": "Start state with two unconditional exits of equal priority",
    "initial_state": "start",
    "persona": "A terse assistant",
    "states": {
        "start": {
            "id": "start",
            "description": "Pick a branch",
            "purpose": "Route the user",
            "response_instructions": "Ask which branch",
            "transitions": [
                {"target_state": "a", "description": "Go to a", "priority": 100},
                {"target_state": "b", "description": "Go to b", "priority": 100},
            ],
        },
        "a": {
            "id": "a",
            "description": "Branch a",
            "purpose": "Finish on a",
            "response_instructions": "Confirm a",
        },
        "b": {
            "id": "b",
            "description": "Branch b",
            "purpose": "Finish on b",
            "response_instructions": "Confirm b",
        },
    },
}


class TestRemovedCoreNames:
    """One absence test per removed core name."""

    def test_manager_cleanup_stale_conversations_is_gone(self):
        assert not hasattr(FSMManager, "cleanup_stale_conversations")
        # The live idle sweep and the lock bookkeeping stay.
        assert callable(API.cleanup_stale_conversations)
        assert callable(FSMManager.prune_orphaned_locks)

    def test_classification_result_is_low_confidence_is_gone(self):
        result = ClassificationResult(reasoning="r", intent="a", confidence=0.1)
        with pytest.raises(AttributeError):
            result.is_low_confidence  # noqa: B018
        assert result.is_below_default_threshold is True

    def test_schema_validation_error_is_gone(self):
        assert "SchemaValidationError" not in fsm_llm.__all__
        with pytest.raises(ImportError):
            from fsm_llm import SchemaValidationError
        with pytest.raises(ImportError):
            from fsm_llm.definitions import SchemaValidationError  # noqa: F401

    @pytest.mark.parametrize(
        "field",
        ["ambiguity_threshold", "minimum_confidence", "evidence_conditions_normalizer"],
    )
    def test_transition_evaluator_config_noop_field_is_rejected(self, field):
        with pytest.raises(TypeError):
            TransitionEvaluatorConfig(**{field: 0.3})
        kept = TransitionEvaluatorConfig(
            strict_condition_matching=False, detailed_logging=True
        )
        assert kept.strict_condition_matching is False

    def test_from_definition_requires_fsm_definition(self, mock_llm2_interface):
        # `definition=` used to be an alias; it is now just an unknown kwarg
        # and the real parameter is required.
        with pytest.raises(TypeError):
            API.from_definition(definition=_FSM, llm_interface=mock_llm2_interface)
        with pytest.raises(TypeError):
            API.from_definition(llm_interface=mock_llm2_interface)
        api = API.from_definition(_FSM, llm_interface=mock_llm2_interface)
        assert api.fsm_definition.name == "two_way"

    def test_handler_system_close_is_gone(self, mock_llm2_interface):
        assert not hasattr(HandlerSystem, "close")
        # API.close still ends every conversation without it.
        api = API.from_definition(_FSM, llm_interface=mock_llm2_interface)
        conv_id, _ = api.start_conversation()
        api.close()
        assert conv_id not in api.active_conversations

    @pytest.mark.parametrize(
        "name",
        [
            "_looks_like_credential_value",
            "_token_value_is_credential",
            "_TOKEN_VALUE_SCAN_NAME_RE",
            "_CREDENTIAL_VALUE_PREFIXES",
            "_shannon_entropy",
        ],
    )
    def test_private_security_helper_not_importable_from_constants(self, name):
        constants = importlib.import_module("fsm_llm.constants")
        security = importlib.import_module("fsm_llm.security")
        assert not hasattr(constants, name)
        assert hasattr(security, name)
        # The documented public path stays.
        assert constants.has_internal_prefix is security.has_internal_prefix
        assert (
            constants.is_forbidden_context_entry is security.is_forbidden_context_entry
        )

    def test_pipeline_provenance_alias_is_gone(self):
        pipeline = importlib.import_module("fsm_llm.pipeline")
        assert not hasattr(pipeline, "_PROVENANCE_KEY")

    def test_classification_module_does_not_reexport_intent_definition(self):
        with pytest.raises(ImportError):
            from fsm_llm.classification import IntentDefinition  # noqa: F401
        from fsm_llm import IntentDefinition as public
        from fsm_llm.definitions import IntentDefinition as home

        assert public is home

    def test_transition_classification_record_lives_in_metadata_only(
        self, mock_llm2_interface
    ):
        constants = importlib.import_module("fsm_llm.constants")
        assert not hasattr(constants, "CONTEXT_KEY_CLASSIFICATION_RESULT")

        # The transition classifier takes its model from the interface.
        mock_llm2_interface.model = "gpt-4"
        api = API.from_definition(_FSM, llm_interface=mock_llm2_interface)
        conv_id, _ = api.start_conversation()
        with patch("fsm_llm.pipeline.Classifier") as mock_cls:
            mock_cls.return_value.classify.return_value = ClassificationResult(
                reasoning="picked a", intent="a", confidence=0.95
            )
            api.converse("a please", conv_id)

        assert api.get_current_state(conv_id) == "a"
        context = api.fsm_manager.instances[conv_id].context
        assert "_transition_classification_result" not in context.data
        record = context.metadata[constants.METADATA_KEY_TRANSITION_CLASSIFICATION]
        assert record["intent"] == "a"
        assert record["confidence"] == pytest.approx(0.95)
