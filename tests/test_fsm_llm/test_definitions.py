"""Definition-model behaviour: history summary, not-found error, entity coercer, version and session docs."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from fsm_llm import API
from fsm_llm.core.definitions import FieldExtractionConfig, FSMDefinition, FSMError
from fsm_llm.core.definitions.schema import SCHEMA_VERSION_DEFAULT
from fsm_llm.core.verify import FSMValidator
from tests.test_fsm_llm.support import _a2026_09_22_api, _fsm
from tests.test_fsm_llm.test_completion_state import _RESULT, _tool_fsm


class TestStep7HistorySizeZeroSummary:
    """P1-2: ``max_history_size=0`` digests exchanges before clearing them."""

    def test_p1_2_size_zero_summarises_before_clearing(self):
        from fsm_llm.core.definitions import Conversation

        conv = Conversation(max_history_size=0)
        conv.add_user_message("hello")
        conv.add_system_message("hi there")
        conv.add_user_message("how are you")
        conv.add_system_message("good")
        assert conv.exchanges == []
        assert conv.summary is not None
        assert "user: hello" in conv.summary
        assert "system: good" in conv.summary

    def test_p1_2_size_zero_keeps_the_summary_cap(self):
        from fsm_llm.core.definitions import Conversation

        conv = Conversation(max_history_size=0, summary="x" * 1990)
        conv.add_user_message("hello")
        conv.add_system_message("hi there")
        assert conv.summary is not None
        assert len(conv.summary) == 2000
        assert conv.summary.startswith("x" * 1990 + " | user: ")

    def test_p1_2_summary_already_at_cap_stays_at_cap(self):
        from fsm_llm.core.definitions import Conversation

        conv = Conversation(max_history_size=0, summary="y" * 2000)
        conv.add_user_message("hello")
        conv.add_system_message("hi")
        assert conv.summary == "y" * 2000
        assert conv.exchanges == []


class TestStep7FSMDefinitionNotFoundError:
    """Typed error for the registry-less id branch of ``load_fsm_definition``."""

    def test_id_branch_raises_typed_error_that_is_fsm_and_value_error(self):
        from fsm_llm import FSMDefinitionNotFoundError
        from fsm_llm.core.loaders import load_fsm_definition

        with pytest.raises(FSMDefinitionNotFoundError) as exc:
            load_fsm_definition("no-such-id")
        err = exc.value
        assert isinstance(err, FSMError)
        assert isinstance(err, ValueError)
        assert err.fsm_id == "no-such-id"
        msg = str(err)
        assert "Unknown FSM ID" in msg
        assert "no FSM registry" in msg
        assert "'no-such-id'" in msg

    def test_error_pickles_with_its_attributes(self):
        import pickle

        from fsm_llm.core.definitions import FSMDefinitionNotFoundError

        err = FSMDefinitionNotFoundError("abc123")
        clone = pickle.loads(pickle.dumps(err))
        assert type(clone) is FSMDefinitionNotFoundError
        assert str(clone) == str(err)
        assert clone.fsm_id == "abc123"
        assert clone.details == {"fsm_id": "abc123"}

    def test_api_loader_fallback_raises_the_typed_error(self):
        from fsm_llm.core.definitions import FSMDefinitionNotFoundError

        api = _a2026_09_22_api()
        with pytest.raises(FSMDefinitionNotFoundError):
            api.fsm_manager.fsm_loader("evicted-content-hash")

    def test_exported_in_all(self):
        import fsm_llm

        assert "FSMDefinitionNotFoundError" in fsm_llm.__all__


class TestStep7SharedEntityCoercer:
    """Both entity validators delegate to one module-level coercer."""

    def test_both_validators_delegate_to_the_shared_function(self):
        import fsm_llm.core.definitions.classification as d

        real = d._coerce_entity_values
        with patch.object(d, "_coerce_entity_values", wraps=real) as spy:
            d.IntentScore(intent="a", confidence=0.5, entities={"k": 1})
            d.ClassificationResult(
                reasoning="r", intent="a", confidence=0.5, entities={"k": [1, 2]}
            )
        assert [c.args for c in spy.call_args_list] == [({"k": 1},), ({"k": [1, 2]},)]

    def test_shared_function_behaviour(self):
        from fsm_llm.core.definitions.classification import _coerce_entity_values

        assert _coerce_entity_values({"a": None, "b": 3, "c": ["x", 1]}) == {
            "a": None,
            "b": "3",
            "c": "x, 1",
        }
        assert _coerce_entity_values("junk") == {}


class TestStep7VersionAndSessionDocs:
    def test_architecture_is_2_pass(self):
        from fsm_llm import get_version_info

        assert get_version_info()["architecture"] == "2-pass"

    def test_p1_6_stack_depth_documented_as_advisory(self):
        from fsm_llm.engine.session import SessionState

        description = SessionState.model_fields["stack_depth"].description or ""
        assert "advisory" in description
        assert "never read by restore" in description


def _with_rules(rules: dict[str, Any]) -> dict[str, Any]:
    data = _fsm()
    data["states"]["a"]["field_extractions"] = [
        {
            "field_name": "age",
            "field_type": "int",
            "extraction_instructions": "the age",
            "validation_rules": rules,
        }
    ]
    return data


class TestValidationRuleBoundsAreTypedAtLoad:
    """F-DV-02: a mistyped min_value / max_value no longer disables the check."""

    @pytest.mark.parametrize("key", ["min_value", "max_value"])
    @pytest.mark.parametrize("bad", ["eighteen", None, True, [1], {"x": 1}, "nan"])
    def test_a_non_numeric_bound_is_refused(self, key, bad):
        with pytest.raises(ValueError, match=key):
            FieldExtractionConfig(
                field_name="age",
                extraction_instructions="the age",
                validation_rules={key: bad},
            )

    @pytest.mark.parametrize("good", [18, 18.5, "18", "-0.5", 0])
    def test_a_numeric_bound_still_loads(self, good):
        config = FieldExtractionConfig(
            field_name="age",
            extraction_instructions="the age",
            validation_rules={"min_value": good, "max_value": good},
        )
        assert config.validation_rules == {"min_value": good, "max_value": good}

    def test_loader_and_validator_agree_on_the_bad_bound(self):
        data = _with_rules({"min_value": "eighteen"})
        with pytest.raises(ValidationError):
            FSMDefinition(**data)
        assert FSMValidator(data).validate().is_valid is False


class TestFSMInstancePersonaDefault:
    """F-DV-10: a bare FSMInstance must not out-rank the definition's persona
    (``prompts`` reads ``instance.persona or fsm_definition.persona``)."""

    def test_the_default_is_unset(self):
        from fsm_llm.core.definitions import FSMInstance

        assert FSMInstance(fsm_id="x", current_state="a").persona is None


_REPO = Path(__file__).resolve().parents[2]


# The ids of the 14 example FSMs, computed with this rule over the models
# of e1f63a9 (the iteration-1 close, before `completion` existed): the
# same ids as at HEAD, so the fields this plan added changed none.
#
# Plan 9013b0e3 D-004 re-pinned seven of them (the examples that declare
# `"version": "4.1"` explicitly: classified_transitions x2, smart_helpdesk x2,
# adaptive_quiz, math_tutor, order_processing): the default moved to 4.2, so
# an explicit "4.1" is no longer a default and `version` is now hashed. The
# other seven (six declare "3.0", one omits `version`) kept their ids.
_E1F63A9_IDS: dict[str, str] = {
    "examples/advanced/yoga_instructions/fsm.json": (
        "fsm_Adaptive Yoga Instruction_bb1015b5"
    ),
    "examples/basic/form_filling/fsm.json": "fsm_Form Filling FSM_fbe5e0cc",
    "examples/basic/multi_turn_extraction/fsm.json": (
        "fsm_MultiTurnExtraction_d84a9e85"
    ),
    "examples/basic/simple_greeting/fsm.json": "fsm_Simple Greeting FSM_51af6495",
    "examples/basic/story_time/fsm.json": (
        "fsm_Three Little Pigs Interactive Story_aae22383"
    ),
    "examples/classification/classified_transitions/fsm.json": (
        "fsm_Classified Transitions Demo_1eea499b"
    ),
    "examples/classification/classified_transitions/fsm_manual.json": (
        "fsm_Classified Transitions Manual Mode_6f84fe4f"
    ),
    "examples/classification/smart_helpdesk/fsm_account.json": (
        "fsm_account_management_flow_95ee2191"
    ),
    "examples/classification/smart_helpdesk/fsm_troubleshooting.json": (
        "fsm_troubleshooting_flow_60018c91"
    ),
    "examples/intermediate/adaptive_quiz/fsm.json": "fsm_adaptive_quiz_cb228ef9",
    "examples/intermediate/book_recommendation/fsm.json": (
        "fsm_Book Recommendation System_1bd2d04b"
    ),
    "examples/intermediate/product_recommendation/fsm.json": (
        "fsm_Product Recommendation System_5f2f95d5"
    ),
    "examples/reasoning/math_tutor/fsm.json": "fsm_math_tutor_09bfbec2",
    "examples/workflows/order_processing/fsm_order_form.json": (
        "fsm_order_form_24b14bf2"
    ),
}


class TestFsmIdIgnoresDefaults:
    """Review area 2, concern 3 (D-030): the id hashed the full dump, so
    every optional field added to the models (``completion``) re-ided every
    FSM. It now hashes ``model_dump(exclude_defaults=True)``."""

    @pytest.mark.parametrize("path", sorted(_E1F63A9_IDS))
    def test_example_ids_did_not_move_with_the_new_fields(self, path: str):
        raw = json.loads((_REPO / path).read_text(encoding="utf-8"))
        _, fsm_id = API.process_fsm_definition(raw)
        assert fsm_id == _E1F63A9_IDS[path]

    def test_an_explicit_default_and_an_omitted_field_hash_the_same(self):
        omitted = _tool_fsm()
        explicit = copy.deepcopy(omitted)
        explicit["version"] = SCHEMA_VERSION_DEFAULT
        explicit["states"]["intake"]["completion"] = None
        explicit["states"]["intake"]["extraction_retries"] = 1
        explicit["states"]["call_model"]["completion"]["result_key"] = _RESULT
        assert (
            API.process_fsm_definition(explicit)[1]
            == API.process_fsm_definition(omitted)[1]
        )

    def test_an_omitted_version_keeps_its_id_across_the_default_bump(self):
        # Plan 9013b0e3 D-004: the default moved 4.1 -> 4.2. The id of a
        # definition that omits `version` is the one computed under 4.1
        # (pinned below), and an explicit default hashes the same as omitted.
        omitted = _tool_fsm()
        explicit = copy.deepcopy(omitted)
        explicit["version"] = SCHEMA_VERSION_DEFAULT
        assert API.process_fsm_definition(omitted)[1] == "fsm_ToolTurns_4913b97e"
        assert (
            API.process_fsm_definition(explicit)[1]
            == API.process_fsm_definition(omitted)[1]
        )

    def test_an_explicit_older_version_has_its_own_id(self):
        # The documented cost of the bump (plan 9013b0e3 D-004): "4.1" is no
        # longer the default, so it is hashed and moves the id.
        omitted = _tool_fsm()
        older = copy.deepcopy(omitted)
        older["version"] = "4.1"
        assert (
            API.process_fsm_definition(older)[1]
            != API.process_fsm_definition(omitted)[1]
        )

    def test_different_content_still_gets_a_different_id(self):
        changed = _tool_fsm()
        changed["states"]["intake"]["extraction_retries"] = 2
        assert (
            API.process_fsm_definition(changed)[1]
            != API.process_fsm_definition(_tool_fsm())[1]
        )

    def test_construction_paths_agree(self, tmp_path: Path):
        raw = _tool_fsm()
        path = tmp_path / "fsm.json"
        path.write_text(json.dumps(raw), encoding="utf-8")
        ids = {
            API.process_fsm_definition(raw)[1],
            API.process_fsm_definition(FSMDefinition(**raw))[1],
            API.process_fsm_definition(str(path))[1],
        }
        assert len(ids) == 1
