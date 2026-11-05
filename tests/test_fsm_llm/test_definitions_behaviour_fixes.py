"""Definitions, validator and loader behaviour fixes (plan 5980fbb7, iteration 1, step 6).

Every test here fails on the parent commit 8c8162b.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from fsm_llm.definitions import FieldExtractionConfig, FSMDefinition
from fsm_llm.validator import FSMValidator, validate_fsm_from_file


def _fsm() -> dict[str, Any]:
    return {
        "name": "t",
        "description": "d",
        "initial_state": "a",
        "states": {
            "a": {
                "id": "a",
                "description": "a",
                "purpose": "p",
                "transitions": [{"target_state": "b", "description": "go"}],
            },
            "b": {"id": "b", "description": "b", "purpose": "p"},
        },
    }


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


class TestValidatorAllowListCoversTheLoaderErrorTypes:
    """F-DV-04: shapes the loader refuses are errors in fsm-llm-validate."""

    @staticmethod
    def _agrees_on_refusal(data: dict[str, Any], expected_type: str) -> None:
        with pytest.raises(ValidationError) as caught:
            FSMDefinition(**data)
        assert {e["type"] for e in caught.value.errors()} == {expected_type}
        assert FSMValidator(data).validate().is_valid is False

    def test_an_oversized_integer_string(self):
        data = _fsm()
        data["states"]["a"]["transitions"][0]["priority"] = "9" * 5000
        self._agrees_on_refusal(data, "int_parsing_size")

    def test_a_lone_surrogate_in_a_string(self):
        data = _fsm()
        data["name"] = "\ud800"
        self._agrees_on_refusal(data, "string_unicode")

    def test_a_fractional_float_for_an_integer(self):
        data = _fsm()
        data["states"]["a"]["transitions"][0]["priority"] = 1.5
        self._agrees_on_refusal(data, "int_from_float")

    def test_an_infinite_number(self):
        data = _fsm()
        data["states"]["a"]["transitions"][0]["priority"] = float("inf")
        self._agrees_on_refusal(data, "finite_number")


class TestEachStructuralDefectIsReportedOnce:
    """F-DV-07: the Schema line and the structure stage no longer both say it."""

    @staticmethod
    def _errors(data: dict[str, Any]) -> list[str]:
        return FSMValidator(data).validate().errors

    def test_a_dangling_target_is_named_once(self):
        data = _fsm()
        data["states"]["a"]["transitions"][0]["target_state"] = "zz"
        named = [e for e in self._errors(data) if "'zz'" in e]
        assert len(named) == 1, named

    def test_an_orphan_is_named_once(self):
        data = _fsm()
        data["states"]["c"] = {"id": "c", "description": "c", "purpose": "p"}
        named = [e for e in self._errors(data) if "Orphaned" in e]
        assert len(named) == 1, named

    def test_an_unknown_initial_state_is_named_once(self):
        data = _fsm()
        data["initial_state"] = "zz"
        named = [e for e in self._errors(data) if "'zz'" in e]
        assert len(named) == 1, named

    def test_an_empty_definition_reports_the_initial_state_once(self):
        named = [e for e in self._errors({}) if "initial" in e.lower()]
        assert len(named) == 1, named

    def test_the_verdict_and_the_error_set_of_a_second_defect_survive(self):
        data = _fsm()
        data["states"]["a"]["transitions"][0]["target_state"] = "zz"
        data["states"]["c"] = {"id": "c", "description": "c", "purpose": "p"}
        result = FSMValidator(data).validate()
        assert result.is_valid is False
        assert any("'zz'" in e for e in result.errors)
        assert any("Orphaned" in e for e in result.errors)


class TestTransitionClassificationKeysNamingNoTargetWarn:
    """F-DV-08: a typo'd key is a WARNING, never an error."""

    def test_a_key_that_is_no_target_warns(self):
        data = _fsm()
        data["states"]["a"]["transition_classification"] = {
            "typo_state": {"description": "x"}
        }
        result = FSMValidator(data).validate()
        assert result.is_valid is True
        assert any("typo_state" in w for w in result.warnings), result.warnings

    def test_a_key_naming_a_target_does_not_warn(self):
        data = _fsm()
        data["states"]["a"]["transition_classification"] = {"b": {"description": "x"}}
        result = FSMValidator(data).validate()
        assert not any("transition_classification" in w for w in result.warnings)


class TestValidateFromFileGivesCleanMessages:
    """F-DV-16: a non-object file and undecodable bytes have a clean message."""

    def test_a_json_array_file(self, tmp_path: Path):
        path = tmp_path / "list.json"
        path.write_text("[]", encoding="utf-8")
        result = validate_fsm_from_file(str(path))
        assert result.is_valid is False
        assert result.errors == ["FSM definition must be a JSON object"]

    def test_undecodable_bytes(self, tmp_path: Path):
        path = tmp_path / "bad.json"
        path.write_bytes(b"\xff\xfe\x00{")
        result = validate_fsm_from_file(str(path))
        assert result.is_valid is False
        assert len(result.errors) == 1
        assert "UTF-8" in result.errors[0]
        assert "codec" not in result.errors[0]

    def test_a_valid_file_is_untouched(self, tmp_path: Path):
        path = tmp_path / "ok.json"
        path.write_text(json.dumps(copy.deepcopy(_fsm())), encoding="utf-8")
        assert validate_fsm_from_file(str(path)).is_valid is True


class TestFSMInstancePersonaDefault:
    """F-DV-10: a bare FSMInstance must not out-rank the definition's persona
    (``prompts`` reads ``instance.persona or fsm_definition.persona``)."""

    def test_the_default_is_unset(self):
        from fsm_llm.definitions import FSMInstance

        assert FSMInstance(fsm_id="x", current_state="a").persona is None


class TestSharedGraphWalkAndSchema:
    """F-DV-14 / F-DV-05: the one reachability walk and the one cap."""

    def test_reachable_from_follows_edges_and_tolerates_unknown_nodes(self):
        from fsm_llm.definitions import reachable_from

        adjacency = {"a": ["b", "ghost"], "b": ["a", "c"], "c": [], "d": ["a"]}
        assert reachable_from(adjacency, ["a"]) == {"a", "b", "c", "ghost"}
        assert reachable_from(adjacency, ["c"]) == {"c"}
        assert reachable_from(adjacency, ["c", "d"]) == {"a", "b", "c", "d", "ghost"}

    def test_the_config_inherits_the_schema_and_its_cap(self):
        from fsm_llm.definitions import (
            MAX_CLASSIFICATION_INTENTS,
            ClassificationExtractionConfig,
            ClassificationSchema,
        )

        assert issubclass(ClassificationExtractionConfig, ClassificationSchema)
        intents = [
            {"name": f"i{n}", "description": "d"}
            for n in range(MAX_CLASSIFICATION_INTENTS + 1)
        ]
        with pytest.raises(ValueError):
            ClassificationExtractionConfig(
                field_name="f", intents=intents, fallback_intent="i0"
            )
        with pytest.raises(ValueError):
            ClassificationSchema(intents=intents, fallback_intent="i0")
