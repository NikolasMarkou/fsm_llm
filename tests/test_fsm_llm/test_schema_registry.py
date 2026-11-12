"""
Tests for ``fsm_llm.schema_registry``: the field-introduction registry.

The registry is proven complete by WALKING the annotations reachable from
``FSMDefinition`` (not by a hand list), so a new field or a new nested model
that is not registered fails here. Every definition below declares its version
explicitly, so none of these tests depends on the default version.
"""

import typing
from typing import Any

import pytest
from pydantic import BaseModel

from fsm_llm.definitions import (
    ClassificationExtractionConfig,
    CompletionStateConfig,
    ContextScope,
    FieldExtractionConfig,
    FSMDefinition,
    IntentDefinition,
    State,
    SynthesizeConfig,
    Transition,
    TransitionCondition,
)
from fsm_llm.schema_registry import (
    FIELD_VERSIONS,
    SCHEMA_VERSION_DEFAULT,
    SchemaFinding,
    newer_fields,
    parse_version,
)

REGISTERED_MODELS = {
    FSMDefinition,
    State,
    Transition,
    TransitionCondition,
    FieldExtractionConfig,
    ClassificationExtractionConfig,
    IntentDefinition,
    ContextScope,
    CompletionStateConfig,
    SynthesizeConfig,
}


def _models_in(annotation: Any) -> set[type[BaseModel]]:
    """Every BaseModel class named anywhere inside a type annotation."""
    found: set[type[BaseModel]] = set()
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        found.add(annotation)
    for arg in typing.get_args(annotation):
        found |= _models_in(arg)
    return found


def _reachable(root: type[BaseModel]) -> set[type[BaseModel]]:
    seen: set[type[BaseModel]] = set()
    pending = [root]
    while pending:
        model = pending.pop()
        if model in seen:
            continue
        seen.add(model)
        for field in model.model_fields.values():
            pending.extend(_models_in(field.annotation))
    return seen


def _state(state_id: str, **extra: Any) -> dict[str, Any]:
    state: dict[str, Any] = {
        "id": state_id,
        "description": "d",
        "purpose": "p",
    }
    state.update(extra)
    return state


def _definition(version: str, **states: dict[str, Any]) -> FSMDefinition:
    """A loadable definition declaring `version`, with the given extra states.

    `start` always exists and points at `end`; every extra state is reachable
    from `start`, so the loader's reachability check passes.
    """
    all_states: dict[str, Any] = {
        "start": _state(
            "start",
            transitions=[
                {"target_state": name, "description": "go", "priority": i}
                for i, name in enumerate(["end", *states], start=1)
            ],
        ),
        "end": {"id": "end", "description": "d", "purpose": "p"},
    }
    all_states.update(states)
    return FSMDefinition(
        name="Registry",
        description="d",
        initial_state="start",
        version=version,
        states=all_states,
    )


def _synthesize_state(state_id: str = "solve") -> dict[str, Any]:
    return _state(
        state_id,
        synthesize={
            "goal": "Book a table",
            "inputs": ["topic"],
            "return_keys": ["answer"],
            "on_failure": "end",
        },
        transitions=[
            {
                "target_state": "end",
                "description": "failed or aborted",
                "conditions": [
                    {
                        "description": "status failed or aborted",
                        "requires_context_keys": ["synthesis_result"],
                        "logic": {
                            "in": [
                                {"var": "synthesis_result.status"},
                                ["failed", "aborted"],
                            ]
                        },
                    }
                ],
            }
        ],
    )


def _found(definition: FSMDefinition) -> set[tuple[str, str, str | None]]:
    return {(f.model, f.field, f.state_id) for f in newer_fields(definition)}


class TestRegistryCompleteness:
    def test_the_reachable_models_are_exactly_the_registered_ones(self):
        # An 11th model reachable from FSMDefinition fails here until it is
        # registered (and added to REGISTERED_MODELS above).
        assert _reachable(FSMDefinition) == REGISTERED_MODELS

    def test_every_field_of_every_reachable_model_is_registered(self):
        expected = {
            f"{model.__name__}.{name}"
            for model in _reachable(FSMDefinition)
            for name in model.model_fields
        }
        assert expected - set(FIELD_VERSIONS) == set()

    def test_no_registry_entry_is_stale(self):
        expected = {
            f"{model.__name__}.{name}"
            for model in _reachable(FSMDefinition)
            for name in model.model_fields
        }
        assert set(FIELD_VERSIONS) - expected == set()

    def test_every_version_parses_and_is_at_most_the_default(self):
        default = parse_version(SCHEMA_VERSION_DEFAULT)
        assert default is not None
        for key, version in FIELD_VERSIONS.items():
            parsed = parse_version(version)
            assert parsed is not None, key
            assert parsed <= default, key

    def test_the_inherited_classification_fields_are_registered(self):
        # ClassificationExtractionConfig inherits from ClassificationSchema;
        # the inherited names are part of its own model_fields.
        for name in ("intents", "fallback_intent", "confidence_threshold"):
            assert f"ClassificationExtractionConfig.{name}" in FIELD_VERSIONS


class TestRegistryTiers:
    @pytest.mark.parametrize(
        ("key", "version"),
        [
            ("State.synthesize", "4.2"),
            ("SynthesizeConfig.goal", "4.2"),
            ("SynthesizeConfig.min_lifecycle", "4.2"),
            ("FSMDefinition.handler_only_keys", "4.1"),
            ("State.completion", "4.1"),
            ("State.extraction_instructions", "4.1"),
            ("Transition.llm_description", "4.1"),
            ("IntentDefinition.name", "4.1"),
            ("State.purpose", "4.0"),
            ("FSMDefinition.version", "4.0"),
            ("Transition.priority", "4.0"),
            ("TransitionCondition.logic", "4.0"),
        ],
    )
    def test_tier_pin(self, key, version):
        assert FIELD_VERSIONS[key] == version

    def test_every_synthesize_config_field_is_4_2(self):
        names = [k for k in FIELD_VERSIONS if k.startswith("SynthesizeConfig.")]
        assert names
        assert {FIELD_VERSIONS[k] for k in names} == {"4.2"}


class TestParseVersion:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("4.1", (4, 1)),
            ("4.0", (4, 0)),
            ("3.0", (3, 0)),
            ("10.25", (10, 25)),
            ("4", None),
            ("4.1.1", None),
            ("v4.1", None),
            ("", None),
            (" 4.1", None),
            ("4.1 ", None),
            ("4.x", None),
            ("-4.1", None),
            ("٤.١", None),
        ],
    )
    def test_only_exact_major_dot_minor_digits_parse(self, text, expected):
        assert parse_version(text) == expected


class TestSchemaFinding:
    def test_message_names_field_versions_and_state(self):
        finding = SchemaFinding("State", "synthesize", "4.2", "4.1", "solve")
        assert "State.synthesize" in finding.message
        assert "4.2" in finding.message
        assert "4.1" in finding.message
        assert "solve" in finding.message

    def test_message_without_a_state_omits_it(self):
        finding = SchemaFinding(
            "FSMDefinition", "handler_only_keys", "4.1", "4.0", None
        )
        assert "FSMDefinition.handler_only_keys" in finding.message
        assert "state" not in finding.message

    def test_a_finding_is_frozen(self):
        finding = SchemaFinding("State", "synthesize", "4.2", "4.1", "solve")
        with pytest.raises(AttributeError):
            finding.field = "other"  # type: ignore[misc]


class TestNewerFields:
    def test_a_4_1_file_using_synthesize_warns(self):
        findings = newer_fields(_definition("4.1", solve=_synthesize_state()))
        assert len(findings) == 1
        finding = findings[0]
        assert (finding.model, finding.field) == ("State", "synthesize")
        assert (finding.introduced, finding.declared) == ("4.2", "4.1")
        assert finding.state_id == "solve"
        assert "State.synthesize" in finding.message
        assert "solve" in finding.message

    def test_the_children_of_a_newer_container_are_not_listed_again(self):
        fields = {
            f.field for f in newer_fields(_definition("4.1", s=_synthesize_state("s")))
        }
        assert fields == {"synthesize"}

    def test_a_4_2_file_is_quiet(self):
        assert newer_fields(_definition("4.2", solve=_synthesize_state())) == ()

    @pytest.mark.parametrize("declared", ["3.0", "v4.1", "4", "4.1.1", "5.0", "3.9"])
    def test_an_unparseable_older_or_newer_declaration_is_quiet(self, declared):
        assert newer_fields(_definition(declared, solve=_synthesize_state())) == ()

    def test_a_4_0_file_using_4_1_fields_warns_for_each(self):
        definition = _definition(
            "4.0",
            a=_state(
                "a",
                extraction_instructions="get x",
                transitions=[
                    {
                        "target_state": "end",
                        "description": "go",
                        "llm_description": "why",
                        "conditions": [
                            {
                                "description": "c",
                                "evaluation_priority": 5,
                                "logic": {"==": [1, 1]},
                            }
                        ],
                    }
                ],
            ),
        )
        assert _found(definition) == {
            ("State", "extraction_instructions", "a"),
            ("Transition", "llm_description", "a"),
            ("TransitionCondition", "evaluation_priority", "a"),
        }

    def test_a_4_0_file_using_only_4_0_fields_is_quiet(self):
        definition = _definition(
            "4.0",
            a=_state(
                "a",
                response_instructions=None,
                required_context_keys=["x"],
                transitions=[{"target_state": "end", "description": "go"}],
            ),
        )
        # `response_instructions` is a 4.1 field: supplied as None it is unused.
        assert newer_fields(definition) == ()

    def test_a_supplied_value_equal_to_the_default_is_not_use(self):
        default_retries = State.model_fields["extraction_retries"].default
        quiet = _state("a", extraction_retries=default_retries)
        quiet["transitions"] = [
            {"target_state": "end", "description": "go", "llm_description": None}
        ]
        definition = _definition("4.0", a=quiet)
        assert ("State", "extraction_retries", "a") not in _found(definition)
        assert ("Transition", "llm_description", "a") not in _found(definition)

    def test_a_non_default_value_of_the_same_field_is_use(self):
        default_retries = State.model_fields["extraction_retries"].default
        definition = _definition(
            "4.0", a=_state("a", extraction_retries=default_retries + 1)
        )
        assert ("State", "extraction_retries", "a") in _found(definition)

    def test_a_top_level_field_has_no_state_id(self):
        definition = FSMDefinition(
            name="Top",
            description="d",
            initial_state="end",
            version="4.0",
            handler_only_keys=["secret"],
            states={"end": {"id": "end", "description": "d", "purpose": "p"}},
        )
        assert _found(definition) == {("FSMDefinition", "handler_only_keys", None)}

    def test_the_same_field_in_many_transitions_is_one_finding_per_state(self):
        transitions = [
            {
                "target_state": "end",
                "description": "go",
                "priority": i,
                "llm_description": "why",
            }
            for i in range(1, 4)
        ]
        definition = _definition(
            "4.0",
            a=_state("a", transitions=transitions),
            b=_state("b", transitions=transitions),
        )
        findings = [f for f in newer_fields(definition) if f.field == "llm_description"]
        assert sorted(f.state_id for f in findings) == ["a", "b"]

    def test_nested_extraction_models_are_walked(self):
        definition = _definition(
            "4.0",
            a=_state(
                "a",
                field_extractions=[
                    {
                        "field_name": "age",
                        "field_type": "int",
                        "extraction_instructions": "the age",
                    }
                ],
                classification_extractions=[
                    {
                        "field_name": "intent",
                        "intents": [
                            {"name": "buy", "description": "buys"},
                            {"name": "browse", "description": "looks"},
                        ],
                        "fallback_intent": "browse",
                    }
                ],
            ),
        )
        found = _found(definition)
        assert ("State", "field_extractions", "a") in found
        assert ("State", "classification_extractions", "a") in found

    def test_the_walk_reaches_models_nested_below_a_current_field(self, monkeypatch):
        # No shipped nested field is newer than its container, so pretend one is.
        monkeypatch.setitem(FIELD_VERSIONS, "IntentDefinition.description", "4.2")
        definition = _definition(
            "4.1",
            a=_state(
                "a",
                classification_extractions=[
                    {
                        "field_name": "intent",
                        "intents": [
                            {"name": "buy", "description": "buys"},
                            {"name": "browse", "description": "looks"},
                        ],
                        "fallback_intent": "browse",
                    }
                ],
            ),
        )
        assert _found(definition) == {("IntentDefinition", "description", "a")}

    def test_the_result_is_deterministic_and_a_tuple(self):
        definition = _definition(
            "4.1", b=_synthesize_state("b"), a=_synthesize_state("a")
        )
        first = newer_fields(definition)
        assert isinstance(first, tuple)
        assert first == newer_fields(definition)
        assert [f.state_id for f in first] == ["b", "a"]

    def test_the_definition_is_not_mutated(self):
        definition = _definition("4.1", solve=_synthesize_state())
        before = definition.model_dump()
        newer_fields(definition)
        assert definition.model_dump() == before


class TestDefaultVersion:
    def test_the_model_default_is_the_registry_constant(self):
        assert FSMDefinition.model_fields["version"].default == SCHEMA_VERSION_DEFAULT
        assert SCHEMA_VERSION_DEFAULT == "4.2"

    def test_an_omitted_version_loads_as_the_default(self):
        definition = FSMDefinition(
            name="d",
            description="d",
            initial_state="a",
            states={
                "a": {"id": "a", "description": "a", "purpose": "a"},
            },
        )
        assert definition.version == SCHEMA_VERSION_DEFAULT
