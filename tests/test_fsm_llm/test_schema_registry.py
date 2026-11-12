"""
Tests for ``fsm_llm.schema_registry``: the field-introduction registry.

The registry is proven complete by WALKING the annotations reachable from
``FSMDefinition`` (not by a hand list), so a new field or a new nested model
that is not registered fails here. Every definition below declares its version
explicitly, so none of these tests depends on the default version.
"""

import contextlib
import json
import os
import stat
import sys
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
            ("IntentDefinition.name", "4.1"),
            ("Transition.llm_description", "4.0"),
            ("TransitionCondition.evaluation_priority", "4.0"),
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
                response_instructions="say y",
            ),
        )
        assert _found(definition) == {
            ("State", "extraction_instructions", "a"),
            ("State", "response_instructions", "a"),
        }

    def test_llm_description_and_evaluation_priority_are_4_0_fields(self):
        # Both exist in the 4.0 schema (the 2-pass package at commit 586f412f,
        # version default "4.0"), so an honest 4.0 file using them is quiet.
        definition = _definition(
            "4.0",
            a=_state(
                "a",
                required_context_keys=["x"],
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
        assert newer_fields(definition) == ()

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
        quiet["response_instructions"] = None
        definition = _definition("4.0", a=quiet)
        assert ("State", "extraction_retries", "a") not in _found(definition)
        assert ("State", "response_instructions", "a") not in _found(definition)

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

    def test_the_same_field_in_many_states_is_one_finding_per_state(self):
        definition = _definition(
            "4.0",
            a=_state("a", extraction_instructions="get a"),
            b=_state("b", extraction_instructions="get b"),
        )
        findings = [
            f for f in newer_fields(definition) if f.field == "extraction_instructions"
        ]
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


def _declared_body(version: str | None, *, with_synthesize: bool) -> dict[str, Any]:
    """A raw definition dict (what a file holds); `version=None` omits the key."""
    solve = _synthesize_state() if with_synthesize else _state("solve")
    body: dict[str, Any] = {
        "name": "Registry",
        "description": "d",
        "initial_state": "start",
        "states": {
            "start": _state(
                "start",
                transitions=[
                    {"target_state": "solve", "description": "go", "priority": 1},
                    {"target_state": "end", "description": "skip", "priority": 2},
                ],
            ),
            "solve": solve,
            "end": {"id": "end", "description": "d", "purpose": "p"},
        },
    }
    if version is not None:
        body["version"] = version
    return body


@contextlib.contextmanager
def _captured_warnings() -> typing.Iterator[list[str]]:
    """Library logging on, a sink collecting every WARNING line; restored on exit."""
    from fsm_llm.logging import logger

    lines: list[str] = []
    logger.enable("fsm_llm")
    sink_id = logger.add(lambda m: lines.append(str(m)), level="WARNING")
    try:
        yield lines
    finally:
        logger.remove(sink_id)
        logger.disable("fsm_llm")


def _loader_warnings(body: dict[str, Any]) -> list[str]:
    """Build the definition with library logging on; return its WARNING lines."""
    with _captured_warnings() as lines:
        FSMDefinition(**body)
    return lines


class TestLoaderWarning:
    """The loader names a field newer than the declared version, once, never raising."""

    def test_a_declared_older_version_logs_one_warning_naming_the_field(self):
        lines = _loader_warnings(_declared_body("4.1", with_synthesize=True))
        assert len(lines) == 1
        assert "State.synthesize" in lines[0]
        assert "'solve'" in lines[0]
        assert "4.1" in lines[0]
        assert "4.2" in lines[0]

    def test_the_definition_still_loads(self):
        body = _declared_body("4.1", with_synthesize=True)
        definition = FSMDefinition(**body)
        assert definition.version == "4.1"
        assert "synthesize" in definition.states["solve"].model_fields_set

    @pytest.mark.parametrize(
        "version", ["4.2", None, "3.0", "v4.1", "4", "4.1.1", "5.0"]
    )
    def test_no_warning_when_nothing_is_newer_or_the_version_is_unchecked(
        self, version
    ):
        assert _loader_warnings(_declared_body(version, with_synthesize=True)) == []

    def test_no_warning_for_a_definition_that_uses_no_newer_field(self):
        assert _loader_warnings(_declared_body("4.1", with_synthesize=False)) == []

    def test_one_warning_lists_every_finding(self):
        body = _declared_body("4.1", with_synthesize=True)
        other = _synthesize_state("other")
        body["states"]["other"] = other
        body["states"]["start"]["transitions"].append(
            {"target_state": "other", "description": "go", "priority": 3}
        )
        lines = _loader_warnings(body)
        assert len(lines) == 1
        assert "'solve'" in lines[0] and "'other'" in lines[0]


class TestLoaderWarnsOncePerDefinitionObject:
    """D-003: a definition object reports its findings once, not once per frame.

    pydantic runs the after-validator of `FSMDefinition` again whenever an
    instance is validated as a field of another model (`FSMStackFrame` in
    `API.start_conversation` and `push_fsm`), so without a per-object flag a
    server with library logging on logged one WARNING per conversation.
    """

    @staticmethod
    def _body() -> dict[str, Any]:
        body = _declared_body("4.0", with_synthesize=False)
        body["states"]["solve"]["extraction_instructions"] = "pull a name"
        return body

    @staticmethod
    def _version_lines(lines: list[str]) -> list[str]:
        return [line for line in lines if "needs definition version" in line]

    def test_one_warning_over_construct_start_converse_and_push(self):
        from fsm_llm import API
        from tests.conftest import MockLLM2Interface

        with _captured_warnings() as lines:
            definition = FSMDefinition(**self._body())
            assert len(self._version_lines(lines)) == 1

            api = API.from_definition(definition, llm_interface=MockLLM2Interface())
            first, _ = api.start_conversation()
            second, _ = api.start_conversation()
            api.converse("hello", first)
            api.converse("hello again", second)
            assert len(self._version_lines(lines)) == 1

            # a pushed definition object was already reported at its construction
            api.push_fsm(first, definition)
            assert len(self._version_lines(lines)) == 1

    def test_a_pushed_dict_is_a_new_definition_and_warns_once(self):
        from fsm_llm import API
        from tests.conftest import MockLLM2Interface

        api = API.from_definition(_definition("4.2"), llm_interface=MockLLM2Interface())
        conversation_id, _ = api.start_conversation()
        with _captured_warnings() as lines:
            api.push_fsm(conversation_id, self._body())
        assert len(self._version_lines(lines)) == 1

    def test_a_second_independent_definition_warns_again(self):
        with _captured_warnings() as lines:
            FSMDefinition(**self._body())
            FSMDefinition(**self._body())
        assert len(self._version_lines(lines)) == 2

    def test_a_collected_definition_leaves_no_entry_behind(self):
        import gc

        from fsm_llm import definitions

        definition = FSMDefinition(**self._body())
        key = id(definition)
        assert key in definitions._REPORTED_DEFINITIONS
        del definition
        gc.collect()
        assert key not in definitions._REPORTED_DEFINITIONS

    def test_the_flag_changes_neither_dump_nor_equality_nor_id(self):
        from fsm_llm.api import API

        reported = FSMDefinition(**self._body())
        fresh = FSMDefinition(**self._body())
        assert reported == fresh
        assert reported.model_dump() == fresh.model_dump()
        assert not any(key.startswith("_") for key in reported.model_dump())
        assert (
            API.process_fsm_definition(reported)[1]
            == (API.process_fsm_definition(self._body())[1])
        )


class TestValidatorWarning:
    """`FSMValidator` carries the same fact as a WARNING and never an error."""

    def test_the_report_warns_about_the_newer_field(self):
        from fsm_llm.validator import FSMValidator

        result = FSMValidator(_declared_body("4.1", with_synthesize=True)).validate()
        hits = [w for w in result.warnings if "State.synthesize" in w]
        assert len(hits) == 1
        assert "'solve'" in hits[0] and "4.1" in hits[0] and "4.2" in hits[0]
        assert not any("State.synthesize" in e for e in result.errors)

    def test_the_verdict_is_what_it_was_without_the_warning(self):
        from fsm_llm.validator import FSMValidator

        older = FSMValidator(_declared_body("4.1", with_synthesize=True)).validate()
        current = FSMValidator(_declared_body("4.2", with_synthesize=True)).validate()
        assert older.is_valid is current.is_valid is True
        assert older.errors == current.errors
        assert not any("needs definition version" in w for w in current.warnings)

    @pytest.mark.parametrize("version", ["4.2", None, "3.0", "v4.1", "4", "5.0"])
    def test_no_warning_in_the_report_when_nothing_is_newer(self, version):
        from fsm_llm.validator import FSMValidator

        result = FSMValidator(_declared_body(version, with_synthesize=True)).validate()
        assert not any("needs definition version" in w for w in result.warnings)

    @pytest.mark.parametrize(
        "version", ["3.0", "4.0", "4.1", "4.2", "v4.1", "4.1.1", "5.0"]
    )
    @pytest.mark.parametrize("with_synthesize", [True, False])
    def test_never_stricter_than_the_loader(self, version, with_synthesize):
        from pydantic import ValidationError

        from fsm_llm.validator import FSMValidator

        body = _declared_body(version, with_synthesize=with_synthesize)
        try:
            FSMDefinition(**body)
            loads = True
        except ValidationError:
            loads = False
        assert FSMValidator(body).validate().is_valid is loads

    def test_a_definition_the_loader_refuses_gets_no_version_warning(self):
        from fsm_llm.validator import FSMValidator

        body = _declared_body("4.1", with_synthesize=True)
        body["initial_state"] = "nowhere"
        result = FSMValidator(body).validate()
        assert result.is_valid is False
        assert not any("needs definition version" in w for w in result.warnings)


def _example_definition_bodies() -> list[tuple[str, dict[str, Any]]]:
    from pathlib import Path

    root = Path(__file__).parent.parent.parent / "examples"
    found = []
    for path in sorted(root.glob("**/*.json")):
        import json

        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and {"initial_state", "states"} <= data.keys():
            found.append((str(path.relative_to(root)), data))
    return found


def _agent_definition_bodies() -> list[tuple[str, dict[str, Any]]]:
    from fsm_llm.agents import fsm_definitions as fd
    from fsm_llm.agents.tools import ToolRegistry

    registry = ToolRegistry()
    registry.register_function(lambda: "ok", name="search", description="Search")
    schemas = [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "Run lookup",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    return [
        ("orchestrator", fd.build_orchestrator_fsm()),
        ("adapt", fd.build_adapt_fsm()),
        ("reflexion", fd.build_reflexion_fsm(registry)),
        ("plan_execute", fd.build_plan_execute_fsm(registry)),
        ("react", fd.build_react_fsm(registry)),
        ("react_approval", fd.build_react_fsm(registry, include_approval_state=True)),
        ("self_consistency", fd.build_self_consistency_fsm()),
        ("debate", fd.build_debate_fsm()),
        ("rewoo", fd.build_rewoo_fsm(registry)),
        ("evalopt", fd.build_evalopt_fsm()),
        ("maker_checker", fd.build_maker_checker_fsm("make it", "check it")),
        ("meta_builder", fd.build_meta_builder_fsm()),
        ("native_fc", fd.build_native_fc_fsm(schemas, instructions="Use the tools.")),
    ]


class TestShippedDefinitionsUseNoNewerField:
    """Read-only sweep: the baseline tiers are right when no shipped file is flagged."""

    def test_the_sweeps_find_something_to_check(self):
        assert len(_example_definition_bodies()) >= 10
        assert len(_agent_definition_bodies()) >= 10

    def test_no_example_declares_a_version_older_than_its_fields(self):
        flagged = {}
        for label, body in _example_definition_bodies():
            findings = newer_fields(FSMDefinition(**body))
            if findings:
                flagged[label] = [f.message for f in findings]
        assert not flagged

    def test_no_agent_built_definition_uses_a_field_newer_than_4_1(self):
        # The builders omit `version`, so they would take the default and could
        # never be flagged; declare 4.1 so the check bites.
        flagged = {}
        for label, body in _agent_definition_bodies():
            findings = newer_fields(FSMDefinition(**{**body, "version": "4.1"}))
            if findings:
                flagged[label] = [f.message for f in findings]
        assert not flagged


# --------------------------------------------------------------
# fsm-llm-validate --upgrade (plan 9013b0e3 D-012)
# --------------------------------------------------------------


def _write_definition(path, version: Any = "4.1", **extra: Any):
    """Write a definition file; `version=None` omits the key. Returns the dict."""
    body = _declared_body(None, with_synthesize=False)
    if version is not None:
        body["version"] = version
    body.update(extra)
    path.write_text(json.dumps(body), encoding="utf-8")
    return body


def _without_version(data: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in data.items() if k != "version"}


class TestUpgradeFsmFile:
    """`upgrade_fsm_file` stamps the current version and nothing else."""

    @pytest.mark.parametrize("version", ["3.0", "4.0", "4.1"])
    def test_an_older_version_is_rewritten_and_loads_clean(self, tmp_path, version):
        from fsm_llm.validator import upgrade_fsm_file

        path = tmp_path / "fsm.json"
        original = _write_definition(path, version)
        assert upgrade_fsm_file(str(path)) is True
        after = json.loads(path.read_text(encoding="utf-8"))
        assert after["version"] == SCHEMA_VERSION_DEFAULT
        assert _without_version(after) == _without_version(original)
        assert list(after) == list(original)  # key order kept
        assert _loader_warnings(after) == []
        assert FSMDefinition(**after).version == SCHEMA_VERSION_DEFAULT

    def test_an_unparseable_string_version_is_rewritten(self, tmp_path):
        from fsm_llm.validator import upgrade_fsm_file

        path = tmp_path / "fsm.json"
        _write_definition(path, "v4.1")
        assert upgrade_fsm_file(str(path)) is True
        assert json.loads(path.read_text())["version"] == SCHEMA_VERSION_DEFAULT

    @pytest.mark.parametrize("version", [SCHEMA_VERSION_DEFAULT, "5.0", "4.10"])
    def test_a_current_or_newer_version_is_never_touched(self, tmp_path, version):
        from fsm_llm.validator import upgrade_fsm_file

        path = tmp_path / "fsm.json"
        _write_definition(path, version)
        before = path.read_bytes()
        assert upgrade_fsm_file(str(path)) is False
        assert path.read_bytes() == before

    def test_an_omitted_version_is_left_omitted(self, tmp_path):
        from fsm_llm.validator import upgrade_fsm_file

        path = tmp_path / "fsm.json"
        _write_definition(path, None)
        before = path.read_bytes()
        assert upgrade_fsm_file(str(path)) is False
        assert path.read_bytes() == before

    @pytest.mark.parametrize("version", [4.1, 4, None, ["4.1"]])
    def test_a_non_string_version_is_left_alone(self, tmp_path, version):
        from fsm_llm.validator import upgrade_fsm_file

        path = tmp_path / "fsm.json"
        body = _declared_body(None, with_synthesize=False)
        body["version"] = version
        path.write_text(json.dumps(body), encoding="utf-8")
        before = path.read_bytes()
        assert upgrade_fsm_file(str(path)) is False
        assert path.read_bytes() == before

    def test_the_second_run_writes_nothing(self, tmp_path):
        from fsm_llm.validator import upgrade_fsm_file

        path = tmp_path / "fsm.json"
        _write_definition(path, "4.0")
        assert upgrade_fsm_file(str(path)) is True
        stamp = path.stat().st_mtime_ns
        data = path.read_bytes()
        os.utime(path, ns=(stamp - 10**10, stamp - 10**10))
        old = path.stat().st_mtime_ns
        assert upgrade_fsm_file(str(path)) is False
        assert path.read_bytes() == data
        assert path.stat().st_mtime_ns == old

    def test_the_text_is_indent_two_with_a_trailing_newline(self, tmp_path):
        from fsm_llm.validator import upgrade_fsm_file

        path = tmp_path / "fsm.json"
        _write_definition(path, "4.1")
        upgrade_fsm_file(str(path))
        text = path.read_text(encoding="utf-8")
        assert text.endswith("}\n") and not text.endswith("\n\n")
        assert text == json.dumps(json.loads(text), indent=2, ensure_ascii=False) + "\n"

    def test_non_ascii_text_is_kept_as_is(self, tmp_path):
        from fsm_llm.validator import upgrade_fsm_file

        path = tmp_path / "fsm.json"
        _write_definition(path, "4.1", persona="Zoë, 北京 — ok")
        upgrade_fsm_file(str(path))
        text = path.read_text(encoding="utf-8")
        assert "Zoë, 北京 — ok" in text
        assert "\\u" not in text

    def test_the_file_mode_is_kept(self, tmp_path):
        from fsm_llm.validator import upgrade_fsm_file

        path = tmp_path / "fsm.json"
        _write_definition(path, "4.1")
        path.chmod(0o644)
        upgrade_fsm_file(str(path))
        assert stat.S_IMODE(path.stat().st_mode) == 0o644
        path.chmod(0o600)
        _write_definition(path, "4.0")
        path.chmod(0o600)
        upgrade_fsm_file(str(path))
        assert stat.S_IMODE(path.stat().st_mode) == 0o600

    def test_a_symlink_keeps_its_link_and_the_target_is_rewritten(self, tmp_path):
        from fsm_llm.validator import upgrade_fsm_file

        real_dir = tmp_path / "real"
        real_dir.mkdir()
        target = real_dir / "fsm.json"
        _write_definition(target, "4.1")
        link = tmp_path / "link.json"
        link.symlink_to(target)
        assert upgrade_fsm_file(str(link)) is True
        assert link.is_symlink()
        assert json.loads(target.read_text())["version"] == SCHEMA_VERSION_DEFAULT
        assert sorted(p.name for p in tmp_path.iterdir()) == ["link.json", "real"]

    def test_invalid_json_is_left_byte_identical(self, tmp_path):
        from fsm_llm.validator import upgrade_fsm_file

        path = tmp_path / "fsm.json"
        path.write_bytes(b'{"version": "4.1", ')
        assert upgrade_fsm_file(str(path)) is False
        assert path.read_bytes() == b'{"version": "4.1", '

    def test_a_non_object_json_is_left_byte_identical(self, tmp_path):
        from fsm_llm.validator import upgrade_fsm_file

        path = tmp_path / "fsm.json"
        path.write_bytes(b'["4.1"]')
        assert upgrade_fsm_file(str(path)) is False
        assert path.read_bytes() == b'["4.1"]'

    def test_a_missing_file_and_non_utf8_do_not_raise(self, tmp_path):
        from fsm_llm.validator import upgrade_fsm_file

        assert upgrade_fsm_file(str(tmp_path / "nope.json")) is False
        bad = tmp_path / "bad.json"
        bad.write_bytes(b"\xff\xfe\x00")
        assert upgrade_fsm_file(str(bad)) is False
        assert bad.read_bytes() == b"\xff\xfe\x00"

    @pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory modes")
    def test_a_read_only_directory_leaves_the_file_untouched(self, tmp_path):
        from fsm_llm.validator import upgrade_fsm_file

        path = tmp_path / "fsm.json"
        _write_definition(path, "4.1")
        before = path.read_bytes()
        tmp_path.chmod(0o555)
        try:
            assert upgrade_fsm_file(str(path)) is False
        finally:
            tmp_path.chmod(0o755)
        assert path.read_bytes() == before
        assert [p.name for p in tmp_path.iterdir()] == ["fsm.json"]

    def test_a_failed_replace_leaves_the_original_and_no_temp_file(
        self, tmp_path, monkeypatch
    ):
        from fsm_llm.validator import upgrade_fsm_file

        path = tmp_path / "fsm.json"
        _write_definition(path, "4.1")
        before = path.read_bytes()

        def boom(src, dst):
            raise OSError("disk on fire")

        monkeypatch.setattr(os, "replace", boom)
        assert upgrade_fsm_file(str(path)) is False
        assert path.read_bytes() == before
        assert [p.name for p in tmp_path.iterdir()] == ["fsm.json"]


class TestValidateMainUpgrade:
    """`main(path, upgrade=True)` and `--upgrade` rewrite, then validate."""

    @pytest.fixture(autouse=True)
    def _keep_logging_state(self, monkeypatch):
        # `main` re-enables library logging and `main_cli` installs a stderr
        # handler (process-global loguru state). A handler bound to pytest's
        # capture stream would break every later test that prints a record, so
        # stub the handler setup and restore the library's disabled default.
        import fsm_llm.validator as validator_module
        from fsm_llm.logging import logger

        monkeypatch.setattr(validator_module, "setup_cli_logging", lambda level: 0)
        yield
        logger.disable("fsm_llm")

    def test_main_rewrites_then_validates_and_stdout_is_the_report(
        self, tmp_path, capsys
    ):
        from fsm_llm.validator import main

        path = tmp_path / "fsm.json"
        _write_definition(path, "4.1")
        assert main(str(path), upgrade=True) == 0
        report = json.loads(capsys.readouterr().out)
        assert report["is_valid"] is True
        assert json.loads(path.read_text())["version"] == SCHEMA_VERSION_DEFAULT

    def test_the_report_describes_the_upgraded_file(self, tmp_path, capsys):
        from fsm_llm.validator import main

        path = tmp_path / "fsm.json"
        body = _declared_body("4.1", with_synthesize=True)
        path.write_text(json.dumps(body), encoding="utf-8")
        assert main(str(path), upgrade=False) == 0
        before = json.loads(capsys.readouterr().out)
        assert any("needs definition version" in w for w in before["warnings"])
        assert main(str(path), upgrade=True) == 0
        after = json.loads(capsys.readouterr().out)
        assert not any("needs definition version" in w for w in after["warnings"])

    def test_without_the_flag_nothing_is_written(self, tmp_path, capsys):
        from fsm_llm.validator import main

        path = tmp_path / "fsm.json"
        _write_definition(path, "4.1")
        before = path.read_bytes()
        assert main(str(path)) == 0
        json.loads(capsys.readouterr().out)
        assert path.read_bytes() == before

    @pytest.mark.parametrize("content", [b'{"version": "4.1", ', b"[1]"])
    def test_a_bad_file_is_reported_not_crashed_on(self, tmp_path, capsys, content):
        from fsm_llm.validator import main

        path = tmp_path / "fsm.json"
        path.write_bytes(content)
        assert main(str(path), upgrade=True) == 1
        report = json.loads(capsys.readouterr().out)
        assert report["is_valid"] is False
        assert path.read_bytes() == content

    def test_main_cli_honours_the_flag(self, tmp_path, capsys, monkeypatch):
        from fsm_llm.validator import main_cli

        path = tmp_path / "fsm.json"
        _write_definition(path, "4.0")
        monkeypatch.setattr(
            sys, "argv", ["fsm-llm-validate", "--fsm", str(path), "--upgrade"]
        )
        with pytest.raises(SystemExit) as exit_info:
            main_cli()
        assert exit_info.value.code == 0
        json.loads(capsys.readouterr().out)
        assert json.loads(path.read_text())["version"] == SCHEMA_VERSION_DEFAULT

    def test_main_cli_without_the_flag_writes_nothing(
        self, tmp_path, capsys, monkeypatch
    ):
        from fsm_llm.validator import main_cli

        path = tmp_path / "fsm.json"
        _write_definition(path, "4.0")
        before = path.read_bytes()
        monkeypatch.setattr(sys, "argv", ["fsm-llm-validate", "--fsm", str(path)])
        with pytest.raises(SystemExit):
            main_cli()
        assert path.read_bytes() == before
