"""The executable envelope, its two identities and its shipped schema (plan ace7a8c8, C7).

Every full-FSM built-in of the corpus is wrapped (``tests.executables_corpus.wrap``) and must
parse, round-trip and keep its FSM identity; the rest are the refusals of the format, the
instantiation record and the schema file. The module under test is lazy on purpose:
``fsm_llm/__init__.py`` and ``core/definitions/__init__.py`` never import it
(``test_import_laziness`` holds the pin).
"""

from __future__ import annotations

import copy
import hashlib
import importlib.resources
import json
import os
import subprocess
import sys
import types
import typing
from typing import Any

import pydantic
import pytest
from pydantic import BaseModel

from fsm_llm.core.codes import Lifecycle
from fsm_llm.core.definitions import FSMDefinition, definition_digest, limits
from fsm_llm.core.definitions import executable as ex
from fsm_llm.core.definitions.schema import (
    EXECUTABLE_SCHEMA_ID,
    SchemaFinding,
    newer_fields,
)
from fsm_llm.core.results import Identity
from fsm_llm.synthesis.constants import LIFECYCLES
from tests import executables_corpus as corpus
from tests.test_fsm_llm import kind_corpus as kc

_CASES = corpus.fsm_cases()
_INDEX = json.loads(corpus.INDEX_PATH.read_text(encoding="utf-8"))
_REGENERATE = (
    "regenerate: python -m tests.executables_corpus --write "
    f"(the bytes depend on the pinned pydantic, here {pydantic.VERSION})"
)


def _fsm(**state_a: Any) -> dict[str, Any]:
    """A two-state FSM; ``state_a`` overrides fields of the first state."""
    return kc._fsm(
        "Mini",
        "a",
        kc._state("a", transitions=[kc._go("b")], **state_a),
        kc._state("b"),
    )


def _doc(fsm: dict[str, Any] | None = None, **header: Any) -> dict[str, Any]:
    """A small ``fsm`` document; ``header`` overrides or adds top-level keys."""
    return {
        "schema": "fsm-llm/executable",
        "kind": "fsm",
        "id": "mini",
        "body": {"fsm": fsm if fsm is not None else _fsm()},
        **header,
    }


def _refused(data: Any) -> ex.ExecutableDocumentError:
    with pytest.raises(ex.ExecutableDocumentError) as info:
        ex.parse_executable(data)
    return info.value


class TestEveryBuiltInWrapped:
    @pytest.mark.parametrize("case", sorted(_CASES))
    def test_it_parses_round_trips_and_keeps_the_fsm_identity(self, case):
        document = corpus.wrap(case, _CASES[case])
        envelope = ex.parse_executable(document)

        dumped = envelope.model_dump(by_alias=True, mode="json")
        assert ex.parse_executable(json.loads(json.dumps(dumped))) == envelope
        assert dumped["schema"] == "fsm-llm/executable"
        assert "identity" not in dumped

        definition, record = ex.instantiate_envelope(envelope)
        assert definition == FSMDefinition(**_CASES[case])
        assert record == {"options": {}, "variables": {}}
        identity = ex.identity_of(envelope, definition)
        assert identity.instance_identity == _INDEX[case]["digest64"]
        assert identity.template_identity not in (None, identity.instance_identity)
        assert len(identity.template_identity) == 64

    @pytest.mark.parametrize("case", sorted(_CASES))
    def test_the_template_identity_ignores_key_order(self, case):
        document = corpus.wrap(case, _CASES[case])
        shuffled = {key: document[key] for key in reversed(document)}
        shuffled["body"] = {k: document["body"][k] for k in reversed(document["body"])}

        assert (
            ex.identity_of(ex.parse_executable(shuffled)).template_identity
            == ex.identity_of(ex.parse_executable(document)).template_identity
        )

    def test_reasoning_strategies_are_strategy_documents(self):
        kinds = {
            case: ex.parse_executable(corpus.wrap(case, raw)).kind
            for case, raw in _CASES.items()
        }

        assert kinds["reasoning/analytical"] == "strategy"
        assert kinds["reasoning/classifier"] == "fsm"
        assert kinds["agents/react"] == "fsm"


class TestTheTemplateIdentity:
    def test_a_default_given_explicitly_moves_nothing(self):
        plain = ex.parse_executable(_doc())
        explicit = ex.parse_executable(
            _doc(schema_version="1.0", lifecycle="draft", version="1", variables=[])
        )

        assert ex.identity_of(plain) == ex.identity_of(explicit)

    def test_a_document_of_required_keys_only_hashes_as_itself(self):
        # exclude_defaults: a defaulted field (added later, say) is not in the canonical dump
        document = _doc()

        assert ex.identity_of(
            ex.parse_executable(document)
        ).template_identity == ex.canonical_digest(document)

    def test_a_changed_field_moves_the_template_and_not_the_instance(self):
        base = ex.parse_executable(_doc())
        other = ex.parse_executable(_doc(description="another text"))

        definition = ex.instantiate_envelope(base)[0]
        assert (
            ex.identity_of(base, definition).template_identity
            != ex.identity_of(other, definition).template_identity
        )
        assert (
            ex.identity_of(base, definition).instance_identity
            == ex.identity_of(other, definition).instance_identity
            == definition_digest(definition)
        )

    def test_the_kind_is_part_of_the_template_identity(self):
        fsm = ex.parse_executable(_doc())
        procedure = ex.parse_executable(_doc(kind="procedure"))

        assert ex.identity_of(fsm) != ex.identity_of(procedure)

    def test_a_mapping_instance_is_hashed_by_the_canonical_recipe(self):
        envelope = ex.parse_executable(_doc())
        body = {"b": 1, "a": [1, 2]}

        text = json.dumps(body, sort_keys=True)
        assert (
            ex.identity_of(envelope, body).instance_identity
            == hashlib.sha256(text.encode()).hexdigest()
            == ex.canonical_digest({"a": [1, 2], "b": 1})
        )
        assert ex.identity_of(envelope) == Identity(
            template_identity=ex.identity_of(envelope).template_identity
        )

    def test_one_instance_has_one_identity_as_a_definition_or_as_a_dict(self):
        # D-054: a dict body that loads as an FSM is normalised through definition_digest
        envelope = ex.parse_executable(_doc())
        definition = FSMDefinition(**_fsm())
        spelled_out = definition.model_dump(mode="json")  # every default written
        as_written = _fsm()

        by_definition = ex.identity_of(envelope, definition)
        assert spelled_out != as_written
        assert ex.identity_of(envelope, spelled_out) == by_definition
        assert ex.identity_of(envelope, as_written) == by_definition
        assert by_definition.instance_identity == definition_digest(definition)

    def test_a_dict_that_is_no_fsm_is_hashed_as_written_and_never_raises(self):
        envelope = ex.parse_executable(_doc())

        for body in ({"steps": []}, {"name": "x"}, {1: "a"}, {}):
            assert ex.identity_of(
                envelope, body
            ).instance_identity == ex.canonical_digest(
                {str(k): v for k, v in body.items()}
            )

    def test_the_canonical_digest_of_an_fsm_dump_is_the_fsm_recipe(self):
        definition = FSMDefinition(**_fsm())

        assert ex.canonical_digest(
            definition.model_dump(exclude_defaults=True)
        ) == definition_digest(definition)


class TestPerKindRefusals:
    def test_a_strategy_needs_its_strategy_block(self):
        data = _doc(kind="strategy")
        assert any("strategy" in line for line in _refused(data).problems)

        data["body"]["strategy"] = {"solution_key": "answer"}
        assert ex.parse_executable(data).kind == "strategy"

    def test_an_fsm_document_with_a_strategy_block_is_refused(self):
        data = _doc()
        data["body"]["strategy"] = {"solution_key": "answer"}

        assert any("strategy" in line for line in _refused(data).problems)

    def test_a_procedure_carries_provenance_and_no_strategy(self):
        data = _doc(kind="procedure")
        data["body"]["provenance"] = {"approved_by": None, "gates_passed": ["G1"]}
        assert ex.parse_executable(data).kind == "procedure"

        data["body"]["strategy"] = {"solution_key": "answer"}
        _refused(data)

    @pytest.mark.parametrize("kind", ["pattern", "workflow"])
    def test_a_kind_not_built_yet_is_refused_by_name(self, kind):
        problems = _refused(_doc(kind=kind)).problems

        assert problems == (
            f"kind: {kind!r} is not supported by this version "
            "(known kinds: fsm, strategy, procedure)",
        )

    def test_an_unknown_kind_an_unknown_key_and_a_wrong_schema_are_refused(self):
        _refused(_doc(kind="bogus"))
        _refused(_doc(surprise=1))
        _refused({**_doc(), "schema": "fsm-llm/other"})
        bad_body = _doc()
        bad_body["body"]["handlers"] = [{"handler": "x"}]
        _refused(bad_body)

    @pytest.mark.parametrize("data", [None, [], "text", 7, {}])
    def test_a_document_that_is_no_object_or_is_empty_is_refused(self, data):
        _refused(data)

    def test_every_problem_is_one_line_of_plain_text(self):
        problems = _refused(_doc(id="", lifecycle="holy", surprise=1)).problems

        assert len(problems) == 3
        assert all("\n" not in line and line.isascii() for line in problems)

    def test_a_deep_document_is_refused_without_a_crash(self):
        deep: dict[str, Any] = {}
        cursor = deep
        for _ in range(100_000):
            cursor["a"] = {}
            cursor = cursor["a"]

        problems = _refused(_doc(metadata=deep)).problems

        assert problems and "deep" in problems[0]

    def test_a_problem_line_never_carries_raw_document_text(self):
        # D-054: key names, the kind value and messages are capped and stripped
        evil_key = "evil\nkey\u202e\x1b[31m" + "k" * 500
        long_kind = "x" * 600 + "\n\u202eend"

        for data in (_doc(**{evil_key: 1}), _doc(kind=long_kind)):
            error = _refused(data)
            texts = [*error.problems, str(error)]
            assert error.problems
            for text in texts:
                assert "\n" not in text and "\u202e" not in text and "\x1b" not in text
                assert all(ch.isprintable() for ch in text)
            assert all(len(line) <= ex.PROBLEM_LINE_CHARS for line in error.problems)
            assert len(str(error)) < ex.PROBLEM_LINE_CHARS + 100

    def test_a_non_dict_mapping_or_a_tuple_document_is_bounded_too(self):
        # a Mapping root and tuples skipped the depth bound and hit RecursionError
        deep: Any = "x"
        for _ in range(5_000):
            deep = [deep]
        as_tuples: Any = "x"
        for _ in range(5_000):
            as_tuples = (as_tuples,)

        for fsm_value in (deep, as_tuples):
            data = types.MappingProxyType(_doc(metadata={"v": fsm_value}))
            assert "deep" in _refused(data).problems[0]
        assert limits.nests_deeper_than((((1,),),), 2)
        assert not limits.nests_deeper_than((((1,),),), 3)
        assert limits.nests_deeper_than(
            types.MappingProxyType({"a": types.MappingProxyType({"b": [1]})}), 2
        )
        assert not limits.nests_deeper_than("a" * 10_000, 1)
        assert not limits.nests_deeper_than(b"abc", 1)

    def test_a_document_as_deep_as_a_real_fsm_body_is_read(self):
        value: Any = "x"
        for _ in range(150):  # more than the 108 levels of a JsonLogic maximum
            value = {"a": value}
        envelope = ex.parse_executable(_doc(metadata=value))

        assert ex.identity_of(envelope).template_identity


class TestHeaderFields:
    def test_the_defaults_and_the_alias(self):
        envelope = ex.parse_executable(_doc())

        assert envelope.schema_ == EXECUTABLE_SCHEMA_ID
        assert envelope.schema_version == ex.ENVELOPE_SCHEMA_VERSION_DEFAULT == "1.0"
        assert envelope.lifecycle is Lifecycle.DRAFT
        assert envelope.requires.tools == []
        assert envelope.contract.inputs == []
        built = ex.FsmEnvelope(
            schema_=EXECUTABLE_SCHEMA_ID, kind="fsm", id="x", body={"fsm": {}}
        )
        assert built.kind == "fsm"
        with pytest.raises(pydantic.ValidationError):  # the kind has no default
            ex.FsmEnvelope(schema_=EXECUTABLE_SCHEMA_ID, id="x", body={"fsm": {}})

    def test_model_dump_lists_exactly_the_fields_and_no_identity(self):
        dumped = ex.parse_executable(_doc()).model_dump(by_alias=True)

        assert set(dumped) == {
            "schema",
            "schema_version",
            "kind",
            "id",
            "version",
            "name",
            "description",
            "contract",
            "requires",
            "capabilities",
            "lifecycle",
            "variables",
            "body",
            "metadata",
        }

    def test_requires_tools_is_a_list_or_none_or_some(self):
        for tools in (["a", "b"], "none", "some"):
            data = _doc(requires={"tools": tools})
            assert ex.parse_executable(data).requires.tools == tools
        _refused(_doc(requires={"tools": "all"}))
        _refused(_doc(requires={"tools": 3}))

    def test_capabilities_are_a_closed_vocabulary(self):
        data = _doc(capabilities=["tools", "hitl"])
        assert [c.value for c in ex.parse_executable(data).capabilities] == [
            "tools",
            "hitl",
        ]
        _refused(_doc(capabilities=["root"]))
        assert [c.value for c in ex.Capability] == [
            "tools",
            "hitl",
            "streaming",
            "strategy",
            "conversational",
        ]
        assert ex.Capability.from_string("hitl") is ex.Capability.HITL
        with pytest.raises(ValueError):
            ex.Capability.from_string("nope")

    def test_lifecycle_is_the_pinned_vocabulary(self):
        for value in (*LIFECYCLES, "builtin"):
            assert ex.parse_executable(_doc(lifecycle=value)).lifecycle == value
        _refused(_doc(lifecycle="Trusted"))

    def test_an_id_is_a_short_plain_name(self):
        for good in ("react", "agents/react", "a.b-c_d:1"):
            ex.parse_executable(_doc(id=good))
        for bad in ("", " x", "x y", "x\ny", "a" * 129, "-x"):
            _refused(_doc(id=bad))


class TestPrinciple14InTheModels:
    def test_a_variable_may_not_be_named_like_an_input(self):
        problems = _refused(
            _doc(contract={"inputs": ["task"]}, variables=["task"])
        ).problems

        assert any("task" in line and "input" in line for line in problems)

    def test_a_variable_may_not_be_named_like_an_input_template_placeholder(self):
        contract = {
            "inputs": ["task"],
            "input_templates": {"task": "Review {{file}} against {{criteria}}"},
        }

        assert ex.parse_executable(_doc(contract=contract)).contract.input_templates
        _refused(_doc(contract=contract, variables=["file"]))
        ex.parse_executable(_doc(contract=contract, variables=["tone"]))

    def test_an_input_template_belongs_to_a_declared_input(self):
        _refused(_doc(contract={"inputs": ["task"], "input_templates": {"x": "y"}}))

    def test_a_variable_name_has_the_placeholder_shape(self):
        for bad in ("Tone", "1x", "a-b", "", "a b"):
            _refused(_doc(variables=[bad]))


class TestSchemaVersion:
    def test_an_omitted_version_takes_the_default(self):
        assert ex.parse_executable(_doc()).schema_version == "1.0"

    @pytest.mark.parametrize("value", ["2.0", "1", "", "1.0.0", "v1.0", 1, 1.0, None])
    def test_a_malformed_or_foreign_major_version_is_refused(self, value):
        _refused(_doc(schema_version=value))

    def test_a_field_newer_than_the_declared_version_is_a_finding(self, monkeypatch):
        monkeypatch.setitem(
            ex.ENVELOPE_FIELD_VERSIONS, "ExecutableEnvelope.name", "1.1"
        )

        old = ex.parse_executable(_doc(schema_version="1.0", name="Mini"))
        new = ex.parse_executable(_doc(schema_version="1.1", name="Mini"))
        omitted = ex.parse_executable(_doc(schema_version="1.0"))

        assert ex.newer_envelope_fields(old) == (
            SchemaFinding("ExecutableEnvelope", "name", "1.1", "1.0"),
        )
        assert ex.newer_envelope_fields(new) == ()
        assert ex.newer_envelope_fields(omitted) == ()

    def test_a_nested_field_is_found_through_its_class(self, monkeypatch):
        monkeypatch.setitem(ex.ENVELOPE_FIELD_VERSIONS, "Requires.callables", "1.1")
        data = _doc(requires={"callables": ["f"]})

        assert [
            f.field for f in ex.newer_envelope_fields(ex.parse_executable(data))
        ] == ["callables"]

    def test_the_author_label_is_not_the_format_version(self):
        # `version` is the author's label; the body carries its own FSM version
        fsm = _fsm()
        fsm["version"] = "4.0"
        fsm["handler_only_keys"] = ["note"]  # a 4.1 field in a 4.0 body
        envelope = ex.parse_executable(_doc(fsm, version="2"))

        assert ex.newer_envelope_fields(envelope) == ()
        findings = newer_fields(ex.instantiate_envelope(envelope)[0])
        assert [(f.model, f.field) for f in findings] == [
            ("FSMDefinition", "handler_only_keys")
        ]

    def test_every_reachable_field_is_registered(self):
        seen: set[type[BaseModel]] = set()

        def walk(annotation: Any) -> None:
            if isinstance(annotation, type) and issubclass(annotation, BaseModel):
                if annotation not in seen:
                    seen.add(annotation)
                    for field in annotation.model_fields.values():
                        walk(field.annotation)
            for arg in typing.get_args(annotation):
                walk(arg)

        for cls in (
            ex.ExecutableEnvelope,
            ex.FsmEnvelope,
            ex.StrategyEnvelope,
            ex.ProcedureEnvelope,
        ):
            walk(cls)

        for cls in seen:
            for name in cls.model_fields:
                keys = [f"{k.__name__}.{name}" for k in cls.__mro__]
                assert any(k in ex.ENVELOPE_FIELD_VERSIONS for k in keys), (
                    cls.__name__,
                    name,
                )
        for key in ex.ENVELOPE_FIELD_VERSIONS:
            model, _, name = key.partition(".")
            assert any(c.__name__ == model and name in c.model_fields for c in seen), (
                f"stale entry {key}"
            )


class TestInstantiation:
    @staticmethod
    def _templated(**header: Any) -> ex.ExecutableEnvelope:
        fsm = _fsm(response_instructions="Greet in a {{tone}} tone about {{topic}}.")
        return ex.parse_executable(_doc(fsm, variables=["tone", "topic"], **header))

    def test_variables_change_the_instance_and_not_the_template(self):
        envelope = self._templated()
        a, record_a = ex.instantiate_envelope(
            envelope, variables={"tone": "calm", "topic": "x"}
        )
        b, _ = ex.instantiate_envelope(
            envelope, variables={"tone": "loud", "topic": "x"}
        )

        assert a.states["a"].response_instructions == "Greet in a calm tone about x."
        assert record_a == {
            "options": {},
            "variables": {"tone": "calm", "topic": "x"},
        }
        ia, ib = ex.identity_of(envelope, a), ex.identity_of(envelope, b)
        assert ia.instance_identity != ib.instance_identity
        assert ia.template_identity == ib.template_identity

    def test_a_directly_built_envelope_with_a_deep_body_is_refused_not_a_crash(self):
        deep: Any = "x"
        for _ in range(5_000):
            deep = [deep]
        envelope = ex.FsmEnvelope(
            schema="fsm-llm/executable",
            kind="fsm",
            id="deep",
            body=ex.FsmBody(fsm={"initial_state": deep}),
        )

        with pytest.raises(ex.InstantiationError, match="deep"):
            ex.instantiate_envelope(envelope)

    def test_substitution_and_scan_read_tuples_and_mappings(self):
        node = {
            "a": ("Hi {{v}}", types.MappingProxyType({"k": "{{v}}"})),
            "b": ["{{v}}"],
        }

        out = ex.substitute_variables(node, {"v": "X"})

        assert out == {"a": ("Hi X", {"k": "X"}), "b": ["X"]}
        assert ex.scan_placeholders(node) == ({"v"}, set())

    def test_a_declared_variable_without_a_value_is_refused(self):
        with pytest.raises(ex.InstantiationError, match="tone"):
            ex.instantiate_envelope(self._templated(), variables={"topic": "x"})

    def test_a_value_for_an_undeclared_name_is_refused(self):
        with pytest.raises(ex.InstantiationError, match="color"):
            ex.instantiate_envelope(
                self._templated(), variables={"tone": "a", "topic": "b", "color": "c"}
            )

    def test_options_are_refused_until_a_kind_has_an_option_schema(self):
        with pytest.raises(ex.InstantiationError, match="options"):
            ex.instantiate_envelope(
                ex.parse_executable(_doc()), options={"approval": True}
            )

    def test_substitution_touches_string_leaves_only_and_never_rescans(self):
        node = {
            "{{x}}": "key stays",
            "text": "a {{x}} b {{y}} {{z}}",
            "list": ["{{x}}", 3, None, True],
            "deep": {"{{y}}": ["{{y}}"]},
        }

        out = ex.substitute_variables(node, {"x": "{{y}}", "y": "Y"})

        assert out == {
            "{{x}}": "key stays",
            "text": "a {{y}} b Y {{z}}",
            "list": ["{{y}}", 3, None, True],
            "deep": {"{{y}}": ["Y"]},
        }
        assert node["text"] == "a {{x}} b {{y}} {{z}}"  # the input is not mutated

    def test_the_placeholder_scan_reports_leaves_and_keys(self):
        leaves, keys = ex.scan_placeholders(
            {"{{k}}": "x", "a": ["{{v}} and {{W}} and {{ w }}", {"b": "{{u_2}}"}]}
        )

        assert (leaves, keys) == ({"v", "u_2"}, {"k"})

    def test_a_caller_input_is_never_a_variable_and_never_in_an_identity(self):
        """Principle 14 as a behaviour: a caller's task, passed as an input through the real
        calls the loader and instantiation offer, is refused each time, is never echoed in an
        error, and moves neither identity."""
        from fsm_llm.core.verify.executable import load_executable

        document = _doc(contract={"inputs": ["task"]}, variables=["tone"])
        envelope = ex.parse_executable(document)
        definition, record = ex.instantiate_envelope(envelope, variables={"tone": "a"})
        before = ex.identity_of(envelope, definition)

        for i in range(10_000):
            task = f"caller input {i} sk-SECRET-{i}"
            with pytest.raises(ex.InstantiationError) as info:
                ex.instantiate_envelope(envelope, variables={"tone": "a", "task": task})
            assert "task" in str(info.value) and task not in str(info.value)
            if i % 1_000 == 0:  # the loader is the heavier call
                loaded, result = load_executable(
                    document, variables={"tone": "a", "task": task}
                )
                assert loaded is None and not result.is_valid
                assert any("task" in m for m in result.errors)
                assert not any(task in m for m in (*result.errors, *result.warnings))
                with pytest.raises(ex.InstantiationError):
                    ex.rebuild_instance(
                        envelope,
                        {"options": {}, "variables": {"tone": "a", "task": task}},
                        before.instance_identity or "",
                    )

        after_definition, after_record = ex.instantiate_envelope(
            envelope, variables={"tone": "a"}
        )
        assert (ex.identity_of(envelope, after_definition), after_record) == (
            before,
            record,
        )
        assert "task" not in json.dumps(after_record)

    def test_a_refused_body_names_error_types_and_never_a_variable_value(self):
        fsm = _fsm()
        fsm["initial_state"] = "{{first}}"
        envelope = ex.parse_executable(_doc(fsm, variables=["first"]))
        value = "sk-SECRET-123\x1b[31m\u202e"

        with pytest.raises(ex.InstantiationError) as info:
            ex.instantiate_envelope(envelope, variables={"first": value})

        text = str(info.value)
        assert "not a valid FSM definition" in text and "value_error" in text
        assert "SECRET" not in text and "\x1b" not in text and "\u202e" not in text
        assert len(text) < 400

    def test_rebuild_matches_and_a_tampered_record_fails(self):
        envelope = self._templated()
        definition, record = ex.instantiate_envelope(
            envelope, variables={"tone": "calm", "topic": "x"}
        )
        expected = definition_digest(definition)

        assert ex.rebuild_instance(envelope, record, expected) == definition

        tampered = {"options": {}, "variables": {"tone": "calm", "topic": "y"}}
        with pytest.raises(ex.InstanceIdentityMismatchError):
            ex.rebuild_instance(envelope, tampered, expected)
        with pytest.raises(ex.ExecutableError):
            ex.rebuild_instance(envelope, {"variables": 3}, expected)
        assert issubclass(ex.InstanceIdentityMismatchError, ex.InstantiationError)


class TestTheShippedSchema:
    def test_it_equals_the_models(self):
        shipped = json.loads(ex.SCHEMA_PATH.read_text(encoding="utf-8"))

        assert shipped == ex.executable_json_schema(), _REGENERATE

    def test_its_text_is_the_canonical_text(self):
        text = ex.SCHEMA_PATH.read_text(encoding="utf-8")
        built = json.dumps(
            ex.executable_json_schema(), indent=2, sort_keys=True, ensure_ascii=True
        )

        assert text == built + "\n" == ex.schema_text(), _REGENERATE

    def test_it_is_found_through_the_package(self):
        resource = importlib.resources.files("fsm_llm.core.definitions").joinpath(
            "schemas/executable.schema.json"
        )

        assert resource.read_text(encoding="utf-8") == ex.SCHEMA_PATH.read_text(
            encoding="utf-8"
        )

    def test_it_has_the_discriminated_union_and_no_identity(self):
        schema = ex.executable_json_schema()

        assert len(schema["oneOf"]) == 3
        assert schema["discriminator"]["propertyName"] == "kind"
        assert set(schema["discriminator"]["mapping"]) == {
            "fsm",
            "strategy",
            "procedure",
        }
        defs = schema["$defs"]
        for branch in ("FsmEnvelope", "StrategyEnvelope", "ProcedureEnvelope"):
            assert "schema" in defs[branch]["properties"]
            assert "schema_" not in defs[branch]["properties"]
            assert "identity" not in defs[branch]["properties"]
        assert "kind" not in defs.get("State", {}).get(
            "properties", {}
        )  # d454fd62/D-015

    def test_it_is_ascii_only(self):
        assert ex.SCHEMA_PATH.read_bytes().isascii()

    def test_it_is_the_same_under_three_hash_seeds(self):
        code = (
            "from fsm_llm.core.definitions.executable import schema_text; "
            "print(schema_text(), end='')"
        )
        shipped = ex.SCHEMA_PATH.read_text(encoding="utf-8")
        children = {
            seed: subprocess.Popen(
                [sys.executable, "-c", code],
                env={**os.environ, "PYTHONHASHSEED": seed},
                stdout=subprocess.PIPE,
                text=True,
            )
            for seed in ("0", "1", "12345")
        }
        for (
            seed,
            child,
        ) in children.items():  # started together: one import time, not three
            out, _ = child.communicate(timeout=120)
            assert child.returncode == 0
            assert out == shipped, f"PYTHONHASHSEED={seed}; {_REGENERATE}"


def test_the_schema_id_constant_is_the_literal_of_the_model():
    field = ex.ExecutableEnvelope.model_fields["schema_"]

    assert typing.get_args(field.annotation) == (EXECUTABLE_SCHEMA_ID,)
    assert field.alias == "schema"


def test_a_parsed_envelope_does_not_share_the_callers_dict():
    document = _doc()
    envelope = ex.parse_executable(document)
    before = copy.deepcopy(document)

    ex.instantiate_envelope(envelope)

    assert document == before
