"""The state-kind registry (plan d454fd62 step 46, D-004, D-015).

`fsm_llm.state_kinds` names the kinds a state can have (conversation,
completion, synthesize) and the declarative rules each kind owns. `State.kind`
is a plain property that reads it. These tests pin the registry against the
`State` model (a new kind field without a spec fails), against the rules the
loader applies today, and the hard constraints: `kind` is not a field, not in
`model_dump` and not in the JSON schema, so `fsm_id` and the library digest do
not move; the module is a leaf that imports nothing from `fsm_llm`.
"""

from __future__ import annotations

import ast
import copy
import typing
from pathlib import Path
from typing import Any, get_args, get_origin

import pytest
from pydantic import BaseModel, ValidationError

from fsm_llm import state_kinds as sk
from fsm_llm.api import API
from fsm_llm.definitions import FSMDefinition, State
from tests.test_fsm_llm import kind_corpus as kc

_MODULE = Path(sk.__file__)
# State fields typed `Model | None` that are not a state kind.
_NOT_KIND_FIELDS = {"context_scope"}
_CORPUS = kc.corpus()


def _plain_state(**fields: Any) -> State:
    base: dict[str, Any] = {
        "id": "s",
        "description": "d",
        "purpose": "p",
        "transitions": [{"target_state": "t", "description": "go"}],
    }
    return State(**{**base, **fields})


def _completion() -> dict[str, Any]:
    return {"tools": [{"type": "function", "function": {"name": "f"}}]}


def _synthesize() -> dict[str, Any]:
    return {"goal": "g", "return_keys": ["k"], "on_failure": "t"}


def _model_config_fields() -> set[str]:
    """`State` fields annotated `SomeModel | None`, found by walking the annotations."""
    names = set()
    for name, info in State.model_fields.items():
        args = get_args(info.annotation) if get_origin(info.annotation) else ()
        if type(None) in args and any(
            isinstance(a, type) and issubclass(a, BaseModel) for a in args
        ):
            names.add(name)
    return names


class TestRegistryShape:
    def test_names_are_the_three_built_in_kinds_in_order(self):
        assert [spec.name for spec in sk.KINDS] == [
            "conversation",
            "completion",
            "synthesize",
        ]

    def test_the_default_kind_is_the_first_and_the_only_one_without_a_config_field(
        self,
    ):
        assert sk.KINDS[0] is sk.CONVERSATION
        assert [s.name for s in sk.KINDS if s.config_field is None] == ["conversation"]

    def test_the_vocabulary_enum_equals_the_registry_names(self):
        assert [m.value for m in sk.CompletionStateKind] == [s.name for s in sk.KINDS]
        assert sk.CompletionStateKind.from_string("completion") is (
            sk.CompletionStateKind.COMPLETION
        )
        with pytest.raises(ValueError):
            sk.CompletionStateKind.from_string("Completion ")

    def test_config_fields_equal_the_model_valued_optional_fields_of_state(self):
        """Guards: a new `State` kind field with no spec (and a spec naming no field)."""
        declared = {s.config_field for s in sk.KINDS if s.config_field}
        assert declared == _model_config_fields() - _NOT_KIND_FIELDS

    def test_every_config_field_exists_on_state(self):
        for spec in sk.KINDS:
            if spec.config_field:
                assert spec.config_field in State.model_fields

    def test_specs_are_frozen(self):
        with pytest.raises(Exception):
            sk.COMPLETION.name = "other"  # type: ignore[misc]

    def test_get_kind_finds_by_name_and_refuses_an_unknown_one(self):
        assert sk.get_kind("synthesize") is sk.SYNTHESIZE
        with pytest.raises(KeyError, match="nope"):
            sk.get_kind("nope")


class TestKindOf:
    def test_a_plain_state_is_conversation(self):
        assert sk.kind_of(_plain_state()) is sk.CONVERSATION

    def test_a_completion_state_is_completion(self):
        assert sk.kind_of(_plain_state(completion=_completion())) is sk.COMPLETION

    def test_a_synthesize_state_is_synthesize(self):
        assert sk.kind_of(_plain_state(synthesize=_synthesize())) is sk.SYNTHESIZE

    def test_both_configs_through_model_construct_resolve_in_registry_order(self):
        """Guards: `model_construct` bypasses the validator that refuses both."""
        state = State.model_construct(
            id="s",
            completion=_plain_state(completion=_completion()).completion,
            synthesize=_plain_state(synthesize=_synthesize()).synthesize,
        )
        assert sk.kind_of(state) is sk.COMPLETION

    def test_a_state_like_object_with_no_config_attribute_is_conversation(self):
        class Bare:
            pass

        assert sk.kind_of(Bare()) is sk.CONVERSATION

    def test_the_state_property_returns_the_kind_name(self):
        assert _plain_state().kind == "conversation"
        assert _plain_state(completion=_completion()).kind == "completion"
        assert _plain_state(synthesize=_synthesize()).kind == "synthesize"
        assert type(_plain_state().kind) is str

    def test_every_corpus_state_has_exactly_one_kind_and_the_parent_rule_agrees(self):
        checked = 0
        for name, (raw, _options) in _CORPUS.items():
            try:
                loaded = FSMDefinition(**copy.deepcopy(raw))
            except Exception:  # the 19 refused shapes never load
                continue
            for state_id, state in loaded.states.items():
                configs = [
                    s.name
                    for s in sk.KINDS
                    if s.config_field and getattr(state, s.config_field) is not None
                ]
                assert len(configs) <= 1, (name, state_id, configs)
                parent_rule = (
                    "completion"
                    if state.completion is not None
                    else "synthesize"
                    if state.synthesize is not None
                    else "conversation"
                )
                assert state.kind == parent_rule, (name, state_id)
                checked += 1
        assert checked > 300


class TestKindIsNotData:
    """The hard constraints of D-004: `kind` never reaches a dump, a schema or an id."""

    def test_kind_is_a_plain_property_and_not_a_field(self):
        assert "kind" not in State.model_fields
        assert "kind" not in State.model_computed_fields
        assert isinstance(State.__dict__["kind"], property)
        assert "kind" not in getattr(State, "__private_attributes__", {})

    def test_kind_is_absent_from_model_dump_in_every_mode(self):
        state = _plain_state(completion=_completion())
        for kwargs in ({}, {"exclude_defaults": True}, {"mode": "json"}):
            assert "kind" not in state.model_dump(**kwargs)
        assert "kind" not in state.model_dump_json()

    def test_kind_is_absent_from_the_json_schema_of_state_and_the_definition(self):
        for model in (State, FSMDefinition):
            for mode in ("validation", "serialization"):
                schema = model.model_json_schema(mode=mode)
                assert "kind" not in schema.get("properties", {})
                assert "kind" not in schema.get("$defs", {}).get("State", {}).get(
                    "properties", {}
                )

    def test_a_kind_keyword_is_still_not_a_field_input(self):
        """Guards: a settable `kind` would let a definition declare its own kind."""
        state = _plain_state(kind="completion")
        assert state.kind == "conversation"

    def test_assigning_kind_is_refused(self):
        with pytest.raises((AttributeError, ValidationError, ValueError)):
            _plain_state().kind = "completion"  # type: ignore[misc]

    def test_fsm_id_of_every_kind_bearing_corpus_case_is_the_parents(self):
        """Guards: a `computed_field` kind would move the hash of `model_dump`."""
        baseline = kc.read_baseline()["cases"]
        checked = 0
        for name, (raw, _options) in _CORPUS.items():
            try:
                loaded = FSMDefinition(**copy.deepcopy(raw))
            except Exception:
                continue
            if all(s.kind == "conversation" for s in loaded.states.values()):
                continue
            dump = loaded.model_dump(exclude_defaults=True)
            assert all("kind" not in state for state in dump["states"].values())
            assert (
                API.process_fsm_definition(copy.deepcopy(raw))[1]
                == (baseline[name]["fsm_id"])
            ), name
            checked += 1
        assert checked >= 30


class TestHooks:
    """The declarative hooks equal the rule the code applies today."""

    def test_result_key_reads_the_config_result_key(self):
        done = _plain_state(completion={**_completion(), "result_key": "r1"})
        synth = _plain_state(synthesize={**_synthesize(), "result_key": "r2"})
        assert sk.kind_of(done).result_key(done) == "r1"
        assert sk.kind_of(synth).result_key(synth) == "r2"
        plain = _plain_state()
        assert sk.kind_of(plain).result_key(plain) is None

    def test_result_key_default_names_match_the_config_defaults(self):
        done = _plain_state(completion=_completion())
        synth = _plain_state(synthesize=_synthesize())
        assert sk.COMPLETION.result_key(done) == "completion_result"
        assert sk.SYNTHESIZE.result_key(synth) == "synthesis_result"

    def test_result_key_agrees_with_handler_only_names_over_the_corpus(self):
        from fsm_llm.definitions import handler_only_names

        for name, (raw, _options) in _CORPUS.items():
            try:
                loaded = FSMDefinition(**copy.deepcopy(raw))
            except Exception:
                continue
            via_kinds = frozenset(loaded.handler_only_keys) | {
                key
                for state in loaded.states.values()
                if (key := sk.kind_of(state).result_key(state)) is not None
            }
            assert via_kinds == handler_only_names(loaded), name

    def test_inherits_result_on_push_keeps_the_parent_asymmetry(self):
        """Preserved here, fixed at step 55 (api.py drops only the completion key)."""
        assert sk.CONVERSATION.inherits_result_on_push is True
        assert sk.COMPLETION.inherits_result_on_push is False
        assert sk.SYNTHESIZE.inherits_result_on_push is True

    @pytest.mark.parametrize("spec_name", ["completion", "synthesize"])
    def test_each_forbidden_declaration_is_refused_by_the_state_validator(
        self, spec_name
    ):
        spec = sk.get_kind(spec_name)
        config = _completion() if spec_name == "completion" else _synthesize()
        values: dict[str, Any] = {
            "field_extractions": [
                {
                    "field_name": "a",
                    "field_type": "str",
                    "extraction_instructions": "get a",
                }
            ],
            "classification_extractions": [
                {
                    "field_name": "c",
                    "intents": [
                        {"name": "x", "description": "x"},
                        {"name": "y", "description": "y"},
                    ],
                    "fallback_intent": "y",
                }
            ],
            "required_context_keys": ["k"],
            "extraction_instructions": "extract",
            "completion": _completion(),
        }
        assert spec.forbidden_declarations
        for field in spec.forbidden_declarations:
            with pytest.raises(ValidationError, match="cannot also declare"):
                _plain_state(**{spec_name: config, field: values[field]})

    def test_the_forbidden_declarations_are_state_fields(self):
        for spec in sk.KINDS:
            for field in spec.forbidden_declarations:
                assert field in State.model_fields

    def test_the_forbidden_sets_equal_the_loader_lists(self):
        assert sk.COMPLETION.forbidden_declarations == (
            "field_extractions",
            "classification_extractions",
            "required_context_keys",
            "extraction_instructions",
        )
        assert sk.SYNTHESIZE.forbidden_declarations == (
            "completion",
            "field_extractions",
            "classification_extractions",
            "required_context_keys",
            "extraction_instructions",
        )
        assert sk.CONVERSATION.forbidden_declarations == ()

    def test_dryrun_judged_is_false_only_for_the_two_config_kinds(self):
        assert sk.CONVERSATION.dryrun_judged is True
        assert sk.COMPLETION.dryrun_judged is False
        assert sk.SYNTHESIZE.dryrun_judged is False


class TestLeafModule:
    def _imports(self) -> list[str]:
        tree = ast.parse(_MODULE.read_text(encoding="utf-8"))
        found: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                found += [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                found.append(("." * node.level) + (node.module or ""))
        return found

    def test_it_imports_nothing_from_fsm_llm(self):
        """Guards: the registry is a leaf the loader, the flow analyses and the
        engine all import (D-015); an import back would make a cycle."""
        for name in self._imports():
            assert not name.startswith("."), name
            assert name.split(".")[0] != "fsm_llm", name

    def test_it_imports_only_the_standard_library(self):
        stdlib = set(__import__("sys").stdlib_module_names)
        for name in self._imports():
            assert name.split(".")[0] in stdlib, name

    def test_it_is_not_loaded_by_the_package_init(self):
        tree = ast.parse((_MODULE.parent / "__init__.py").read_text(encoding="utf-8"))
        names = {
            (node.module or "")
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        }
        assert not any("state_kinds" in n for n in names)

    def test_every_name_in_all_exists(self):
        for name in sk.__all__:
            assert hasattr(sk, name), name
        assert typing.get_type_hints(sk.KindSpec)
