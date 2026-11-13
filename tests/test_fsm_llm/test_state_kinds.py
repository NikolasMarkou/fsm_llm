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
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar, get_args, get_origin

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

    def test_a_result_owning_kind_drops_its_inherited_result_on_push(self):
        """Both result-owning kinds drop the key (the step 55 fix of the old asymmetry)."""
        assert sk.CONVERSATION.inherits_result_on_push is True
        assert sk.COMPLETION.inherits_result_on_push is False
        assert sk.SYNTHESIZE.inherits_result_on_push is False

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


def _function_node(path: Path, name: str, owner: str | None = None) -> ast.AST:
    """The `def` named *name* (inside class *owner* when given) of a source file."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    scope: ast.AST = tree
    if owner is not None:
        scope = next(
            n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == owner
        )
    return next(
        n for n in ast.walk(scope) if isinstance(n, ast.FunctionDef) and n.name == name
    )


def _config_attributes(node: ast.AST) -> set[str]:
    """The kind-config attributes (`.completion`, `.synthesize`) a function reads."""
    return {
        n.attr
        for n in ast.walk(node)
        if isinstance(n, ast.Attribute) and n.attr in {"completion", "synthesize"}
    }


class TestResultKeyReaders:
    """Plan d454fd62 step 47: the readers of a kind's result key go through the
    registry, so a new kind with a result key needs no edit in them."""

    def _loaded(self):
        for name, (raw, _options) in _CORPUS.items():
            try:
                yield name, FSMDefinition(**copy.deepcopy(raw))
            except Exception:
                continue

    def test_the_readers_equal_the_old_inline_rule_over_the_corpus(self):
        """Guards: the hook output is the rule the readers applied before the
        registry (`state.completion or state.synthesize`), for every corpus state."""
        from fsm_llm.definitions import handler_only_names
        from fsm_llm.flow_analysis import _result_key

        checked = 0
        for name, loaded in self._loaded():
            old = set()
            for state in loaded.states.values():
                config = state.completion or state.synthesize
                expected = config.result_key if config is not None else None
                assert _result_key(state) == expected, (name, state.id)
                if expected is not None:
                    old.add(expected)
                    checked += 1
            assert handler_only_names(loaded) == (
                frozenset(loaded.handler_only_keys) | old
            ), name
        assert checked > 20  # the corpus holds completion and synthesize states

    @pytest.mark.parametrize(
        ("path", "name", "owner"),
        [
            ("definitions.py", "handler_only_names", None),
            ("flow_analysis.py", "_result_key", None),
            ("api.py", "_build_push_context", "API"),
        ],
    )
    def test_no_reader_branches_on_a_kind_config(self, path, name, owner):
        """Guards (D-015): these readers ask the kind registry, never
        `.completion` or `.synthesize` (a new kind would need an edit in each)."""
        node = _function_node(_MODULE.parent / path, name, owner)
        assert _config_attributes(node) == set(), (path, name)

    def test_the_result_key_collision_check_asks_the_registry_for_completion(self):
        """Guards: `FSMDefinition._check_synthesize_states` builds `owned` from the
        registry; only the synthesize edge rules (step 50) still read `.synthesize`."""
        node = _function_node(
            _MODULE.parent / "definitions.py",
            "_check_synthesize_states",
            "FSMDefinition",
        )
        assert "completion" not in _config_attributes(node)

    def test_a_result_key_collision_names_the_owning_kind(self):
        """The refusal text is unchanged: the label is built from the kind name."""
        done = {
            "id": "ask",
            "description": "d",
            "purpose": "p",
            "completion": {**_completion(), "result_key": "answer_slot"},
            "transitions": [{"target_state": "work", "description": "go"}],
        }
        work = {
            "id": "work",
            "description": "d",
            "purpose": "p",
            "synthesize": {
                **_synthesize(),
                "return_keys": ["answer_slot"],
                "on_failure": "end",
            },
            "transitions": [
                {
                    "target_state": "end",
                    "description": "x",
                    "conditions": [
                        {
                            "description": "failed",
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
        }
        end = {"id": "end", "description": "d", "purpose": "p"}
        with pytest.raises(ValueError, match="result_key of a completion state"):
            FSMDefinition(
                name="n",
                description="d",
                initial_state="ask",
                persona="p",
                states={"ask": done, "work": work, "end": end},
            )

    def test_push_drops_the_result_key_of_each_kind_that_does_not_inherit_it(self):
        """The parent's completion and synthesize result keys are both dropped
        from a pushed child's inherited context (step 55 fixed the old asymmetry
        that kept the synthesize one), and the registry flag decides."""
        calls = {
            "completion": {"kind": "final", "text": "parent", "calls": []},
            "synthesis": {"status": "ok"},
        }
        fsm = {
            "name": "child",
            "description": "d",
            "initial_state": "a",
            "persona": "p",
            "states": {
                "a": {
                    "id": "a",
                    "description": "d",
                    "purpose": "p",
                    "completion": {**_completion(), "result_key": "completion_result"},
                    "transitions": [{"target_state": "b", "description": "go"}],
                },
                "b": {
                    "id": "b",
                    "description": "d",
                    "purpose": "p",
                    "synthesize": {**_synthesize(), "on_failure": "c"},
                    "transitions": [
                        {
                            "target_state": "c",
                            "description": "x",
                            "conditions": [
                                {
                                    "description": "failed",
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
                },
                "c": {"id": "c", "description": "d", "purpose": "p"},
            },
        }
        child = FSMDefinition(**fsm)
        api = API.__new__(API)

        class _Manager:
            @staticmethod
            def get_conversation_data(_conversation_id: str) -> dict[str, Any]:
                return {
                    "completion_result": calls["completion"],
                    "synthesis_result": calls["synthesis"],
                    "other": 1,
                }

        api.fsm_manager = _Manager()  # type: ignore[assignment]
        context = api._build_push_context(
            "conv", None, False, True, child_definition=child
        )
        assert context == {"other": 1}


class TestMintingHooks:
    """Plan d454fd62 step 48: the minting rule and the produced-keys rule live in
    the registry; the pipeline and the flow analyses read the hooks, and the hand
    mirror in flow_analysis is gone."""

    @staticmethod
    def _old_minted(state: State) -> list[str]:
        """Frozen oracle: the rule `MessagePipeline._build_field_configs_from_state`
        spelled before the hook (ordered list of the config field names)."""
        if state.completion is not None or state.synthesize is not None:
            return []
        required: list[str] = list(state.required_context_keys or [])
        for transition in state.transitions or []:
            for condition in transition.conditions or []:
                for key in condition.requires_context_keys or []:
                    if key not in required:
                        required.append(key)
        owned = {c.field_name for c in state.classification_extractions or []}
        required = [k for k in required if k not in owned]
        names: list[str] = []
        if required:
            names = list(required)
        if state.field_extractions:
            explicit = {fc.field_name for fc in state.field_extractions}
            names = [n for n in names if n not in explicit]
            names.extend(fc.field_name for fc in state.field_extractions)
        return names

    @staticmethod
    def _old_produced(state: State, handler_only: frozenset[str]) -> set[str]:
        """Frozen oracle: `flow_analysis._produced` and `_minted_keys` before the hooks."""
        minted: set[str] = set()
        if state.completion is None and state.synthesize is None:
            minted = set(state.required_context_keys or [])
            for transition in state.transitions or []:
                for condition in transition.conditions or []:
                    minted.update(condition.requires_context_keys or [])
            owned = {c.field_name for c in state.classification_extractions or []}
            minted = (minted - owned) | {
                f.field_name for f in state.field_extractions or []
            }
        keys = minted | {c.field_name for c in state.classification_extractions or []}
        keys -= handler_only
        config = state.completion or state.synthesize
        return keys | {config.result_key} if config is not None else keys

    def _states(self):
        for name, (raw, _options) in _CORPUS.items():
            try:
                loaded = FSMDefinition(**copy.deepcopy(raw))
            except Exception:
                continue
            for state in loaded.states.values():
                yield name, loaded, state

    def test_minted_keys_equal_the_old_pipeline_rule_over_the_corpus(self):
        from fsm_llm.pipeline import MessagePipeline

        checked = minting = 0
        for name, _loaded, state in self._states():
            checked += 1
            hook = list(sk.kind_of(state).minted_keys(state))
            assert hook == self._old_minted(state), (name, state.id)
            configs = MessagePipeline._build_field_configs_from_state(state)
            assert [c.field_name for c in configs] == hook, (name, state.id)
            minting += bool(hook)
        assert checked > 300 and minting > 50  # the corpus holds both shapes

    def test_produced_keys_equal_the_old_flow_rule_over_the_corpus(self):
        from fsm_llm.definitions import handler_only_names

        checked = 0
        for name, loaded, state in self._states():
            handler_only = handler_only_names(loaded)
            assert sk.kind_of(state).produced_keys(state, handler_only) == frozenset(
                self._old_produced(state, handler_only)
            ), (name, state.id)
            checked += 1
        assert checked > 300

    def test_a_kind_that_mints_nothing_gets_no_config_even_with_explicit_fields(self):
        """A completion state built around the loader (`model_construct`) with an
        explicit extraction still mints nothing: the kind decides, as before."""
        from fsm_llm.definitions import FieldExtractionConfig
        from fsm_llm.pipeline import MessagePipeline

        state = _plain_state(completion=_completion())
        bypass = state.model_copy(
            update={
                "field_extractions": [
                    FieldExtractionConfig(
                        field_name="a",
                        field_type="str",
                        extraction_instructions="get a",
                    )
                ],
                "required_context_keys": ["k"],
            }
        )
        assert sk.kind_of(bypass).name == "completion"
        assert sk.kind_of(bypass).minted_keys(bypass) == ()
        assert MessagePipeline._build_field_configs_from_state(bypass) == []

    def test_the_hooks_of_a_new_kind_default_to_minting_nothing(self):
        spec = sk.KindSpec(name="probe", config_field="probe")
        state = _plain_state(required_context_keys=["k"])
        assert spec.minted_keys(state) == ()

    def test_the_mirror_in_flow_analysis_is_gone(self):
        from fsm_llm import flow_analysis

        assert not hasattr(flow_analysis, "_minted_keys")

    @pytest.mark.parametrize(
        ("path", "name", "owner"),
        [
            ("pipeline.py", "_build_field_configs_from_state", "MessagePipeline"),
            ("flow_analysis.py", "_produced", None),
            ("flow_analysis.py", "declared_available", None),
        ],
    )
    def test_no_reader_branches_on_a_kind_config(self, path, name, owner):
        """Guards (D-015): the minting and arrival rules ask the registry, never
        `.completion` or `.synthesize`."""
        node = _function_node(_MODULE.parent / path, name, owner)
        assert _config_attributes(node) == set(), (path, name)

    @pytest.mark.parametrize(
        ("path", "name", "owner"),
        [
            ("pipeline.py", "_build_field_configs_from_state", "MessagePipeline"),
            ("flow_analysis.py", "_produced", None),
            ("flow_analysis.py", "declared_available", None),
        ],
    )
    def test_the_minting_rule_is_not_spelled_outside_the_registry(
        self, path, name, owner
    ):
        """Guards (plan 2cefeb61 D-017 item 7): the rule's inputs are read in
        `state_kinds.py` only, so there is no second spelling to keep in step."""
        node = _function_node(_MODULE.parent / path, name, owner)
        read = {
            n.attr
            for n in ast.walk(node)
            if isinstance(n, ast.Attribute)
            and n.attr in {"required_context_keys", "requires_context_keys"}
        }
        assert read == set(), (path, name)


class TestStateValidationHooks:
    """Plan d454fd62 step 49: the State validator asks the registry for the rules
    of a kind (forbidden declarations, the per-kind state check), so a new kind
    needs no edit in it; every refusal text is byte-equal to the parent's."""

    _FIELDS: ClassVar[dict[str, Any]] = {
        "field_extractions": [
            {"field_name": "a", "field_type": "str", "extraction_instructions": "a"}
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
    }

    @staticmethod
    def _old_refusal(raw: dict[str, Any]) -> str | None:
        """Frozen oracle: the text the State validator built for the completion
        and synthesize rules before the hooks (None when it accepts)."""
        sid = raw["id"]
        completion, synthesize = raw.get("completion"), raw.get("synthesize")
        extraction = (
            "field_extractions",
            "classification_extractions",
            "required_context_keys",
            "extraction_instructions",
        )
        if completion is not None:
            conflicts = [n for n in extraction if raw.get(n)]
            if conflicts:
                return (
                    f"State '{sid}': a completion state cannot also declare {conflicts}"
                )
            if not raw.get("transitions"):
                return (
                    f"State '{sid}': a completion state needs a transition "
                    "(a terminal state runs no turn)"
                )
        if synthesize is not None:
            conflicts = [n for n in ("completion", *extraction) if raw.get(n)]
            if conflicts:
                return (
                    f"State '{sid}': a synthesize state cannot also declare {conflicts}"
                )
            targets = {t["target_state"] for t in raw.get("transitions") or []}
            if synthesize["on_failure"] not in targets:
                return (
                    f"State '{sid}': synthesize on_failure "
                    f"{synthesize['on_failure']!r} must be one of the state's "
                    "transition targets (a terminal state has none)"
                )
        return None

    @staticmethod
    def _shapes():
        """Every combination of kind configs, declared channels, and edges."""
        import itertools

        channels = list(TestStateValidationHooks._FIELDS)
        for has_completion, has_synthesize in itertools.product([0, 1], repeat=2):
            for picked in [(), *((c,) for c in channels), tuple(channels)]:
                for edges in ("none", "t", "other"):
                    raw: dict[str, Any] = {
                        "id": "s",
                        "description": "d",
                        "purpose": "p",
                    }
                    if has_completion:
                        raw["completion"] = _completion()
                    if has_synthesize:
                        raw["synthesize"] = _synthesize()
                    for name in picked:
                        raw[name] = copy.deepcopy(
                            TestStateValidationHooks._FIELDS[name]
                        )
                    if edges != "none":
                        raw["transitions"] = [
                            {
                                "target_state": "t" if edges == "t" else "u",
                                "description": "go",
                            }
                        ]
                    yield raw

    def test_the_validator_reads_no_kind_config(self):
        """Guards (D-015): `State._validate_extraction_configs` never reads
        `.completion` or `.synthesize` (a new kind would need an edit there)."""
        node = _function_node(
            _MODULE.parent / "definitions.py", "_validate_extraction_configs", "State"
        )
        assert _config_attributes(node) == set()

    def test_every_refusal_text_equals_the_old_rule_text(self):
        """Guards: over every kind and declaration shape the State refuses with
        exactly the text the inline rules built, and accepts what they accepted."""
        refused = accepted = 0
        for raw in self._shapes():
            expected = self._old_refusal(raw)
            try:
                State(**copy.deepcopy(raw))
            except ValidationError as exc:
                text = exc.errors()[0]["msg"]
                if expected is None:
                    # another rule of the same validator or model refused it
                    for rule in (
                        "a completion state",
                        "a synthesize state",
                        "on_failure",
                    ):
                        assert rule not in text, raw
                    continue
                assert text == f"Value error, {expected}", raw
                refused += 1
            else:
                assert expected is None, raw
                accepted += 1
        assert refused > 50 and accepted > 5

    def test_state_problem_is_the_one_refusal_rule(self):
        done = _plain_state(completion=_completion())
        assert sk.state_problem(done) is None
        bare = State.model_construct(
            id="s", completion=_completion_config(), transitions=[]
        )
        assert sk.state_problem(bare) == (
            "a completion state needs a transition (a terminal state runs no turn)"
        )

    def test_a_state_holding_two_configs_is_judged_by_both_kinds_in_order(self):
        """`kind_of` names one kind, but the rules of every set config apply, in
        registry order (completion first): the refusal of the pair is the
        synthesize one, as before the hooks."""
        both = State.model_construct(
            id="s",
            completion=_completion_config(),
            synthesize=_synthesize_config(),
            transitions=[_edge("t")],
            field_extractions=None,
            classification_extractions=None,
            required_context_keys=None,
            extraction_instructions=None,
        )
        assert sk.state_problem(both) == (
            "a synthesize state cannot also declare ['completion']"
        )

    def test_a_kind_with_its_own_rules_needs_no_edit_in_the_validator(self):
        """A spec registered at runtime brings its own forbidden declarations and
        its own state check; `state_problem` applies both."""
        from dataclasses import replace

        marker = replace(
            sk.CONVERSATION,
            name="marker",
            config_field="marker",
            forbidden_declarations=("extraction_instructions",),
            check_state=lambda s: None if s.transitions else "a marker needs an edge",
        )
        # the hooks read a state by getattr only, so any object with the fields works
        state = SimpleNamespace(
            id="s", marker=object(), extraction_instructions="x", transitions=[]
        )
        original = sk.KINDS
        sk.KINDS = (*original, marker)
        try:
            assert sk.state_problem(state) == (
                "a marker state cannot also declare ['extraction_instructions']"
            )
            state.extraction_instructions = None
            assert sk.state_problem(state) == "a marker needs an edge"
        finally:
            sk.KINDS = original
        assert sk.KINDS == original

    def test_the_default_check_accepts_everything(self):
        plain = _plain_state()
        assert sk.CONVERSATION.check_state(plain) is None
        assert sk.state_problem(plain) is None


def _host_condition(logic, requires=("synthesis_result",)):
    return {
        "description": "c",
        "requires_context_keys": list(requires),
        "logic": logic,
    }


_OK = {"==": [{"var": "synthesis_result.status"}, "ok"]}
_FAILED = {"in": [{"var": "synthesis_result.status"}, ["failed", "aborted"]]}


def _host_edge(target, *conditions):
    return {"target_state": target, "description": "go", "conditions": list(conditions)}


def _host_definition(
    edges, synth=None, handler_only=(), extra_states=None
) -> dict[str, Any]:
    """A start state, a synthesize state `work` (edges *edges*, failure state
    `fail`) and the extra states, each reachable from `start`."""
    extra = extra_states or {}
    states: dict[str, Any] = {
        "start": {
            "id": "start",
            "description": "d",
            "purpose": "p",
            "transitions": [
                {"target_state": t, "description": "go"} for t in ["work", *extra]
            ],
        },
        "work": {
            "id": "work",
            "description": "d",
            "purpose": "p",
            "synthesize": {
                "goal": "g",
                "return_keys": ["k"],
                "on_failure": "fail",
                **(synth or {}),
            },
            "transitions": edges,
        },
        "end": {"id": "end", "description": "d", "purpose": "p"},
        "fail": {"id": "fail", "description": "d", "purpose": "p"},
        **extra,
    }
    return {
        "name": "n",
        "description": "d",
        "initial_state": "start",
        "persona": "p",
        "handler_only_keys": list(handler_only),
        "states": states,
    }


def _status(value):
    return {"==": [{"var": "synthesis_result.status"}, value]}


def _abort_edge_only(target):
    return _host_edge(
        target,
        _host_condition({"var": "procedure_abort"}, ("procedure_abort",)),
    )


def _host_scenarios() -> dict[str, dict[str, Any]]:
    ok_edge = _host_edge("end", _host_condition(_OK))
    fail_edge = _host_edge("fail", _host_condition(_FAILED))
    other = {"==": [{"var": "x"}, 1]}
    answer_slot = {
        "ask": {
            "id": "ask",
            "description": "d",
            "purpose": "p",
            "completion": {**_completion(), "result_key": "answer_slot"},
            "transitions": [{"target_state": "end", "description": "go"}],
        }
    }
    nested = {
        "nested": {
            "id": "nested",
            "description": "d",
            "purpose": "p",
            "synthesize": {
                "goal": "g",
                "return_keys": ["z"],
                "on_failure": "fail",
                "result_key": "other_result",
            },
            "transitions": [
                _host_edge(
                    "end",
                    _host_condition(
                        {"==": [{"var": "other_result.status"}, "ok"]},
                        ("other_result",),
                    ),
                ),
                _host_edge(
                    "fail",
                    _host_condition(
                        {"in": [{"var": "other_result.status"}, ["failed", "aborted"]]},
                        ("other_result",),
                    ),
                ),
            ],
        }
    }
    return {
        "valid": _host_definition([ok_edge, fail_edge]),
        "edge-reads-other-key": _host_definition(
            [_host_edge("end", _host_condition(other, ("x",))), fail_edge]
        ),
        "edge-without-conditions": _host_definition(
            [{"target_state": "end", "description": "go"}, fail_edge]
        ),
        "failed-only": _host_definition(
            [ok_edge, _host_edge("fail", _host_condition(_status("failed")))]
        ),
        "aborted-only": _host_definition(
            [ok_edge, _host_edge("fail", _host_condition(_status("aborted")))]
        ),
        "negated-fires-on-absent": _host_definition(
            [
                ok_edge,
                _host_edge(
                    "fail", _host_condition({"!": [{"var": "synthesis_result"}]}, ())
                ),
                fail_edge,
            ]
        ),
        "fires-on-running": _host_definition(
            [
                _host_edge(
                    "end",
                    _host_condition({"!=": [{"var": "synthesis_result.status"}, "ok"]}),
                ),
                fail_edge,
            ]
        ),
        "no-failure-edge-passes": _host_definition(
            [ok_edge, _host_edge("fail", _host_condition(_status("never")))]
        ),
        "and-with-other-key": _host_definition(
            [
                ok_edge,
                _host_edge(
                    "fail",
                    _host_condition(_FAILED),
                    _host_condition({"==": [{"var": "note"}, "x"]}, ("note",)),
                ),
            ]
        ),
        "handler-only-edge-alone": _host_definition(
            [_abort_edge_only("end"), fail_edge], handler_only=("procedure_abort",)
        ),
        "handler-only-failure-edge": _host_definition(
            [ok_edge, _abort_edge_only("fail")], handler_only=("procedure_abort",)
        ),
        "return-key-handler-only": _host_definition(
            [ok_edge, fail_edge],
            synth={"return_keys": ["flag"]},
            handler_only=("flag",),
        ),
        "input-handler-only-ok": _host_definition(
            [ok_edge, fail_edge], synth={"inputs": ["flag"]}, handler_only=("flag",)
        ),
        "return-key-is-completion-result": _host_definition(
            [ok_edge, fail_edge],
            synth={"return_keys": ["answer_slot"]},
            extra_states=answer_slot,
        ),
        "input-is-synthesize-result-of-other-state": _host_definition(
            [ok_edge, fail_edge],
            synth={"inputs": ["other_result"]},
            extra_states=nested,
        ),
    }


# The refusal text (or None: the definition loads) the loader produced for each
# scenario BEFORE the edge checks moved into the registry (recorded on the parent
# commit b409a0ea of plan d454fd62, never regenerated).
_HOST_EXPECTED: dict[str, str | None] = {
    "valid": None,
    "edge-reads-other-key": "Value error, State 'work': the transition to 'end' must read the synthesize result key 'synthesis_result' (a condition that requires it or reads it in its logic), or it fires before the runner has produced a result",
    "edge-without-conditions": "Value error, State 'work': the transition to 'end' must read the synthesize result key 'synthesis_result' (a condition that requires it or reads it in its logic), or it fires before the runner has produced a result",
    "failed-only": "Value error, State 'work': no transition to 'fail' fires when the procedure failed or aborted (it does not pass when 'synthesis_result' is aborted, {'status': 'aborted', 'reason': 'budget_spent'})",
    "aborted-only": "Value error, State 'work': no transition to 'fail' fires when the procedure failed or aborted (it does not pass when 'synthesis_result' is failed, {'status': 'failed', 'reason': 'compile_failed'})",
    "negated-fires-on-absent": "Value error, State 'work': the transition to 'fail' fires before the runner has produced a result (it passes when 'synthesis_result' is absent)",
    "fires-on-running": "Value error, State 'work': the transition to 'end' fires before the runner has produced a result (it passes when 'synthesis_result' is running)",
    "no-failure-edge-passes": "Value error, State 'work': no transition to 'fail' fires when the procedure failed or aborted (it does not pass when 'synthesis_result' is failed, {'status': 'failed', 'reason': 'compile_failed'})",
    "and-with-other-key": None,
    "handler-only-edge-alone": None,
    "handler-only-failure-edge": "Value error, State 'work': no transition to 'fail' fires when the procedure failed or aborted (it does not pass when 'synthesis_result' is failed, {'status': 'failed', 'reason': 'compile_failed'})",
    "return-key-handler-only": "Value error, State 'work': synthesize return_keys entry 'flag' collides with a handler_only_keys entry",
    "input-handler-only-ok": None,
    "return-key-is-completion-result": "Value error, State 'work': synthesize return_keys entry 'answer_slot' collides with result_key of a completion state",
    "input-is-synthesize-result-of-other-state": "Value error, State 'work': synthesize inputs entry 'other_result' collides with result_key of a synthesize state",
}


class TestDefinitionCheckHooks:
    """Plan d454fd62 step 50: the loader's cross-state rules for a synthesize
    state (its edges read the result key, run on sample results, and the host
    keys do not collide with a runner-owned key) live in the kind spec; the
    loader asks the registry and keeps only the call. Every refusal text is
    byte-equal to the parent's."""

    def test_the_loader_reads_no_kind_config_and_asks_the_registry(self):
        """Guards (D-015): `_check_synthesize_states` names neither kind config nor
        the sample constant; it calls `definition_problem` (a new kind with a
        cross-state rule needs no edit in the loader)."""
        node = _function_node(
            _MODULE.parent / "definitions.py",
            "_check_synthesize_states",
            "FSMDefinition",
        )
        assert _config_attributes(node) == set()
        names = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
        assert "_SYNTHESIZE_SAMPLES" not in names
        assert "definition_problem" in names

    @pytest.mark.parametrize("name", sorted(_HOST_EXPECTED))
    def test_every_refusal_text_equals_the_recorded_parent_text(self, name):
        raw = _host_scenarios()[name]
        expected = _HOST_EXPECTED[name]
        if expected is None:
            FSMDefinition(**copy.deepcopy(raw))
            return
        with pytest.raises(ValidationError) as excinfo:
            FSMDefinition(**copy.deepcopy(raw))
        assert excinfo.value.errors()[0]["msg"] == expected

    def test_the_scenarios_cover_the_refusals_and_the_acceptances(self):
        assert set(_host_scenarios()) == set(_HOST_EXPECTED)
        refused = [v for v in _HOST_EXPECTED.values() if v is not None]
        assert len(refused) >= 9 and len(refused) < len(_HOST_EXPECTED)

    def test_the_synthesize_kind_owns_the_samples_and_the_definition_check(self):
        state = _plain_state(synthesize=_synthesize())
        samples = sk.SYNTHESIZE.sample_results(state)
        assert [label for label, _ in samples] == [
            "absent",
            "running",
            "failed",
            "aborted",
        ]
        assert samples[0][1] is None
        assert sk.CONVERSATION.sample_results(_plain_state()) == ()
        assert (
            sk.COMPLETION.sample_results(_plain_state(completion=_completion())) == ()
        )

    def test_the_sample_constant_of_the_loader_is_the_hook_data(self):
        """`test_codes` reads `definitions._SYNTHESIZE_SAMPLES`; the name stays and
        its value is the one tuple the hook returns (the leaf cannot import
        `codes.Status`, so test_codes pins the strings against the enum)."""
        from fsm_llm import definitions
        from fsm_llm.codes import Status

        state = _plain_state(synthesize=_synthesize())
        assert definitions._SYNTHESIZE_SAMPLES is sk.SYNTHESIZE.sample_results(state)
        assert sk._RUNNING == Status.RUNNING.value

    def test_only_the_synthesize_kind_has_a_definition_check(self):
        for spec in (sk.CONVERSATION, sk.COMPLETION):
            assert spec.check_definition is sk._no_definition_problem
        assert sk.SYNTHESIZE.check_definition is not sk._no_definition_problem

    def test_the_evaluator_is_passed_in_and_nothing_else_runs_edges(self):
        """The hook module never imports the evaluator: the one handed to
        `definition_problem` is the one that judges the sample results."""
        from fsm_llm.definitions import logic_referenced_keys

        raw = _host_scenarios()["valid"]
        definition = FSMDefinition(**copy.deepcopy(raw))
        state = definition.states["work"]
        seen: list[str] = []

        class Probe:
            def transition_passes(self, transition, context):
                seen.append(sorted(context))
                return False

        problem = sk.definition_problem(
            definition, state, Probe(), logic_referenced_keys
        )
        assert problem is not None and "no transition to 'fail' fires" in problem
        assert seen and all(isinstance(keys, list) for keys in seen)

    def test_a_kind_registered_at_runtime_brings_its_own_cross_state_rule(self):
        """The loader asks the registry: replace the synthesize spec's check and
        the loader refuses with the replacement's message, unedited."""
        from dataclasses import replace

        stub = replace(
            sk.SYNTHESIZE,
            check_definition=lambda d, s, e, k: f"stub rule on {s.id}",
        )
        original = sk.KINDS
        sk.KINDS = (sk.CONVERSATION, sk.COMPLETION, stub)
        try:
            with pytest.raises(ValidationError) as excinfo:
                FSMDefinition(**copy.deepcopy(_host_scenarios()["valid"]))
            assert (
                excinfo.value.errors()[0]["msg"]
                == "Value error, State 'work': stub rule on work"
            )
        finally:
            sk.KINDS = original
        FSMDefinition(**copy.deepcopy(_host_scenarios()["valid"]))

    def test_a_state_holding_two_configs_is_checked_by_every_kind(self):
        """Behaviour kept: a completion-and-synthesize state built around the
        validator still gets the synthesize edge rules (as `config is not None`
        did), not only the rules of `kind_of`'s kind."""
        both = State.model_construct(
            id="s",
            completion=_completion_config(),
            synthesize=_synthesize_config(),
            transitions=[_edge("t")],
        )
        definition = SimpleNamespace(states={"s": both}, handler_only_keys=[], name="n")
        from fsm_llm.definitions import logic_referenced_keys
        from fsm_llm.transition_evaluator import TransitionEvaluator

        problem = sk.definition_problem(
            definition, both, TransitionEvaluator(), logic_referenced_keys
        )
        assert problem is not None and "must read the synthesize result key" in problem

    def test_owned_result_keys_labels_each_key_with_its_kind(self):
        done = _plain_state(completion={**_completion(), "result_key": "slot"})
        host = _plain_state(synthesize=_synthesize())
        assert sk.owned_result_keys([done, host, _plain_state()]) == {
            "slot": "result_key of a completion state",
            "synthesis_result": "result_key of a synthesize state",
        }


class TestVerifierHooks:
    """Plan d454fd62 step 51: G4 (through the produced keys), the dry-run gate G6a
    and the raw-dict lint of the validator read the registry, so a new kind needs
    no edit in them; every verdict is equal to the parent's."""

    _ROOT = _MODULE.parent

    @staticmethod
    def _old_mentions(state: State) -> list[str]:
        """Frozen oracle: the names `dryrun._mentions` collected before the hook."""
        return [
            *(state.required_context_keys or []),
            *(f.field_name for f in state.field_extractions or []),
            *(c.field_name for c in state.classification_extractions or []),
            *(
                key
                for t in state.transitions or []
                for c in t.conditions or []
                for key in c.requires_context_keys or []
            ),
        ]

    @staticmethod
    def _old_typed(state: State) -> dict[str, tuple[str, tuple[str, ...]]]:
        """Frozen oracle: the typed keys `dryrun._narrowing` built before the hook."""
        kinds = {"bool": "bool", "int": "number", "float": "number"}
        found = {
            c.field_name: ("intent", tuple(i.name for i in c.intents))
            for c in state.classification_extractions or []
        }
        for f in state.field_extractions or []:
            found[f.field_name] = (kinds.get(f.field_type, "any"), ())
        return found

    @staticmethod
    def _old_kind_warnings(fsm_data: dict[str, Any]) -> list[str]:
        """Frozen oracle: the warnings `FSMValidator._validate_completion_states`
        added before the hooks, in order."""
        results = ("calls", "final", "malformed")
        structured = ("final", "malformed")

        def literals(node: Any) -> set[str]:
            if isinstance(node, str):
                return {node}
            if isinstance(node, dict):
                return set().union(*(literals(v) for v in node.values()))
            if isinstance(node, list):
                return set().union(*(literals(v) for v in node))
            return set()

        out: list[str] = []
        states = fsm_data.get("states", {})
        result_keys: set[str] = set()
        for state_id, state in states.items():
            completion = state.get("completion")
            if not isinstance(completion, dict):
                continue
            result_keys.add(completion.get("result_key") or "completion_result")
            kinds = results if completion.get("tools") else structured
            transitions = state.get("transitions") or []
            if any(not t.get("conditions") for t in transitions):
                continue
            named: set[str] = set()
            for transition in transitions:
                for condition in transition.get("conditions") or []:
                    named |= literals(condition.get("logic"))
            missing = [kind for kind in kinds if kind not in named]
            if missing:
                out.append(
                    f"Completion state '{state_id}' has no transition for result "
                    f"kind(s) {missing} and no unconditional transition: such a "
                    "turn stays in the state and makes no further call"
                )
        handler_only = set(fsm_data.get("handler_only_keys") or [])
        for state_id, state in states.items():
            declared = [
                ("field_extractions", fe.get("field_name"))
                for fe in state.get("field_extractions") or []
            ]
            declared += [
                ("classification_extractions", ce.get("field_name"))
                for ce in state.get("classification_extractions") or []
            ]
            declared += [
                ("required_context_keys", key)
                for key in state.get("required_context_keys") or []
            ]
            for channel, key in declared:
                if key in result_keys or (
                    key in handler_only and channel != "classification_extractions"
                ):
                    out.append(
                        f"State '{state_id}' {channel} names '{key}', a key only "
                        "handlers or a completion call write: it is never "
                        "extracted"
                    )
        return out

    _WANTED = ("Completion state '", "a key only handlers or a completion call")

    @staticmethod
    def _stage_warnings(fsm_data: dict[str, Any]) -> list[str]:
        """The warnings of the validator's kind stage alone (raw shapes that never
        load reach it too): `_validate_kind_states`, which the parent spelled
        `_validate_completion_states`."""
        from fsm_llm.validator import FSMValidator

        validator = FSMValidator(copy.deepcopy(fsm_data))
        validator._validate_kind_states()
        return list(validator.result.warnings)

    @staticmethod
    def _kind_warnings(fsm_data: dict[str, Any]) -> list[str]:
        """The kind warnings of a whole `validate()` run."""
        from fsm_llm.validator import FSMValidator

        warnings = FSMValidator(copy.deepcopy(fsm_data)).validate().warnings
        return [w for w in warnings if any(p in w for p in TestVerifierHooks._WANTED)]

    def _states(self):
        for name, (raw, _options) in _CORPUS.items():
            try:
                loaded = FSMDefinition(**copy.deepcopy(raw))
            except Exception:
                continue
            for state in loaded.states.values():
                yield name, loaded, state

    # ---- no kind branch in the readers -------------------------------

    @pytest.mark.parametrize("name", ["_search", "_narrowing", "_mentions"])
    def test_the_dryrun_readers_read_no_kind_config(self, name):
        node = _function_node(self._ROOT / "synthesis" / "dryrun.py", name)
        assert _config_attributes(node) == set(), name

    @pytest.mark.parametrize("name", ["_narrowing", "_mentions"])
    def test_the_dryrun_readers_read_no_extraction_field(self, name):
        """The extraction fields are read by the kind's hooks, not by the gate."""
        node = _function_node(self._ROOT / "synthesis" / "dryrun.py", name)
        fields = {
            n.attr
            for n in ast.walk(node)
            if isinstance(n, ast.Attribute)
            and n.attr
            in {
                "field_extractions",
                "classification_extractions",
                "required_context_keys",
            }
        }
        assert fields == set(), name

    @pytest.mark.parametrize("name", ["analyze_data_flow", "analyze_liveness"])
    def test_the_flow_analyses_read_no_kind_config(self, name):
        node = _function_node(self._ROOT / "flow_analysis.py", name)
        assert _config_attributes(node) == set(), name

    def test_the_validator_asks_the_kinds_for_its_raw_rules(self):
        """`FSMValidator` never names a kind config: no `.get("completion")`."""
        tree = ast.parse((self._ROOT / "validator.py").read_text(encoding="utf-8"))
        cls = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.ClassDef) and n.name == "FSMValidator"
        )
        named = {
            call.args[0].value
            for call in ast.walk(cls)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "get"
            and call.args
            and isinstance(call.args[0], ast.Constant)
            and call.args[0].value in {"completion", "synthesize"}
        }
        assert named == set()

    # ---- equality with the parent rules ------------------------------

    def test_mentioned_keys_equal_the_old_rule_over_the_corpus(self):
        checked = 0
        for name, _loaded, state in self._states():
            hook = list(sk.kind_of(state).mentioned_keys(state))
            assert hook == self._old_mentions(state), (name, state.id)
            checked += 1
        assert checked > 300

    def test_typed_keys_equal_the_old_rule_over_the_corpus(self):
        checked = typed = 0
        for name, _loaded, state in self._states():
            hook = sk.kind_of(state).typed_keys(state)
            assert hook == self._old_typed(state), (name, state.id)
            checked += 1
            typed += any(kind != "any" for kind, _ in hook.values())
        assert checked > 300 and typed > 10

    def test_typed_keys_keep_a_later_field_over_a_classification_of_its_name(self):
        state = _plain_state(
            classification_extractions=[
                {
                    "field_name": "k",
                    "intents": [
                        {"name": "x", "description": "x"},
                        {"name": "y", "description": "y"},
                    ],
                    "fallback_intent": "y",
                }
            ],
            field_extractions=[
                {
                    "field_name": "k",
                    "field_type": "int",
                    "extraction_instructions": "k",
                },
                {
                    "field_name": "s",
                    "field_type": "str",
                    "extraction_instructions": "s",
                },
            ],
        )
        assert sk.CONVERSATION.typed_keys(state) == {
            "k": ("number", ()),
            "s": ("any", ()),
        }

    def test_the_dryrun_verdicts_over_the_corpus_are_judged_by_the_hook(self):
        """`dryrun_judged` is False for exactly the kinds that were skipped."""
        for name, _loaded, state in self._states():
            skipped = state.completion is not None or state.synthesize is not None
            assert sk.kind_of(state).dryrun_judged is (not skipped), (name, state.id)

    def test_the_raw_lint_equals_the_old_validator_rule_over_the_corpus(self):
        """Through `validate()`: a definition the structure stage refuses never
        reaches the lint (no warning), every other one says what it said."""
        from fsm_llm.validator import FSMValidator

        checked = warned = 0
        for name, (raw, _options) in _CORPUS.items():
            reached = FSMValidator(copy.deepcopy(raw)).validate()
            expected = self._old_kind_warnings(raw) if reached.is_valid else []
            assert self._kind_warnings(raw) == expected, name
            assert self._stage_kind_warnings(raw) == self._old_kind_warnings(raw), name
            checked += 1
            warned += bool(expected)
        assert checked > 100 and warned >= 2

    def _stage_kind_warnings(self, raw: dict[str, Any]) -> list[str]:
        return [
            w for w in self._stage_warnings(raw) if any(p in w for p in self._WANTED)
        ]

    @staticmethod
    def _raw_shapes():
        edge = {"target_state": "b", "description": "go"}
        gated = {
            **edge,
            "conditions": [
                {
                    "description": "g",
                    "logic": {
                        "or": [
                            {"==": [{"var": "completion_result.kind"}, "final"]},
                            {"==": [{"var": "completion_result.kind"}, "malformed"]},
                        ]
                    },
                }
            ],
        }
        all_three = {
            **edge,
            "conditions": [
                {
                    "description": "g",
                    "logic": {
                        "in": [
                            {"var": "completion_result.kind"},
                            ["calls", "final", "malformed"],
                        ]
                    },
                }
            ],
        }
        tools = [{"type": "function", "function": {"name": "f"}}]
        for label, completion in {
            "tools": {"tools": tools},
            "structured": {"response_format_key": "_fmt"},
            "custom-key": {"tools": tools, "result_key": "res"},
        }.items():
            for edges_label, edges in {
                "gated-two": [gated],
                "gated-all": [all_three],
                "unconditional": [edge],
                "mixed": [gated, edge],
                "none": [],
            }.items():
                yield f"{label}/{edges_label}", completion, edges

    def test_the_raw_lint_equals_the_old_rule_over_raw_shapes(self):
        """Raw dicts that never load (no transitions, odd keys) are linted too:
        the validator reads raw dicts and must say what it said before."""
        checked = warned = 0
        for label, completion, edges in self._raw_shapes():
            for channel in (None, "completion_result", "res", "handlers"):
                state: dict[str, Any] = {
                    "id": "a",
                    "description": "d",
                    "purpose": "p",
                    "completion": completion,
                    "transitions": edges,
                }
                other: dict[str, Any] = {"id": "b", "description": "d", "purpose": "p"}
                if channel:
                    other["required_context_keys"] = [channel]
                    other["field_extractions"] = [
                        {"field_name": channel, "field_type": "str"}
                    ]
                raw = {
                    "name": "N",
                    "description": "d",
                    "initial_state": "a",
                    "handler_only_keys": ["handlers"],
                    "states": {"a": state, "b": other},
                }
                expected = self._old_kind_warnings(raw)
                assert self._stage_kind_warnings(raw) == expected, (label, channel)
                checked += 1
                warned += bool(expected)
        assert checked == 60 and warned >= 30

    def test_a_non_dict_completion_is_not_linted(self):
        raw = {
            "name": "N",
            "description": "d",
            "initial_state": "a",
            "states": {"a": {"id": "a", "completion": "nope"}},
        }
        assert self._stage_kind_warnings(raw) == []
        assert self._old_kind_warnings(raw) == []

    # ---- a new kind needs no edit in the readers ---------------------

    def _swap_conversation(self, monkeypatch, **changes):
        from dataclasses import replace

        spec = replace(sk.CONVERSATION, **changes)
        monkeypatch.setattr(sk, "CONVERSATION", spec)
        monkeypatch.setattr(sk, "KINDS", (spec, *sk.KINDS[1:]))

    @staticmethod
    def _dead_edge_definition() -> FSMDefinition:
        return FSMDefinition(
            name="T",
            description="t",
            initial_state="ask",
            states={
                "ask": {
                    "id": "ask",
                    "description": "d",
                    "purpose": "p",
                    "field_extractions": [
                        {
                            "field_name": "ok",
                            "field_type": "bool",
                            "extraction_instructions": "ok",
                        }
                    ],
                    "transitions": [
                        {
                            "target_state": "done",
                            "description": "go",
                            "conditions": [
                                {
                                    "description": "g",
                                    "logic": {"==": [{"var": "ok"}, "yes"]},
                                }
                            ],
                        }
                    ],
                },
                "done": {"id": "done", "description": "d", "purpose": "p"},
            },
        )

    def test_the_gate_judges_the_edges_of_the_kinds_the_hook_names(self, monkeypatch):
        from fsm_llm.synthesis.dryrun import analyze_dryrun

        definition = self._dead_edge_definition()
        assert {f.code for f in analyze_dryrun(definition)} == {
            "edge-unfireable",
            "terminal-unreachable",
        }
        self._swap_conversation(monkeypatch, dryrun_judged=False)
        assert analyze_dryrun(definition) == []

    def test_the_gate_narrows_the_keys_the_hook_types(self, monkeypatch):
        from fsm_llm.synthesis.dryrun import analyze_dryrun

        self._swap_conversation(monkeypatch, typed_keys=lambda state: {})
        assert analyze_dryrun(self._dead_edge_definition()) == []

    def test_the_gate_counts_the_keys_the_hook_mentions(self, monkeypatch):
        """A key another state mentions is not narrowed: the hook says who does."""
        from fsm_llm.synthesis.dryrun import analyze_dryrun

        definition = self._dead_edge_definition()
        done = definition.states["done"]
        self._swap_conversation(
            monkeypatch,
            mentioned_keys=lambda state: ("ok",) if state is done else (),
        )
        # `ask` no longer mentions `ok` itself: the mention of `done` alone is
        # another state's, so the key is not narrowed and the edge can fire.
        assert analyze_dryrun(definition) == []

    def test_the_liveness_pass_reads_the_produced_keys_of_the_kind(self, monkeypatch):
        """G4 counts a gate read as satisfiable through `produced_keys`: the same
        loop is fine while the kind produces `k`, and `cycle-no-exit` when it
        produces nothing."""
        from fsm_llm.flow_analysis import analyze_liveness

        loop = {"target_state": "a", "description": "again", "priority": 2}
        leave = {
            "target_state": "b",
            "description": "go",
            "priority": 1,
            "conditions": [
                {"description": "g", "logic": {"==": [{"var": "k"}, "done"]}}
            ],
        }

        definition = FSMDefinition(
            name="T",
            description="t",
            initial_state="a",
            states={
                "a": {
                    "id": "a",
                    "description": "d",
                    "purpose": "p",
                    "field_extractions": [
                        {
                            "field_name": "k",
                            "field_type": "str",
                            "extraction_instructions": "k",
                        }
                    ],
                    "transitions": [loop, leave],
                },
                "b": {"id": "b", "description": "d", "purpose": "p"},
            },
        )
        assert [f.code for f in analyze_liveness(definition)] == []
        self._swap_conversation(
            monkeypatch, produced_keys=lambda state, handler_only: frozenset()
        )
        assert [f.code for f in analyze_liveness(definition)] == ["cycle-no-exit"]

    def test_a_kind_with_its_own_raw_lint_needs_no_edit_in_the_validator(
        self, monkeypatch
    ):
        from dataclasses import replace

        def lint(state_id, state, fsm_data, services):
            return [f"marker state '{state_id}' says {sorted(state['marker'])}"]

        marker = replace(
            sk.CONVERSATION,
            name="marker",
            config_field="marker",
            lint_raw=lint,
            raw_result_key=lambda state, services: state["marker"].get("key"),
        )
        raw = {
            "name": "N",
            "description": "d",
            "initial_state": "a",
            "states": {
                "a": {
                    "id": "a",
                    "description": "d",
                    "purpose": "p",
                    "marker": {"key": "mk", "x": 1},
                    "transitions": [{"target_state": "b", "description": "go"}],
                },
                "b": {
                    "id": "b",
                    "description": "d",
                    "purpose": "p",
                    "required_context_keys": ["mk"],
                },
            },
        }
        from fsm_llm.validator import FSMValidator

        monkeypatch.setattr(sk, "KINDS", (*sk.KINDS, marker))
        warnings = FSMValidator(raw).validate().warnings
        assert "marker state 'a' says ['key', 'x']" in warnings
        assert any("State 'b' required_context_keys names 'mk'" in w for w in warnings)

    def test_the_raw_hooks_default_to_nothing(self):
        services = sk.RawLintServices(("calls",), "k", lambda node: set())
        assert sk.CONVERSATION.lint_raw("s", {}, {}, services) == []
        assert sk.SYNTHESIZE.lint_raw("s", {"synthesize": {}}, {}, services) == []
        assert sk.CONVERSATION.raw_result_key({}, services) is None
        assert sk.SYNTHESIZE.raw_result_key({"synthesize": {}}, services) is None
        assert sk.COMPLETION.raw_result_key({"completion": {}}, services) == "k"
        assert (
            sk.COMPLETION.raw_result_key({"completion": {"result_key": "r"}}, services)
            == "r"
        )

    def test_a_kind_that_forbids_bulk_extraction_cannot_raise_the_bulk_finding(self):
        """`analyze_data_flow` asks the kind (its forbidden declarations) instead of
        testing `.completion`: a state built around the loader with both is silent
        for the kinds that refuse the declaration, and flagged for the default."""
        from fsm_llm.flow_analysis import analyze_data_flow

        def definition(**extra):
            return FSMDefinition(
                name="T",
                description="t",
                initial_state="a",
                states={
                    "a": {
                        "id": "a",
                        "description": "d",
                        "purpose": "p",
                        "extraction_instructions": "get it",
                        "transitions": [{"target_state": "b", "description": "go"}],
                    },
                    "b": {"id": "b", "description": "d", "purpose": "p"},
                },
            )

        loaded = definition()
        assert [f.code for f in analyze_data_flow(loaded)] == ["bulk-extraction"]
        bypass = loaded.states["a"].model_copy(
            update={"completion": _completion_config()}
        )
        loaded.states["a"] = bypass
        assert analyze_data_flow(loaded) == []


class TestToolAndProcedureHooks:
    """Plan d454fd62 step 52: the declared tool names (three copies before) and the
    "a procedure cannot hold this kind" rule read the registry; every answer is
    equal to the parent's."""

    _ROOT = _MODULE.parent / "synthesis"

    # Frozen oracles of the three parent copies.
    @staticmethod
    def _old_synthesizer(definition: FSMDefinition) -> list[str]:
        states = definition.states.values()
        return sorted(
            {
                t["function"]["name"]
                for s in states
                if s.completion and s.completion.tools
                for t in s.completion.tools
            }
        )

    @staticmethod
    def _old_library(definition: FSMDefinition) -> list[str]:
        return sorted(
            {
                t["function"]["name"]
                for s in definition.states.values()
                if s.completion and s.completion.tools
                for t in s.completion.tools
                if isinstance(t.get("function"), dict)
                and isinstance(t["function"].get("name"), str)
            }
        )

    @staticmethod
    def _old_policy(completion: dict[str, Any]) -> list[str | None]:
        tools = completion.get("tools")
        entries = [] if tools is None else tools if isinstance(tools, list) else [tools]
        names: list[str | None] = []
        for tool in entries:
            fn = tool.get("function") if isinstance(tool, dict) else None
            name = fn.get("name") if isinstance(fn, dict) else None
            names.append(name if isinstance(name, str) and name else None)
        return names

    _RAW_TOOLS: ClassVar[list[Any]] = [
        None,
        [],
        [{"type": "function", "function": {"name": "a"}}],
        [
            {"type": "function", "function": {"name": "a"}},
            {"type": "function", "function": {"name": "b"}},
        ],
        [{"type": "function", "function": {"name": ""}}],
        [{"type": "function", "function": {"name": 3}}],
        [{"type": "function", "function": "a"}],
        [{"type": "function"}],
        ["a", 3, None, {"function": {"name": "z"}}],
        {"type": "function", "function": {"name": "solo"}},
        "not a list",
        [{"function": {"name": "a"}}, {"function": {"name": "a"}}],
    ]

    def _loaded(self):
        for name, (raw, _options) in _CORPUS.items():
            try:
                yield name, FSMDefinition(**copy.deepcopy(raw))
            except Exception:
                continue

    # ---- no kind branch in the readers -------------------------------

    @pytest.mark.parametrize("module", ["synthesizer", "library"])
    def test_the_typed_readers_read_no_completion_config(self, module):
        tree = ast.parse((self._ROOT / f"{module}.py").read_text(encoding="utf-8"))
        assert [
            n.lineno
            for n in ast.walk(tree)
            if isinstance(n, ast.Attribute) and n.attr == "completion"
        ] == [], module

    def test_the_policy_keeps_no_copy_of_the_tool_reader(self):
        tree = ast.parse((self._ROOT / "policy.py").read_text(encoding="utf-8"))
        defined = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        assert "_completion_tool_names" not in defined

    def test_the_runner_refusal_reads_the_hook_not_the_config(self):
        node = _function_node(self._ROOT / "runner.py", "runnable_reason")
        assert _config_attributes(node) == set()

    # ---- equal to the parent over the corpus -------------------------

    def test_the_typed_hook_equals_both_parent_copies_over_the_corpus(self):
        seen = 0
        for name, definition in self._loaded():
            states = definition.states.values()
            got = sorted(sk.declared_tool_names(states))
            assert got == self._old_synthesizer(definition), name
            assert got == self._old_library(definition), name
            seen += len(got)
        assert seen > 0, "the corpus holds no completion tool"

    def test_the_raw_hook_equals_the_parent_policy_reader(self):
        for tools in self._RAW_TOOLS:
            completion = {"tools": tools}
            assert list(
                sk.COMPLETION.raw_declared_tools({"completion": completion})
            ) == (self._old_policy(completion)), tools
        assert list(sk.COMPLETION.raw_declared_tools({"completion": {}})) == []

    def test_the_raw_hook_equals_the_parent_over_the_corpus_states(self):
        for name, (raw, _options) in _CORPUS.items():
            for state in (raw.get("states") or {}).values():
                if not isinstance(state, dict):
                    continue
                completion = state.get("completion")
                if isinstance(completion, dict):
                    assert list(
                        sk.COMPLETION.raw_declared_tools(state)
                    ) == self._old_policy(completion), name

    def test_the_typed_hook_reads_a_state_around_the_loader(self):
        state = _plain_state().model_copy(
            update={
                "completion": SimpleNamespace(tools=[{"function": {"name": "x"}}, 3])
            }
        )
        assert sk.COMPLETION.declared_tools(state) == ("x",)

    def test_a_kind_without_tools_declares_none(self):
        state = _plain_state(synthesize=_synthesize())
        assert sk.CONVERSATION.declared_tools(state) == ()
        assert sk.SYNTHESIZE.declared_tools(state) == ()
        assert sk.CONVERSATION.raw_declared_tools({}) == ()
        assert sk.SYNTHESIZE.raw_declared_tools({"synthesize": {}}) == ()
        assert sk.declared_tool_names([state]) == frozenset()

    # ---- supported_in_procedure --------------------------------------

    def test_only_the_completion_kind_is_unsupported_in_a_procedure(self):
        assert {spec.name: spec.supported_in_procedure for spec in sk.KINDS} == {
            "conversation": True,
            "completion": False,
            "synthesize": True,
        }

    def test_the_runner_refusal_equals_the_parent_over_the_corpus(self):
        from fsm_llm.synthesis.policy import SynthesisPolicy
        from fsm_llm.synthesis.runner import runnable_reason

        refused = 0
        for name, definition in self._loaded():
            old = any(s.completion for s in definition.states.values())
            got = runnable_reason(
                definition,
                "verified",
                min_lifecycle="verified",
                policy=SynthesisPolicy(),
                inputs=(),
                return_keys=(),
            )
            assert (got == "completion_unsupported") is old, name
            refused += old
        assert refused > 0

    # ---- a new kind needs no edit in the readers ---------------------

    def test_the_runner_refuses_the_kinds_the_hook_names(self, monkeypatch):
        from fsm_llm.synthesis.policy import SynthesisPolicy
        from fsm_llm.synthesis.runner import runnable_reason

        definition = FSMDefinition(
            **_host_definition(
                [
                    _host_edge("end", _host_condition(_OK)),
                    _host_edge("fail", _host_condition(_FAILED)),
                ]
            )
        )

        def reason():
            return runnable_reason(
                definition,
                "verified",
                min_lifecycle="verified",
                policy=SynthesisPolicy(),
                inputs=(),
                return_keys=(),
            )

        assert reason() != "completion_unsupported"
        spec = replace(sk.SYNTHESIZE, supported_in_procedure=False)
        monkeypatch.setattr(sk, "SYNTHESIZE", spec)
        monkeypatch.setattr(sk, "KINDS", (*sk.KINDS[:2], spec))
        assert reason() == "completion_unsupported"

    def _swap_completion(self, monkeypatch, **changes):
        spec = replace(sk.COMPLETION, **changes)
        monkeypatch.setattr(sk, "COMPLETION", spec)
        monkeypatch.setattr(sk, "KINDS", (sk.KINDS[0], spec, sk.KINDS[2]))

    def test_the_library_reads_the_tools_the_hook_names(self, monkeypatch, tmp_path):
        from fsm_llm.synthesis.library import FSMLibrary

        definition = FSMDefinition(
            name="T",
            description="t",
            initial_state="a",
            states={
                "a": {
                    "id": "a",
                    "description": "d",
                    "purpose": "p",
                    "completion": _completion(),
                    "transitions": [{"target_state": "t", "description": "go"}],
                },
                "t": {"id": "t", "description": "d", "purpose": "p"},
            },
        )

        def tools_used():
            lib = FSMLibrary(tmp_path / str(len(list(tmp_path.iterdir()))))
            record = lib.put(
                definition, goal="g", inputs_required=[], return_keys=["k"]
            )
            return record.signature["tools_used"]

        assert tools_used() == ["f"]
        self._swap_completion(monkeypatch, declared_tools=lambda state: ("zz", "aa"))
        assert tools_used() == ["aa", "zz"]

    def test_the_approval_request_names_the_tools_the_hook_names(self, monkeypatch):
        from fsm_llm.synthesis.synthesizer import _approval_request

        definition = FSMDefinition(
            name="T",
            description="t",
            initial_state="a",
            states={
                "a": {
                    "id": "a",
                    "description": "d",
                    "purpose": "p",
                    "completion": _completion(),
                    "transitions": [{"target_state": "t", "description": "go"}],
                },
                "t": {"id": "t", "description": "d", "purpose": "p"},
            },
        )
        result = SimpleNamespace(
            definition=definition,
            report=SimpleNamespace(verdicts={}),
            simulation=None,
        )

        def summary():
            return _approval_request("g", ("k",), result, "d" * 64).summary

        assert summary().endswith("tools: f")
        self._swap_completion(monkeypatch, declared_tools=lambda state: ("zz",))
        assert summary().endswith("tools: zz")

    def test_the_policy_reads_the_raw_tools_the_hook_names(self, monkeypatch):
        from fsm_llm.synthesis.policy import (
            SynthesisPolicy,
            check_policy,
            needs_approval,
        )

        data = {
            "name": "T",
            "initial_state": "a",
            "states": {
                "a": {
                    "id": "a",
                    "completion": {"response_format_key": "_shape"},
                    "transitions": [{"target_state": "t"}],
                },
                "t": {"id": "t"},
            },
        }
        policy = SynthesisPolicy()
        assert needs_approval(data, policy) is False
        assert [f.code for f in check_policy(data, policy)] == []
        self._swap_completion(monkeypatch, raw_declared_tools=lambda state: ("evil",))
        assert needs_approval(data, policy) is True
        assert "completion-tool-not-allowed" in [
            f.code for f in check_policy(data, policy)
        ]


def _completion_config():
    from fsm_llm.definitions import CompletionStateConfig

    return CompletionStateConfig(**_completion())


def _synthesize_config():
    from fsm_llm.definitions import SynthesizeConfig

    return SynthesizeConfig(**_synthesize())


def _edge(target: str):
    from fsm_llm.definitions import Transition

    return Transition(target_state=target, description="go")


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
