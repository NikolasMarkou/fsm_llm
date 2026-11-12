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

    def test_push_drops_only_the_kinds_that_do_not_inherit_their_result(self):
        """Behaviour kept (the asymmetry is fixed at step 55): the parent's
        completion result key is dropped from a pushed child's inherited context,
        the synthesize one is not, and the registry flag decides."""
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
        assert context == {"synthesis_result": {"status": "ok"}, "other": 1}


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
