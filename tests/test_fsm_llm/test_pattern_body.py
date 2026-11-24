"""The ``pattern`` kind of the executable envelope: body, holes, options, variants, identity.

Plan a555dcfe step 5 (D-011). The module under test is lazy on purpose. A pattern document is a
template: typed ``{"$hole": name}`` leaves, a closed options-schema subset, named RFC 6902
variants and a run block. These rows hold the format: what is refused (one line each), the order
of operations, the key order of the instance (the bytes of the E0.1 fixtures depend on it), the
two identities and the replay of a record.
"""

from __future__ import annotations

import copy
import json
from typing import Any

import pytest

from fsm_llm.core.definitions import executable as ex
from fsm_llm.core.definitions import executable_instance as ei
from fsm_llm.core.definitions import executable_pattern as ep
from fsm_llm.core.definitions.fsm import FSMDefinition, definition_digest
from fsm_llm.core.verify.executable import load_executable

_ADD_STATE = {
    "op": "add",
    "path": "/states/gate",
    "value": {
        "id": "gate",
        "description": "gate {{tone}}",
        "purpose": "hold",
        "response_instructions": "wait {{tone}}",
        "required_context_keys": {"$hole": "keys"},
    },
}


def _fsm() -> dict[str, Any]:
    return {
        "name": "Pat",
        "description": "static text",
        "initial_state": "start",
        "persona": "a helper {{tone}}",
        "states": {
            "start": {
                "id": "start",
                "description": "start",
                "purpose": "begin {{tone}}",
                "response_instructions": "say hi",
                "required_context_keys": {"$hole": "keys"},
                "transitions": [
                    {
                        "target_state": "done",
                        "description": "finish",
                        "priority": {"$hole": "priority"},
                    }
                ],
            },
            "done": {
                "id": "done",
                "description": "done",
                "purpose": "end",
                "response_instructions": "bye",
            },
        },
    }


def _body(**over: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "fsm": _fsm(),
        "options_schema": {
            "properties": {
                "keys": {"type": "array", "items": {"type": "string"}, "maxItems": 4},
                "priority": {"type": "integer", "minimum": 1, "maximum": 900},
            }
        },
        "options": {"keys": [], "priority": 100},
        "variants": {
            "first": [
                {
                    "op": "add",
                    "path": "/states/gate",
                    "value": {
                        "id": "gate",
                        "description": "gate",
                        "purpose": "hold",
                        "response_instructions": "wait",
                        "transitions": [
                            {
                                "target_state": "done",
                                "description": "go on",
                                "priority": 5,
                            }
                        ],
                    },
                },
                {
                    "op": "replace",
                    "path": "/states/start/transitions/0/target_state",
                    "value": "gate",
                },
            ],
            "second": [
                {
                    "op": "replace",
                    "path": "/states/gate/response_instructions",
                    "value": "waited",
                },
            ],
        },
    }
    body.update(over)
    return body


def _doc(**over: Any) -> dict[str, Any]:
    doc: dict[str, Any] = {
        "schema": "fsm-llm/executable",
        "kind": "pattern",
        "id": "demo/pattern",
        "variables": ["tone"],
        "body": _body(),
    }
    doc.update(over)
    return doc


def _envelope(**over: Any) -> ex.PatternEnvelope:
    envelope = ex.parse_executable(_doc(**over))
    assert isinstance(envelope, ex.PatternEnvelope)
    return envelope


def _refused(doc: object) -> ex.ExecutableDocumentError:
    with pytest.raises(ex.ExecutableDocumentError) as caught:
        ex.parse_executable(doc)
    return caught.value


def _only_problem(doc: object) -> str:
    problems = _refused(doc).problems
    assert len(problems) == 1, problems
    return problems[0]


TONE = {"tone": "calm"}


class TestAPatternDocumentParses:
    def test_it_is_a_pattern_envelope_with_a_pattern_body(self):
        envelope = _envelope()

        assert envelope.kind == "pattern"
        assert isinstance(envelope.body, ex.PatternBody)
        assert list(envelope.body.variants) == ["first", "second"]
        assert envelope.body.run == ep.RunBlock()

    def test_exactly_one_of_fsm_and_generator(self):
        both = _body(generator="agents.prompt_chain")
        neither = {k: v for k, v in _body().items() if k != "fsm"}

        assert "exactly one of fsm and generator" in _only_problem(_doc(body=both))
        assert "exactly one of fsm and generator" in _only_problem(_doc(body=neither))

    def test_a_generator_body_parses_and_cannot_be_instantiated_yet(self):
        body = {"generator": "agents.prompt_chain", "options_schema": {}}
        envelope = _envelope(body=body)

        with pytest.raises(
            ei.GeneratorUnavailableError, match="no generator namespace"
        ):
            ei.instantiate_body(envelope, variables=TONE)
        with pytest.raises(ei.InstantiationError):
            ei.instantiate_envelope(envelope, variables=TONE)

    def test_a_generator_body_takes_no_variant(self):
        body = {"generator": "g", "variants": {"a": [{"op": "remove", "path": "/x"}]}}

        assert "no variants" in _only_problem(_doc(body=body))

    def test_the_header_fields_of_a_pattern_are_the_header_of_every_kind(self):
        envelope = _envelope(
            lifecycle="draft",
            capabilities=["tools", "hitl"],
            requires={"handlers": ["h"]},
        )

        assert [c.value for c in envelope.capabilities] == ["tools", "hitl"]
        assert envelope.requires.handlers == ["h"]

    def test_an_unknown_key_is_refused_at_every_level(self):
        for over in (
            {"body": _body(surprise=1)},
            {"body": _body(run={"surprise": 1})},
            {"body": _body(handlers=[{"handler": "h", "on": "entry", "surprise": 1}])},
            {"body": _body(options_schema={"properties": {}, "surprise": 1})},
        ):
            _refused(_doc(**over))

    def test_workflow_is_still_refused_by_name(self):
        problems = _refused({**_doc(), "kind": "workflow"}).problems

        assert problems == (
            "kind: 'workflow' is not supported by this version "
            "(known kinds: fsm, strategy, procedure, pattern)",
        )


class TestHolesAreWholeValues:
    def test_a_hole_in_a_key_position_beside_other_keys_is_refused(self):
        fsm = _fsm()
        fsm["states"]["start"] = {"$hole": "keys", "id": "start"}

        line = _only_problem(_doc(body=_body(fsm=fsm)))

        assert "fsm/states/start" in line and "never a key" in line

    def test_a_nested_hole_is_refused(self):
        fsm = _fsm()
        fsm["states"]["start"]["purpose"] = {"$hole": {"$hole": "keys"}}

        assert "never inside another hole" in _only_problem(_doc(body=_body(fsm=fsm)))

    def test_a_hole_holds_one_option_name(self):
        for bad in (3, ["keys"], {"x": 1}, "Not A Name", None):
            fsm = _fsm()
            fsm["states"]["start"]["purpose"] = {"$hole": bad}

            assert "one option name" in _only_problem(_doc(body=_body(fsm=fsm)))

    def test_a_hole_names_a_declared_option(self):
        fsm = _fsm()
        fsm["states"]["start"]["purpose"] = {"$hole": "missing"}

        assert "undeclared options ['missing']" in _only_problem(
            _doc(body=_body(fsm=fsm))
        )

    def test_the_root_of_the_template_is_never_a_hole(self):
        line = _only_problem(_doc(body=_body(fsm={"$hole": "keys"})))

        assert "root" in line

    def test_a_hole_inside_a_variant_value_is_checked_too(self):
        bad = copy.deepcopy(_ADD_STATE)
        bad["value"]["purpose"] = {"$hole": "ghost"}
        line = _only_problem(_doc(body=_body(variants={"v": [bad]})))
        assert "undeclared options ['ghost']" in line

        bad["value"]["purpose"] = {"$hole": "keys", "id": "x"}
        assert "variants/v" in _only_problem(_doc(body=_body(variants={"v": [bad]})))

    def test_a_variant_cannot_plant_a_hole_key_at_instantiation(self):
        # the op path names `$hole`: the parse-time scan sees no hole, the fill pass does
        plant = [{"op": "add", "path": "/states/start/$hole", "value": "keys"}]
        envelope = _envelope(body=_body(variants={"plant": plant}))

        with pytest.raises(ei.InstantiationError, match="malformed hole"):
            ei.instantiate_body(envelope, variants=["plant"], variables=TONE)

    def test_an_option_may_not_be_named_like_a_caller_input(self):
        doc = _doc(contract={"inputs": ["keys"]})

        assert "named like a contract input" in _only_problem(doc)


class TestTheOptionsSubset:
    def test_a_value_of_the_wrong_type_is_refused_with_the_option_named(self):
        envelope = _envelope()

        for given in ({"priority": "100"}, {"priority": 1.5}, {"priority": True}):
            with pytest.raises(ei.InstantiationError, match="option 'priority' is not"):
                ei.instantiate_body(envelope, given, variables=TONE)
        with pytest.raises(ei.InstantiationError, match="option 'keys' is not"):
            ei.instantiate_body(envelope, {"keys": "abc"}, variables=TONE)
        with pytest.raises(ei.InstantiationError, match="option 'keys' is not"):
            ei.instantiate_body(envelope, {"keys": [1]}, variables=TONE)

    def test_an_unknown_option_is_refused(self):
        with pytest.raises(ei.InstantiationError, match="unknown option"):
            ei.instantiate_body(_envelope(), {"approval": True}, variables=TONE)

    def test_a_value_above_or_below_its_bound_is_refused(self):
        envelope = _envelope()

        for given in (
            {"priority": 901},
            {"priority": 0},
            {"keys": ["a", "b", "c", "d", "e"]},
        ):
            with pytest.raises(ei.InstantiationError, match="is not valid"):
                ei.instantiate_body(envelope, given, variables=TONE)
        assert ei.instantiate_body(envelope, {"priority": 900}, variables=TONE)

    def test_enum_text_length_number_and_boolean(self):
        schema = {
            "properties": {
                "mode": {"type": "string", "enum": ["a", "b"]},
                "name": {"type": "string", "maxLength": 3},
                "ratio": {"type": "number", "minimum": 0, "maximum": 1},
                "flag": {"type": "boolean"},
            }
        }
        fsm = _fsm()
        fsm["states"]["start"]["purpose"] = {"$hole": "name"}
        fsm["states"]["start"]["transitions"][0]["priority"] = 1
        fsm["states"]["start"].pop("required_context_keys")
        envelope = _envelope(
            body=_body(fsm=fsm, options_schema=schema, options={}, variants={})
        )
        base = {"mode": "a", "name": "abc", "ratio": 0.5, "flag": False}

        assert (
            ei.instantiate_body(envelope, base, variables=TONE)["states"]["start"][
                "purpose"
            ]
            == "abc"
        )
        for bad in (
            {"mode": "c"},
            {"mode": 1},
            {"name": "abcd"},
            {"ratio": 1.5},
            {"ratio": True},
            {"flag": 1},
        ):
            with pytest.raises(ei.InstantiationError, match="is not valid"):
                ei.instantiate_body(envelope, {**base, **bad}, variables=TONE)
        assert ei.instantiate_body(envelope, {**base, "ratio": 1}, variables=TONE)

    def test_an_option_with_no_default_is_required(self):
        envelope = _envelope(body=_body(options={"priority": 100}))

        with pytest.raises(ei.InstantiationError, match=r"\['keys'\] have no value"):
            ei.instantiate_body(envelope, variables=TONE)
        assert ei.instantiate_body(envelope, {"keys": ["k"]}, variables=TONE)

    def test_a_default_that_breaks_its_own_schema_is_refused_at_parse(self):
        assert "option 'priority' is not valid" in _only_problem(
            _doc(body=_body(options={"keys": [], "priority": 5000}))
        )
        assert "undeclared options" in _only_problem(
            _doc(body=_body(options={"keys": [], "priority": 1, "ghost": 1}))
        )

    @pytest.mark.parametrize(
        "spec",
        [
            {"type": "object"},
            {"type": "string", "$ref": "#/x"},
            {"type": "integer", "maxLength": 3},
            {"type": "string", "minimum": 1},
            {"type": "integer", "minimum": 5, "maximum": 1},
            {"type": "boolean", "enum": [True]},
            {"type": "array"},
            {"type": "array", "items": {"type": "array"}},
            {"type": "array", "items": {"type": "string"}, "enum": ["a"]},
            {"type": "string", "items": {"type": "string"}},
            {"type": "string", "maxLength": 10**9},
            {"type": "array", "items": {"type": "string"}, "maxItems": 10**6},
            {"type": "integer", "default": 1},
        ],
    )
    def test_the_subset_is_closed(self, spec):
        body = _body(
            options_schema={
                "properties": {"keys": spec, "priority": {"type": "integer"}}
            },
            options={},
        )

        _refused(_doc(body=body))

    def test_an_option_name_is_a_plain_name(self):
        body = _body(options_schema={"properties": {"Bad Name": {"type": "integer"}}})

        _refused(_doc(body=body))

    def test_the_given_options_are_never_mutated(self):
        envelope = _envelope()
        given = {"keys": ["a", "b"], "priority": 7}
        snapshot = copy.deepcopy(given)

        built = ei.instantiate_body(envelope, given, variables=TONE)
        built["states"]["start"]["required_context_keys"].append("z")

        assert given == snapshot


class TestTheOrderOfOperations:
    def test_variants_come_first_so_a_hole_and_a_variable_they_add_are_filled(self):
        spec = _body()
        spec["variants"] = {"gate": [_ADD_STATE]}
        envelope = _envelope(body=spec)

        built = ei.instantiate_body(
            envelope, {"keys": ["k1"]}, variants=["gate"], variables=TONE
        )

        gate = built["states"]["gate"]
        assert gate["required_context_keys"] == ["k1"]
        assert (
            gate["description"] == "gate calm"
            and gate["response_instructions"] == "wait calm"
        )

    def test_a_variant_may_replace_a_hole_leaf(self):
        replace = [
            {
                "op": "replace",
                "path": "/states/start/transitions/0/priority",
                "value": 3,
            }
        ]
        envelope = _envelope(body=_body(variants={"fixed": replace}))

        built = ei.instantiate_body(envelope, variants=["fixed"], variables=TONE)

        assert built["states"]["start"]["transitions"][0]["priority"] == 3

    def test_variants_apply_in_declaration_order_whatever_the_selection_order(self):
        envelope = _envelope()

        forward = ei.instantiate_body(
            envelope, variants=["first", "second"], variables=TONE
        )
        backward = ei.instantiate_body(
            envelope, variants=["second", "first"], variables=TONE
        )

        assert forward == backward
        assert forward["states"]["gate"]["response_instructions"] == "waited"
        with pytest.raises(ei.InstantiationError, match="variant 'second' was refused"):
            ei.instantiate_body(envelope, variants=["second"], variables=TONE)

    def test_an_undeclared_variant_is_refused_and_no_option_selects_one(self):
        envelope = _envelope()

        with pytest.raises(ei.InstantiationError, match="unknown variant"):
            ei.instantiate_body(envelope, variants=["approval"], variables=TONE)
        with pytest.raises(ei.InstantiationError, match="unknown option"):
            ei.instantiate_body(envelope, {"variants": ["first"]}, variables=TONE)
        with pytest.raises(ei.InstantiationError, match="unknown option"):
            ei.instantiate_body(envelope, {"first": True}, variables=TONE)

    def test_a_hole_value_is_never_rescanned_for_text_variables(self):
        schema = {"properties": {"label": {"type": "string"}}}
        fsm = _fsm()
        fsm["states"]["start"]["purpose"] = {"$hole": "label"}
        fsm["states"]["start"]["description"] = "about {{tone}}"
        fsm["states"]["start"].pop("required_context_keys")
        fsm["states"]["start"]["transitions"][0]["priority"] = 1
        envelope = _envelope(
            body=_body(fsm=fsm, options_schema=schema, options={}, variants={})
        )

        built = ei.instantiate_body(
            envelope, {"label": "keep {{tone}} here"}, variables=TONE
        )

        assert built["states"]["start"]["purpose"] == "keep {{tone}} here"
        assert built["states"]["start"]["description"] == "about calm"

    def test_variables_are_checked_as_for_every_kind(self):
        envelope = _envelope()

        with pytest.raises(ei.InstantiationError, match="tone"):
            ei.instantiate_body(envelope)
        with pytest.raises(ei.InstantiationError, match="color"):
            ei.instantiate_body(envelope, variables={**TONE, "color": "red"})


class TestTheInstanceIsAFreshOrderedDict:
    def test_key_order_after_replace_and_add_equals_the_documents(self):
        envelope = _envelope()

        built = ei.instantiate_body(
            envelope, variants=["first", "second"], variables=TONE
        )

        assert list(built) == [
            "name",
            "description",
            "initial_state",
            "persona",
            "states",
        ]
        assert list(built["states"]) == ["start", "done", "gate"]  # add appends
        assert list(built["states"]["start"]) == list(_fsm()["states"]["start"])
        assert list(built["states"]["gate"]) == [
            "id",
            "description",
            "purpose",
            "response_instructions",
            "transitions",
        ]
        text = json.dumps(built)
        assert (
            text.index('"states": {"start"')
            < text.index('"done": {')
            < text.index('"gate": {')
        )

    def test_keys_that_sort_differently_keep_the_documents_order(self):
        fsm = _fsm()
        fsm["states"] = {k: fsm["states"][k] for k in ("start", "done")}
        reordered = {"zeta": 1, "alpha": 2, "mid": 3}
        fsm["states"]["start"]["metadata"] = reordered
        envelope = _envelope(body=_body(fsm=fsm, variants={}))

        built = ei.instantiate_body(envelope, variables=TONE)

        assert list(built["states"]["start"]["metadata"]) == ["zeta", "alpha", "mid"]
        assert list(built["states"]) == ["start", "done"]

    def test_two_calls_give_equal_independent_dicts(self):
        envelope = _envelope()
        template = copy.deepcopy(envelope.body.fsm)

        first = ei.instantiate_body(
            envelope, {"keys": ["a"]}, variants=["first"], variables=TONE
        )
        second = ei.instantiate_body(
            envelope, {"keys": ["a"]}, variants=["first"], variables=TONE
        )

        assert first == second and first is not second
        first["states"]["start"]["required_context_keys"].append("x")
        first["states"]["start"]["transitions"][0]["priority"] = -1
        assert second["states"]["start"]["required_context_keys"] == ["a"]
        assert envelope.body.fsm == template

    def test_a_hole_used_twice_gives_two_independent_values(self):
        fsm = _fsm()
        fsm["states"]["done"]["required_context_keys"] = {"$hole": "keys"}
        envelope = _envelope(body=_body(fsm=fsm, variants={}))

        built = ei.instantiate_body(envelope, {"keys": ["a"]}, variables=TONE)
        built["states"]["start"]["required_context_keys"].append("b")

        assert built["states"]["done"]["required_context_keys"] == ["a"]

    def test_the_walker_does_not_recurse(self):
        deep: Any = "leaf"
        for _ in range(20_000):
            deep = [deep]

        out = ep.fill_holes({"a": deep}, {}, str.upper)

        node = out["a"]
        for _ in range(20_000):
            node = node[0]
        assert node == "LEAF"

    def test_the_instance_is_an_ordinary_loadable_definition(self):
        envelope = _envelope()

        definition, _ = ei.instantiate_envelope(
            envelope,
            options={"keys": ["k"]},
            variants=["first", "second"],
            variables=TONE,
        )

        assert isinstance(definition, FSMDefinition)
        assert definition.states["gate"].response_instructions == "waited"


class TestTheTwoIdentities:
    def test_template_identity_never_moves_with_options_or_variants(self):
        envelope = _envelope()
        a, _ = ei.instantiate_envelope(
            envelope, options={"priority": 5}, variables=TONE
        )
        b, _ = ei.instantiate_envelope(
            envelope, options={"priority": 6}, variants=["first"], variables=TONE
        )

        ia, ib = ei.identity_of(envelope, a), ei.identity_of(envelope, b)

        assert ia.template_identity == ib.template_identity
        assert ia.instance_identity != ib.instance_identity

    def test_the_instance_identity_is_the_definition_digest_of_the_instance(self):
        envelope = _envelope()
        built = ei.instantiate_body(envelope, {"priority": 9}, variables=TONE)
        definition = FSMDefinition(**built)

        identity = ei.identity_of(envelope, built)

        assert identity.instance_identity == definition_digest(definition)
        assert ei.identity_of(envelope, definition).instance_identity == (
            identity.instance_identity
        )

    def test_an_option_that_changes_nothing_changes_no_identity(self):
        schema = {"properties": {"unused": {"type": "integer"}}}
        fsm = _fsm()
        fsm["states"]["start"].pop("required_context_keys")
        fsm["states"]["start"]["transitions"][0]["priority"] = 1
        envelope = _envelope(
            body=_body(
                fsm=fsm, options_schema=schema, options={"unused": 1}, variants={}
            )
        )

        a = ei.instantiate_body(envelope, {"unused": 1}, variables=TONE)
        b = ei.instantiate_body(envelope, {"unused": 2}, variables=TONE)

        assert ei.identity_of(envelope, a) == ei.identity_of(envelope, b)

    def test_the_template_identity_reads_the_template_not_the_key_order_of_the_file(
        self,
    ):
        doc = _doc()
        flipped = {k: doc[k] for k in reversed(list(doc))}
        flipped["body"] = {k: doc["body"][k] for k in reversed(list(doc["body"]))}

        assert ei.identity_of(ex.parse_executable(doc)).template_identity == (
            ei.identity_of(ex.parse_executable(flipped)).template_identity
        )
        assert ei.identity_of(_envelope(id="other/id")).template_identity != (
            ei.identity_of(_envelope()).template_identity
        )

    def test_a_variant_or_a_default_changes_the_template_identity(self):
        base = ei.identity_of(_envelope()).template_identity
        changed_default = ei.identity_of(
            _envelope(body=_body(options={"keys": [], "priority": 101}))
        ).template_identity
        spec = _body()
        spec["variants"]["first"][0]["value"]["purpose"] = "other"
        changed_variant = ei.identity_of(_envelope(body=spec)).template_identity

        assert len({base, changed_default, changed_variant}) == 3

    def test_a_generator_pattern_has_a_template_identity(self):
        envelope = _envelope(body={"generator": "agents.prompt_chain"})

        assert ei.identity_of(envelope).instance_identity is None
        assert len(ei.identity_of(envelope).template_identity) == 64


class TestTheRecordReplays:
    def test_the_record_holds_the_resolved_options_and_the_variants(self):
        envelope = _envelope()

        _, record = ei.instantiate_envelope(
            envelope,
            options={"priority": 7},
            variants=["second", "first"],
            variables=TONE,
        )

        assert record == {
            "options": {"keys": [], "priority": 7},
            "variables": TONE,
            "variants": ["first", "second"],
        }

    def test_rebuild_instance_round_trips(self):
        envelope = _envelope()
        definition, record = ei.instantiate_envelope(
            envelope,
            options={"keys": ["k"], "priority": 7},
            variants=["first"],
            variables=TONE,
        )
        expected = ei.identity_of(envelope, definition).instance_identity
        assert expected is not None

        rebuilt = ei.rebuild_instance(envelope, record, expected)

        assert definition_digest(rebuilt) == expected

    def test_a_tampered_record_is_refused(self):
        envelope = _envelope()
        definition, record = ei.instantiate_envelope(
            envelope, options={"priority": 7}, variants=["first"], variables=TONE
        )
        expected = ei.identity_of(envelope, definition).instance_identity
        assert expected is not None

        for key, value in (
            ("options", {**record["options"], "priority": 8}),
            ("variants", []),
            ("variables", {"tone": "loud"}),
        ):
            with pytest.raises(ei.InstanceIdentityMismatchError):
                ei.rebuild_instance(envelope, {**record, key: value}, expected)
        with pytest.raises(ex.ExecutableError):
            ei.rebuild_instance(envelope, {**record, "variants": "first"}, expected)
        with pytest.raises(ex.ExecutableError):
            ei.rebuild_instance(envelope, {**record, "extra": 1}, expected)

    def test_a_record_with_a_missing_option_cannot_be_replayed(self):
        envelope = _envelope(body=_body(options={}))

        with pytest.raises(ei.InstantiationError, match="have no value"):
            ei.rebuild_instance(
                envelope, {"options": {"priority": 7}, "variables": TONE}, "0" * 64
            )

    def test_an_fsm_record_keeps_its_shape_and_refuses_variants(self):
        doc = {
            "schema": "fsm-llm/executable",
            "kind": "fsm",
            "id": "plain",
            "body": {"fsm": ei.substitute_variables(_fsm(), {})},
        }
        doc["body"]["fsm"]["states"]["start"].pop("required_context_keys")
        doc["body"]["fsm"]["states"]["start"]["transitions"][0]["priority"] = 1
        envelope = ex.parse_executable(doc)

        _, record = ei.instantiate_envelope(envelope)

        assert record == {"options": {}, "variables": {}}
        with pytest.raises(ei.InstantiationError, match="variants"):
            ei.instantiate_envelope(envelope, variants=["first"])
        with pytest.raises(ei.InstantiationError, match="options"):
            ei.instantiate_body(envelope)


class TestTheRunBlock:
    def test_a_run_block_parses_with_bindings_and_hooks(self):
        run = {
            "agent_type": "react",
            "answer_keys": ["final_answer"],
            "success_keys": ["final_answer"],
            "evidence_keys": ["tool_results"],
            "judged": True,
            "run_output_keys": ["final_answer"],
            "context_seed": {"observations": [], "mode": "x", "n": 3},
            "step_budget": 40,
            "seconds_exempt": ["act"],
            "hooks": [{"hook": "agents.verify_retry", "at": "after_run"}],
        }
        handlers = [
            {"handler": "agents.tool_executor", "on": "entry", "states": ["act"]},
            {
                "handler": "agents.k",
                "on": "context_update",
                "keys": ["a"],
                "priority": 50,
                "params": {"limit": 3},
            },
        ]

        envelope = _envelope(body=_body(run=run, handlers=handlers))

        assert envelope.body.run.hooks[0].at == "after_run"
        assert envelope.body.handlers[1].params == {"limit": 3}

    @pytest.mark.parametrize(
        "seed",
        [
            {"_approval_granted": True},
            {"system_note": "x"},
            {"internal_x": 1},
            {"api_key": "sk-1234567890abcdef1234567890"},
        ],
    )
    def test_a_file_can_never_seed_a_driver_key_or_a_secret(self, seed):
        line = _only_problem(_doc(body=_body(run={"context_seed": seed})))

        assert "context_seed key" in line

    def test_a_binding_or_hook_with_an_unknown_position_is_refused(self):
        _refused(_doc(body=_body(handlers=[{"handler": "h", "on": "sideways"}])))
        _refused(_doc(body=_body(run={"hooks": [{"hook": "h", "at": "never"}]})))
        _refused(_doc(body=_body(run={"step_budget": 0})))


class TestUntilTheLoadSequenceExists:
    def test_no_route_loads_a_pattern_document_yet(self):
        for origin in ("operator", "model"):
            envelope, result = load_executable(_doc(), origin=origin)

            assert envelope is None
            assert not result.is_valid
            assert len(result.errors) == 1 and "[schema]" in result.errors[0]


class TestTheShippedSchemaNamesThePattern:
    def test_the_schema_has_the_pattern_branch_and_the_hole_text(self):
        schema = ex.executable_json_schema()

        assert len(schema["oneOf"]) == 4
        assert "pattern" in schema["discriminator"]["mapping"]
        body = schema["$defs"]["PatternBody"]
        assert set(body["properties"]) == {
            "fsm",
            "generator",
            "handlers",
            "options_schema",
            "options",
            "variants",
            "run",
        }
        assert "maxLength" in schema["$defs"]["OptionSpec"]["properties"]
