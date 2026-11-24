"""The generator seam of a ``pattern`` body (plan a555dcfe step 11, D-019).

A pattern whose FSM depends on its input declares ``generator: <name>`` instead of ``fsm``. Core
imports no namespace, so the caller passes a lookup ``name -> function | None``; the function gets
the RESOLVED options and nothing else, and what it writes is checked (a mapping, within the depth
cap, a valid FSM) before anything runs it. These rows hold the seam, V2 on the default instance,
the replay of a record and the V7 refusal of a generator in a document a model wrote.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from fsm_llm.core.definitions import executable as ex
from fsm_llm.core.definitions import executable_instance as ei
from fsm_llm.core.definitions.fsm import FSMDefinition, definition_digest
from fsm_llm.core.verify.executable import load_executable
from tests.test_fsm_llm import kind_corpus as kc

_SECRET = "an option value that must never reach an error"


def _write(options: Any) -> dict[str, Any]:
    """A generator: one state per ``names`` entry, a linear chain ending in ``end``."""
    names = options["names"]
    if not names:
        raise ValueError(_SECRET)
    states = [
        kc._state(
            name,
            response_instructions=f"say {name}",
            transitions=[kc._go(names[i + 1] if i + 1 < len(names) else "end")],
        )
        for i, name in enumerate(names)
    ]
    return kc._fsm("Chain", names[0], *states, kc._state("end"))


def _lookup(name: str) -> Any:
    return {"test.chain": _write}.get(name)


def _doc(**body: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "generator": "test.chain",
        "options_schema": {
            "properties": {
                "names": {"type": "array", "items": {"type": "string"}, "maxItems": 5}
            }
        },
        "options": {"names": ["a", "b"]},
    }
    base.update(body)
    return {
        "schema": "fsm-llm/executable",
        "kind": "pattern",
        "id": "demo/generator",
        "body": base,
    }


def _envelope(**body: Any) -> ex.PatternEnvelope:
    envelope = ex.parse_executable(_doc(**body))
    assert isinstance(envelope, ex.PatternEnvelope)
    return envelope


class TestAGeneratorWritesTheFsmFromTheResolvedOptions:
    def test_the_default_and_a_given_option_each_write_their_own_fsm(self):
        envelope = _envelope()

        default = ei.instantiate_body(envelope, generators=_lookup)
        given = ei.instantiate_body(envelope, {"names": ["x"]}, generators=_lookup)

        assert list(default["states"]) == ["a", "b", "end"]
        assert list(given["states"]) == ["x", "end"]

    def test_two_calls_are_equal_and_share_nothing(self):
        envelope = _envelope()
        first = ei.instantiate_body(envelope, generators=_lookup)
        second = ei.instantiate_body(envelope, generators=_lookup)

        assert first == second
        first["states"]["a"]["purpose"] = "changed"
        assert second["states"]["a"]["purpose"] != "changed"

    def test_the_generator_gets_a_copy_of_the_options(self):
        seen: list[Any] = []

        def spy(options: Any) -> dict[str, Any]:
            seen.append(options)
            options["names"].append(
                "mutated"
            )  # must not reach the envelope or the record
            return _write({"names": ["a"]})

        envelope = _envelope()
        definition, record = ei.instantiate_envelope(envelope, generators=lambda n: spy)

        assert seen and record["options"] == {"names": ["a", "b"]}
        assert envelope.body.options == {"names": ["a", "b"]}
        assert isinstance(definition, FSMDefinition)

    def test_the_record_replays_to_the_same_instance_identity(self):
        envelope = _envelope()
        definition, record = ei.instantiate_envelope(
            envelope, options={"names": ["p", "q", "r"]}, generators=_lookup
        )

        again = ei.rebuild_instance(
            envelope, record, definition_digest(definition), generators=_lookup
        )

        assert definition_digest(again) == definition_digest(definition)

    def test_the_options_schema_still_judges_the_options(self):
        with pytest.raises(ei.InstantiationError):
            ei.instantiate_body(
                _envelope(), {"names": ["a"] * 6}, generators=_lookup
            )  # maxItems 5
        with pytest.raises(ei.InstantiationError):
            ei.instantiate_body(_envelope(), {"nope": 1}, generators=_lookup)

    def test_a_generator_takes_no_variant(self):
        with pytest.raises(ei.InstantiationError, match="no variants"):
            ei.instantiate_body(_envelope(), variants=("any",), generators=_lookup)


class TestAGeneratorIsCheckedBeforeAnythingRunsIt:
    @pytest.mark.parametrize("lookup", [None, lambda name: None])
    def test_no_lookup_or_an_unknown_name_fails_closed(self, lookup):
        with pytest.raises(ei.GeneratorUnavailableError, match="no lookup"):
            ei.instantiate_body(_envelope(), generators=lookup)

    def test_an_exception_of_the_generator_names_its_type_never_its_text(self):
        with pytest.raises(ei.InstantiationError) as caught:
            ei.instantiate_body(_envelope(), {"names": []}, generators=_lookup)

        assert "ValueError" in str(caught.value)
        assert _SECRET not in str(caught.value)

    @pytest.mark.parametrize(
        "written",
        [
            ["not", "a", "mapping"],
            {"name": "x"},  # a mapping that is no FSM
            {"states": {}},
        ],
    )
    def test_an_output_that_is_no_fsm_is_refused(self, written):
        with pytest.raises(ei.InstantiationError):
            ei.instantiate_body(_envelope(), generators=lambda n: lambda o: written)

    def test_an_output_nested_past_the_cap_is_refused(self):
        deep: Any = "leaf"
        for _ in range(ex.MAX_DOCUMENT_DEPTH + 2):
            deep = {"k": deep}

        with pytest.raises(ei.InstantiationError, match="levels"):
            ei.instantiate_body(_envelope(), generators=lambda n: lambda o: deep)

    def test_an_output_is_returned_as_a_copy_of_what_the_generator_kept(self):
        kept = _write({"names": ["a"]})

        out = ei.instantiate_body(_envelope(), generators=lambda n: lambda o: kept)
        out["states"]["a"]["purpose"] = "changed"

        assert kept["states"]["a"]["purpose"] != "changed"


class TestLoadRunsTheGeneratorOnItsDefaultInstance:
    def test_a_good_generator_loads_and_a_bad_default_does_not(self):
        envelope, report = load_executable(
            _doc(), resolver=lambda c, n: True, generators=_lookup
        )
        assert envelope is not None and report.is_valid, report.errors

        bad = _doc(options={"names": []})
        envelope, report = load_executable(
            bad, resolver=lambda c, n: True, generators=_lookup
        )
        assert envelope is None
        assert any("[body]" in line and "ValueError" in line for line in report.errors)

    def test_without_a_lookup_the_body_is_not_instantiated_and_says_so(self):
        envelope, report = load_executable(_doc(), resolver=lambda c, n: True)

        assert envelope is not None
        assert any("nothing to check" in line for line in report.info)

    def test_the_default_instance_is_judged_as_an_fsm(self):
        def broken(
            options: Any,
        ) -> dict[str, Any]:  # loads as an FSM, fails the validator
            return kc._fsm("Orphan", "a", kc._state("a"), kc._state("lost"))

        envelope, report = load_executable(
            _doc(), resolver=lambda c, n: True, generators=lambda n: broken
        )

        assert envelope is None
        assert report.errors


class TestAModelNeverNamesAGenerator:
    def test_v7_refuses_the_field_whatever_the_document_claims(self):
        doc = copy.deepcopy(_doc())
        doc["requires"] = {"tools": "none"}
        doc["capabilities"] = []
        doc["lifecycle"] = "draft"

        envelope, report = load_executable(
            doc, origin="model", resolver=lambda c, n: True, generators=_lookup
        )

        assert envelope is None
        assert any("generator" in line for line in report.errors)

        operator, ok = load_executable(
            doc, resolver=lambda c, n: True, generators=_lookup
        )
        assert operator is not None and ok.is_valid
