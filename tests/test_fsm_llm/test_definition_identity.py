"""The identity recipe of an FSM definition is ONE core function (plan ace7a8c8, F1.1).

``fsm_id_of`` (the id ``API`` mints), ``definition_digest`` (the 64-hex content
digest) and ``definition_canonical`` (the digest with its canonical text) live in
``fsm_llm.core.definitions``. Before the move the recipe stood inline in
``API.process_fsm_definition`` and again in ``FSMLibrary``. The literal recipe
below is written out here on purpose: it is the independent reference every row
compares against, so a change of the recipe fails here before it re-ids a
conversation.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from fsm_llm import API
from fsm_llm.core.definitions import (
    FSMDefinition,
    definition_canonical,
    definition_digest,
    fsm_id_of,
)
from fsm_llm.synthesis import FSMLibrary, digest_of
from tests.test_fsm_llm import kind_corpus as kc

_SRC = Path(__file__).resolve().parents[2] / "src" / "fsm_llm"


def _definitions() -> dict[str, dict[str, Any]]:
    handler_only = kc._fsm(
        "HandlerOnly",
        "a",
        kc._state("a", handler_only_keys=["secret_note"], transitions=[kc._go("b")]),
        kc._state("b", response_instructions="Bye."),
    )
    plain = kc._fsm(
        "Plain",
        "a",
        kc._state("a", transitions=[kc._go("b")]),
        kc._state("b", response_instructions="Bye."),
    )
    return {
        "plain": plain,
        "classification": kc._plain_fsm(),
        "completion": kc._tool_fsm(),
        "synthesize": kc._synth_fsm(),
        "handler_only_keys": handler_only,
    }


_DEFINITIONS = _definitions()


def _literal(definition: FSMDefinition) -> tuple[str, str]:
    """The recipe as it stood inline in ``API`` and ``FSMLibrary``: (digest, text)."""
    text = json.dumps(definition.model_dump(exclude_defaults=True), sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest(), text


@pytest.mark.parametrize("name", sorted(_DEFINITIONS))
class TestTheRecipe:
    def test_the_canonical_pair_is_the_literal_recipe(self, name):
        definition = FSMDefinition(**_DEFINITIONS[name])
        digest, text = _literal(definition)
        assert definition_canonical(definition) == (digest, text)
        assert definition_digest(definition) == digest
        assert len(digest) == 64
        # the digest is of the very text returned, never of a re-dumped copy
        assert hashlib.sha256(text.encode()).hexdigest() == digest

    def test_the_id_is_name_and_eight_hex_of_the_digest(self, name):
        definition = FSMDefinition(**_DEFINITIONS[name])
        digest, _ = _literal(definition)
        assert fsm_id_of(definition) == f"fsm_{definition.name}_{digest[:8]}"

    def test_api_gives_the_same_id_for_a_dict_an_object_and_a_file(
        self, name, tmp_path
    ):
        data = _DEFINITIONS[name]
        path = tmp_path / "fsm.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        definition = FSMDefinition(**data)
        ids = {
            API.process_fsm_definition(data)[1],
            API.process_fsm_definition(definition)[1],
            API.process_fsm_definition(str(path))[1],
        }
        assert ids == {fsm_id_of(definition)}

    def test_the_library_keys_and_stores_the_core_digest_and_text(self, name, tmp_path):
        definition = FSMDefinition(**_DEFINITIONS[name])
        digest, text = definition_canonical(definition)
        assert digest_of(definition) == digest
        library = FSMLibrary(tmp_path / "lib")
        record = library.put(definition, goal="g", inputs_required=[], return_keys=[])
        assert record.digest == digest
        stored = (tmp_path / "lib" / "definitions" / f"{digest}.json").read_bytes()
        assert stored == text.encode()


class TestDefaultsAreNotHashed:
    """D-030 of plan 944e2692: a field at its default, given or omitted, is not in the hash."""

    def test_explicit_default_equals_omitted_and_a_non_default_differs(self):
        base = _DEFINITIONS["plain"]
        default = FSMDefinition.model_fields["version"].default
        omitted = definition_digest(FSMDefinition(**base))
        explicit = definition_digest(FSMDefinition(**base, version=default))
        other = definition_digest(FSMDefinition(**base, persona="A pirate"))
        assert explicit == omitted
        assert other != omitted

    def test_a_changed_prompt_changes_the_id_and_the_digest(self):
        base = FSMDefinition(**_DEFINITIONS["plain"])
        changed = base.model_copy(deep=True)
        changed.states["b"].response_instructions = "Goodbye."
        assert definition_digest(changed) != definition_digest(base)
        assert fsm_id_of(changed) != fsm_id_of(base)


# The recipe has no second copy: a function that hashes a definition's default-free
# dump is the one in core, and the detector below is proven on both shapes.
_INLINE_RECIPE = (
    "def f(d):\n"
    "    return hashlib.sha256(json.dumps(d.model_dump(exclude_defaults=True),"
    " sort_keys=True).encode()).hexdigest()\n"
)
_SPLIT_RECIPE = (
    "def g(d):\n"
    "    text = json.dumps(d.model_dump(exclude_defaults=True), sort_keys=True)\n"
    "    return hashlib.sha256(text.encode()).hexdigest(), text\n"
)
_OTHER_HASH = (
    "def h(d, ops):\n"
    "    d.model_dump(exclude_defaults=True)\n"
    "    return hashlib.sha256(json.dumps(ops, sort_keys=True).encode()).hexdigest()\n"
)


def _is_default_free_dump(call: ast.AST) -> bool:
    return (
        isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "model_dump"
        and any(
            k.arg == "exclude_defaults"
            and isinstance(k.value, ast.Constant)
            and k.value.value is True
            for k in call.keywords
        )
    )


def _inline_recipes(source: str) -> list[str]:
    """Functions of *source* that call ``sha256`` and also ``json.dumps`` a
    ``model_dump(exclude_defaults=True)`` (the old recipe, however the two are split)."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        calls = [n for n in ast.walk(node) if isinstance(n, ast.Call)]
        hashes = any(
            isinstance(c.func, ast.Attribute) and c.func.attr == "sha256" for c in calls
        )
        dumps = any(
            isinstance(c.func, ast.Attribute)
            and c.func.attr == "dumps"
            and any(_is_default_free_dump(n) for n in ast.walk(c))
            for c in calls
        )
        if hashes and dumps:
            found.append(node.name)
    return found


class TestOneRecipe:
    def test_the_detector_sees_both_old_shapes_and_not_a_call_to_core(self):
        assert _inline_recipes(_INLINE_RECIPE) == ["f"]
        assert _inline_recipes(_SPLIT_RECIPE) == ["g"]
        assert _inline_recipes(_OTHER_HASH) == []
        assert _inline_recipes("def f(d):\n    return fsm_id_of(d)\n") == []

    def test_no_module_outside_core_hashes_a_definition_dump(self):
        offenders = []
        for path in sorted(_SRC.rglob("*.py")):
            if path.relative_to(_SRC).parts[0] == "core":
                continue
            names = _inline_recipes(path.read_text(encoding="utf-8"))
            offenders += [f"{path.relative_to(_SRC).as_posix()}::{n}" for n in names]
        assert offenders == []

    def test_the_decision_anchors_stand_where_the_text_now_lives(self):
        core = (_SRC / "core" / "definitions" / "fsm.py").read_text(encoding="utf-8")
        api = (_SRC / "engine" / "api" / "conversation.py").read_text(encoding="utf-8")
        assert "plan-2026-10-01T093600-944e2692/D-030" in core
        assert "plan-2026-09-20T114608-a8e47b88/D-016" in api
