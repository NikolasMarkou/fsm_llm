"""The identity recipe of an FSM definition is ONE core function (plan ace7a8c8, F1).

``fsm_id_of``, ``definition_digest`` and ``definition_canonical`` live in
``fsm_llm.core.definitions``; the recipe once stood inline in ``API`` and in
``FSMLibrary``. ``_literal`` below writes it out on purpose: the independent
reference every row compares with.
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
from tests import executables_corpus as corpus
from tests.test_fsm_llm import kind_corpus as kc

_SRC = Path(__file__).resolve().parents[2] / "src" / "fsm_llm"
_CORPUS = corpus.fsm_cases()


def _two_states(name: str, **first: Any) -> dict[str, Any]:
    return kc._fsm(
        name,
        "a",
        kc._state("a", transitions=[kc._go("b")], **first),
        kc._state("b", response_instructions="Bye."),
    )


_DEFINITIONS = {
    "plain": _two_states("Plain"),
    "classification": kc._plain_fsm(),
    "completion": kc._tool_fsm(),
    "synthesize": kc._synth_fsm(),
    "handler_only_keys": _two_states("HandlerOnly", handler_only_keys=["note"]),
}


def _literal(definition: FSMDefinition) -> tuple[str, str]:
    """The recipe as it stood inline: (digest, canonical text)."""
    text = json.dumps(definition.model_dump(exclude_defaults=True), sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest(), text


@pytest.mark.parametrize("name", sorted(_DEFINITIONS))
class TestTheRecipe:
    def test_the_pair_and_the_id_are_the_literal_recipe(self, name):
        definition = FSMDefinition(**_DEFINITIONS[name])
        digest, text = _literal(definition)
        assert definition_canonical(definition) == (digest, text)
        assert definition_digest(definition) == digest
        assert fsm_id_of(definition) == f"fsm_{definition.name}_{digest[:8]}"
        assert len(digest) == 64

    def test_api_gives_one_id_for_a_dict_an_object_and_a_file(self, name, tmp_path):
        data = _DEFINITIONS[name]
        path = tmp_path / "fsm.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        definition = FSMDefinition(**data)
        got = {
            API.process_fsm_definition(arg)[1] for arg in (data, definition, str(path))
        }
        assert got == {fsm_id_of(definition)}

    def test_the_library_keys_and_stores_the_core_digest_and_text(self, name, tmp_path):
        definition = FSMDefinition(**_DEFINITIONS[name])
        digest, text = definition_canonical(definition)
        record = FSMLibrary(tmp_path).put(
            definition, goal="g", inputs_required=[], return_keys=[]
        )
        stored = tmp_path / "definitions" / f"{digest}.json"
        assert (digest_of(definition), record.digest) == (digest, digest)
        assert stored.read_bytes() == text.encode()


@pytest.mark.parametrize("case", sorted(_CORPUS))
def test_the_core_recipe_is_the_literal_recipe_over_every_builtin(case):
    definition = FSMDefinition(**_CORPUS[case])
    digest, text = _literal(definition)
    assert definition_canonical(definition) == (digest, text)
    assert fsm_id_of(definition) == f"fsm_{definition.name}_{digest[:8]}"


class TestDefaultsAreNotHashed:
    """D-030 of plan 944e2692: a field at its default, given or omitted, is not hashed."""

    def test_explicit_default_equals_omitted_and_a_non_default_differs(self):
        base = _DEFINITIONS["plain"]
        default = FSMDefinition.model_fields["version"].default
        omitted = definition_digest(FSMDefinition(**base))
        assert definition_digest(FSMDefinition(**base, version=default)) == omitted
        assert definition_digest(FSMDefinition(**base, persona="A pirate")) != omitted


_OLD_SHAPES = {
    "one expression": "def f(d):\n return hashlib.sha256(json.dumps("
    "d.model_dump(exclude_defaults=True), sort_keys=True).encode()).hexdigest()\n",
    "split": "def f(d):\n text = json.dumps(d.model_dump(exclude_defaults=True))\n"
    " return hashlib.sha256(text.encode()).hexdigest(), text\n",
}
_NOT_THE_RECIPE = (
    "def f(d, ops):\n d.model_dump(exclude_defaults=True)\n"
    " return hashlib.sha256(json.dumps(ops).encode()).hexdigest()\n"
)


def _named(call: ast.AST, attr: str) -> bool:
    return (
        isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == attr
    )


def _default_free_dump(node: ast.AST) -> bool:
    return _named(node, "model_dump") and any(
        k.arg == "exclude_defaults" for k in node.keywords
    )


def _inline_recipes(source: str) -> list[str]:
    """Functions that call ``sha256`` and ``json.dumps`` a default-free ``model_dump``."""
    found = []
    for fn in ast.walk(ast.parse(source)):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call)]
        hashes = any(_named(c, "sha256") for c in calls)
        dumps = any(
            _named(c, "dumps") and any(_default_free_dump(n) for n in ast.walk(c))
            for c in calls
        )
        if hashes and dumps:
            found.append(fn.name)
    return found


class TestOneRecipe:
    def test_the_detector_sees_both_old_shapes_and_not_other_hashing(self):
        assert {k: _inline_recipes(v) for k, v in _OLD_SHAPES.items()} == {
            "one expression": ["f"],
            "split": ["f"],
        }
        assert _inline_recipes(_NOT_THE_RECIPE) == []

    def test_no_module_outside_core_hashes_a_definition_dump(self):
        offenders = [
            f"{path.relative_to(_SRC).as_posix()}::{name}"
            for path in sorted(_SRC.rglob("*.py"))
            if path.relative_to(_SRC).parts[0] != "core"
            for name in _inline_recipes(path.read_text(encoding="utf-8"))
        ]
        assert offenders == []

    def test_the_decision_anchors_stand_where_the_text_now_lives(self):
        core = (_SRC / "core" / "definitions" / "fsm.py").read_text(encoding="utf-8")
        api = (_SRC / "engine" / "api" / "conversation.py").read_text(encoding="utf-8")
        assert "plan-2026-10-01T093600-944e2692/D-030" in core
        assert "plan-2026-09-20T114608-a8e47b88/D-016" in api
